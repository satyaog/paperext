# Read sources and tree.
# Find source with closest hamming distance.
# Give 10 proposition to merge them.
# Otherwise move to next best match.
from __future__ import annotations

import argparse
import asyncio
import copy
import itertools
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from pprint import pprint
from typing import List, Tuple

import instructor
import Levenshtein
import openai
import pandas as pd
import pydantic_core
from openai.types.chat.chat_completion import CompletionUsage
from paperoni.config import load_config, papconf

from .download_convert import download_and_convert_paper
from .models.model_category import (
    _FIRST_MESSAGE,
    ExtractionResponse,
    ModelCategoryExtraction,
)
from .stats import load_analysis


def get_model_category_from_research_paper(
    message: str,
    model_names: list[str],
) -> Tuple[ModelCategoryExtraction, CompletionUsage]:
    """Extract Models, Datasets and Frameworks names from a research paper."""
    client = instructor.from_openai(openai.OpenAI())

    model_names = ",".join(f"`{model_name}`" for model_name in model_names)

    retries = [True] * 2
    while True:
        try:
            (
                extractions,
                completion,
            ) = client.chat.completions.create_with_completion(
                model="gpt-4o",
                response_model=ModelCategoryExtraction,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Your role is to categorize deep learning models and algorithms. "
                            "The model categories to which the model belongs. "
                            f"It may be values like {model_names} or other "
                            "model categories. "
                            "If the given name is not one of a model but rather of an algorithm, "
                            "it should have the category `algorithm`. Reinforcement learning "
                            "algorithms should also have the category `reinforcement learning`. "
                            "Optimization algorithms should also have the category `optimization`. "
                            "If the name is not one of a model nor an algorithm, then it should have "
                            "the category `others`."
                        ),
                        #  f"The Models, Datasets and Frameworks must be used in the paper "
                        #  f"and / or the comparison analysis of the results of the "
                        #  f"paper. The papers provided will be a convertion from pdf to text, which could imply some formatting issues.",
                    },
                    {
                        "role": "user",
                        "content": message,
                    },
                ],
                max_retries=2,
            )
            return extractions, completion.usage
        except openai.RateLimitError as e:
            time.sleep(60)
            if retries:
                retries.pop()
                continue
            raise e


def extract_model_categories(
    paper_text_path: Path, model_name: str, models: dict
) -> ModelCategoryExtraction:

    paper = paper_text_path.name

    count = 0
    for line in paper_text_path.read_text().splitlines():
        count += len([w for w in line.strip().split() if w])

    f = (papconf.paths.cache.parents[0] / "queries/models") / paper
    f = f.with_stem(f"{f.stem}_{model_name}").with_suffix(".json")

    try:
        response = ExtractionResponse.model_validate_json(f.read_text())
    except (FileNotFoundError, pydantic_core._pydantic_core.ValidationError):
        models = copy.deepcopy(models)
        models.pop("ignore")
        models["others"] = {}

        top_models = build_models_dataframe(truncate_dict(models, depth=2))
        top_models = top_models["model"].tolist()

        print(sorted(top_models))
        message = _FIRST_MESSAGE.format(
            model_name=model_name,
            paper_text=paper_text_path.read_text(),
        )

        extractions, usage = get_model_category_from_research_paper(message, top_models)

        response = ExtractionResponse(
            paper=paper,
            words=count,
            extractions=extractions,
            usage=usage,
        )

        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(response.model_dump_json(indent=2))

    return response.extractions


def get_paper_json(paper_title):

    from paperoni.sources.scrapers.semantic_scholar import SemanticScholarScraper

    try:
        for paper in SemanticScholarScraper(papconf, papconf.database).query(
            title=paper_title, limit=1
        ):
            return paper
    except KeyError:
        pass
    except Exception as e:
        print(traceback.format_exc())
        pass

    print("Could not find paper for title:")
    print(paper_title)
    corrected_title = input("Enter corrected title or leave blank to skip: ")
    if corrected_title:
        return get_paper_json(corrected_title)

    return


def get_external_paper_text(paper_title: str) -> Path | None:
    paper = get_paper_json(paper_title)
    if paper is None:
        print("Paperoni unable to find paper")
        return None

    text, _ = download_and_convert_paper(
        None,
        [link.dict() for link in paper.links],
        papconf.paths.cache,
        check_only=False,
    )

    if text is None:
        print("Could not download paper")
        return None

    return text


def get_model_description(
    paper_title: str, model_name: str, models: dict
) -> ModelCategoryExtraction | None:
    text = get_external_paper_text(paper_title)

    if text is None:
        print("Cannot get text")
        return None

    return extract_model_categories(text, model_name, models)


def build_models_dataframe(models):
    def build_models_tree(models, parent=None) -> List:
        rows = []
        for key in models:
            rows.append({"model": key})
            if models[key]:
                submodels = build_models_tree(models[key])
                for submodel in submodels:
                    submodel[f"parent{len(submodel)}"] = key
                rows.extend(submodels)

        return rows

    return pd.DataFrame(build_models_tree(models))


def is_probably_an_acronym(model_name):
    return len(model_name) < 5 and " " not in model_name


def levenshtein_for_sequence(remaining, model):
    remaining = remaining.replace("-", " ")
    model = model.replace("-", " ")
    dists = defaultdict(list)
    for a in remaining.split(" "):
        for b in model.split(" "):
            dists[b].append(Levenshtein.distance(a, b))

    dists = [(min(v), k) for k, v in dists.items()]
    sorted(dists)

    return sum(dists)


def get_proposition(remainings, models, k=5, df=None):
    if df is None:
        df = build_models_dataframe(models)
    propositions = []
    for remaining in remainings:
        remaining_model_is_probably_an_acronym = is_probably_an_acronym(remaining)
        distances = []
        for model in list(df["model"]):
            if remaining_model_is_probably_an_acronym and not is_probably_an_acronym(
                model
            ):
                distance = Levenshtein.distance(
                    remaining, "".join(m[0] for m in model.split(" "))
                )
            elif remaining_model_is_probably_an_acronym:
                continue
            else:
                distance = Levenshtein.distance(remaining, model)

            distances.append((distance, model))

        distances = sorted(distances)
        propositions.append(
            (distances[0][0], remaining) + tuple(d[1] for d in distances[:k])
        )

    # longest_first = [p for p in propositions if len(p[1]) > 3]
    # if longest_first:
    #     propositions = longest_first

    return sorted(propositions)


def get_model(model, models, df=None):
    if df is None:
        df = build_models_dataframe(models)

    try:
        model_row = df[df["model"] == model].iloc[0]
    except:
        import pdb

        pdb.set_trace()
    for parent in model_row[::-1]:
        if pd.isnull(parent):
            continue
        models = models[parent]

    return models


def truncate_dict(d, depth):
    if depth == 0:
        return {}

    if isinstance(d, dict):
        return {k: truncate_dict(v, depth - 1) for k, v in d.items()}
    else:
        return d


def get_paper_url(paper):
    for link in paper["links"]:
        if "url" not in link:
            continue
        if link["url"].endswith("pdf"):
            return link["url"]

    return


def get_paper_text(paper, folder):
    for link in paper["links"]:
        paper_path = (
            folder / "cache" / link["type"].replace(".pdf", "") / f"{link['link']}.txt"
        )
        if os.path.exists(paper_path):
            return open(paper_path).read()

    return


def get_paper_model_excerpt(paper, folder, model):
    text = get_paper_text(paper, folder)
    if not text:
        return []

    text = text.lower().replace("\n", " ")

    # TODO return more than one excerpt
    # return [text[text.find(model) - 75 : text.find(model) + 75]]
    return [text[m.start() - 75 : m.start() + 75] for m in re.finditer(model, text)]


def chose_proposition(
    remaining, propositions, paper_models, models, paperoni, df
) -> Tuple[str, str]:

    related_papers = paper_models[paper_models["name"] == remaining]
    papers_exploded = related_papers.explode("aliases")
    aliases = papers_exploded[papers_exploded["name"] == remaining]["aliases"]
    pd.options.display.max_rows = 999
    pd.options.display.max_colwidth = 300

    print()
    print("--------------------")
    print()
    print(f"Categorizing '{remaining}'")
    print()

    print("Concurrent alias:")
    # related_papers = paper_models[paper_models["paper_id"].isin(paper_ids)]
    # related_models = list(related_papers["name"].explode().unique())
    # related_models.remove(remaining)
    # model_names = related_papers["name"].explode()
    # counts = model_names[model_names != remaining].value_counts()
    counts = aliases.value_counts()
    counts /= related_papers.shape[0]
    print(
        counts[counts != remaining]
        .to_frame()
        .reset_index()
        .to_string(index=False, header=False)
    )

    print()
    print(related_papers.explode("research_fields")["research_fields"].value_counts())

    for paper_id in related_papers["paper_id"]:
        paper = [paper for paper in paperoni if paper["paper_id"] == paper_id]
        if not paper:
            import pdb

            pdb.set_trace()
        # print("\n".join(get_paper_model_excerpt(paper[0], Path("data"), remaining)))
        print(get_paper_url(paper[0]))
        print(paper[0]["title"])

    print()
    print("References for this model:")
    print(related_papers["referenced_paper_title"])

    valid_choices = list(range(len(propositions)))
    choice = None
    while choice not in valid_choices:
        if choice is not None:
            print()
            print(f"Categorizing '{remaining}'")
        print()
        print(f"Propositions:")
        print("\n".join(f"{i}: {d}" for i, d in enumerate(propositions)))
        print("d: drop")
        print("s: skip")
        print("l: list titles for this model")
        print("r: search")
        print("t{number}: print current models tree with depth 'number'")
        print("wq: save and exit")
        print("q: exit without saving")
        print("h: get help from gpt")
        print("pdb: debug")
        print()

        choice = input("Choose a proposition: ")
        if choice == "s":
            return "", ""
        elif choice == "wq":
            return "wq", "wq"
        elif choice == "q":
            return "q", "q"
        elif choice.startswith("t"):
            try:
                tmp = copy.deepcopy(models)
                tmp.pop("ignore")
                pprint(truncate_dict(tmp, int(choice[1:])))
            except:
                print("Invalid depth")
        elif choice == "l":
            pd.options.display.max_rows = 999
            pd.options.display.max_colwidth = 300
            print(paper_models[paper_models["name"] == remaining][["title", "aliases"]])
            continue
        elif choice == "r":
            look_for = input("Search: ")
            propositions = get_proposition([look_for], models, df=df)
            propositions = propositions[0][2:]
            continue
        elif choice == "h":
            if (
                related_papers.iloc[0]["is_contributed"]
                or not related_papers.iloc[0]["referenced_paper_title"]
            ):
                model_categories = get_model_description(
                    related_papers.iloc[0]["title"], remaining, models
                )
            else:
                model_categories = get_model_description(
                    related_papers.iloc[0]["referenced_paper_title"], remaining, models
                )

            if model_categories is None:
                print("Cannot get help from gpt. :'(")
                continue

            print(f"Asked for `{remaining}`, answered for `{model_categories.name}`:")
            for model_category in model_categories.categories:
                print()
                print(model_category.value)
                print(model_category.justification)
                # print(model_category.quote)
                if (df["model"] == model_category.value).any():
                    propositions = [model_category.value] + list(propositions)
        elif choice == "d":
            return "drop", "drop"
        elif choice == "pdb":
            import pdb

            pdb.set_trace()

        try:
            choice = int(choice)
        except:
            continue

        if choice >= len(propositions):
            continue

        chosen_model = propositions[choice]
        parent = df[df["model"] == chosen_model]["parent1"]
        if parent.isnull().any():
            parent = ""
        else:
            print(parent)
            parent = parent.item()

        return chosen_model, parent


def chose_action(chosen_model, parent, models, df):

    print()
    print(f"Chosen {chosen_model}, which is in current context:")
    if parent:
        pprint({parent: get_model(parent, models, df)})
    else:
        pprint({chosen_model: get_model(chosen_model, models, df)})

    valid_choices = list("uapns")
    choice = None

    while choice not in valid_choices:
        print()
        print("u: under chosen model")
        print("a: above chosen model")
        print("p: above all sibling of chosen model")
        print("n: next to chosen model")
        print("s: skip")

        choice = input("Choose an action: ")

    return choice


def handle_propositions(
    remaining, propositions, models, paper_models, paperoni, df=None
):
    if df is None:
        df = build_models_dataframe(models)

    models_groups = [key for key in models.keys() if key != "ignore"]
    top_models = sum([list(models[key].keys()) for key in models_groups], [])
    propositions = propositions + tuple(sorted(top_models)) + tuple(models_groups)

    chosen_model, parent = chose_proposition(
        remaining, propositions, paper_models, models, paperoni, df
    )

    if not chosen_model:
        return "skip"

    if chosen_model == "wq" and parent == "wq":
        return "wq"

    if chosen_model == "q" and parent == "q":
        return "q"

    if chosen_model == "drop" and parent == "drop":
        if "ignore" not in models:
            models["ignore"] = {}

        models["ignore"][remaining] = {}
        return "drop"

    choice = chose_action(chosen_model, parent, models, df)

    print(choice)

    # add under
    if choice == "u":
        get_model(chosen_model, models, df)[remaining] = {}
    # add above
    elif choice == "a":
        parent_group = get_model(parent, models, df)
        chosen_group = parent_group.pop(chosen_model)
        parent_group[remaining] = {chosen_model: chosen_group}
    # add above siblings
    elif choice == "p":
        grand_parent = df[df["model"] == chosen_model]["parent2"]
        grand_parent_group = get_model(grand_parent.item(), models, df)
        parent_group = grand_parent_group.pop(parent)
        grand_parent_group[parent] = {remaining: parent_group}
    # Add with siblings
    elif choice == "n":
        get_model(parent, models, df)[remaining] = {}
    # Skip for now
    elif choice == "s":
        return "skip"

    grand_parent = df[df["model"] == chosen_model]["parent2"]
    if not grand_parent.isnull().any():
        pprint(get_model(grand_parent.item(), models, df))
    elif parent:
        pprint(get_model(parent, models, df))
    else:
        pprint(get_model(chosen_model, models, df))


def print_status(skipped, remainings, models, df=None):
    if df is None:
        df = build_models_dataframe(models)

    total = len(skipped) + len(remainings) + df.shape[0]
    print(f"{len(skipped) + len(remainings)} / {total} remainings")


def any_remainings(models_df, analysis, skipped):
    return list(
        (set(analysis["models"]["name"]) - set(models_df["model"])) - set(skipped)
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--papers",
        nargs="*",
        type=Path,
        default=[],
        help="Paperoni Json files of papers to analyse",
    )
    parser.add_argument(
        "--analysis-folder",
        type=Path,
        default=Path("data/queries"),
        help="Folder containing the annotations of the papers",
    )
    parser.add_argument(
        "--paperoni-config",
        type=Path,
        default=Path("data/config.yaml"),
        help="Configuration file for paperoni",
    )
    parser.add_argument(
        "--categorized-models",
        type=Path,
        default=Path("data/categorized_models.json"),
        help="Path to categorized models",
    )

    options = parser.parse_args(argv)

    models = json.load(open(options.categorized_models))

    papers = []
    for papers_json_path in options.papers:
        papers.extend(json.load(open(papers_json_path)))

    analysis, _ = load_analysis(papers, options.analysis_folder)

    skipped = []
    force_end = False

    with load_config(options.paperoni_config):

        df = build_models_dataframe(models)
        remainings = any_remainings(df, analysis, skipped)

        while remainings and not force_end:

            print_status(skipped, remainings, models)
            distances = get_proposition(remainings, models, df=df)
            remaining = distances[0][1]
            if (
                distances[0][0] == 0 and distances[0][1] == distances[0][2]
            ):  #  and False:
                print(f"{remaining} already in categorized models. Removing it.")
            else:
                rval = handle_propositions(
                    remaining,
                    distances[0][2:],
                    models,
                    analysis["models"],
                    papers,
                    df=df,
                )
                if rval == "skip":
                    skipped.append(remaining)
                elif rval == "wq":
                    # Will end loop
                    force_end = True
                    skipped.append(remaining)
                elif rval == "q":
                    return

            df = build_models_dataframe(models)
            remainings = any_remainings(df, analysis, skipped)

            # with open(options.non_categorized_models.with_suffix(".tmp"), "w") as f:
            #     json.dump(remainings + skipped, f)

            with open(options.categorized_models.with_suffix(".tmp"), "w") as f:
                json.dump(models, f)

        # os.rename(
        #     options.non_categorized_models.with_suffix(".tmp"),
        #     options.non_categorized_models,
        # )
        if options.categorized_models.with_suffix(".tmp").exists():
            os.rename(
                options.categorized_models.with_suffix(".tmp"),
                options.categorized_models,
            )


if __name__ == "__main__":
    main()
