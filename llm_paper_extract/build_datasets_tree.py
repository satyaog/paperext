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
import numpy as np
import openai
import pandas as pd
import pydantic_core
from openai.types.chat.chat_completion import CompletionUsage
from paperoni.config import load_config, papconf

from .download_convert import download_and_convert_paper
from .models.dataset_category import (_FIRST_MESSAGE,
                                      DatasetCategoryExtraction,
                                      ExtractionResponse)
from .stats import load_analysis


def get_dataset_category_from_research_paper(
    message: str,
    dataset_names: list[str],
) -> Tuple[DatasetCategoryExtraction, CompletionUsage]:
    """Extract Datasets, Datasets and Frameworks names from a research paper."""
    client = instructor.from_openai(openai.OpenAI())

    dataset_names = ",".join(f"`{dataset_name}`" for dataset_name in dataset_names)

    retries = [True] * 2
    while True:
        try:
            (
                extractions,
                completion,
            ) = client.chat.completions.create_with_completion(
                model="gpt-4o",
                response_model=DatasetCategoryExtraction,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Your role is to categorize deep learning datasets and environments."
                        ),
                        #  f"The Datasets, Datasets and Frameworks must be used in the paper "
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


def extract_dataset_categories(
    paper_text_path: Path, dataset_name: str, datasets: dict
) -> DatasetCategoryExtraction:

    paper = paper_text_path.name

    count = 0
    for line in paper_text_path.read_text().splitlines():
        count += len([w for w in line.strip().split() if w])

    f = (papconf.paths.cache.parents[0] / "queries/datasets") / paper
    f = f.with_stem(f"{f.stem}_{dataset_name}").with_suffix(".json")

    try:
        response = ExtractionResponse.model_validate_json(f.read_text())
    except (FileNotFoundError, pydantic_core._pydantic_core.ValidationError):
        datasets = copy.deepcopy(datasets)
        datasets.pop("ignore")
        datasets["others"] = {}

        top_datasets = build_datasets_dataframe(truncate_dict(datasets, depth=2))
        top_datasets = top_datasets["dataset"].tolist()

        print(sorted(top_datasets))
        message = _FIRST_MESSAGE.format(
            dataset_name=dataset_name,
            paper_text=paper_text_path.read_text(),
        )

        extractions, usage = get_dataset_category_from_research_paper(
            message, top_datasets
        )

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

    from paperoni.sources.scrapers.semantic_scholar import \
        SemanticScholarScraper

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


def get_dataset_description(
    paper_title: str, dataset_name: str, datasets: dict
) -> DatasetCategoryExtraction | None:
    text = get_external_paper_text(paper_title)

    if text is None:
        print("Cannot get text")
        return None

    return extract_dataset_categories(text, dataset_name, datasets)


def build_datasets_dataframe(datasets):
    def build_datasets_tree(datasets, parent=None) -> List:
        rows = []
        for key in datasets:
            rows.append({"dataset": key})
            if datasets[key]:
                subdatasets = build_datasets_tree(datasets[key])
                for subdataset in subdatasets:
                    subdataset[f"parent{len(subdataset)}"] = key
                rows.extend(subdatasets)

        return rows

    df = pd.DataFrame(build_datasets_tree(datasets))
    for col in ["parent1", "parent2"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


def is_probably_an_acronym(dataset_name):
    return len(dataset_name) < 5 and " " not in dataset_name


def levenshtein_for_sequence(remaining, dataset):
    remaining = remaining.replace("-", " ")
    dataset = dataset.replace("-", " ")
    dists = defaultdict(list)
    for a in remaining.split(" "):
        for b in dataset.split(" "):
            dists[b].append(Levenshtein.distance(a, b))

    dists = [(min(v), k) for k, v in dists.items()]
    sorted(dists)

    return sum(dists)


def get_proposition(remainings, datasets, k=5, df=None):
    if df is None:
        df = build_datasets_dataframe(datasets)
    propositions = []
    for remaining in remainings:
        remaining_dataset_is_probably_an_acronym = is_probably_an_acronym(remaining)
        distances = []
        for dataset in list(df["dataset"]):
            if remaining_dataset_is_probably_an_acronym and not is_probably_an_acronym(
                dataset
            ):
                distance = Levenshtein.distance(
                    remaining, "".join(m[0] for m in dataset.split(" "))
                )
            # elif remaining_dataset_is_probably_an_acronym:
            #     continue
            else:
                distance = Levenshtein.distance(remaining, dataset)

            distances.append((distance, dataset))

        distances = sorted(distances)
        propositions.append(
            (distances[0][0], remaining) + tuple(d[1] for d in distances[:k])
        )

    # longest_first = [p for p in propositions if len(p[1]) > 3]
    # if longest_first:
    #     propositions = longest_first

    return sorted(propositions)


def get_first_paper_with_non_categorized_dataset(analysis, remainings):
    sorted_datasets = analysis["datasets"].sort_values(by=["paper_id"])
    sorted_datasets = sorted_datasets[sorted_datasets["name"].isin(remainings)]
    return sorted_datasets.iloc[0]["name"]


def get_dataset(dataset, datasets, df=None):
    if df is None:
        df = build_datasets_dataframe(datasets)

    try:
        dataset_row = df[df["dataset"] == dataset].iloc[0]
    except:
        import pdb

        pdb.set_trace()
    for parent in dataset_row[::-1]:
        if pd.isnull(parent):
            continue
        datasets = datasets[parent]

    return datasets


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


def get_paper_dataset_excerpt(paper, folder, dataset):
    text = get_paper_text(paper, folder)
    if not text:
        return []

    text = text.lower().replace("\n", " ")

    # TODO return more than one excerpt
    # return [text[text.find(dataset) - 75 : text.find(dataset) + 75]]
    return [text[m.start() - 75 : m.start() + 75] for m in re.finditer(dataset, text)]


def chose_proposition(
    remaining, propositions, paper_datasets, datasets, paperoni, df
) -> Tuple[str, str]:

    related_papers = paper_datasets[paper_datasets["name"] == remaining]
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
    # related_papers = paper_datasets[paper_datasets["paper_id"].isin(paper_ids)]
    # related_datasets = list(related_papers["name"].explode().unique())
    # related_datasets.remove(remaining)
    # dataset_names = related_papers["name"].explode()
    # counts = dataset_names[dataset_names != remaining].value_counts()
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
        # print("\n".join(get_paper_dataset_excerpt(paper[0], Path("data"), remaining)))
        print(get_paper_url(paper[0]))
        print(paper[0]["title"])

    print()
    print("References for this dataset:")
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
        print("l: list titles for this dataset")
        print("r: search")
        print("t{number}: print current datasets tree with depth 'number'")
        print("wq: save and exit")
        print("q: exit without saving")
        print("h: get help from gpt")
        print("n: new category")
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
                tmp = copy.deepcopy(datasets)
                tmp.pop("ignore")
                pprint(truncate_dict(tmp, int(choice[1:])))
            except:
                print("Invalid depth")
        elif choice == "l":
            pd.options.display.max_rows = 999
            pd.options.display.max_colwidth = 300
            print(
                paper_datasets[paper_datasets["name"] == remaining][
                    ["title", "aliases"]
                ]
            )
            continue
        elif choice == "r":
            look_for = input("Search: ")
            propositions = get_proposition([look_for], datasets, df=df)
            propositions = propositions[0][2:]
            continue
        elif choice == "h":
            if (
                related_papers.iloc[0]["role"].lower() == "contributed"
                or not related_papers.iloc[0]["referenced_paper_title"]
            ):
                dataset_categories = get_dataset_description(
                    related_papers.iloc[0]["title"], remaining, datasets
                )
            else:
                dataset_categories = get_dataset_description(
                    related_papers.iloc[0]["referenced_paper_title"],
                    remaining,
                    datasets,
                )

            if dataset_categories is None:
                print("Cannot get help from gpt. :'(")
                continue

            print(f"Asked for `{remaining}`, answered for `{dataset_categories.name}`:")
            for dataset_category in dataset_categories.categories:
                print()
                print(dataset_category.value)
                print(dataset_category.justification)
                # print(dataset_category.quote)
                if (df["dataset"] == dataset_category.value).any():
                    propositions = [dataset_category.value] + list(propositions)
        elif choice == "n":
            new_category = input("New category: ")
            if new_category not in datasets:
                datasets[new_category] = {}
            df.loc[df.shape[0]] = [new_category] + [
                np.nan for _ in range(len(df.columns) - 1)
            ]
            return new_category, ""
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

        chosen_dataset = propositions[choice]
        parent = df[df["dataset"] == chosen_dataset]["parent1"]
        if parent.isnull().any():
            parent = ""
        else:
            print(parent)
            parent = parent.item()

        return chosen_dataset, parent


def chose_action(chosen_dataset, parent, datasets, df):

    print()
    print(f"Chosen {chosen_dataset}, which is in current context:")
    if parent:
        pprint({parent: get_dataset(parent, datasets, df)})
    else:
        pprint({chosen_dataset: get_dataset(chosen_dataset, datasets, df)})

    valid_choices = list("uapns")
    choice = None

    while choice not in valid_choices:
        print()
        print("u: under chosen dataset")
        print("a: above chosen dataset")
        print("p: above all sibling of chosen dataset")
        print("n: next to chosen dataset")
        print("s: skip")

        choice = input("Choose an action: ")

    return choice


def handle_propositions(
    remaining, propositions, datasets, paper_datasets, paperoni, df=None
):
    if df is None:
        df = build_datasets_dataframe(datasets)

    datasets_groups = [key for key in datasets.keys() if key != "ignore"]
    top_datasets = (
        []
    )  #  sum([list(datasets[key].keys()) for key in datasets_groups], [])
    propositions = propositions + tuple(sorted(top_datasets)) + tuple(datasets_groups)

    chosen_dataset, parent = chose_proposition(
        remaining, propositions, paper_datasets, datasets, paperoni, df
    )

    if not chosen_dataset:
        return "skip"

    if chosen_dataset == "wq" and parent == "wq":
        return "wq"

    if chosen_dataset == "q" and parent == "q":
        return "q"

    if chosen_dataset == "drop" and parent == "drop":
        if "ignore" not in datasets:
            datasets["ignore"] = {}

        datasets["ignore"][remaining] = {}
        return "drop"

    choice = chose_action(chosen_dataset, parent, datasets, df)

    print(choice)

    # add under
    if choice == "u":
        get_dataset(chosen_dataset, datasets, df)[remaining] = {}
    # add above
    elif choice == "a":
        parent_group = get_dataset(parent, datasets, df)
        chosen_group = parent_group.pop(chosen_dataset)
        parent_group[remaining] = {chosen_dataset: chosen_group}
    # add above siblings
    elif choice == "p":
        grand_parent = df[df["dataset"] == chosen_dataset]["parent2"]
        grand_parent_group = get_dataset(grand_parent.item(), datasets, df)
        parent_group = grand_parent_group.pop(parent)
        grand_parent_group[parent] = {remaining: parent_group}
    # Add with siblings
    elif choice == "n":
        get_dataset(parent, datasets, df)[remaining] = {}
    # Skip for now
    elif choice == "s":
        return "skip"

    grand_parent = df[df["dataset"] == chosen_dataset]["parent2"]
    if not grand_parent.isnull().any():
        pprint(get_dataset(grand_parent.item(), datasets, df))
    elif parent:
        pprint(get_dataset(parent, datasets, df))
    else:
        pprint(get_dataset(chosen_dataset, datasets, df))


def print_status(skipped, remainings, datasets, df=None):
    if df is None:
        df = build_datasets_dataframe(datasets)

    total = len(skipped) + len(remainings) + df.shape[0]
    print(f"{len(skipped) + len(remainings)} / {total} remainings")


def any_remainings(datasets_df, analysis, skipped):
    return list(
        (set(analysis["datasets"]["name"]) - set(datasets_df["dataset"])) - set(skipped)
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
        "--categorized-datasets",
        type=Path,
        default=Path("data/categorized_datasets.json"),
        help="Path to categorized datasets",
    )

    options = parser.parse_args(argv)

    datasets = json.load(open(options.categorized_datasets))

    papers = []
    for papers_json_path in options.papers:
        papers.extend(json.load(open(papers_json_path)))

    analysis, _ = load_analysis(papers, options.analysis_folder)

    skipped = []
    force_end = False

    with load_config(options.paperoni_config):

        df = build_datasets_dataframe(datasets)
        remainings = any_remainings(df, analysis, skipped)

        while remainings and not force_end:

            print_status(skipped, remainings, datasets)
            remaining = get_first_paper_with_non_categorized_dataset(
                analysis, remainings
            )
            # distances = get_proposition(remainings, datasets, df=df)
            # remaining = distances[0][1]
            distances = get_proposition([remaining], datasets, df=df)
            if (
                distances[0][0] == 0 and distances[0][1] == distances[0][2]
            ):  #  and False:
                print(f"{remaining} already in categorized datasets. Removing it.")
            else:
                rval = handle_propositions(
                    remaining,
                    distances[0][2:],
                    datasets,
                    analysis["datasets"],
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

            df = build_datasets_dataframe(datasets)
            remainings = any_remainings(df, analysis, skipped)

            # with open(options.non_categorized_datasets.with_suffix(".tmp"), "w") as f:
            #     json.dump(remainings + skipped, f)

            with open(options.categorized_datasets.with_suffix(".tmp"), "w") as f:
                json.dump(datasets, f)

        # os.rename(
        #     options.non_categorized_datasets.with_suffix(".tmp"),
        #     options.non_categorized_datasets,
        # )
        if options.categorized_datasets.with_suffix(".tmp").exists():
            os.rename(
                options.categorized_datasets.with_suffix(".tmp"),
                options.categorized_datasets,
            )


if __name__ == "__main__":
    main()
