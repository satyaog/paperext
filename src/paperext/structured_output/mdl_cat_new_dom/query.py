import argparse
import asyncio
from asyncio.log import logger
from datetime import datetime
import hashlib
import json
import logging
import math
from pathlib import Path

from instructor.exceptions import InstructorRetryException
import numpy as np
from sentence_transformers import SentenceTransformer
import tqdm

from paperext.config import CFG, Config
from paperext.query import PLATFORMS, PROG, batch_queries
from paperext.sanitize_categorization import (
    _flatten_dict,
    _make_sanitized_map,
    _update_sanitized_map,
    sanitize_categories,
)

# from paperext.structured_output import get_struct_module
from paperext.structured_output.mdl.stats.build_domains_tree import (
    any_remainings,
    build_domains_dataframe,
    get_domain,
    get_proposition,
)
from paperext.structured_output.mdl.stats.stats import load_analysis
from paperext.structured_output.mdl_clus_dom.state import (
    _find_min_max_threshold,
    _sort_categories,
    cluster_categories,
)
from paperext.structured_output.mdl_cat_new_dom.state import State
from paperext.structured_output.mdl_cat_new_dom.model import Response
from paperext.structured_output.mdl_find_acr.query import list_domains

# from paperext.utils import Paper


def _iter_tolerance(min_tolerance: int, max_tolerance: int, log_step=4):
    tolerance = max_tolerance

    while tolerance >= min_tolerance:
        tolerance_step = max(0.01, (tolerance - min_tolerance) / log_step)
        tolerance = tolerance - tolerance_step
        yield tolerance


def hierarchical_clustering(categories: list):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    categories = {k: {} for k in _sort_categories(model, categories)}

    min_tolerace, max_tolerance = _find_min_max_threshold(model, sorted(categories))
    tolerance = max_tolerance

    for tolerance in tqdm.tqdm(_iter_tolerance(min_tolerace, max_tolerance)):
        if len(categories) < 2:
            break

        categorisation = cluster_categories(model, categories, tolerance=tolerance)

        for cat, sub_cats in categorisation.items():
            for sub_cat in sub_cats:
                assert not sub_cats.get(sub_cat, None)
                sub_cats[sub_cat] = categories[sub_cat]
            assert not sub_cats.get(cat, None)
            sub_cats[cat] = categories[cat]

        categories = categorisation

    return categories


def _update_sorted_propositions(
    last_propositions: list,
    similarities,
    skipped,
    k=20,
):
    new_domain = last_propositions[0][1]

    if new_domain in skipped:
        return last_propositions[1:]

    propositions = []

    for proposition in last_propositions[1:]:
        remaining = proposition[1]
        distances = sorted(
            [
                (1 - similarities[remaining, domain], domain)
                for domain in proposition[2:] + [new_domain]
            ]
        )
        propositions.append(
            [distances[0][0], remaining] + [d[1] for d in distances[:k]]
        )

    return sorted(propositions)


def main(argv: list = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paperoni",
        nargs="*",
        type=Path,
        help="Paperoni json report of papers to analyse",
    )
    parser.add_argument(
        "--categorized-domains",
        type=Path,
        default=CFG.dir.data / "mdl/categorized_domains.json",
        help="Path to categorized domains",
    )
    parser.add_argument(
        "--accronyms",
        type=Path,
        default=CFG.dir.data / "mdl_find_acr/acronyms_or_abbreviations_domains.json",
        help="Path to categorized domains",
    )

    options = parser.parse_args(argv)

    domains = json.loads(options.categorized_domains.read_text())

    if options.accronyms:
        accronyms_map = json.loads(options.accronyms.read_text().lower())
    else:
        accronyms_map = {}

    papers = []
    for papers_json_path in options.paperoni:
        papers.extend(json.loads(Path(papers_json_path).read_text()))

    analysis, _ = load_analysis(papers, CFG.dir.queries / CFG.platform.select)

    sanitized_map = _make_sanitized_map(set(list_domains(papers)))
    _update_sanitized_map(
        sanitized_map,
        *(
            set(_flatten_dict(domains["abstract_research_topics"]))
            | set(_flatten_dict(domains["application_domains"]))
            | set(accronyms_map.keys())
            | set(accronyms_map.values())
            | set(analysis["attrs"]["research_fields"].explode())
        ),
    )
    accronyms_map = {
        sanitized_map[k]: sanitized_map[v] for k, v in accronyms_map.items()
    }
    sanitized_map = {
        k: accronyms_map.get(sanitized_map[k], v) for k, v in sanitized_map.items()
    }
    domains = {
        k.replace(" ", "_"): v
        for k, v in sanitize_categories(domains, accronyms_map, sanitized_map).items()
    }

    for research_fields in analysis["attrs"]["research_fields"]:
        research_fields[:] = map(lambda x: sanitized_map[x], research_fields)

    skipped = []
    df = build_domains_dataframe(domains)
    remainings = any_remainings(df, analysis, skipped)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    entries = sorted(
        (set(remainings) | set(df["domain"]))
        - set(["abstract_research_topics", "application_domains"])
    )
    embeddings = model.encode(entries)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = model.similarity(embeddings, embeddings)
    similarities = {
        (domain, other): similarities[i, j]
        for i, domain in enumerate(entries)
        for j, other in enumerate(entries)
    }

    Path(options.categorized_domains).with_suffix(".tmp").write_text(
        json.dumps(domains, indent=2, sort_keys=True)
    )

    domains_pool = set((d for d in df["domain"] if d.strip())) - set(
        ["abstract_research_topics", "application_domains"]
    )

    with Config.push():
        # CFG.platform.select = "ollama"
        CFG.platform.struct = Path(__file__).parent.name
        # CFG.ollama.model = "deepseek-r1:14b"
        # CFG.ollama.model = "deepseek-r1:32b"

        LOG_FILE = CFG.dir.log / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logging.basicConfig(
            filename=LOG_FILE.with_suffix(f".{PROG}.{CFG.platform.struct}.dbg"),
            level=logging.DEBUG,
            force=True,
        )

        client = PLATFORMS[CFG.platform.select]()

        all_distances = get_proposition(
            remainings,
            domains,
            k=20,
            df=df,
            similarities=similarities,
            exclude=["abstract_research_topics", "application_domains"],
            domains_pool=domains_pool,
        )

        for _ in tqdm.tqdm(
            list(range(len(all_distances))), desc="Categorizing domains"
        ):
            # all_distances = get_proposition(
            #     remainings,
            #     domains,
            #     k=20,
            #     df=df,
            #     similarities=similarities,
            #     exclude=["abstract_research_topics", "application_domains"],
            #     domains_pool=domains_pool,
            # )

            subject, *propositions = all_distances[0][1:]

            _filename_prefix = "".join(
                [subject[0]] + sorted(set(domain[0] for domain in sorted(propositions)))
            )
            _filename = "_".join(
                [
                    _filename_prefix,
                    hashlib.sha256(
                        "".join([subject] + sorted(propositions)).encode()
                    ).hexdigest(),
                ]
            )

            make_state = lambda *args, **kwargs: State(
                *args,
                **kwargs,
                domain=subject,
                other_domains=propositions,
                sanitized_map=sanitized_map,
            )
            while True:
                try:
                    responses: list[Response] = asyncio.run(
                        batch_queries(
                            client,
                            [(None, Path(_filename))],
                            destination=CFG.dir.data
                            / CFG.platform.struct
                            / "queries"
                            / CFG.platform.select,
                            state_cls=make_state,
                        )
                    )
                    break

                except InstructorRetryException:
                    continue

            for r in responses:
                _update_sanitized_map(
                    sanitized_map, r.analysis.closest_parent_domain.value
                )
                _update_sanitized_map(
                    sanitized_map, r.analysis.closest_child_domain.value
                )
                _update_sanitized_map(
                    sanitized_map, r.analysis.closest_sibling_domain.value
                )

                for r_domains in (
                    r.analysis.semantically_equivalent_domains,
                    r.analysis.parent_domains,
                    r.analysis.child_domains,
                    r.analysis.sibling_domains,
                    r.analysis.unrelated_domains,
                ):
                    for domain in r_domains:
                        _update_sanitized_map(sanitized_map, domain.value)

            _propositions_map = {
                sanitized_map[domain]: (i, domain)
                for i, domain in enumerate(propositions)
            }

            selection = None
            equivalent_matches = []
            parent_matches = []
            child_matches = []
            sibling_matches = []

            for r in responses:
                _equivalent_matches = (
                    sorted(
                        _propositions_map[sanitized_map[_d.value]]
                        for _d in r.analysis.semantically_equivalent_domains
                        if sanitized_map[_d.value] in _propositions_map
                    )
                    if len(r.analysis.semantically_equivalent_domains)
                    / len(propositions)
                    <= 0.2
                    else []
                )

                if _equivalent_matches:
                    equivalent_matches.extend(_equivalent_matches)
                    break

                if (
                    _parent_match := _propositions_map.get(
                        sanitized_map[r.analysis.closest_parent_domain.value], None
                    )
                ) is not None:
                    parent_matches.append(_parent_match)

                if (
                    _child_match := _propositions_map.get(
                        sanitized_map[r.analysis.closest_child_domain.value], None
                    )
                ) is not None:
                    child_matches.append(_child_match)

                if (
                    _sibling_match := _propositions_map.get(
                        sanitized_map[r.analysis.closest_sibling_domain.value], None
                    )
                ) is not None:
                    sibling_matches.append(_sibling_match)

            equivalent_match = next(iter(sorted(equivalent_matches)), (math.inf, None))
            parent_match = next(iter(sorted(parent_matches)), (math.inf, None))
            child_match = next(iter(sorted(child_matches)), (math.inf, None))
            sibling_match = next(iter(sorted(sibling_matches)), (math.inf, None))

            if equivalent_match[1]:
                selection = equivalent_match
            else:
                selection = sorted((parent_match, child_match, sibling_match))[0]

            if selection[1] is None:
                skipped.append(subject)
            elif selection in (equivalent_match, parent_match):
                get_domain(selection[1], domains, df)[subject] = {}
            elif selection is child_match:
                parent = df[df["domain"] == selection[1]]["parent1"].iloc[0]
                parent_group = get_domain(parent, domains, df)
                chosen_group = parent_group.pop(selection[1])
                parent_group[subject] = {selection[1]: chosen_group}
            elif selection is sibling_match:
                parent = df[df["domain"] == selection[1]]["parent1"].iloc[0]
                get_domain(parent, domains, df)[subject] = {}

            df = build_domains_dataframe(domains)
            remainings = any_remainings(df, analysis, skipped)

            Path(options.categorized_domains).with_suffix(".tmp").write_text(
                json.dumps(domains, indent=2, sort_keys=True)
            )

            domains_pool = set(
                sum(
                    (domain_distances[2:] for domain_distances in all_distances[1:]),
                    [],
                )
                + ([subject] if subject not in skipped else [])
            )

            all_distances = _update_sorted_propositions(
                all_distances, similarities, skipped, k=20
            )

    options.categorized_domains.with_suffix(".tmp").rename(options.categorized_domains)


if __name__ == "__main__":
    main()
