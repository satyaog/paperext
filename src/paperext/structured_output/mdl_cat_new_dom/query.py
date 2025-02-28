import argparse
import asyncio
from asyncio.log import logger
from datetime import datetime
import hashlib
import json
import logging
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


def main(argv: list = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--papers",
        nargs="*",
        type=Path,
        default=[],
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
        accronyms_map = None

    papers = []
    for papers_json_path in options.papers:
        papers.extend(json.loads(Path(papers_json_path).read_text()))

    analysis, _ = load_analysis(papers, CFG.dir.queries / CFG.platform.select)

    sanitized_map = _make_sanitized_map(set(list_domains(papers)))
    _update_sanitized_map(
        sanitized_map,
        *set(_flatten_dict(domains["abstract_research_topics"])),
        *set(_flatten_dict(domains["application_domains"])),
        *accronyms_map.keys(),
        *accronyms_map.values(),
        *set(analysis["attrs"]["research_fields"].explode()),
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

    df = build_domains_dataframe(domains)
    remainings = any_remainings(df, analysis, [])

    model = SentenceTransformer("all-MiniLM-L6-v2")
    entries = sorted(set(remainings) | set(df["domain"]))
    embeddings = model.encode(entries)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = model.similarity(embeddings, embeddings)
    similarities = {
        (domain, other): similarities[i, j]
        for i, domain in enumerate(entries)
        for j, other in enumerate(entries)
    }

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

        for _ in tqdm.tqdm(
            list(range(len(remainings))), desc="Finding accronyms/abbreviations"
        ):
            distances = get_proposition(
                remainings, domains, k=20, df=df, similarities=similarities
            )

            subject, *propositions = distances[0][1:]

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
                    sanitized_map, r.extractions.closest_parent_domain.value
                )

                for domains in (
                    r.extractions.semantically_equivalent_domains,
                    r.extractions.parent_domains,
                    r.extractions.child_domains,
                    r.extractions.sibling_domains,
                    r.extractions.unrelated_domains,
                ):
                    for i, domain in enumerate(domains):
                        _update_sanitized_map(sanitized_map, domain)

            for r in responses:
                closest_parent_domain = sanitized_map[
                    r.extractions.closest_parent_domain.value
                ]

                equivalent_match = next(
                    (
                        sanitized_map[_d.value]
                        for _d in r.extractions.semantically_equivalent_domains
                        if sanitized_map[subject] == sanitized_map[_d.value]
                    )
                )

                for domains in (
                    r.extractions.semantically_equivalent_domains,
                    r.extractions.parent_domains,
                    r.extractions.child_domains,
                    r.extractions.sibling_domains,
                    r.extractions.unrelated_domains,
                ):
                    for i, domain in enumerate(domains):
                        domain = sanitized_map[domain]


if __name__ == "__main__":
    main()
