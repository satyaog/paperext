import argparse
import asyncio
from asyncio.log import logger
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path

from instructor.exceptions import InstructorRetryException
from sentence_transformers import SentenceTransformer
import tqdm

from paperext.config import CFG, Config
from paperext.query import PLATFORMS, PROG, batch_queries
from paperext.sanitize_categorization import (
    _dict_heads,
    _flatten_dict,
    _make_sanitized_map,
    _update_sanitized_map,
    sanitize_categories,
)

# from paperext.structured_output import get_struct_module
from paperext.structured_output.mdl_clus_dom.state import (
    _find_min_max_threshold,
    _sort_categories,
    cluster_categories,
)
from paperext.structured_output.mdl_cat_dom.state import State
from paperext.structured_output.mdl_cat_dom.model import Response

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


# def list_domains(papers: list[dict | Paper]):
#     for paper in papers:
#         if not isinstance(paper, Paper):
#             paper = Paper(paper)

#         for query in paper.queries:
#             extractions = (
#                 get_struct_module(CFG.platform.struct)
#                 .model.Response.model_validate_json(query.read_text())
#                 .extractions
#             )

#             for research_field in (
#                 extractions.primary_research_field,
#                 *extractions.sub_research_fields,
#             ):
#                 yield research_field.name.value
#                 yield from research_field.aliases


def main(argv: list = None):
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--papers",
    #     nargs="*",
    #     type=Path,
    #     default=[],
    #     help="Paperoni json report of papers to analyse",
    # )
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

    domains = json.loads(options.categorized_domains.read_text().lower())
    ignored = domains.pop("ignore")
    domains = sorted(
        set(
            sum(
                [list(_flatten_dict(domains[key])) for key in domains],
                [],
            )
        )
    )
    accronyms_map = json.loads(options.accronyms.read_text().lower())

    sanitized_map = _make_sanitized_map(domains)
    _update_sanitized_map(
        sanitized_map, *ignored, *accronyms_map, *accronyms_map.values()
    )

    domains = [sanitized_map[domain] for domain in domains]
    accronyms_map = {
        sanitized_map[k]: sanitized_map[v] for k, v in accronyms_map.items()
    }
    domains = sorted(set(accronyms_map.get(domain, domain) for domain in domains))
    ignored.update({k: {} for k in accronyms_map})

    model = SentenceTransformer("all-MiniLM-L6-v2")
    sorted_domains = _sort_categories(model, domains)
    abstract_research_topics = set()
    application_domains = set()

    with Config.push():
        # CFG.platform.select = "ollama"
        CFG.platform.struct = Path(__file__).parent.name
        # CFG.ollama.model = "deepseek-r1:14b"
        # CFG.ollama.model = "deepseek-r1:70b"

        LOG_FILE = CFG.dir.log / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logging.basicConfig(
            filename=LOG_FILE.with_suffix(f".{PROG}.{CFG.platform.struct}.dbg"),
            level=logging.DEBUG,
            force=True,
        )

        client = PLATFORMS[CFG.platform.select]()

        step = 25
        for start_index in tqdm.tqdm(
            range(0, len(sorted_domains), step), desc="Categorizing"
        ):
            selection = _sort_categories(
                model, sorted_domains[start_index : start_index + step]
            )

            _filename_prefix = "".join(
                sorted(set(_term[0] for _term in sorted(selection)))
            )
            _filename = "_".join(
                [
                    _filename_prefix,
                    hashlib.sha256("".join(sorted(selection)).encode()).hexdigest(),
                ]
            )

            make_state = lambda *args, **kwargs: State(
                *args, **kwargs, domains=selection, sanitized_map=sanitized_map
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

            _abstract_research_topics = set(
                domain.value
                for r in responses
                for domain in r.extractions.abstract_research_topics
            )
            _application_domains = set(
                domain.value
                for r in responses
                for domain in r.extractions.application_domains
            )
            _update_sanitized_map(
                sanitized_map, *(_abstract_research_topics & _application_domains)
            )
            selection = [sanitized_map[domain] for domain in selection]
            _abstract_research_topics = [
                sanitized_map[domain] for domain in _abstract_research_topics
            ]
            _application_domains = [
                sanitized_map[domain] for domain in _application_domains
            ]

            for domain in _abstract_research_topics:
                if domain not in selection:
                    logger.warning(
                        f"Model "
                        f"{CFG.platform.select}:{CFG[CFG.platform.select].model} "
                        f"assigned an unexisting domain [{domain}] to "
                        f"abstract_research_topics. Ignoring"
                    )
                    continue

                abstract_research_topics.add(domain)

            for domain in _application_domains:
                if domain not in selection:
                    logger.warning(
                        f"Model "
                        f"{CFG.platform.select}:{CFG[CFG.platform.select].model} "
                        f"assigned an unexisting domain [{domain}] to "
                        f"application_domains. Ignoring"
                    )
                    continue

                application_domains.add(domain)

    application_domains = application_domains - abstract_research_topics

    assert len(abstract_research_topics ^ application_domains) == len(domains)

    abstract_research_topics = hierarchical_clustering(abstract_research_topics)
    application_domains = hierarchical_clustering(application_domains)

    start_level = 0
    while len(_dict_heads(abstract_research_topics, start_level=start_level + 1)) < 50:
        start_level += 1

    abstract_research_topics = sanitize_categories(
        _dict_heads(abstract_research_topics, start_level=start_level),
    )
    del abstract_research_topics["ignore"]

    start_level = 0
    while len(_dict_heads(application_domains, start_level=start_level + 1)) < 50:
        start_level += 1

    application_domains = sanitize_categories(
        _dict_heads(application_domains, start_level=start_level),
    )
    del application_domains["ignore"]

    options.categorized_domains.write_text(
        json.dumps(
            {
                "abstract_research_topics": abstract_research_topics,
                "application_domains": application_domains,
                "ignore": ignored,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
