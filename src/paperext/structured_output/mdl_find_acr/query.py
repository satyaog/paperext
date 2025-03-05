import argparse
import asyncio
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path

from instructor.exceptions import InstructorRetryException
from sentence_transformers import SentenceTransformer
import tqdm
from paperext.config import CFG, Config
from paperext.log import logger
from paperext.query import PLATFORMS, PROG, batch_queries
from paperext.sanitize_categorization import (
    _flatten_dict,
    _make_sanitized_map,
    _update_sanitized_map,
)
from paperext.structured_output import get_struct_module
from paperext.structured_output.mdl.stats.stats import load_analysis
from paperext.structured_output.mdl_clus_dom.state import _sort_categories
from paperext.structured_output.mdl_find_acr.model import Response
from paperext.structured_output.mdl_find_acr.state import State
from paperext.utils import Paper


def list_domains(papers: list[dict | Paper]):
    for paper in papers:
        if not isinstance(paper, Paper):
            paper = Paper(paper)

        for query in paper.queries:
            extractions = (
                get_struct_module(CFG.platform.struct)
                .model.Response.model_validate_json(query.read_text())
                .analysis
            )

            for research_field in (
                extractions.primary_research_field,
                *extractions.sub_research_fields,
            ):
                yield research_field.name.value
                yield from research_field.aliases


def main(argv: list = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paperoni",
        nargs="*",
        type=Path,
        help="Paperoni json report of papers to analyse",
    )
    parser.add_argument(
        "--categorized-terms",
        type=Path,
        help="Path to categorized terms",
    )
    options = parser.parse_args(argv)

    papers = []
    for papers_json_path in options.paperoni:
        papers.extend(json.loads(papers_json_path.read_text()))

    categorised_terms = json.loads(options.categorized_terms.read_text().lower())
    terms = sorted(
        set(
            sum(
                [
                    list(_flatten_dict(categorised_terms[key]))
                    for key in categorised_terms
                    if key != "ignore"
                ],
                [],
            )
        )
    )
    sanitized_map = _make_sanitized_map(terms)
    _update_sanitized_map(sanitized_map, *set(list_domains(papers)))

    analysis, _ = load_analysis(papers, CFG.dir.queries / CFG.platform.select)
    papers = analysis["attrs"]
    _update_sanitized_map(
        sanitized_map, *papers.explode("research_fields")["research_fields"].unique()
    )

    terms = [sanitized_map[term] for term in terms]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    terms = _sort_categories(model, terms)

    acronyms: dict[str, list[str]] = {}
    left_overs = set()

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

        for term in tqdm.tqdm(terms, desc="Finding acronyms/abbreviations"):
            papers_exploded = papers.explode("research_fields")
            term_aliases = [k for k, v in sanitized_map.items() if v == term]
            titles = list(
                papers_exploded[papers_exploded["research_fields"].isin(term_aliases)][
                    "title"
                ].unique()
            )

            if not titles:
                continue

            related_papers = papers[papers["title"].isin(titles)]
            concurrent_terms = related_papers["research_fields"].explode().unique()
            concurrent_terms = set(sanitized_map[_term] for _term in concurrent_terms)
            concurrent_terms = _sort_categories(
                model, [_term for _term in concurrent_terms if _term]
            )

            _filename_prefix = "".join(
                sorted(set(_term[0] for _term in sorted(concurrent_terms)))
            )
            _filename = "_".join(
                [
                    _filename_prefix,
                    hashlib.sha256(
                        "".join(sorted(concurrent_terms)).encode()
                    ).hexdigest(),
                ]
            )

            make_state = lambda *args, **kwargs: State(
                *args, **kwargs, terms=concurrent_terms, sanitized_map=sanitized_map
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

            _acronyms = {}

            for acr_abb in (acr for r in responses for acr in r.analysis.acronyms):
                acr, full_form = (
                    acr_abb.acronym_abbreviation.value,
                    acr_abb.full_form.value,
                )

                if (acr in concurrent_terms) != (full_form in concurrent_terms):
                    left_overs.add(
                        (tuple(sorted((acr, full_form))), tuple(concurrent_terms))
                    )

                if acr not in concurrent_terms or full_form not in concurrent_terms:
                    logger.warning(
                        f"Model "
                        f"{CFG.platform.select}:{CFG[CFG.platform.select].model} "
                        f"identified an acronym [{acr}:{full_form}] that is "
                        f"missing from the concurrent terms list. Provided terms "
                        f"are {concurrent_terms}. Ignoring"
                    )
                    continue

                if len(acr) == 1:
                    logger.warning(
                        f"Model "
                        f"{CFG.platform.select}:{CFG[CFG.platform.select].model} "
                        f"identified a single char acronym [{acr}:{full_form}] "
                        f"from the concurrent terms list. Provided terms are "
                        f"{concurrent_terms}. Ignoring"
                    )
                    continue

                if acr == full_form:
                    logger.warning(
                        f"Model "
                        f"{CFG.platform.select}:{CFG[CFG.platform.select].model} "
                        f"identified an acronym [{acr}:{full_form}] that stands "
                        f"for the same term. Provided terms are "
                        f"{concurrent_terms}. Ignoring"
                    )
                    continue

                _update_sanitized_map(sanitized_map, acr)
                _update_sanitized_map(sanitized_map, full_form)

                acr = sanitized_map[acr]
                full_form = sanitized_map[full_form]

                _acronyms.setdefault(acr, set())
                _acronyms[acr].add(full_form)

            for k, v in _acronyms.items():
                acronyms.setdefault(k, [])
                acronyms[k].extend(v)

        acronyms = {
            sanitized_map[k]: sorted(
                (
                    (
                        sum(
                            1
                            for other in v
                            if sanitized_map[full_form] == sanitized_map[other]
                        ),
                        sanitized_map[full_form],
                    )
                    for full_form in set(v)
                ),
                reverse=True,
            )
            for k, v in acronyms.items()
        }

        options.categorized_terms.with_stem(
            f"{options.categorized_terms.stem}_acronyms"
        ).write_text(
            json.dumps(
                {k: v[0][1] for k, v in acronyms.items()},
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
