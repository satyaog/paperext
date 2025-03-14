import asyncio
from dataclasses import dataclass
import hashlib
from pathlib import Path

from instructor.exceptions import InstructorRetryException
import pandas as pd
from sentence_transformers import SentenceTransformer
import tqdm
from paperext.config import CFG
from paperext.log import logger
from paperext.query import PLATFORMS, batch_queries
from paperext.sanitize_categorization import (
    _flatten_dict,
    _update_sanitized_map,
)
from paperext.structured_output.mdl_clus_dom.state import _sort_categories
from paperext.structured_output.find_acr_el.model import Response


@dataclass
class AcronymsData:
    categorized_terms: dict[str, dict]
    papers_data: pd.DataFrame

    def iter_categorized_terms(self):
        yield from _flatten_dict(self.categorized_terms)

    def list_paper_terms(self) -> list[str]:
        raise NotImplemented()

    def explode_paper_terms(self) -> pd.DataFrame:
        raise NotImplemented()

    def match_terms(self, terms) -> pd.DataFrame:
        raise NotImplemented()

    def concurrent_terms(self, terms) -> pd.DataFrame:
        raise NotImplemented()


def identify_terms_acronyms(
    acronyms_data: AcronymsData,
    sanitized_map: dict[str, str],
    state_cls,
):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    terms = _sort_categories(
        model, [_term for _term in acronyms_data.list_paper_terms() if _term.strip()]
    )

    acronyms: dict[str, list[str]] = {}
    left_overs = set()

    client = PLATFORMS[CFG.platform.select]()

    for term in tqdm.tqdm(terms, desc="Finding acronyms/abbreviations"):
        term_aliases = [k for k, v in sanitized_map.items() if v == term]
        concurrent_terms = [
            _term
            for _term in acronyms_data.concurrent_terms(term_aliases)
            if _term.strip()
        ]
        if len(concurrent_terms) < 2:
            continue

        concurrent_terms = _sort_categories(model, concurrent_terms)

        _filename_prefix = "".join(
            sorted(set(_term[0] for _term in sorted(concurrent_terms)))
        )
        _filename = "_".join(
            [
                _filename_prefix,
                hashlib.sha256("".join(sorted(concurrent_terms)).encode()).hexdigest(),
            ]
        )

        make_state = lambda *args, **kwargs: state_cls(
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

        _acronyms: dict[str, set] = {}

        for acr_abb in (acr for r in responses for acr in r.analysis.acronyms):
            acr, full_form = (
                acr_abb.acronym_abbreviation.value,
                acr_abb.full_form.value,
            )

            _update_sanitized_map(sanitized_map, acr)
            _update_sanitized_map(sanitized_map, full_form)

        for not_acr in (
            not_acr for r in responses for not_acr in r.analysis.not_acronyms
        ):
            not_acr = not_acr.value
            _update_sanitized_map(sanitized_map, not_acr)

        sanitized_concurrent_terms = [
            sanitized_map[_term] for _term in concurrent_terms
        ]

        for acr_abb in (acr for r in responses for acr in r.analysis.acronyms):
            acr, full_form = (
                sanitized_map[acr_abb.acronym_abbreviation.value],
                sanitized_map[acr_abb.full_form.value],
            )

            acr, full_form = (
                (acr, full_form)
                if len(acr) <= len(full_form) or not full_form
                else (full_form, acr)
            )

            if (acr in sanitized_concurrent_terms) != (
                full_form in sanitized_concurrent_terms
            ):
                left_overs.add(
                    (tuple(sorted((acr, full_form))), tuple(concurrent_terms))
                )

            if (
                acr not in sanitized_concurrent_terms
                or full_form not in sanitized_concurrent_terms
            ):
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

    return acronyms, left_overs
