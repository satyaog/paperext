from pathlib import Path
from typing import Generator

from paperext.sanitize_categorization import (
    _update_sanitized_map,
    default_sanitize_key,
)
from paperext.log import logger
from paperext.structured_output._base import BaseState
from paperext.utils import Paper
from paperext.structured_output.mdl_find_acr.model import (
    FIRST_MESSAGE,
    SYSTEM_MESSAGE,
    Analysis,
    Response,
)


class State(BaseState):
    AnalysisCls: Analysis = Analysis
    ResponseCls: Response = Response

    def __init__(
        self,
        paper: Paper,
        pdf_txt: Path,
        terms: list,
        sanitized_map: dict[str, str],
        **kwargs,
    ):
        super().__init__(
            paper=paper,
            pdf_txt=pdf_txt,
            terms=terms,
            sanitized_map=sanitized_map,
            **kwargs,
        )
        self._terms = terms
        self._sanitized_map = sanitized_map.copy()

    @property
    def categories_refs(self):
        return self._categories_refs

    def format_messages(self) -> Generator[list[dict[str:str]], None, None]:
        terms = set()
        missing = None

        _messages = [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": FIRST_MESSAGE.format(
                    "\n".join([f'"{d}"' for d in self._terms])
                ),
            },
        ]

        while missing is None:
            self._query_data.append(
                {
                    "terms": self._terms,
                    "missing": missing,
                }
            )

            yield _messages[:]

            for acr in self.responses[-1].extractions.acronyms:
                _acr, _full_form = default_sanitize_key(
                    acr.acronym_abbreviation.value
                ), default_sanitize_key(acr.full_form.value)
                _acr, _full_form = (
                    (_acr, _full_form)
                    if len(_acr) <= len(_full_form) or not _full_form
                    else (_full_form, _acr)
                )

                if _acr not in self._terms:
                    logger.warning(
                        f"Model identified an accronym [{_acr}:{_full_form}] "
                        f"that is missing from the terms list. Provided terms are "
                        f"{self._terms}."
                    )
                    continue

                if _acr:
                    _update_sanitized_map(self._sanitized_map, _acr)
                    acr.acronym_abbreviation.value = self._sanitized_map[_acr]
                    terms.add(_acr)

                else:
                    continue

                if _full_form:
                    _update_sanitized_map(self._sanitized_map, _full_form)
                    acr.full_form.value = self._sanitized_map[_full_form]
                    terms.add(_full_form)

            for i, term in enumerate(self.responses[-1].extractions.not_acronyms):
                term = default_sanitize_key(term.value)
                if term:
                    _update_sanitized_map(self._sanitized_map, term)
                    self.responses[-1].extractions.not_acronyms[i].value = (
                        self._sanitized_map[term]
                    )
                    terms.add(term)

            missing = set(self._terms) - terms
