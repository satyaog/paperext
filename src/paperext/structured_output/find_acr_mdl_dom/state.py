from pathlib import Path
from typing import Generator

from paperext.sanitize_categorization import (
    _update_sanitized_map,
    default_sanitize_key,
)
from paperext.log import logger
from paperext.structured_output._base import BaseState
from paperext.utils import Paper
from paperext.structured_output.find_acr_mdl_dom.model import (
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

        self.responses: list[Response]

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

            _messages[0] = {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            }

            yield _messages

            for acr_abb in self.responses[-1].analysis.acronyms:
                acr, full_form = (
                    acr_abb.acronym_abbreviation.value,
                    acr_abb.full_form.value,
                )

                _update_sanitized_map(self._sanitized_map, acr)
                _update_sanitized_map(self._sanitized_map, full_form)

            for term in self.responses[-1].analysis.not_acronyms:
                term = term.value
                _update_sanitized_map(self._sanitized_map, term)

            sanitized_terms = [self._sanitized_map[_term] for _term in self._terms]

            for acr_abb in self.responses[-1].analysis.acronyms:
                acr, full_form = (
                    self._sanitized_map[acr_abb.acronym_abbreviation.value],
                    self._sanitized_map[acr_abb.full_form.value],
                )

                acr, full_form = (
                    (acr, full_form)
                    if len(acr) <= len(full_form) or not full_form
                    else (full_form, acr)
                )

                if acr not in sanitized_terms:
                    logger.warning(
                        f"Model identified an accronym [{acr}:{full_form}] "
                        f"that is missing from the terms list. Provided terms are "
                        f"{sanitized_terms}."
                    )
                    continue

                terms.add(acr)

                if full_form:
                    terms.add(full_form)

            for term in self.responses[-1].analysis.not_acronyms:
                term = self._sanitized_map.get(term.value, None)
                if term:
                    terms.add(term)

            missing = set(sanitized_terms) - terms
