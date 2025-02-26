import math
from pathlib import Path
from typing import Generator

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from paperext.sanitize_categorization import (
    _update_sanitized_map,
    default_sanitize_key,
)
from paperext.log import logger
from paperext.utils import Paper
from paperext.structured_output.mdl_find_acr.model import (
    FIRST_MESSAGE,
    SYSTEM_MESSAGE,
    Analysis,
    Response,
)


class State:
    def __init__(
        self,
        paper: Paper,
        pdf_txt: Path,
        terms: list,
        sanitized_map: dict["str", "str"],
    ):
        self._paper = paper
        self._pdf_txt = pdf_txt
        self._terms = terms
        self._sanitized_map = sanitized_map.copy()

        self.responses: list[Response] = []

    @property
    def categories_refs(self):
        return self._categories_refs

    def format_messages(self) -> Generator[list[dict[str:str]], None, None]:
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

        success = False

        terms = set()

        while not success:
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

            success = not (set(self._terms) - terms)

    def push_response(self, response: Response):
        self.responses.append(response)

    def get_response_cls(self):
        return Response

    def get_response_model(self):
        return Analysis
