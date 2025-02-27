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
from paperext.structured_output.mdl_cat_dom.model import (
    FIRST_MESSAGE,
    RETRY_MESSAGE,
    SYSTEM_MESSAGE,
    Analysis,
    Response,
)


class State:
    def __init__(
        self, paper: Paper, pdf_txt: Path, domains: list, sanitized_map: dict[str, str]
    ):
        self._paper = paper
        self._pdf_txt = pdf_txt
        self._domains = domains
        self._sanitized_map = sanitized_map

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
                    "\n".join([f'"{d}"' for d in self._domains])
                ),
            },
        ]

        domains = set()
        missing = set("empty")

        while missing:
            yield _messages[:]

            for i, domain in enumerate(
                self.responses[-1].extractions.abstract_research_topics
            ):
                if domain := default_sanitize_key(domain.value):
                    _update_sanitized_map(self._sanitized_map, domain)
                    domain = self._sanitized_map[domain]
                    self.responses[-1].extractions.abstract_research_topics[
                        i
                    ].value = domain
                    domains.add(domain)

            for i, domain in enumerate(
                self.responses[-1].extractions.application_domains
            ):
                if domain := default_sanitize_key(domain.value):
                    _update_sanitized_map(self._sanitized_map, domain)
                    domain = self._sanitized_map[domain]
                    self.responses[-1].extractions.application_domains[i].value = domain
                    domains.add(domain)

            missing = (
                set(self._sanitized_map[domain] for domain in self._domains) - domains
            )
            _messages[1] = {
                "role": "user",
                "content": RETRY_MESSAGE.format(
                    "\n".join([f'"{d}"' for d in domains]),
                    "\n".join([f'"{d}"' for d in missing]),
                ),
            }

    def push_response(self, response: Response):
        self.responses.append(response)

    def get_response_cls(self):
        return Response

    def get_response_model(self):
        return Analysis
