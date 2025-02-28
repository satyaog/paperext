from pathlib import Path
from typing import Generator

from paperext.sanitize_categorization import (
    _update_sanitized_map,
    default_sanitize_key,
)
from paperext.log import logger
from paperext.utils import Paper
from paperext.structured_output.mdl_cat_new_dom.model import (
    FIRST_MESSAGE,
    RETRY_MESSAGE,
    SYSTEM_MESSAGE,
    Analysis,
    Response,
)


class State:
    def __init__(
        self,
        paper: Paper,
        pdf_txt: Path,
        domain: str,
        other_domains: list,
        sanitized_map: dict[str, str],
    ):
        self._paper = paper
        self._pdf_txt = pdf_txt
        self._domain = domain
        self._other_domains = other_domains
        self._sanitized_map = sanitized_map

        self.responses: list[Response] = []

    @property
    def categories_refs(self):
        return self._categories_refs

    def format_messages(self) -> Generator[list[dict[str:str]], None, None]:
        _messages = []
        listed_domains = set()
        missing = set("empty")

        _messages = [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": FIRST_MESSAGE.format(
                    f'"{self._domain}"',
                    "\n".join([f'"{d}"' for d in self._other_domains]),
                ),
            },
        ]

        while missing:
            _messages[0] = {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            }

            yield _messages

            if closest_parent_domain := default_sanitize_key(
                self.responses[-1].extractions.closest_parent_domain.value
            ):
                _update_sanitized_map(self._sanitized_map, closest_parent_domain)

            for domains in (
                self.responses[-1].extractions.semantically_equivalent_domains,
                self.responses[-1].extractions.parent_domains,
                self.responses[-1].extractions.child_domains,
                self.responses[-1].extractions.sibling_domains,
                self.responses[-1].extractions.unrelated_domains,
            ):
                for i, domain in enumerate(domains):
                    if domain := default_sanitize_key(domain.value):
                        _update_sanitized_map(self._sanitized_map, domain)

            if closest_parent_domain := self._sanitized_map.get(
                closest_parent_domain, ""
            ):
                listed_domains.add(closest_parent_domain)

            self.responses[-1].extractions.closest_parent_domain.value = ""

            for domains in (
                self.responses[-1].extractions.semantically_equivalent_domains,
                self.responses[-1].extractions.parent_domains,
                self.responses[-1].extractions.child_domains,
                self.responses[-1].extractions.sibling_domains,
                self.responses[-1].extractions.unrelated_domains,
            ):
                for i, domain in enumerate(domains):
                    if domain := self._sanitized_map.get(domain, ""):
                        listed_domains.add(domain)

                    domains[i].value = domain

            missing = (
                set(self._sanitized_map[domain] for domain in self._other_domains)
                - listed_domains
            )
            _messages[1] = {
                "role": "user",
                "content": RETRY_MESSAGE.format(
                    f'"{self._domain}"',
                    "\n".join([f'"{d}"' for d in self._other_domains]),
                    "\n".join([f'"{d}"' for d in missing]),
                ),
            }

    def push_response(self, response: Response):
        self.responses.append(response)

    def get_response_cls(self):
        return Response

    def get_response_model(self):
        return Analysis
