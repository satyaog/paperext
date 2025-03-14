from pathlib import Path
from typing import Generator

from paperext.sanitize_categorization import (
    _update_sanitized_map,
    default_sanitize_key,
)
from paperext.log import logger
from paperext.structured_output._base import BaseState
from paperext.utils import Paper
from paperext.structured_output.cat_new_mdl_dom.model import (
    FIRST_MESSAGE,
    RETRY_MESSAGE,
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
        domain: str,
        other_domains: list,
        sanitized_map: dict[str, str],
        **kwargs,
    ):
        super().__init__(
            paper=paper,
            pdf_txt=pdf_txt,
            domain=domain,
            other_domains=other_domains,
            sanitized_map=sanitized_map,
            **kwargs,
        )
        self._domain = domain
        self._other_domains = other_domains
        self._sanitized_map = sanitized_map.copy()

    def format_messages(self) -> Generator[list[dict[str:str]], None, None]:
        listed_domains = set()
        missing = None

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

        while missing is None or (len(missing) / len(self._other_domains)) > 0.1:
            self._query_data.append(
                {
                    "domain": self._domain,
                    "domain_pool": self._other_domains,
                    "missing": missing,
                }
            )

            _messages[0] = {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            }

            yield _messages

            if closest_parent_domain := default_sanitize_key(
                self.responses[-1].extractions.closest_parent_domain.value
            ):
                _update_sanitized_map(self._sanitized_map, closest_parent_domain)

            if closest_child_domain := default_sanitize_key(
                self.responses[-1].extractions.closest_child_domain.value
            ):
                _update_sanitized_map(self._sanitized_map, closest_child_domain)

            if closest_sibling_domain := default_sanitize_key(
                self.responses[-1].extractions.closest_sibling_domain.value
            ):
                _update_sanitized_map(self._sanitized_map, closest_sibling_domain)

            for domains in (
                self.responses[-1].extractions.semantically_equivalent_domains,
                self.responses[-1].extractions.parent_domains,
                self.responses[-1].extractions.child_domains,
                self.responses[-1].extractions.sibling_domains,
                self.responses[-1].extractions.unrelated_domains,
            ):
                for domain in domains:
                    if domain := default_sanitize_key(domain.value):
                        _update_sanitized_map(self._sanitized_map, domain)

            if closest_parent_domain := self._sanitized_map.get(
                closest_parent_domain, ""
            ):
                self.responses[-1].extractions.closest_parent_domain.value = (
                    closest_parent_domain
                )
                listed_domains.add(closest_parent_domain)

            if closest_child_domain := self._sanitized_map.get(
                closest_child_domain, ""
            ):
                self.responses[-1].extractions.closest_child_domain.value = (
                    closest_child_domain
                )
                listed_domains.add(closest_child_domain)

            if closest_sibling_domain := self._sanitized_map.get(
                closest_sibling_domain, ""
            ):
                self.responses[-1].extractions.closest_sibling_domain.value = (
                    closest_sibling_domain
                )
                listed_domains.add(closest_sibling_domain)

            for domains in (
                self.responses[-1].extractions.semantically_equivalent_domains,
                self.responses[-1].extractions.parent_domains,
                self.responses[-1].extractions.child_domains,
                self.responses[-1].extractions.sibling_domains,
                self.responses[-1].extractions.unrelated_domains,
            ):
                for domain in domains:
                    if _domain := self._sanitized_map.get(domain.value, ""):
                        listed_domains.add(_domain)

                    domain.value = _domain

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
