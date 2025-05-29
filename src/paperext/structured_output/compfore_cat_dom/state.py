from pathlib import Path
from typing import Generator

from paperext.sanitize_categorization import _flatten_dict
from paperext.structured_output._base import BaseState
from paperext.utils import Paper
from paperext.structured_output.compfore_cat_dom.model import (
    FIRST_MESSAGE,
    RETRY_MESSAGE_PARENT,
    RETRY_MESSAGE_SELECTED,
    SYSTEM_MESSAGE,
    Analysis,
    Response,
)
from paperext.structured_output.utils import dict_to_txt, list_to_txt


class State(BaseState):
    AnalysisCls: Analysis = Analysis
    ResponseCls: Response = Response

    def __init__(
        self,
        paper: Paper,
        pdf_txt: Path,
        categorization: dict,
        domains: str,
        **kwargs,
    ):
        super().__init__(
            paper=paper,
            pdf_txt=pdf_txt,
            categorization=categorization,
            domains=domains,
            **kwargs,
        )
        self._categorization = categorization
        self._domains = domains

        self.responses: list[Response]

    def format_messages(self) -> Generator[list[dict[str:str]], None, None]:
        """Format messages for the domain categorization task.

        Yields:
            A list of message dictionaries containing the system and user messages
            for the categorization task.
        """
        selected = None
        parent = None

        _messages = [
            {
                "role": "system",
                "content": "",
            },
            {
                "role": "user",
                "content": FIRST_MESSAGE.format(list_to_txt(self._domains)),
            },
        ]

        while selected is None or parent is None:
            self._query_data.append(
                {
                    "categorization": self._categorization,
                    "domains": self._domains,
                    "selected": selected,
                    "parent": parent,
                }
            )

            _messages[0] = {
                "role": "system",
                "content": SYSTEM_MESSAGE.format(dict_to_txt(self._categorization)),
            }

            yield _messages

            analysis = self.responses[-1].analysis
            selected = analysis.selected_domain  # .value
            parent = analysis.parent_category  # .value

            # If we need to retry, update the user message with the retry prompt
            if parent not in set(_flatten_dict(self._categorization)):
                _messages[1] = {
                    "role": "user",
                    "content": RETRY_MESSAGE_PARENT.format(
                        parent, list_to_txt(self._domains)
                    ),
                }
                parent = None

            if selected not in self._domains:
                _messages[1] = {
                    "role": "user",
                    "content": RETRY_MESSAGE_SELECTED.format(
                        selected, list_to_txt(self._domains)
                    ),
                }
                selected = None
