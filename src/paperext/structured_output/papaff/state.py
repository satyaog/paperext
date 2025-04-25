from pathlib import Path
from typing import Generator

import regex as re

from paperext.structured_output._base import BaseState
from paperext.utils import Paper
from paperext.structured_output.papaff.model import (
    FIRST_MESSAGE,
    SYSTEM_MESSAGE,
    Analysis,
    Response,
)
from paperext.structured_output.parse_doc.model import Response as ParsedDocResponse


class State(BaseState):
    AnalysisCls: Analysis = Analysis
    ResponseCls: Response = Response

    def __init__(
        self,
        paper: Paper,
        pdf_txt: Path,
        **kwargs,
    ):
        super().__init__(
            paper=paper,
            pdf_txt=pdf_txt,
            **kwargs,
        )

        self.responses: list[Response]

    def format_messages(self) -> Generator[list[dict[str:str]], None, None]:
        parsed_doc = self._paper.queries[-1]
        analysis = ParsedDocResponse.model_validate_json(
            parsed_doc.read_text()
        ).analysis

        pages = analysis.pages_txt or analysis.pages_md
        for i, page in enumerate(pages[:10]):
            if (
                re.search(r"(^|[^a-zA-Z])abstract($|[^a-zA-Z])", page.lower())
                is not None
            ):
                break

        first_pages = pages[: i + 2]

        _messages = [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": FIRST_MESSAGE.format("\n---\n".join(first_pages)),
            },
        ]

        self._query_data.append({"parsed_doc": str(parsed_doc)})

        yield _messages

    def push_response(self, response: Response):
        self.responses.append(response)

    def get_response_cls(self):
        return Response

    def get_response_model(self):
        return Analysis
