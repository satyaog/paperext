from pathlib import Path
from typing import Generator

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
        first_page = ParsedDocResponse.model_validate_json(
            parsed_doc.read_text()
        ).analysis.pages_txt[0]

        _messages = [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": FIRST_MESSAGE.format(first_page),
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
