from pathlib import Path
from typing import Generator
from paperext.structured_output._base import BaseState
from paperext.utils import Paper
from .model import FIRST_MESSAGE, SYSTEM_MESSAGE, Response, Analysis


class State(BaseState):
    AnalysisCls: Analysis = Analysis
    ResponseCls: Response = Response

    def __init__(self, paper: Paper, pdf_txt: Path, **kwargs):
        super().__init__(paper=paper, pdf_txt=pdf_txt, **kwargs)
        self.responses: list[Response] = []

    def format_messages(self) -> Generator[list[dict[str:str]], None, None]:
        self._query_data.append({})

        yield [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": FIRST_MESSAGE.format(self._pdf_txt.read_text()),
            },
        ]
