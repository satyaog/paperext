from pathlib import Path
from typing import Generator
from paperext.utils import Paper
from .model import FIRST_MESSAGE, SYSTEM_MESSAGE, Response, PaperExtractions


class State:
    def __init__(self, paper: Paper, pdf_txt: Path):
        self._paper = paper
        self._pdf_txt = pdf_txt
        self.responses: list[Response] = []

    def format_messages(self) -> Generator[list[dict[str:str]], None, None]:
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

    def push_response(self, response: "State.response_type"):
        self.responses.append(response)

    def get_response_cls(self):
        return Response

    def get_response_model(self):
        return PaperExtractions
