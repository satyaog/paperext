from pathlib import Path
from typing import Generator
from paperext.utils import Paper
from . import model


class State:
    def __init__(self, paper: Paper, pdf_txt: Path):
        self._paper = paper
        self._pdf_txt = pdf_txt
        self.responses: list[model.Response] = []

    def format_messages(self) -> Generator[list[dict[str:str]], None, None]:
        yield [
            {
                "role": "system",
                "content": model.SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": model.FIRST_MESSAGE.format(self._pdf_txt.read_text()),
            },
        ]

    def push_response(self, response: "State.response_type"):
        self.responses.append(response)

    def make_response(
        self, paper_name: str, words: int, analysis: model.PaperExtractions, usage: dict
    ):
        return model.Response(
            paper=paper_name,
            words=words,
            extractions=analysis,
            usage=usage,
        )

    def get_response_cls(self):
        return model.Response

    def get_response_model(self):
        return model.PaperExtractions
