from pathlib import Path
from typing import Generator
from paperext.utils import Paper
from .model import FIRST_MESSAGE, SYSTEM_MESSAGE, ExtractionResponse, PaperExtractions


class State:
    def __init__(self, paper: Paper, pdf_txt: Path):
        self._paper = paper
        self._pdf_txt = pdf_txt
        self.responses: list[ExtractionResponse] = []

    def format_messages(self) -> Generator[list[dict[str:str]], None, None]:
        yield [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": FIRST_MESSAGE.format(self.pdf_txt.read_text()),
            },
        ]

    def push_response(self, response: "State.response_type"):
        self.responses.append(response)

    def get_extraction_response(self):
        return ExtractionResponse

    def get_paper_extractions(self):
        return PaperExtractions
