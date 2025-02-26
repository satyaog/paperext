from pathlib import Path
from typing import Generator
from paperext.utils import Paper
from .model import (
    FIRST_MESSAGE,
    # RETRY_MESSAGE,
    SYSTEM_MESSAGE,
    Response,
    PaperExtractions,
)


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
                "content": FIRST_MESSAGE.format(self.pdf_txt.read_text()),
            },
        ]

        # TODO: add a stop_at: int __init__ arg to stop messages at position X, or something like that
        # models = [m.name.value for m in self.responses[-1].extractions.models]
        # datasets = [d.name.value for d in self.responses[-1].extractions.datasets]
        # libraries = [f.name.value for f in self.responses[-1].extractions.libraries]

        # yield [
        #     {
        #         "role": "system",
        #         "content": SYSTEM_MESSAGE,
        #     },
        #     {
        #         "role": "user",
        #         "content": RETRY_MESSAGE.format(
        #             models, datasets, libraries, self.pdf_txt.read_text()
        #         ),
        #     },
        # ]

    def push_response(self, response: "State.response_type"):
        self.responses.append(response)

    def get_response_cls(self):
        return Response

    def get_response_model(self):
        return PaperExtractions
