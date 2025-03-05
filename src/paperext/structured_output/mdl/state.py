from pathlib import Path
from typing import Generator
from paperext.structured_output._base import BaseState
from paperext.utils import Paper
from .model import (
    FIRST_MESSAGE,
    # RETRY_MESSAGE,
    SYSTEM_MESSAGE,
    Response,
    Analysis,
)


class State(BaseState):
    AnalysisCls: Analysis = Analysis
    ResponseCls: Response = Response

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
        #             models, datasets, libraries, self._pdf_txt.read_text()
        #         ),
        #     },
        # ]
