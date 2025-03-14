from pathlib import Path
from typing import Generator

from paperext.sanitize_categorization import (
    _update_sanitized_map,
    default_sanitize_key,
)
from paperext.log import logger
from paperext.structured_output._base import BaseState
from paperext.utils import Paper
from .model import (
    FIRST_MESSAGE,
    SYSTEM_MESSAGE,
    Analysis,
    Response,
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
