from pathlib import Path
from typing import Generator

from paperext.structured_output._base import BaseState
from paperext.utils import Paper
from paperext.structured_output.parse_doc.model import (
    Analysis,
    Response,
)


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
        pdfs = list(
            map(
                lambda x: x.with_suffix(".pdf"),
                [self._pdf_txt, *self._paper.pdfs] if self._paper else [self._pdf_txt],
            )
        )
        for pdf in pdfs:
            if pdf.exists():
                break
        else:
            raise ValueError(f"Could not find a pdf within {pdfs}")

        _messages = [{"pdf": str(pdf)}]
        self._query_data.append(_messages[0])

        yield _messages
