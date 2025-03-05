import copy
import json
from typing import Any, Generator, Optional
from packaging.version import Version
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from paperext.config import CFG
from paperext.log import logger
from paperext.utils import Paper


class ResponseMetadata(BaseModel):
    model_version: Version
    llm_model: Optional[str] = None

    # Serialize Version to string
    @classmethod
    def parse_model_version(cls, v: str) -> Version:
        return Version(v)

    # Deserialize Version from string
    @classmethod
    def serialize_model_version(cls, v: Version) -> str:
        return str(v)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        # Override Pydantic's `.json()` method to handle serialization of
        # Version type
        json_encoders={
            Version: lambda v: str(v)  # Custom serialization logic for Version
        },
    )


class BaseResponse(BaseModel):
    paper: str
    words: int
    analysis: BaseModel
    usage: Optional[Any]
    query_data: Optional[dict] = None
    metadata: Optional[ResponseMetadata] = ResponseMetadata(
        model_version=Version("1.0.0")
    )


class BaseState:
    AnalysisCls = BaseModel
    ResponseCls = BaseResponse

    def __init__(self, paper: Paper, pdf_txt: Path, *args, **kwargs):
        self._paper = paper
        self._pdf_txt = pdf_txt

        self._query_data: list[dict] = []
        self.responses: list[BaseResponse] = []

    def format_messages(self) -> Generator[list[dict[str:str]], None, None]:
        raise NotImplementedError

    def push_response(self, response: BaseResponse):
        self.responses.append(response)

    def make_response(
        self, paper_name: str, words: int, analysis: BaseModel, usage: dict
    ):
        response = self.ResponseCls(
            paper=paper_name,
            words=words,
            analysis=analysis,
            usage=usage,
            query_data=self._query_data[-1],
        )
        response.metadata.llm_model = CFG[CFG.platform.select].model
        return response
