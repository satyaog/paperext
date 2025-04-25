from typing import Any, Generator, Optional
import typing
from packaging.version import Version
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_serializer, model_validator

from paperext.config import CFG
from paperext.utils import Paper

_EMPTY_FLAG = "__EMPTY__"


class ResponseMetadata(BaseModel):
    model_version: Version
    llm_model: Optional[str] = None

    @model_validator(mode="before")
    def parse_model_version(cls, values):
        # Convert the model_version from string to Version if it is a string
        if "model_version" in values and isinstance(values["model_version"], str):
            values["model_version"] = Version(values["model_version"])
        return values

    @field_serializer("model_version")
    def serialize_model_version(self, model_version: Version):
        return str(model_version)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
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


def _is_base(cls, other):
    try:
        return cls.__base__ == other
    except AttributeError:
        return False


def _base_empty_fields(model_cls: type[BaseModel], explained_cls: type[BaseModel]):
    try:
        iter_fields = model_cls.model_fields.items()
    except AttributeError:
        if typing.get_origin(model_cls) == list:
            return [_base_empty_fields(model_cls.__args__[0], explained_cls)]
        else:
            return _EMPTY_FLAG

    if _is_base(model_cls, explained_cls):
        fields = {
            k: (_base_empty_fields(v, explained_cls) if k == "value" else "")
            for k, v in iter_fields
        }
    else:
        fields = {}
        for k, field in iter_fields:
            fields[k] = _base_empty_fields(field.annotation, explained_cls)

    return fields


def _base_empty_response(model_cls: type[BaseResponse], explained_cls: type[BaseModel]):
    empty_fields = _base_empty_fields(model_cls, explained_cls)
    empty_fields["words"] = 0
    empty_fields["query_data"] = {}
    empty_fields["metadata"] = None

    return model_cls(**empty_fields)
