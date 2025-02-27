from __future__ import annotations

import logging
import typing
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

from paperext.sanitize_categorization import split_words
from paperext.utils import str_normalize

logging.basicConfig(level=logging.DEBUG)


SYSTEM_MESSAGE = """You are an Expert in Deep Learning Research. Your task is to categorize Deep Learning Research Domains onto one of the following two categories:

1. **abstract research topics**: Domains focused on theoretical or abstract research topics within the field of Deep Learning.
2. **application domains**: Domains focused on practical applications or real-world use cases of Deep Learning techniques.

### Instructions:
- Carefully analyze each Deep Learning Research Domain in the provided set.
- Consider the relationships between the Deep Learning Research Domains and the overarching themes.
- For each Deep Learning Research Domain, determine whether it aligns better with "abstract research topics" or "application domains".
- Ensure that every provided Deep Learning Research Domain is included in one of the categories.
- Do **not** rename any Deep Learning Research Domains.
- You must not introduce any new domains, nor create new categories; you must choose from the existing list only."""

FIRST_MESSAGE = """### List of Deep Learning Research Domains (one per line):
{}"""

RETRY_MESSAGE = (
    FIRST_MESSAGE
    + "\n"
    + """In your last analyse, the following Deep Learning Research Domains where missing (one per line):
{}"""
)

_EMPTY_FLAG = "__EMPTY__"


T = TypeVar("T")


class Explained(BaseModel, Generic[T]):
    value: T
    justification: str = Field(
        description="A detailed explanation for the choice of the value.",
    )

    def __eq__(self, other: "Explained"):
        return str_normalize(str(self.value)) == str_normalize(str(other.value))

    def __lt__(self, other: "Explained"):
        if isinstance(self.value, bool):
            return not self.value < other.value
        return str_normalize(str(self.value)) < str_normalize(str(other.value))


class Analysis(BaseModel):
    abstract_research_topics: list[Explained[str]] = Field(
        description=(
            "List of theoretical or abstract research topics within the field of Deep Learning."
        )
    )
    application_domains: list[Explained[str]] = Field(
        description=(
            "List of practical applications or real-world use cases of Deep Learning techniques."
        )
    )

    @classmethod
    def parse_obj(cls, obj: dict) -> "Analysis":
        # Create a new dictionary with spaces removed from the keys
        cleaned_obj = {}
        for key, value in obj.items():
            key_words = split_words(key.lower(), separators=" -_")
            cleaned_obj["_".join(key_words)] = value
        return super().model_validate(cleaned_obj)

    class Config:
        # This allows the model to use the cleaned keys for validation
        populate_by_name = False


class Response(BaseModel):
    paper: str
    words: int
    extractions: Analysis
    usage: Optional[Any]


def _is_base(cls, other):
    try:
        return cls.__base__ == other
    except AttributeError:
        return False


def _empty_fields(model_cls: BaseModel):
    try:
        iter_fields = model_cls.model_fields.items()
    except AttributeError:
        if typing.get_origin(model_cls) == list:
            return [_empty_fields(model_cls.__args__[0])]
        else:
            return _EMPTY_FLAG

    if _is_base(model_cls, Explained):
        fields = {k: (_empty_fields(v) if k == "value" else "") for k, v in iter_fields}
    else:
        fields = {}
        for k, field in iter_fields:
            fields[k] = _empty_fields(field.annotation)

    return fields


def empty_model(model_cls):
    empty_fields = _empty_fields(model_cls)
    empty_fields["type"]["value"] = "empirical"
    empty_fields["models"][0]["is_contributed"]["value"] = False
    empty_fields["models"][0]["is_executed"]["value"] = False
    empty_fields["models"][0]["is_compared"]["value"] = False
    empty_fields["datasets"][0]["role"] = "referenced"
    empty_fields["libraries"][0]["role"] = "referenced"

    return model_cls(**empty_fields)
