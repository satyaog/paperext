from __future__ import annotations

import enum
import logging
import typing
from typing import Any, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

from paperext.utils import str_normalize

logging.basicConfig(level=logging.DEBUG)

SYSTEM_MESSAGE = (
    "Your role is to categorize and hierarchize a Deep Learning research domain, "
    "given a research paper context. It's possible for the current hierarchy of "
    "Deep Learning research domains to be incorrect. Feel free to ignore "
    "incorrect placements when returning your categorization. Return a list of "
    "valid category paths in the following format:\n"
    "`category[.subcategory[.subcategory[...]]].researchdomain`\n"
    "If a domain overlaps multiple categories, list all applicable categories. "
    "The current hierarchy of Deep Learning research domains is:\n"
    "{}\n"
)
FIRST_MESSAGE = (
    "Research domain to categorize: {}. Please consider the hierarchy and "
    "categorize accordingly to the following research paper context:\n"
    "{}"
)
_EMPTY_FLAG = "__EMPTY__"


T = TypeVar("T")


class Explained(BaseModel, Generic[T]):
    value: T
    justification: str = Field(
        description="Short justification for the choice of the value",
    )
    quote: str = Field(
        description="The best literal quote from the paper which supports the value",
    )

    def __eq__(self, other: "Explained"):
        return str_normalize(str(self.value)) == str_normalize(str(other.value))

    def __lt__(self, other: "Explained"):
        if isinstance(self.value, bool):
            return not self.value < other.value
        return str_normalize(str(self.value)) < str_normalize(str(other.value))


class PaperExtractions(BaseModel):
    domain_hierarchies: List[Explained[str]] = Field(
        description="Hierarchical paths for research domains"
    )
    domain_hierarchies_aliases: List[Explained[str]] = Field(
        description="Similar domain hierarchies you could find in current "
        "hierarchy of Deep Learning research domains that are differently worded"
    )


class ExtractionResponse(BaseModel):
    paper: str
    words: int
    extractions: PaperExtractions
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
