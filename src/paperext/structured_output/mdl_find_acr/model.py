from __future__ import annotations

import logging
from packaging.version import Version
from typing import Generic, Optional, TypeVar

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from paperext.sanitize_categorization import split_words
from paperext.structured_output._base import (
    BaseModel,
    BaseResponse,
    ResponseMetadata,
    _base_empty_fields,
    _base_empty_response,
)
from paperext.utils import str_normalize

logging.basicConfig(level=logging.DEBUG)


SYSTEM_MESSAGE = """You are an Expert in Deep Learning Research and Linguistics. Your task is to identify expressions containing at least one acronym or abbreviation within a given list of expressions and provide their corresponding full forms, if available in the same list. If a full form cannot be found in the list, leave the full form empty.

### Instructions:
- Carefully analyze the provided list of expressions, considering each term individually.
- Identify which expressions contains at least one acronym or abbreviation.
- If an expression contains an acronym or abbreviation, and a corresponding full form can be found in the list, pair them together.
- If no full form is available in the list for a given acronym or abbreviation, leave the full form as an empty string.
- Ensure that you only use the expressions available in the provided list."""

FIRST_MESSAGE = """### List of expressions (one per line):
{}"""

RETRY_MESSAGE = (
    """Your previous selection "{}" is not an exact match for any expressions in the provided list and must be rejected. You must choose from the list only, without introducing new domains or categories. """
    + FIRST_MESSAGE
)


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


class Acronym(BaseModel):
    acronym_abbreviation: Explained[str] = Field(
        description=(
            "Expression explicitely containing at least one acronym or abbreviation from the provided list of expressions."
        )
    )
    full_form: Explained[str] = Field(
        description=(
            "Corresponding full form from the provided list of expressions. This can remain empty if no full form is found"
        )
    )

    @classmethod
    def parse_obj(cls, obj: dict) -> "Acronym":
        # Create a new dictionary with spaces removed from the keys
        cleaned_obj = {}
        for key, value in obj.items():
            key_words = split_words(key, separators=" -_")
            cleaned_obj["_".join(key_words)] = value
        return super().model_validate(cleaned_obj)

    model_config = ConfigDict(
        populate_by_name=False,
    )


class Analysis(BaseModel):
    acronyms: list[Acronym] = Field(
        description=(
            "List of expressions containing at least one acronym or abbreviation matched with their full form."
        )
    )
    not_acronyms: list[Explained[str]] = Field(
        description=(
            "Exaustive list of all provided expressions not including any acronyms or abbreviations."
        )
    )
    justification: str = Field(
        description="A detailed explanation of the thought process leading to this result.",
    )

    @classmethod
    def parse_obj(cls, obj: dict) -> "Analysis":
        # Create a new dictionary with spaces removed from the keys
        cleaned_obj = {}
        for key, value in obj.items():
            key_words = split_words(key, separators=" -_")
            cleaned_obj["_".join(key_words)] = value
        return super().model_validate(cleaned_obj)

    model_config = ConfigDict(
        populate_by_name=False,
    )


class Response(BaseResponse):
    analysis: Analysis = Field(validation_alias=AliasChoices("analysis", "extractions"))
    metadata: Optional[ResponseMetadata] = ResponseMetadata(
        model_version=Version("1.0.0")
    )


def _empty_fields(model_cls: BaseModel):
    return _base_empty_fields(model_cls, Explained)


def empty_model(model_cls):
    empty_fields = _empty_fields(model_cls)

    return model_cls(**empty_fields)


def empty_response(model_cls):
    return _base_empty_response(model_cls, Explained)
