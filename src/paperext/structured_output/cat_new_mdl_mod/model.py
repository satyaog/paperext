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

SYSTEM_MESSAGE = """You are an Expert Deep Learning Scientist. Your task is to categorize a specified Deep Learning Models within a hierarchical structure, relative to a provided list of models.

### Definitions:
- Given Model: The Given Model is the Deep Learning Model that needs to be categorized.
- Given List: The Given List is the provided list of Deep Learning Models.
- Semantically Equivalent Model: A Semantically Equivalent Model is a Deep Learning Model that is the same semantically to the Given Model. There can be multiple Semantically Equivalent Models.
- Parent Model: A Parent Model is a Deep Learning Model that is broader and more general than the Given Model. There can be multiple Parent Models.
- Closest Parent Model: The Closest Parent Model is the Parent Model that is the closest semantically to the Given Model. There can only be a single Closest Parent Model.
- Child Model: A Child Model is a Deep Learning Model that is more specific and falls under the Given Model. There can be multiple Child Models.
- Closest Child Model: The Closest Child Model is the Child Model that is the closest semantically to the Given Model. There can only be a single Closest Child Model.
- Sibling Model: A Sibling Model is a Deep Learning Model that is not a Semantically Equivalent Model and not a Child Model but should still fall under the Closest Parent Model. There can be multiple Sibling Models.
- Closest Sibling Model: The Closest Sibling Model is the Sibling Model that is the closest semantically to the Given Model. There can only be a single Closest Sibling Model.
- Unrelated Model: An Unrelated Model is a Deep Learning Model that is not related to the Given Model.

### Instructions:
1.  Thoroughly examine the definitions above
2.  Thoroughly examine the Given Model and the Given List.
3.  Identify the list of Semantically Equivalent Models from the Given List.
4.  Justify your choices of Semantically Equivalent Models with explainations.
5.  Identify the list of Parent Models from the Given List.
6.  Justify your choices of Parent Models with explainations.
7.  From the selection of Parent Models, identify the Closest Parent Model.
8.  Justify your choice for the Closest Parent Model with explainations.
9.  Identify the list of Child Models from the Given List.
10. Justify your choices of Child Models with explainations.
11. From the selection of Child Models, identify the Closest Child Model.
12. Justify your choice for the Closest Child Model with explainations.
13. Identify the list of Sibling Models from the Given List.
14. Justify your choices of Sibling Models with explainations.
15. From the selection of Sibling Models, identify the Closest Sibling Model.
16. Justify your choice for the Closest Sibling Model with explainations.
17. **Do not** rename any Deep Learning Models or create new ones.
18. Use only the existing Deep Learning Models in the Given List for categorization.

### Reminder: 
You are not to introduce new categories or models. Your focus is on categorizing within the existing Given List only."""

FIRST_MESSAGE = """### The Given Model:
{}

### Given List (one Deep Learning Model per line):
{}"""

RETRY_MESSAGE = (
    FIRST_MESSAGE
    + "\n"
    + """In your last analyse, the following Deep Learning Models where missing (one per line):
{}"""
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


class Analysis(BaseModel):
    semantically_equivalent_models: list[Explained[str]] = Field(
        description=("List of Semantically Equivalent Models.")
    )
    parent_models: list[Explained[str]] = Field(description=("List of Parent Models."))
    closest_parent_model: Explained[str] = Field(
        description=("The Closest Parent Model.")
    )
    child_models: list[Explained[str]] = Field(description=("List of Child Models."))
    closest_child_model: Explained[str] = Field(
        description=("The Closest Child Model.")
    )
    sibling_models: list[Explained[str]] = Field(
        description=("List of Sibling Models.")
    )
    closest_sibling_model: Explained[str] = Field(
        description=("The Closest Sibling Model.")
    )
    unrelated_models: list[Explained[str]] = Field(
        description=("List of Unrelated Models.")
    )

    @classmethod
    def parse_obj(cls, obj: dict) -> "Analysis":
        # Create a new dictionary with spaces removed from the keys
        cleaned_obj = {}
        for key, value in obj.items():
            key_words = split_words(key.lower(), separators=" -_")
            cleaned_obj["_".join(key_words)] = value
        return super().model_validate(cleaned_obj)

    model_config = ConfigDict(
        populate_by_name=False,
    )


class Response(BaseResponse):
    analysis: Analysis = Field(validation_alias=AliasChoices("analysis", "extractions"))
    query_data: dict
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
