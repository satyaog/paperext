from __future__ import annotations

import logging
from packaging.version import Version
import typing
from typing import Any, Generic, Optional, TypeVar

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from paperext.sanitize_categorization import split_words
from paperext.structured_output._base import (
    BaseModel,
    BaseResponse,
    ResponseMetadata,
)
from paperext.utils import str_normalize

logging.basicConfig(level=logging.DEBUG)

SYSTEM_MESSAGE = """You are an Expert in Deep Learning Research. Your task is to categorize a specified Deep Learning Research Domain within a hierarchical structure, relative to a provided list of research domains.

### Definitions:
- Given Domain: The Given Domain is the Deep Learning Research Domain that needs to be categorized.
- Given List: The Given List is the provided list of Deep Learning Research Domains.
- Semantically Equivalent Domain: A Semantically Equivalent Domain is a Deep Learning Research Domain that is the same semantically to the Given Domain. There can be multiple Semantically Equivalent Domains.
- Parent Domain: A Parent Domain is a Deep Learning Research Domain that is broader and more general than the Given Domain. There can be multiple Parent Domains.
- Closest Parent Domain: The Closest Parent Domain is the Parent Domain that is the closest semantically to the Given Domain. There can only be a single Closest Parent Domain.
- Child Domain: A Child Domain is a Deep Learning Research Domain that is more specific and falls under the Given Domain. There can be multiple Child Domains.
- Closest Child Domain: The Closest Child Domain is the Child Domain that is the closest semantically to the Given Domain. There can only be a single Closest Child Domain.
- Sibling Domain: A Sibling Domain is a Deep Learning Research Domain that is not a Semantically Equivalent Domain and not a Child Domain but should still fall under the Closest Parent Domain. There can be multiple Sibling Domains.
- Closest Sibling Domain: The Closest Sibling Domain is the Sibling Domain that is the closest semantically to the Given Domain. There can only be a single Closest Sibling Domain.
- Unrelated Domain: An Unrelated Domain is a Deep Learning Research Domain that is not not related to the Given Domain.

### Instructions:
1.  Thoroughly examine the definitions above
2.  Thoroughly examine the Given Domain and the Given List.
3.  Identify the list of Semantically Equivalent Domains from the Given List.
4.  Justify your choices of Semantically Equivalent Domains with explainations.
5.  Identify the list of Parent Domains from the Given List.
6.  Justify your choices of Parent Domains with explainations.
7.  From the selection of Parent Domains, identify the Closest Parent Domain.
8.  Justify your choice for the Closest Parent Domain with explainations.
9.  Identify the list of Child Domains from the Given List.
10. Justify your choices of Child Domains with explainations.
11. From the selection of Child Domains, identify the Closest Child Domain.
12. Justify your choice for the Closest Child Domain with explainations.
13. Identify the list of Sibling Domains from the Given List.
14. Justify your choices of Sibling Domains with explainations.
15. From the selection of Sibling Domains, identify the Closest Sibling Domain.
16. Justify your choice for the Closest Sibling Domain with explainations.
17. **Do not** rename any Deep Learning Research Domains or create new ones.
18. Use only the existing Deep Learning Research Domains in the Given List for categorization.

### Reminder: 
You are not to introduce new categories or domains. Your focus is on categorizing within the existing Given List only."""

FIRST_MESSAGE = """### The Given Domain:
{}

### Given List (one Deep Learning Research Domain per line):
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
    semantically_equivalent_domains: list[Explained[str]] = Field(
        description=("List of Semantically Equivalent Domains.")
    )
    parent_domains: list[Explained[str]] = Field(
        description=("List of Parent Domains.")
    )
    closest_parent_domain: Explained[str] = Field(
        description=("The Closest Parent Domain.")
    )
    child_domains: list[Explained[str]] = Field(description=("List of Child Domains."))
    closest_child_domain: Explained[str] = Field(
        description=("The Closest Child Domain.")
    )
    sibling_domains: list[Explained[str]] = Field(
        description=("List of Sibling Domains.")
    )
    closest_sibling_domain: Explained[str] = Field(
        description=("The Closest Sibling Domain.")
    )
    unrelated_domains: list[Explained[str]] = Field(
        description=("List of Unrelated Domains.")
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

    return model_cls(**empty_fields)
