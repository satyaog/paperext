from __future__ import annotations

import logging
import typing
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

from paperext.sanitize_categorization import split_words
from paperext.utils import str_normalize

logging.basicConfig(level=logging.DEBUG)

SYSTEM_MESSAGE = """You are an Expert in Deep Learning Research. Your task is to categorize a specified Deep Learning Research Domain within a hierarchical structure, relative to a provided list of research domains.

### Definitions:
- Given Domain: The Given Domain is the Deep Learning Research Domain that needs to be categorized.
- Given List: The Given List is the provided list of Deep Learning Research Domains.
- Semantically Equivalent Domain: A Semantically Equivalent Domain is a Deep Learning Research Domain that is the same semantically to the Given Domain. There can be multiple Semantically Equivalent Domains.
- Parent Domain: A Parent domain is a Deep Learning Research Domain that is broader and more general than the Given Domain. There can be multiple Parent Domains.
- Closest Parent Domain: The Closest Parent Domain is the Parent Domain that is the closest semantically to the Given Domain. There can only be a single Closest Parent Domain.
- Child Domain: A Child domain is a Deep Learning Research Domain that is more specific and falls under the Given Domain. There can be multiple Child Domains.
- Sibling Domain: A Sibling domain is a Deep Learning Research Domain that is not a Semantically Equivalent Domain and not a Child Domain but should still fall under the Closest Parent Domain. There can be multiple Sibling Domains.
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
11. Identify the list of Sibling Domains from the Given List.
12. Justify your choices of Sibling Domains with explainations.
13. **Do not** rename any Deep Learning Research Domains or create new ones.
14. Use only the existing Deep Learning Research Domains in the Given List for categorization.

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
    sibling_domains: list[Explained[str]] = Field(
        description=("List of Sibling Domains.")
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
