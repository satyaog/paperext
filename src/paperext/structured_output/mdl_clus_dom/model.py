from __future__ import annotations

import logging
import re
import typing
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

from paperext.utils import str_normalize

logging.basicConfig(level=logging.DEBUG)


# SYSTEM_MESSAGE = """You are an Expert in Deep Learning Research. Your task is to
# perform hierarchical clustering on a list of Deep Learning Research Domains. You
# must organize the domains into three primary root categories:

# 1. **abstract_research_topics**: This category includes domains that are related to theoretical or abstract research topics within the field of Deep Learning.
# 2. **application_domains**: This category includes domains that focus on practical applications or real-world use cases of Deep Learning techniques.
# 3. **ignore**: Place any entries that are acronyms, abbreviations, or anything that does not represent a valid Deep Learning Research Domain.

# ### Instructions:
# - Use a bottom-up hierarchical clustering approach: Each domain starts in its own cluster and clusters are merged based on similarity.
# - Do **not** rename any Deep Learning Research Domains. Only group them into the specified categories.
# - Provide justifications for any classifications you make.
# - Ensure that every provided domain is included in one of the categories.

# Your result should be structured as a dictionary containing the three
# categories. The clustering should respect the relationships between the domains
# and ensure the integrity of each category's structure.

# ### Format for your response:
# - A dictionary containing three keys: `abstract_research_topics`, `application_domains`, and `ignore`.
# - Each key should contain the hierarchical clusters as nested dictionaries, with justifications for each grouping."""
# FIRST_MESSAGE = """The list of Deep Learning Research Domains to cluster is as
# follows:

# {}"""
# _EMPTY_FLAG = "__EMPTY__"


# T = TypeVar("T")


# class Explained(BaseModel, Generic[T]):
#     value: T
#     justification: str = Field(
#         description="A detailed explanation for the choice of the value.",
#     )

#     def __eq__(self, other: "Explained"):
#         return str_normalize(str(self.value)) == str_normalize(str(other.value))

#     def __lt__(self, other: "Explained"):
#         if isinstance(self.value, bool):
#             return not self.value < other.value
#         return str_normalize(str(self.value)) < str_normalize(str(other.value))


# class HierarchicalClustering(BaseModel):
#     abstract_research_topics: list[list[str]] = Field(
#         description=(
#             "Hierarchical clustering of theoretical or abstract research topics "
#             "related to Deep Learning."
#         )
#     )
#     application_domains: list[list[str]] = Field(
#         description=(
#             "Hierarchical clustering of practical applications or real-world use "
#             "cases of Deep Learning."
#         )
#     )
#     ignore: list[str] = Field(
#         description=(
#             "Domains or entries that do not fit into the relevant Deep Learning "
#             "categories."
#         )
#     )
#     # domain_hierarchies_aliases: List[Explained[tuple[str, List[str]]]] = Field(
#     #     description=(
#     #         "Mapping of close to intentical Deep Learning Research Domains from "
#     #         "the provided list."
#     #     )
#     # )

SYSTEM_MESSAGE = """You are an Expert in Deep Learning Research. Your task is to select the most generic and comprehensive Research Domain from a provided list of Research Domains.

### Instructions:
- Carefully analyze the provided list of Research Domains to understand each term individually
- Identify how the concepts relate to each others
- Identify the Research Domain from the list that is the most general, meaning it can encompass or cover a broad range of the other Research Domains listed.
- Justify your choice by explaining why this Research Domain is the most generic and how it encompasses the others.
- Identify the remaining rejected Research Domains from the list
- You must not introduce any new domains, nor create new categories; you must choose from the existing list only.
"""

FIRST_MESSAGE = """Select the most generic and comprehensive Research Domain from the following list of Research Domains with each Research Domain sitting on its own line:
{}"""

RETRY_MESSAGE = (
    """Your previous selection "{}" is not an exact match for any Research Domain in the provided list and must be rejected. You must choose from the list only, without introducing new domains or categories. """
    + FIRST_MESSAGE
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


class GenericDomain(BaseModel):
    generic_domain: Explained[str] = Field(
        description=(
            "The most generic Research Domain that best encompasses the "
            "provided list of domains."
        )
    )
    rejected_domains: list[str] = Field(description="The list of rejected domains.")

    @classmethod
    def parse_obj(cls, obj: dict) -> "GenericDomain":
        # Create a new dictionary with spaces removed from the keys
        cleaned_obj = {
            key.lower().replace(" ", ""): value for key, value in obj.items()
        }
        return super().model_validate(cleaned_obj)

    class Config:
        # This allows the model to use the cleaned keys for validation
        populate_by_name = False


class Response(BaseModel):
    paper: str
    words: int
    extractions: GenericDomain
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
