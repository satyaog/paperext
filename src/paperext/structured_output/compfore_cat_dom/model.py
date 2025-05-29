from __future__ import annotations

import logging
from packaging.version import Version
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field

from paperext.structured_output._base import (
    BaseModel,
    BaseResponse,
    ResponseMetadata,
    _base_empty_fields,
    _base_empty_response,
)
from paperext.utils import str_normalize

logging.basicConfig(level=logging.DEBUG)


SYSTEM_MESSAGE = """You are an Expert in Deep Learning Research with extensive knowledge of research domains, methodologies, and their hierarchical relationships. Your task is to accurately categorize Deep Learning Research Domains within a provided hierarchical structure.

### Your Role and Expertise:
- You possess deep understanding of Deep Learning research areas, their interconnections, and hierarchical relationships
- You can identify subtle distinctions between related research domains
- You understand both theoretical foundations and practical applications in Deep Learning
- You are precise in matching domains to their exact categories

### Task Instructions:
1. Analysis Phase:
   - Carefully examine the provided hierarchical structure
   - Review the list of Deep Learning Research Domains to categorize
   - Consider both explicit and implicit relationships between domains

2. Categorization Process:
   - Select ONLY ONE domain from the provided list that you are most confident about its categorization within the hierarchical structure
   - Identify its exact location in the hierarchical structure
   - Determine the direct parent category that exists in the hierarchical structure
   - Ensure the categorization maintains logical consistency with the existing hierarchical structure
   - Provide clear reasoning for your categorization decisions

3. Quality Requirements:
   - The selected domain MUST be an exact match within the provided list
   - The parent category MUST be an exact match within the hierarchical structure
   - Do not introduce new domains or categories
   - Make selections based on highest confidence
   - Provide clear reasoning for your categorization
   - Maintain consistency with the existing hierarchical structure

### The Hierarchical Structure:

{}
"""

FIRST_MESSAGE = """### List of Deep Learning Research Domains to Categorize:

{}
"""

RETRY_MESSAGE_SELECTED = (
    """Your previous domain selection "{}" was not an exact match for any domain in the provided list. Please follow these guidelines:
- The selected domain MUST be an exact match within the provided list
- The parent category MUST be an exact match within the hierarchical structure
- Do not introduce new domains or categories

"""
    + FIRST_MESSAGE
)

RETRY_MESSAGE_PARENT = (
    """Your previous identified parent category "{}" was not an exact match for any category in the provided hierarchical structure. Please follow these guidelines:
- The selected domain MUST be an exact match within the provided list
- The parent category MUST be an exact match within the hierarchical structure
- Do not introduce new domains or categories

"""
    + FIRST_MESSAGE
)


T = TypeVar("T")


class Explained(BaseModel, Generic[T]):
    value: T
    reasoning: str = Field(
        description="A detailed explanation for the choice of the value.",
    )

    def __eq__(self, other: "Explained"):
        return str_normalize(str(self.value)) == str_normalize(str(other.value))

    def __lt__(self, other: "Explained"):
        if isinstance(self.value, bool):
            return not self.value < other.value
        return str_normalize(str(self.value)) < str_normalize(str(other.value))


class Analysis(BaseModel):
    selected_domain: str = Field(
        description="The Deep Learning Research Domain selected from the provided list.",
    )
    parent_category: str = Field(
        description="The direct parent category in the hierarchical structure where the selected domain belongs.",
    )
    reasoning: str = Field(
        description="A detailed explanation for the choice of the domain and parent category.",
    )


class Response(BaseResponse):
    analysis: Analysis
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
