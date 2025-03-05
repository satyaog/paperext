from __future__ import annotations

import logging
from packaging.version import Version
import typing
from typing import Any, Generic, Optional, TypeVar

from pydantic import AliasChoices, BaseModel, Field

from paperext.sanitize_categorization import split_words
from paperext.structured_output._base import (
    BaseModel,
    BaseResponse,
    ResponseMetadata,
)
from paperext.utils import str_normalize

logging.basicConfig(level=logging.DEBUG)


SYSTEM_MESSAGE = """You are an Expert in Deep Learning Research. Your task is to retreive the title from a given research paper."""

FIRST_MESSAGE = """Retreive the title from the following research paper:
{}"""


class Analysis(BaseModel):
    title: str = Field(description=("The title of the given paper."))


class Response(BaseResponse):
    analysis: Analysis
