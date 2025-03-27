from __future__ import annotations

import logging
from packaging.version import Version
from typing import Optional

from pydantic import BaseModel


from paperext.structured_output._base import (
    BaseResponse,
    ResponseMetadata,
)

logging.basicConfig(level=logging.DEBUG)


class Analysis(BaseModel):
    pages: list[str]


class Response(BaseResponse):
    analysis: Analysis
    metadata: Optional[ResponseMetadata] = ResponseMetadata(
        model_version=Version("1.0.0")
    )
