from __future__ import annotations

import enum
import logging
from typing import Any, Generic, List, Optional, TypeVar
import typing

from pydantic import BaseModel, Field

from ..utils import str_normalize

logging.basicConfig(level=logging.DEBUG)

_FIRST_MESSAGE = (
    "Which Deep Learning Models, Datasets and Libraries can you find in the "
    "following research paper:\n"
    "{}"
)
_RETRY_MESSAGE = (
    "Given your precedent list of Models\n"
    "{}\n"
    "your precedent list of Datasets\n"
    "{}\n"
    "your precedent list of Libraries\n"
    "{}\n"
    "which might be incomplete or erroneous, please find more Deep Learning "
    "Models, Datasets and Libraries in the same research paper:\n"
    "{}"
)
_EMPTY_FLAG = "__EMPTY__"


class ResearchType(str, enum.Enum):
    EMPIRICAL = "empirical"
    THEORETICAL = "theoretical"


# TODO: make list of contributed, used, referenced models, datasets and libraries
class Role(str, enum.Enum):
    CONTRIBUTED = "contributed"
    USED = "used"
    REFERENCED = "referenced"


T = TypeVar("T")
class Explained(BaseModel, Generic[T]):
    value: T | str
    justification: str = Field(
        description="Short justification for the choice of the value",
    )
    quote: str = Field(
        description="The best literal quote from the paper which supports the value",
    )

    def __eq__(self, other:"Explained"):
        return str_normalize(str(self.value)) == str_normalize(str(other.value))

    def __lt__(self, other:"Explained"):
        if isinstance(self.value, bool):
            return not self.value < other.value
        return str_normalize(str(self.value)) < str_normalize(str(other.value))


class Model(BaseModel):
    name: Explained[str] = Field(
        description="Name of the Model",
    )
    caracteristics: List[Explained[str]] = Field(
        description="List of carateristics of the Model like convolution layers, transformer modules",
    )
    is_contributed: Explained[bool] = Field(
        description="Was the Model a contribution to the research field, in the scope of the paper",
    )
    is_executed: Explained[bool] = Field(
        description="Was the Model executed on GPU or CPU, in the scope of the paper",
    )
    is_compared: Explained[bool] = Field(
        description="Was the Model compared numerically to other models, in the scope of the paper",
    )
    referenced_paper_title: Explained[str] = Field(
        description="Title of reference paper of the Model, found in the references section",
    )

    def __lt__(self, other:"Explained"):
        for (k, v), (ok, ov) in zip(self, other):
            if k != ok:
                break
            if k in ("caracteristics","referenced_paper_title",):
                continue
            if v == ov:
                continue
            return v < ov
        return False


class Dataset(BaseModel):
    name: Explained[str] = Field(
        description="Name of the Dataset",
    )
    role: Role | str = Field(
        description=f"Was the Dataset {' or '.join([role.value.lower() for role in Role])} in the scope of the paper"
    )
    referenced_paper_title: Explained[str] = Field(
        description="Title of reference paper of the Dataset, found in the references section",
    )

    def __lt__(self, other:"Explained"):
        for (k, v), (ok, ov) in zip(self, other):
            if k != ok:
                break
            if k in ("caracteristics","referenced_paper_title",):
                continue
            if v == ov:
                continue
            return v < ov
        return False


class Library(BaseModel):
    name: Explained[str] = Field(
        description="Name of the Library",
    )
    role: Role | str = Field(
        description=f"Was the Library {' or '.join([role.value.lower() for role in Role])} in the scope of the paper"
    )
    referenced_paper_title: Explained[str] = Field(
        description="Title of reference paper of the Library, found in the references section",
    )

    def __lt__(self, other:"Explained"):
        for (k, v), (ok, ov) in zip(self, other):
            if k != ok:
                break
            if k in ("caracteristics","referenced_paper_title",):
                continue
            if v == ov:
                continue
            return v < ov
        return False


class PaperExtractions(BaseModel):
    title: Explained[str] = Field(
        description="Title of the paper",
    )
    description: str = Field(
        description="Short description of the paper",
    )
    type: Explained[ResearchType] = Field(
        description=f"Is the paper an {' or a '.join([rt.value.lower() + ' study' for rt in ResearchType])}",
    )
    primary_research_field: Explained[str] = Field(
        description="Primary Deep Learning research field of the paper",
    )
    sub_research_fields: List[Explained[str]] = Field(
        description="List of Deep Learning sub-research fields and research sub-fields of the paper, order by importance",
    )
    models: List[Model] = Field(
        description="All Models found in the paper"
    )
    datasets: List[Dataset] = Field(
        description="All Datasets found in the paper"
    )
    libraries: List[Library] = Field(
        description="All Deep Learning Libraries found in the paper"
    )


# PaperExtractions = fix_explained_fields()


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


def _empty_fields(model_cls:BaseModel):
    try:
        iter_fields = model_cls.model_fields.items()
    except AttributeError:
        if typing.get_origin(model_cls) == list:
            return [_empty_fields(model_cls.__args__[0])]
        else:
            return _EMPTY_FLAG

    if _is_base(model_cls, Explained):
        fields = {
            k: (_empty_fields(v) if k == "value" else "")
            for k, v in iter_fields
        }
    else:
        fields = {}
        for k, field in iter_fields:
            fields[k] = _empty_fields(field.annotation)

    return fields


def empty_model(model_cls):
    return model_cls(
        **_empty_fields(model_cls)
    )
