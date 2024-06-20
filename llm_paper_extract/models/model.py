from __future__ import annotations

import enum
import logging
from typing import Any, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

from .. import ROOT_DIR

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
    "which might be incomplete or erroneous, please find more Deep Learning Models, Datasets and Libraries in the "
    "same research paper:\n"
    "{}"
)
_EMPTY_FLAG = "__EMPTY__"


def _caseinsensitive_missing_(cls:enum.Enum, value):
    if isinstance(value, str):
        value = value.strip().lower()
    for member in cls:
        if member.lower() == value:
            return member
    # try:
    #     # Counting on the string version to save us here
    #     value = max(0, int(value) - 1)
    #     if value < 0:
    #         raise IndexError
    #     return list(cls)[value]
    # except ValueError:
    #     pass
    # except IndexError:
    #     pass
    return None


class ResearchType(str, enum.Enum):
    EMPIRICAL = "empirical"
    THEORETICAL = "theoretical"

    @classmethod
    def _missing_(cls, value):
        return _caseinsensitive_missing_(cls, value)


class ModelMode(str, enum.Enum):
    TRAINED = "trained"
    FINE_TUNED = "fine-tuned"
    INFERENCE = "inference"

    @classmethod
    def _missing_(cls, value):
        return _caseinsensitive_missing_(cls, value)


# TODO: make list of contributed, used, referenced models, datasets and libraries
class Role(str, enum.Enum):
    CONTRIBUTED = "contributed"
    USED = "used"
    REFERENCED = "referenced"

    @classmethod
    def _missing_(cls, value):
        return _caseinsensitive_missing_(cls, value)


T = TypeVar("T")
class Explained(BaseModel, Generic[T]):
    value: T | str
    # value_str: str = Field(description="Literal conversion of the value")
    justification: str = Field(
        description="Short justification for the choice of the value",
    )
    quote: str = Field(
        # description=f"Short literal quote from the paper on which the choice of the value was made",
        description="The best literal quote from the paper which supports the value",
    )

    def __eq__(self, other:"Explained"):
        if isinstance(self.value, str):
            return self.value.lower() == other.value.lower()
        else:
            return self.value == other.value

    def __lt__(self, other:"Explained"):
        return self.value < other.value


class Model(BaseModel):
    name: Explained[str] = Field(
        description="Name of the Model",
    )
    # list of carateristics for the model like convolution layers, transformer modules
    type: Explained[str] = Field(
        description="Type of the Model",
    )
    role: Role | str = Field(
        description=f"Was the Model {' or '.join([role.value.lower() for role in Role])} in the scope of the paper"
    )
    mode: ModelMode | str = Field(
        description=f"Was the Model {' or '.join([mode.value.lower() for mode in ModelMode])} in the scope of the paper"
    )
    referenced_paper_title: Explained[str] = Field(
        description="Title of the Model reference paper found in the references section",
    )


class Dataset(BaseModel):
    name: Explained[str] = Field(
        description="Name of the Dataset",
    )
    role: Role | str = Field(
        description=f"Was the Dataset {' or '.join([role.value.lower() for role in Role])} in the scope of the paper"
    )
    referenced_paper_title: Explained[str] = Field(
        description="Title of the Dataset reference paper found in the references section",
    )


class Library(BaseModel):
    name: Explained[str] = Field(
        description="Name of the Library",
    )
    role: Role | str = Field(
        description=f"Was the Library {' or '.join([role.value.lower() for role in Role])} in the scope of the paper"
    )
    referenced_paper_title: Explained[str] = Field(
        description="Title of the Library reference paper found in the references section",
    )


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
    # Should be renamed to primary research field
    research_field: Explained[str] = Field(
        description="Deep Learning research field of the paper",
    )
    # This should have been a list
    research_fields: List[Explained[str] | str] = Field(
        description="List of Deep Learning research fields and research sub-fields of the paper, order by importance",
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


def get_fields(model_cls:BaseModel):
    fields = {}
    for field, info in model_cls.model_fields.items():
        if typing.get_origin(info.annotation) == list:
            try:
                sub_fields = get_fields(info.annotation.__args__[0])
            except AttributeError:
                fields[field] = (info.annotation, Field(description=info.description))
                continue
            fields[field] = (
                List[create_model(field, **sub_fields)],
                Field(description=info.description)
            )
            continue

        try:
            sub_fields = get_fields(info.annotation)
        except AttributeError:
            fields[field] = (info.annotation, Field(description=info.description))
            continue

        if info.annotation.__base__ == Explained:
            sub_fields = {
                field:(sub_fields["value"][0], Field(description=info.description)),
                **{k:v for k,v in sub_fields.items() if k != "value"}
            }
            cls_name = f"{Explained.__name__}[{field}]"
        else:
            cls_name = info.annotation.__name__
        fields[field] = (create_model(cls_name, **sub_fields), Field(description=info.description))

    return fields


def fix_explained_fields():
    import pdb ; pdb.set_trace()
    fields = get_fields(PaperExtractions)
    return create_model(PaperExtractions.__name__, **fields)


def print_model(model_cls:BaseModel, indent = 0):
    for field, info in model_cls.model_fields.items():
        print(" " * indent, field, info)
        if typing.get_origin(info.annotation) == list:
            print_model(info.annotation.__args__[0], indent+2)
            continue

        try:
            print_model(info.annotation, indent+2)
        except AttributeError:
            pass


# PaperExtractions = fix_explained_fields()


class ExtractionResponse(BaseModel):
    paper: str
    words: int
    extractions: PaperExtractions
    usage: Optional[Any]


def empty_paperextractions():
    empty_explained = Explained[str](
        value=_EMPTY_FLAG,
        justification="",
        quote=""
    )
    empty_explained_kwargs = (lambda:{k:v for k,v in empty_explained})()

    empty_explained_modelmode = Explained[ModelMode](**empty_explained_kwargs)
    empty_explained_str = Explained[str](**empty_explained_kwargs)
    empty_explained_researchtype = Explained[ResearchType](**empty_explained_kwargs)
    empty_explained_role = Explained[Role](**empty_explained_kwargs)

    return PaperExtractions(
        description=_EMPTY_FLAG,
        title=empty_explained_str,
        type=empty_explained_researchtype,
        research_field=empty_explained_str,
        sub_research_field=empty_explained_str,
        models=[
            Model(
                name=empty_explained_str,
                role=_EMPTY_FLAG,
                type=empty_explained_str,
                mode=_EMPTY_FLAG
            )
        ],
        datasets=[
            Dataset(
                name=empty_explained_str,
                role=_EMPTY_FLAG,
            )
        ],
        libraries=[
            Library(
                name=empty_explained_str,
                role=_EMPTY_FLAG,
            )
        ],
    )
