from __future__ import annotations

import asyncio
import enum
import logging
from pathlib import Path
from typing import Any, Generic, List, Optional, Tuple, TypeVar

import instructor
import openai
from openai.types.chat.chat_completion import CompletionUsage
from pydantic import BaseModel, Field
import pydantic_core

from llm_paper_extract import ROOT_DIR

logging.basicConfig(level=logging.DEBUG)

_FIRST_MESSAGE = (
    "Which Deep Learning Models, Datasets and Frameworks can you find in the "
    "following research paper:\n"
    "{}"
)
_RETRY_MESSAGE = (
    "Given your precedent list of Models\n"
    "{}\n"
    "your precedent list of Datasets\n"
    "{}\n"
    "your precedent list of Frameworks\n"
    "{}\n"
    "please find more Deep Learning Models, Datasets and Frameworks in the "
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
    try:
        value = int(value)
        if value < 0:
            raise IndexError
        return list(cls)[value]
    except ValueError:
        pass
    except IndexError:
        pass
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
    confidence: float = Field(
        description="Confidence level between 0.0 and 1.0 that the value is correct",
    )
    justification: str = Field(
        description="Short justification for the choice of the value",
    )
    quote: str = Field(
        # description=f"Short literal quote from the paper on which the choice of the value was made",
        description="The best literal quote from the paper which supports the value",
    )

    def __eq__(self, other:"Explained"):
        return self.value == other.value

    def __lt__(self, other:"Explained"):
        return self.value < other.value or (
            self.value == other.value and self.confidence > other.confidence
        )


class _EQ():
    def __eq__(self, other):
        for (k1,v1), (k2,v2) in zip(self, other):
            if k1 != k2:
                return False
            if k1 == "keywords":
                continue
            if v1 != v2:
                return False
        else:
            return True

    def __ne__(self, other) -> bool:
        return not self == other


class Model(_EQ, BaseModel):
    name: Explained[str] = Field(
        description="Name of the Model",
    )
    role: Role | str = Field(
        description=f"Was the Model {' or '.join([role.value.lower() for role in Role])} in the scope of the paper"
    )
    type: Explained[str] = Field(
        description="Type of the Model",
    )
    mode: ModelMode | str = Field(
        description=f"Was the Model {' or '.join([mode.value.lower() for mode in ModelMode])} in the scope of the paper"
    )


class Dataset(_EQ, BaseModel):
    name: Explained[str] = Field(
        description="Name of the Dataset",
    )
    role: Role | str = Field(
        description=f"Was the Dataset {' or '.join([role.value.lower() for role in Role])} in the scope of the paper"
    )


class Framework(_EQ, BaseModel):
    name: Explained[str] = Field(
        description="Name of the Framework",
    )
    role: Role | str = Field(
        description=f"Was the Framework {' or '.join([role.value.lower() for role in Role])} in the scope of the paper"
    )


class PaperExtractions(BaseModel):
    description: str = Field(
        description="Short description of the paper",
    )
    title: Explained[str] = Field(
        description="Title of the paper",
    )
    type: Explained[ResearchType] = Field(
        description=f"Is the paper an {' or a '.join([rt.value.lower() + ' study' for rt in ResearchType])}",
    )
    research_field: Explained[str] = Field(
        description="Deep Learning research field of the paper",
    )
    sub_research_field: Explained[str] = Field(
        description="Deep Learning sub-research field of the paper",
    )
    models: List[Model] = Field(
        description="All Models found in the paper"
    )
    datasets: List[Dataset] = Field(
        description="All Datasets found in the paper"
    )
    frameworks: List[Framework] = Field(
        description="All Frameworks found in the paper"
    )


class ExtractionResponse(BaseModel):
    paper: str
    words: int
    extractions: PaperExtractions
    usage: Optional[Any]


def empty_paperextractions():
    empty_explained = Explained[str](
        value=_EMPTY_FLAG,
        confidence=0.0,
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
        frameworks=[
            Framework(
                name=empty_explained_str,
                role=_EMPTY_FLAG,
            )
        ],
    )


async def extract_from_research_paper(
        client: instructor.client.Instructor | instructor.client.AsyncInstructor,
        message: str,
) -> Tuple[PaperExtractions, CompletionUsage]:
    """Extract Models, Datasets and Frameworks names from a research paper."""
    retries = [True] * 2
    while True:
        try:
            extractions, completion = await client.chat.completions.create_with_completion(
                model="gpt-4o",
                # model="gpt-3.5-turbo",
                response_model=PaperExtractions,
                messages=[
                    {
                        "role": "system",
                        "content": f"Your role is to extract Deep Learning Models, Datasets and Frameworks from a given research paper."
                                  #  f"The Models, Datasets and Frameworks must be used in the paper "
                                  #  f"and / or the comparison analysis of the results of the "
                                  #  f"paper. The papers provided will be a convertion from pdf to text, which could imply some formatting issues.",
                    },
                    {
                        "role": "user",
                        "content": message,
                    },
                ],
                max_retries=0
            )
            return extractions, completion.usage
        except openai.RateLimitError as e:
            asyncio.sleep(60)
            if retries:
                retries.pop()
                continue
            raise e


async def batch_extract_models_names(
        client: instructor.client.Instructor | instructor.client.AsyncInstructor,
        papers_fn: List[Path]
) -> List[ExtractionResponse]:
    for paper_fn in papers_fn:
        paper = paper_fn.name

        count = 0
        for line in paper_fn.read_text().splitlines():
            count += len([w for w in line.strip().split() if w])

        data = []

        for i, message in enumerate((_FIRST_MESSAGE, _RETRY_MESSAGE)):
            f = (ROOT_DIR / "data/queries/") / paper
            f = f.with_stem(f"{f.stem}_{i:02}").with_suffix(".json")

            try:
                response = ExtractionResponse.model_validate_json(f.read_text())
            except (
                FileNotFoundError,
                pydantic_core._pydantic_core.ValidationError
            ):
                message = message.format(*data, paper_fn.read_text())

                extractions, usage = await extract_from_research_paper(client, message)

                response = ExtractionResponse(
                    paper=paper,
                    words=count,
                    extractions=extractions,
                    usage=usage,
                )

                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(response.model_dump_json(indent=2))

            print(response.model_dump_json(indent=2))

            models = [
                m.name.value for m in response.extractions.models
            ]
            datasets = [
                d.name.value for d in response.extractions.datasets
            ]
            frameworks = [
                f.name.value for f in response.extractions.frameworks
            ]

            data = [models, datasets, frameworks]
