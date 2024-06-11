import asyncio
import enum
import logging
from pathlib import Path
from typing import Any, Generic, List, Optional, Tuple, TypeVar

import instructor
from openai.types.chat.chat_completion import ChatCompletion, CompletionUsage
from pydantic import BaseModel, Field

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
        description=f"Confidence level between 0.0 and 1.0 that the value is correct",
    )
    justification: str = Field(
        description=f"Short justification for the choice of the value",
    )
    quote: str = Field(
        # description=f"Short literal quote from the paper on which the choice of the value was made",
        description=f"The best literal quote from the paper which supports the value",
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
        description=f"Name of the Model",
    )
    role: Role | str = Field(
        description=f"Was the Model {' or '.join([role.value.lower() for role in Role])} in the scope of the paper"
    )
    type: str = Field(
        description=f"Type of the Model",
    )
    mode: ModelMode | str = Field(
        description=f"Was the Model {' or '.join([mode.value.lower() for mode in ModelMode])} in the scope of the paper"
    )


class Dataset(_EQ, BaseModel):
    name: Explained[str] = Field(
        description=f"Name of the Dataset",
    )
    role: Role | str = Field(
        description=f"Was the Dataset {' or '.join([role.value.lower() for role in Role])} in the scope of the paper"
    )


class Framework(_EQ, BaseModel):
    name: Explained[str] = Field(
        description=f"Name of the Framework",
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
        description="Complete list of Models found in the paper"
    )
    datasets: List[Dataset] = Field(
        description="Complete list of Datasets found in the paper"
    )
    frameworks: List[Framework] = Field(
        description="Complete list of Frameworks found in the paper"
    )


# class MultiAttempts(BaseModel):
#     attempts: List[PaperExtractions] = Field(
#         description="3 attempts in completing the task"
#     )


class ExtractionResponse(BaseModel):
    paper: str
    words: int
    extractions: PaperExtractions
    usage: Optional[Any]


async def extract_from_research_paper(
        client: instructor.client.Instructor | instructor.client.AsyncInstructor,
        *data: str,
        message: str = _FIRST_MESSAGE,
):
    """Extract Models, Datasets and Frameworks names from a research paper."""
    message = message.format(*data)

    return await client.chat.completions.create_with_completion(
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


async def batch_extract_models_names(
        client: instructor.client.Instructor | instructor.client.AsyncInstructor,
        papers_fn: List[Path]
) -> List[ExtractionResponse]:
    papers_fn = list(papers_fn)
    papers = [paper_fn.name for paper_fn in papers_fn]
    words = []
    for paper_fn in papers_fn:
        count = 0
        for line in paper_fn.read_text().split("\n"):
            count += len([w for w in line.strip().split() if w])
        words.append(count)

    extractions:List[Tuple[PaperExtractions, Any]] = await asyncio.gather(
        *[
            extract_from_research_paper(client, paper_fn.read_text())
            for paper_fn in papers_fn
        ]
    )

    models = [
        [m.name.value for m in e.models]
        for (e,_) in extractions
    ]
    datasets = [
        [d.name.value for d in e.datasets]
        for (e,_) in extractions
    ]
    frameworks = [
        [f.name.value for f in e.frameworks]
        for (e,_) in extractions
    ]

    second_extractions:List[PaperExtractions] = await asyncio.gather(
        *[
            extract_from_research_paper(
                client,
                m,d,f,
                paper_fn.read_text(),
                message=_RETRY_MESSAGE
            )
            for paper_fn, m,d,f in zip(papers_fn, models, datasets, frameworks)
        ]
    )

    all_extractions = zip(extractions, second_extractions)

    return [
        ExtractionResponse(
            paper=p,
            words=w,
            extractions=e,
            usage=c.usage,
        )
        for p, w, paper_extractions in zip(papers, words, all_extractions)
        for (e,c) in paper_extractions
    ]
