import asyncio
import enum
from pathlib import Path
from typing import Generic, List, TypeVar

import instructor
from pydantic import BaseModel, Field


class ResearchType(str, enum.Enum):
    """Enumeration for scientific paper type"""
    EMPIRICAL = "empirical"
    THEORETICAL = "theoretical"


class ModelMode(str, enum.Enum):
    """Enumeration for Deep Learning Model mode"""
    TRAINED = "trained"
    FINE_TUNED = "fine-tuned"
    INFERENCE = "inference"


class Role(str, enum.Enum):
    CONTRIBUTED = "Contributed"
    USED = "Used"
    REFERENCE = "Reference"


T = TypeVar("T")
class Explained(BaseModel, Generic[T]):
    value: T
    confidence: float = Field(
        description=f"Confidence level between 0.0 and 1.0 that the value is correct",
    )
    justification: str = Field(
        description=f"Short justification for the choice of the value",
    )
    quote: str = Field(
        description=f"Short litteral quote from the paper on which the choice of the value was made",
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
        description=f"Name of the Dataset",
    )
    role: Role = Field(
        description="Role of the Model in the scope of the paper"
    )
    type: str = Field(
        description=f"Type of the Model",
    )
    mode: ModelMode = Field(
        description=f"Was the model {' or '.join([mode.value.lower() for mode in ModelMode])} in the scope of the paper"
    )
    keywords: List[str] = Field(
        description="Keywords related to the Model"
    )


class Dataset(_EQ, BaseModel):
    name: Explained[str] = Field(
        description=f"Name of the Dataset",
    )
    role: Role = Field(
        description="Role of the Dataset in the scope of the paper"
    )
    keywords: List[str] = Field(
        description="Keywords related to the Dataset"
    )


class Framework(_EQ, BaseModel):
    name: Explained[str] = Field(
        description=f"Name of the Framework",
    )
    role: Role = Field(
        description="Role of the Framework in the scope of the paper"
    )
    keywords: List[str] = Field(
        description="Keywords related to the Framework"
    )


class PaperExtractions(BaseModel):
    type: ResearchType = Field(
        description=f"Is the paper an {' or a '.join([rt.value.lower() + ' study' for rt in ResearchType])}",
    )
    title: Explained[str] = Field(
        description="Title of the paper",
    )
    description: str = Field(
        description="Short description of the paper",
    )
    research_field: Explained[str] = Field(
        description="Deep Learning research field, like Natural Language Processing, of the paper",
    )
    sub_research_field: Explained[str] = Field(
        description="Deep Learning sub-research field of the paper",
    )
    models: List[Model] = Field(
        description="List of Models found in the paper"
    )
    datasets: List[Dataset] = Field(
        description="List of Datasets found in the paper"
    )
    frameworks: List[Framework] = Field(
        description="List of Frameworks found in the paper"
    )


class ExtractionResponse(BaseModel):
    paper: str
    words: int
    extractions: PaperExtractions


async def extract_from_research_paper(
        client: instructor.client.Instructor | instructor.client.AsyncInstructor,
        data: str
):
    """Extract Models, Datasets and Frameworks names from a research paper."""
    return await client.chat.completions.create(
        model="gpt-4o",
        # model="gpt-3.5-turbo",
        # stream=True,
        response_model=PaperExtractions,
        messages=[
            {
                "role": "system",
                "content": f"Your role is to extract Deep Learning Models, Datasets and Frameworks from a given Deep Learning research paper. "
                          #  f"The Models, Datasets and Frameworks must be used in the paper "
                          #  f"and / or the comparison analysis of the results of the "
                          #  f"paper. The papers provided will be a convertion from pdf to text, which could imply some formatting issues.",
            },
            {
                "role": "user",
                "content": f"Which Deep Learning Models, Datasets and Frameworks can you find in the following research paper : {data}",
            },
        ]
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
    extractions = await asyncio.gather(
        *[extract_from_research_paper(client, paper_fn.read_text()) for paper_fn in papers_fn]
    )
    return [
        ExtractionResponse(
            paper=p,
            words=w,
            extractions=e,
        )
        for p, w, e in zip(papers, words, extractions)
    ]
