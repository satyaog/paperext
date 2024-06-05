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


class _Role(str, enum.Enum):
    CONTRIBUTED = "Contributed"
    USED = "Used"
    REFERENCE = "Reference"


T = TypeVar("T")
class Explained(BaseModel, Generic[T]):
    value: T
    excerpt: str = Field(
        description=f"Short litteral excerpt from the paper justifying the choice of the value",
    )


class Model(BaseModel):
    name: Explained[str] = Field(
        description=f"Name of the Dataset",
    )
    role: _Role = Field(
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


class Dataset(BaseModel):
    name: Explained[str] = Field(
        description=f"Name of the Dataset",
    )
    role: _Role = Field(
        description="Role of the Dataset in the scope of the paper"
    )
    keywords: List[str] = Field(
        description="Keywords related to the Dataset"
    )


class Framework(BaseModel):
    name: Explained[str] = Field(
        description=f"Name of the Framework",
    )
    role: _Role = Field(
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
        description="List of Models referenced in the paper"
    )
    datasets: List[Dataset] = Field(
        description="List of Datasets referenced in the paper"
    )
    frameworks: List[Framework] = Field(
        description="List of Frameworks referenced in the paper"
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
