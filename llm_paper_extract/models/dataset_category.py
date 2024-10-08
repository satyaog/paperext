from typing import Any, List, Optional

from pydantic import BaseModel, Field

from .model import Explained

_FIRST_MESSAGE = (
    "To which dataset or environment categories belong the dataset or environment `{dataset_name}` "
    "in the following research paper:\n"
    "{paper_text}"
)


class CategoryExtraction(BaseModel):
    name: Explained[str] = Field(description="")


class DatasetCategoryExtraction(BaseModel):
    name: str = Field(
        description="Name of the dataset for which we want to know the dataset categories to which it belongs."
    )
    categories: List[Explained] = Field(
        description=(
            "The dataset categories to which the dataset belongs. "
            "It may be values like `Computer Vision`, `Images`, `Video`, "
            "`Natural Language Processing`, `Text`, `Dialog`, "
            "`Reinforcement Learning Environment`, `Biology and Chemistry` or other "
            "dataset categories. "
        )
    )


class ExtractionResponse(BaseModel):
    paper: str
    words: int
    extractions: DatasetCategoryExtraction
    usage: Optional[Any]
