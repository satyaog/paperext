from typing import Any, List, Optional

from pydantic import BaseModel, Field

from .model import Explained

_FIRST_MESSAGE = (
    "To which model or algorithm categories belong the model or algorithm `{model_name}` "
    "following research paper:\n"
    "{paper_text}"
)


class CategoryExtraction(BaseModel):
    name: Explained[str] = Field(description="")


class ModelCategoryExtraction(BaseModel):
    name: str = Field(
        description="Name of the model for which we want to know the model categories to which it belongs."
    )
    categories: List[Explained] = Field(
        description=(
            "The model categories to which the model belongs. "
            "It may be values like `transformer`, `neural network`, "
            "`graph neural network`, `autoencoder`, `diffusion model`, "
            "`adversarial neural network`, `multi layer perceptron` or other "
            "model categories. "
            "If the given name is not one of a model but rather of an algorithm, "
            "it should have the category `algorithm`. Reinforcement learning "
            "algorithms should also have the category `reinforcement learning`. "
            "Optimization algorithms should also have the category `optimization`. "
            "If the name is not one of a model nor an algorithm, then it should have "
            "the category `others`."
        )
    )


class ExtractionResponse(BaseModel):
    paper: str
    words: int
    extractions: ModelCategoryExtraction
    usage: Optional[Any]
