import asyncio

import instructor
import openai

from . import ROOT_DIR
from .model import batch_extract_models_names
from .utils import build_validation_set


if __name__ == "__main__":
    validation_set = build_validation_set(ROOT_DIR / "data/")

    client = instructor.from_openai(
        openai.AsyncOpenAI()
    )

    asyncio.run(
        batch_extract_models_names(
            client,
            [ROOT_DIR / "data/cache/arxiv/2402.05468.txt"]
        )
    )
