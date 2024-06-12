import asyncio
import bdb
from pathlib import Path
from typing import List
import warnings

import instructor
import openai

from . import ROOT_DIR
from .model import batch_extract_models_names
from .utils import build_validation_set


async def ignore_exceptions(validation_set:List[Path]):
    for paper in validation_set:
        try:
            await batch_extract_models_names(client, [paper])
        except bdb.BdbQuit:
            raise
        except Exception as e:
            warnings.warn(f"Failed to extract paper informations from {paper.name}:\n{e}")


if __name__ == "__main__":
    validation_set = build_validation_set(ROOT_DIR / "data/")
    print(*validation_set, sep="\n")

    client = instructor.from_openai(
        openai.AsyncOpenAI()
    )

    asyncio.run(ignore_exceptions(validation_set))
