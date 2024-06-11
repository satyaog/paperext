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

    responses = [
        asyncio.run(
            batch_extract_models_names(
                client,
                [ROOT_DIR / "data/cache/arxiv/2305.18283.txt"]
            )
        )
    ]
    responses = sum(responses, [])

    for response in responses:
        f = (ROOT_DIR / "data/queries/") / response.paper
        f = f.with_suffix(".json")
        i = 0
        while f.with_stem(f"{f.stem}_{i:02}").with_suffix(".json").exists():
            i += 1
        f = f.with_stem(f"{f.stem}_{i:02}").with_suffix(".json")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(response.model_dump_json(indent=2))
        print(response.model_dump_json(indent=2))
