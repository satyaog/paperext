import argparse
import asyncio
import bdb
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import instructor
import pydantic_core

from paperext import LOG_DIR as _LOG_DIR
from paperext import ROOT_DIR
from paperext.models.model import (_FIRST_MESSAGE, ExtractionResponse,
                                   PaperExtractions)
from paperext.utils import Paper, build_validation_set

# Set logging to DEBUG to print OpenAI requests
LOG_DIR = _LOG_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_DIR.mkdir(parents=True)
logging.basicConfig(
    filename=LOG_DIR / f"{Path(__file__).stem}.dbg", level=logging.DEBUG, force=True
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler(LOG_DIR / "query.out"))
logger.setLevel(logging.INFO)

PROG = f"{Path(__file__).stem.replace('_', '-')}"

DESCRIPTION = """
Utility to query Chat-GPT on papers

Logs will be written in logs/query.dbg and logs/query.out
"""

EPILOG = f"""
Example:
  $ {PROG} --input data/query_set.txt
"""

PLATFORMS = {}

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel

    vertexai.init(project=os.environ.get("PAPEREXT_VERTEX_PROJECT"))

    def _client():
        model = "models/gemini-1.5-pro"
        client = instructor.from_vertexai(GenerativeModel(model_name=model))
        _create_with_completion = client.chat.completions.create_with_completion

        def _wrap(*args, **kwargs):
            # Gemini does not support "system" role
            system_messages = []
            for message in kwargs["messages"][:]:
                if message["role"] == "system":
                    system_messages.append(message["content"])
                    kwargs["messages"].remove(message)
                    continue
                if system_messages:
                    message["content"] = "\n".join(
                        (*system_messages, message["content"])
                    )
                    system_messages = []
            extractions, completion = _create_with_completion(*args, **kwargs)
            # completion.usage_metadata doesn't seams to be serializable
            # Unable to serialize unknown type: <class
            # 'google.cloud.aiplatform_v1beta1.types.prediction_service.GenerateContentResponse.UsageMetadata'>
            usage = {
                "cached_content_token_count": completion.usage_metadata.cached_content_token_count,
                "candidates_token_count": completion.usage_metadata.cached_content_token_count,
                "prompt_token_count": completion.usage_metadata.cached_content_token_count,
                "total_token_count": completion.usage_metadata.total_token_count,
            }
            return extractions, usage

        client.chat.completions.create_with_completion = _wrap
        return client

    PLATFORMS["vertexai"] = _client
except ModuleNotFoundError as e:
    logging.info(e, exc_info=True)

try:
    import openai
    from openai.types.chat.chat_completion import CompletionUsage

    def _client():
        model = "gpt-4o"
        client = instructor.from_openai(
            # TODO: update to use the new feature Mode.TOOLS_STRICT
            # https://openai.com/index/introducing-structured-outputs-in-the-api/
            openai.AsyncOpenAI(),
            mode=instructor.Mode.TOOLS_STRICT,
        )
        _create_with_completion = client.chat.completions.create_with_completion

        async def _wrap(*args, **kwargs):
            extractions, completion = await _create_with_completion(
                model=model, *args, **kwargs
            )
            return extractions, completion.usage

        client.chat.completions.create_with_completion = _wrap
        return client

    PLATFORMS["openai"] = _client
except ModuleNotFoundError as e:
    logging.info(e, exc_info=True)


async def extract_from_research_paper(
    client: instructor.client.Instructor | instructor.client.AsyncInstructor,
    message: str,
) -> Tuple[PaperExtractions, CompletionUsage]:
    """Extract Models, Datasets and Frameworks names from a research paper."""
    retries = [True] * 1
    while True:
        try:
            result = client.chat.completions.create_with_completion(
                # model="gpt-4o",
                response_model=PaperExtractions,
                messages=[
                    {
                        "role": "system",
                        "content": f"Your role is to extract Deep Learning Models, Datasets and Deep Learning Libraries from a given research paper.",
                        #  f"The Models, Datasets and Frameworks must be used in the paper "
                        #  f"and / or the comparison analysis of the results of the "
                        #  f"paper. The papers provided will be a convertion from pdf to text, which could imply some formatting issues.",
                    },
                    {
                        "role": "user",
                        "content": message,
                    },
                ],
                max_retries=1,
            )

            try:
                extractions, usage = result
            except TypeError:
                extractions, usage = await result

            return extractions, usage
        except openai.RateLimitError as e:
            asyncio.sleep(60)
            if retries:
                retries.pop()
                continue
            raise e


async def batch_extract_models_names(
    client: instructor.client.Instructor | instructor.client.AsyncInstructor,
    papers_fn: List[Path],
    destination: Path = (ROOT_DIR / "data/queries/"),
) -> List[ExtractionResponse]:
    destination.mkdir(parents=True, exist_ok=True)

    for paper_fn in papers_fn:
        paper = paper_fn.name

        count = 0
        for line in paper_fn.read_text().splitlines():
            count += len([w for w in line.strip().split() if w])

        data = []

        for i, message in enumerate((_FIRST_MESSAGE,)):
            f = destination / paper
            f = f.with_stem(f"{f.stem}_{i:02}").with_suffix(".json")

            try:
                response = ExtractionResponse.model_validate_json(f.read_text())
            except (
                FileNotFoundError,
                pydantic_core._pydantic_core.ValidationError,
            ) as e:
                logging.error(e, exc_info=True)

                message = message.format(*data, paper_fn.read_text())

                extractions, usage = await extract_from_research_paper(client, message)

                f.parent.mkdir(parents=True, exist_ok=True)

                try:
                    response = ExtractionResponse(
                        paper=paper,
                        words=count,
                        extractions=extractions,
                        usage=usage,
                    )
                    f.write_text(response.model_dump_json(indent=2))

                except pydantic_core._pydantic_core.PydanticSerializationError:
                    response = ExtractionResponse(
                        paper=paper,
                        words=count,
                        extractions=extractions,
                        usage=None,
                    )
                    f.write_text(response.model_dump_json(indent=2))

            logger.info(response.model_dump_json(indent=2))

            models = [m.name.value for m in response.extractions.models]
            datasets = [d.name.value for d in response.extractions.datasets]
            libraries = [f.name.value for f in response.extractions.libraries]

            data = [models, datasets, libraries]


async def ignore_exceptions(
    client: instructor.client.Instructor | instructor.client.AsyncInstructor,
    validation_set: List[Path],
    *args,
    **kwargs,
):
    for paper in validation_set:
        try:
            await batch_extract_models_names(client, [paper], *args, **kwargs)
        except bdb.BdbQuit:
            raise
        except Exception as e:
            logging.error(
                f"Failed to extract paper information from {paper.name}: {e}",
                exc_info=True,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--platform",
        type=str,
        choices=sorted(PLATFORMS.keys()),
        default="openai",
        help="Platform to use",
    )
    parser.add_argument(
        "--papers", nargs="*", type=str, default=None, help="Papers to analyse"
    )
    parser.add_argument(
        "--input",
        metavar="TXT",
        type=Path,
        default=None,
        help="List of papers to analyse",
    )
    parser.add_argument(
        "--paperoni",
        metavar="JSON",
        type=Path,
        default=None,
        help="Paperoni json output of papers to query on converted pdfs -> txts",
    )
    options = parser.parse_args(argv)

    if options.paperoni:
        papers = [
            Paper(p).get_link_id_pdf() for p in json.loads(options.paperoni.read_text())
        ]
        papers = [p for p in papers if p is not None]
    elif options.input:
        papers = [
            Path(paper)
            for paper in Path(options.input).read_text().splitlines()
            if paper.strip()
        ]
    elif options.papers:
        papers = [Path(paper) for paper in options.papers if paper.strip()]
    else:
        papers = build_validation_set(ROOT_DIR / "data/")
        for p in papers:
            logger.info(p)

    if not all(map(lambda p: p.exists(), papers)):
        papers = [Path(ROOT_DIR / f"data/cache/arxiv/{paper}.txt") for paper in papers]

    assert all(map(lambda p: p.exists(), papers))

    client = PLATFORMS[options.platform]()

    asyncio.run(
        ignore_exceptions(
            client,
            [paper.absolute() for paper in papers],
            destination=ROOT_DIR / f"data/queries/{options.platform}",
        )
    )


if __name__ == "__main__":
    main()
