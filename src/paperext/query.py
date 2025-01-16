import argparse
import asyncio
import bdb
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

import instructor
import pydantic_core

from paperext import CFG
from paperext.log import logger
from paperext.structured_output import STRUCT_MODULES, ai4hcat, mdl
from paperext.utils import Paper, build_validation_set


def get_first_message() -> str:
    return STRUCT_MODULES[CFG.platform.struct].FIRST_MESSAGE


def get_extraction_response() -> (
    ai4hcat.model.ExtractionResponse | mdl.model.ExtractionResponse
):
    return STRUCT_MODULES[CFG.platform.struct].ExtractionResponse


def get_paper_extractions() -> (
    ai4hcat.model.PaperExtractions | mdl.model.PaperExtractions
):
    return STRUCT_MODULES[CFG.platform.struct].PaperExtractions


PROG = f"{Path(__file__).stem.replace('_', '-')}"

DESCRIPTION = """
Utility to query Chat-GPT on papers

Queries logs will be written in ${PAPEREXT_DIR_LOG}/DATE.query.dbg
"""

EPILOG = f"""
Example:
  $ {PROG} --input data/query_set.txt
"""

PLATFORMS = {}

try:
    import openai
    from openai.types.chat.chat_completion import CompletionUsage

    def _client():
        model = CFG.openai.model
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
    logger.info(e, exc_info=True)
    logging.info(e, exc_info=True)

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel

    vertexai.init(project=CFG.vertexai.project)

    def _client():
        model = CFG.vertexai.model
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
    logger.info(e, exc_info=True)
    logging.info(e, exc_info=True)


async def extract_from_research_paper(
    client: instructor.client.Instructor | instructor.client.AsyncInstructor,
    message: str,
) -> Tuple[Any, CompletionUsage]:
    """Extract Models, Datasets and Frameworks names from a research paper."""
    retries = [True] * 1
    while True:
        try:
            result = client.chat.completions.create_with_completion(
                # model="gpt-4o",
                response_model=get_paper_extractions(),
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
    destination: Path = CFG.dir.queries,
) -> List:
    destination.mkdir(parents=True, exist_ok=True)

    for paper_fn in papers_fn:
        paper = paper_fn.name

        count = 0
        for line in paper_fn.read_text().splitlines():
            count += len([w for w in line.strip().split() if w])

        data = []

        for i, message in enumerate((get_first_message(),)):
            f = destination / paper
            f = f.with_stem(f"{f.stem}_{i:02}").with_suffix(".json")

            try:
                response = get_extraction_response().model_validate_json(f.read_text())
            except (
                FileNotFoundError,
                pydantic_core._pydantic_core.ValidationError,
            ) as e:
                logger.error(e, exc_info=True)
                logging.error(e, exc_info=True)

                message = message.format(*data, paper_fn.read_text())

                extractions, usage = await extract_from_research_paper(client, message)

                f.parent.mkdir(parents=True, exist_ok=True)

                try:
                    response = get_extraction_response()(
                        paper=paper,
                        words=count,
                        extractions=extractions,
                        usage=usage,
                    )
                    f.write_text(response.model_dump_json(indent=2))

                except pydantic_core._pydantic_core.PydanticSerializationError:
                    response = get_extraction_response()(
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
            logger.error(
                f"Failed to extract paper information from {paper.name}: {e}",
                exc_info=True,
            )
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
        default=CFG.platform.select,
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

    CFG.platform.select = options.platform

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
        papers = build_validation_set()
        for p in papers:
            logger.info(p)

    if not all(map(lambda p: p.exists(), papers)):
        papers = [Path(CFG.dir.cache / f"arxiv/{paper}.txt") for paper in papers]

    assert all(map(lambda p: p.exists(), papers))

    client = PLATFORMS[CFG.platform.select]()

    # Set logging to DEBUG to print OpenAI requests
    # TODO: there must be a better way that would not impact other usage of
    # logging
    LOG_FILE = CFG.dir.log / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        filename=LOG_FILE.with_suffix(f".{PROG}.dbg"), level=logging.DEBUG, force=True
    )

    asyncio.run(
        ignore_exceptions(
            client,
            [paper.absolute() for paper in papers],
            destination=CFG.dir.queries / CFG.platform.select,
        )
    )


if __name__ == "__main__":
    main()
