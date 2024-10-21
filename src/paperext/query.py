import argparse
import asyncio
import bdb
import json
import logging
from pathlib import Path
from typing import List, Tuple

import instructor
import openai
import pydantic_core
from openai.types.chat.chat_completion import CompletionUsage

from paperext import LOG_DIR, ROOT_DIR
from paperext.models.model import (_FIRST_MESSAGE, ExtractionResponse,
                                   PaperExtractions)
from paperext.utils import Paper, build_validation_set, python_module

# Set logging to DEBUG to print OpenAI requests
logging.basicConfig(
    filename=LOG_DIR / f"{Path(__file__).stem}.out", level=logging.DEBUG
)

PROG = f"python3 -m {python_module(__file__)}"

DESCRIPTION = """
Utility to query Chat-GPT on papers
"""

EPILOG = f"""
Example:
  $ {PROG} --input data/query_set.txt > query.out
"""


async def extract_from_research_paper(
    client: instructor.client.Instructor | instructor.client.AsyncInstructor,
    message: str,
) -> Tuple[PaperExtractions, CompletionUsage]:
    """Extract Models, Datasets and Frameworks names from a research paper."""
    retries = [True] * 2
    while True:
        try:
            extractions, completion = (
                await client.chat.completions.create_with_completion(
                    model="gpt-4o",
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
                    max_retries=2,
                )
            )
            return extractions, completion.usage
        except openai.RateLimitError as e:
            asyncio.sleep(60)
            if retries:
                retries.pop()
                continue
            raise e


async def batch_extract_models_names(
    client: instructor.client.Instructor | instructor.client.AsyncInstructor,
    papers_fn: List[Path],
) -> List[ExtractionResponse]:
    for paper_fn in papers_fn:
        paper = paper_fn.name

        count = 0
        for line in paper_fn.read_text().splitlines():
            count += len([w for w in line.strip().split() if w])

        data = []

        for i, message in enumerate((_FIRST_MESSAGE,)):
            f = (ROOT_DIR / "data/queries/") / paper
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

                response = ExtractionResponse(
                    paper=paper,
                    words=count,
                    extractions=extractions,
                    usage=usage,
                )

                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(response.model_dump_json(indent=2))

            print(response.model_dump_json(indent=2))

            models = [m.name.value for m in response.extractions.models]
            datasets = [d.name.value for d in response.extractions.datasets]
            libraries = [f.name.value for f in response.extractions.libraries]

            data = [models, datasets, libraries]


async def ignore_exceptions(
    client: instructor.client.Instructor | instructor.client.AsyncInstructor,
    validation_set: List[Path],
):
    for paper in validation_set:
        try:
            await batch_extract_models_names(client, [paper])
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
        print(*papers, sep="\n")

    if not all(map(lambda p: p.exists(), papers)):
        papers = [Path(ROOT_DIR / f"data/cache/arxiv/{paper}.txt") for paper in papers]

    assert all(map(lambda p: p.exists(), papers))

    client = instructor.from_openai(
        # TODO: update to use the new feature Mode.TOOLS_STRICT
        # https://openai.com/index/introducing-structured-outputs-in-the-api/
        openai.AsyncOpenAI()  # , mode=instructor.Mode.TOOLS_STRICT
    )

    asyncio.run(ignore_exceptions(client, [paper.absolute() for paper in papers]))


if __name__ == "__main__":
    main()
