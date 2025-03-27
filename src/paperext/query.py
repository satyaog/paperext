import argparse
import asyncio
import bdb
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

import instructor
from pydantic import BaseModel
import pydantic_core

from paperext import CFG
from paperext.log import logger
from paperext.structured_output import get_struct_module, mdl
from paperext.utils import Paper, build_validation_set


def get_state_cls():
    return get_struct_module(CFG.platform.struct).state.State


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
    from openai import AsyncOpenAI, RateLimitError
    from openai.types.chat.chat_completion import CompletionUsage

    def _client():
        model = CFG.openai.model
        client = instructor.from_openai(
            AsyncOpenAI(),
            mode=instructor.Mode.TOOLS_STRICT,
        )
        _create_with_completion = client.chat.completions.create_with_completion

        async def _wrap(*args, **kwargs):
            extractions, completion = await _create_with_completion(
                model=model, *args, **{"max_retries": 1, **kwargs}
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
            extractions, completion = _create_with_completion(
                *args, **{"max_retries": 2, **kwargs}
            )

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

try:
    from openai import AsyncOpenAI, RateLimitError
    from openai.types.chat.chat_completion import CompletionUsage

    def _client():
        model = CFG.ollama.model
        client = instructor.from_openai(
            AsyncOpenAI(
                base_url=CFG.ollama.url,
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
        )
        _create_with_completion = client.chat.completions.create_with_completion

        async def _wrap(*args, **kwargs):
            extractions, completion = await _create_with_completion(
                model=model, *args, **{"max_retries": 2, **kwargs}
            )
            return extractions, completion.usage

        client.chat.completions.create_with_completion = _wrap
        return client

    PLATFORMS["ollama"] = _client

except ModuleNotFoundError as e:
    logger.info(e, exc_info=True)
    logging.info(e, exc_info=True)

try:
    from llama_cloud_services import LlamaParse
    from llama_index.core import SimpleDirectoryReader

    def _client():
        model = CFG.llamaparse.model

        class Mock:
            def __getattribute__(self, name: str):
                try:
                    return object.__getattribute__(self, name)

                except AttributeError:
                    self.__dict__[name] = Mock()

                return object.__getattribute__(self, name)

        llama_parse_args = {"result_type": "markdown"}
        match model:
            case "balance":
                pass
            case "fast":
                llama_parse_args["fast_mode"] = True
                llama_parse_args["result_type"] = "text"
            case "premium":
                llama_parse_args["premium_mode"] = True
            case _:
                llama_parse_args["parse_mode"] = model

        client = Mock()

        async def create_with_completion(
            *args, response_model: BaseModel, messages: list[dict], **kwargs
        ):
            assert len(messages) == 1
            parser = LlamaParse(**llama_parse_args)
            file_extractor = {".pdf": parser}
            documents = await SimpleDirectoryReader(
                input_files=[m["pdf"] for m in messages],
                file_extractor=file_extractor,
            ).aload_data()
            analysis = response_model(pages=[d.text for d in documents])
            return analysis, {
                "metadata": documents[0].metadata,
                "doc_id": [d.doc_id for d in documents],
            }

        client.chat.completions.create_with_completion = create_with_completion

        return client

    PLATFORMS["llamaparse"] = _client

except ModuleNotFoundError as e:
    logger.info(e, exc_info=True)
    logging.info(e, exc_info=True)


async def query(
    client: instructor.client.Instructor | instructor.client.AsyncInstructor,
    state: mdl.state.State,
    messages: list[dict[str:str]],
) -> Tuple[Any, CompletionUsage]:
    """Extract Models, Datasets and Frameworks names from a research paper."""
    retries = [True] * 1
    while True:
        try:
            result = client.chat.completions.create_with_completion(
                response_model=state.AnalysisCls,
                messages=messages,
            )

            try:
                extractions, usage = result
            except TypeError:
                extractions, usage = await result

            return extractions, usage

        except RateLimitError as e:
            asyncio.sleep(60)
            if retries:
                retries.pop()
                continue
            raise e


def _additional_attempts(state, force=tuple()):
    yield from state.format_messages()

    while force:
        yield from state.format_messages()


async def batch_queries(
    client: instructor.client.Instructor | instructor.client.AsyncInstructor,
    papers_w_pdf_txt: List[Path],
    destination: Path = CFG.dir.queries,
    state_cls=None,
    force=False,
) -> List:
    state_cls = state_cls or get_state_cls()
    destination.mkdir(parents=True, exist_ok=True)

    responses = []

    for paper, pdf_txt in papers_w_pdf_txt:
        paper_name = pdf_txt.name

        count = 0
        # for line in pdf_txt.read_text().splitlines():
        #     count += len([w for w in line.strip().split() if w])

        state = state_cls(paper, pdf_txt)

        force = [True] * force

        for i, messages in enumerate(_additional_attempts(state, force=force)):
            f = destination / paper_name
            f = f.with_stem(f"{f.stem}_{i:02}").with_suffix(".json")

            try:
                response = state.ResponseCls.model_validate_json(f.read_text())

            except (
                FileNotFoundError,
                pydantic_core._pydantic_core.ValidationError,
            ) as e:
                logger.error(e, exc_info=True)
                logging.error(e, exc_info=True)

                if force:
                    force.pop()

                analysis, usage = await query(client, state, messages)

                f.parent.mkdir(parents=True, exist_ok=True)

                try:
                    response = state.make_response(
                        paper_name=paper_name,
                        words=count,
                        analysis=analysis,
                        usage=usage,
                    )
                    f.write_text(response.model_dump_json(indent=2))

                except pydantic_core._pydantic_core.PydanticSerializationError:
                    response = state.make_response(
                        paper_name=paper_name,
                        words=count,
                        analysis=analysis,
                        usage=None,
                    )
                    f.write_text(response.model_dump_json(indent=2))

            state.push_response(response)
            # logger.info(response.model_dump_json(indent=2))

        responses.extend(state.responses)

    return responses


async def ignore_exceptions(
    client: instructor.client.Instructor | instructor.client.AsyncInstructor,
    validation_set: List[Path],
    *args,
    **kwargs,
):
    for paper in validation_set:
        try:
            await batch_queries(client, [paper], *args, **kwargs)
        except bdb.BdbQuit:
            raise
        except Exception as e:
            logger.error(
                f"Failed to extract paper information from {paper[1].name}: {e}",
                exc_info=True,
            )
            logging.error(
                f"Failed to extract paper information from {paper[1].name}: {e}",
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
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Paperoni json output of papers to query on converted pdfs -> txts",
    )
    options = parser.parse_args(argv)

    CFG.platform.select = options.platform or CFG.platform.select

    if options.paperoni:
        papers = [Paper(p) for p in json.loads(options.paperoni.read_text())]
        papers = [
            (p, p.get_link_id_pdf()) for p in papers if p.get_link_id_pdf() is not None
        ]
    elif options.input:
        papers = [
            (None, Path(paper.strip()))
            for paper in Path(options.input).read_text().splitlines()
            if paper.strip()
        ]
    elif options.papers:
        papers = [(None, Path(paper)) for paper in options.papers if paper.strip()]
    else:
        papers = [
            (None, Path(paper)) for paper in build_validation_set() if paper.strip()
        ]
        for _, p in papers:
            logger.info(p)

    if not all([pdf_txt.exists() for _, pdf_txt in papers]):
        papers = [
            (p, Path(CFG.dir.cache / f"arxiv/{pdf_txt}.txt")) for p, pdf_txt in papers
        ]

    assert all([pdf_txt.exists() for _, pdf_txt in papers])

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
            [
                (paper, pdf_txt.absolute())
                for paper, pdf_txt in sorted(
                    papers, key=lambda x: x[0]._paper_id if x[0] else ""
                )
            ],
            destination=CFG.dir.queries / CFG.platform.select,
            force=options.force,
        )
    )


if __name__ == "__main__":
    main()
