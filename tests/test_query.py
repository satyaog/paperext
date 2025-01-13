from unittest.mock import MagicMock

import pytest

from paperext.models import STRUCT_MODULES
from paperext.models.mdl.model import PaperExtractions
import paperext.query
from paperext.query import (
    main,
    get_first_message,
    get_extraction_response,
    get_paper_extractions,
)


@pytest.fixture(autouse=True)
def set_cfg(cfg, monkeypatch):
    monkeypatch.setattr(paperext.query, "CFG", cfg)


@pytest.fixture(scope="function", autouse=True)
def clean_up(cfg):
    yield

    for query_file in cfg.dir.queries.glob(f"*/new_*.json"):
        query_file.unlink(missing_ok=True)


@pytest.mark.parametrize("model_struct", ["ai4hcat", "mdl"])
def test_model_struct_from_cfg(monkeypatch, cfg, model_struct):
    cfg.platform.struct = model_struct

    monkeypatch.setattr(paperext.query, "CFG", cfg)

    assert get_first_message() is STRUCT_MODULES[model_struct]._FIRST_MESSAGE
    assert get_extraction_response() is STRUCT_MODULES[model_struct].ExtractionResponse
    assert get_paper_extractions() is STRUCT_MODULES[model_struct].PaperExtractions


@pytest.mark.parametrize("platform", ["openai", "vertexai"])
def test_query(
    platform, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Test that the query function:
    * works correctly for different platforms
    * does not retry a request if the query result file exists
    * creates a query result file on success
    """
    if platform == "openai":
        monkeypatch.setattr(paperext.query.openai, "AsyncOpenAI", MagicMock)

    elif platform == "vertexai":
        monkeypatch.setattr(paperext.query, "GenerativeModel", MagicMock)

    monkeypatch.setattr(paperext.query.instructor, f"from_{platform}", MagicMock())

    monkeypatch.setattr(paperext.query.openai, "AsyncOpenAI", MagicMock)
    monkeypatch.setattr(paperext.query.instructor, "from_openai", MagicMock())

    main(["--platform", platform, "--papers", "2401.14487"])

    if platform == "openai":

        async def create_with_completion(*_a, **_kwa):
            magicmock = MagicMock(spec=PaperExtractions)
            magicmock.models = []
            magicmock.datasets = []
            magicmock.libraries = []
            return magicmock, MagicMock()

    elif platform == "vertexai":

        def create_with_completion(*_a, **_kwa):
            magicmock = MagicMock(spec=PaperExtractions)
            magicmock.models = []
            magicmock.datasets = []
            magicmock.libraries = []
            return magicmock, MagicMock()

    def AsyncOpenAI(*_a, **_kwa):
        magicmock = MagicMock()
        magicmock.chat.completions.create_with_completion.side_effect = (
            create_with_completion
        )
        return magicmock

    monkeypatch.setattr(paperext.query.instructor, f"from_{platform}", AsyncOpenAI)

    main(["--platform", platform, "--papers", "new_1234.12345"])
    assert (
        len(
            list(
                paperext.query.CFG.dir.queries.glob(f"{platform}/new_1234.12345_*.json")
            )
        )
        == 1
    )

    for record in filter(lambda r: r.levelname == "ERROR", caplog.records):
        assert "Failed to extract paper information" not in record.message


def test_query_error_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Test that the query function:
    * logs an error when a query fails and continues with the next paper
    """

    async def create_with_completion(*_a, **_kwa):
        raise Exception("Expected exception")

    def AsyncOpenAI(*_a, **_kwa):
        magicmock = MagicMock()
        magicmock.chat.completions.create_with_completion.side_effect = (
            create_with_completion
        )
        return magicmock

    monkeypatch.setattr(paperext.query.openai, "AsyncOpenAI", MagicMock)
    monkeypatch.setattr(paperext.query.instructor, f"from_openai", AsyncOpenAI)

    papers = ["new_1234.12345", "new_1234.23456"]
    main(["--platform", "openai", "--papers", *papers])

    error_msg_cnt = 0
    for record in filter(lambda r: r.levelname == "ERROR", caplog.records):
        if (
            sum(
                text in record.message
                for text in [
                    "Failed to extract paper information",
                    *papers,
                    "Expected exception",
                ]
            )
            == 3
        ):
            error_msg_cnt += 1

    assert error_msg_cnt == len(papers)
