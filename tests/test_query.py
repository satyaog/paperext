from pathlib import Path
from unittest.mock import MagicMock

import pytest

import paperext.query
from paperext.query import get_state_cls, main
from paperext.structured_output import ai4hcat, mdl
from paperext.structured_output.mdl.model import Analysis

from .mock.structured_output import test as test_struct_module


@pytest.fixture(autouse=True)
def set_cfg(cfg, monkeypatch):
    monkeypatch.setattr(paperext.query, "CFG", cfg)


@pytest.fixture(scope="function", autouse=True)
def clean_up(cfg):
    yield

    for query_file in cfg.dir.queries.glob(f"*/new_*.json"):
        query_file.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("model_struct", "state_cls"),
    [("ai4hcat", ai4hcat.state.State), ("mdl", mdl.state.State)],
)
def test_model_struct_from_cfg(cfg, model_struct, state_cls):
    cfg.platform.struct = model_struct

    assert get_state_cls() is state_cls


@pytest.mark.usefixtures("no_query")
@pytest.mark.parametrize("platform", ["openai", "vertexai", "ollama"])
def test_no_query(platform, cfg, monkeypatch, tmp_path):
    def get_struct_module(struct: str):
        match struct:
            case "test":
                return test_struct_module
            case _:
                raise ValueError(f"Invalid structureed output {struct}")

    paper_id = "2401.14487"

    cfg.platform.struct = "test"
    cfg.dir.queries = tmp_path / "queries"

    monkeypatch.setattr(paperext.query, f"get_struct_module", get_struct_module)

    main(["--platform", platform, "--papers", paper_id])

    assert not list(cfg.dir.queries.glob(f"**/{paper_id}_00.json"))


def test_query(cfg, monkeypatch, tmp_path: Path):
    def get_struct_module(struct: str):
        match struct:
            case "test":
                return test_struct_module
            case _:
                raise ValueError(f"Invalid structureed output {struct}")

    assert cfg.platform.select == "ollama"
    paper_id = "2401.14487"

    cfg.platform.struct = "test"
    cfg.dir.queries = tmp_path / "queries"

    monkeypatch.setattr(paperext.query, f"get_struct_module", get_struct_module)

    # 5 times should be enough for the small model to return a valid JSON
    for _ in range(5):
        main(["--papers", paper_id])

    response = test_struct_module.model.Response.model_validate_json(
        list(cfg.dir.queries.glob(f"**/{paper_id}_00.json"))[0].read_text()
    )
    assert response.metadata.llm_model == cfg.ollama.model


@pytest.mark.usefixtures("no_query")
@pytest.mark.parametrize("platform", ["openai", "vertexai"])
def test_query_mock(
    platform,
    cfg,
    no_query,
    tmp_path,
    caplog: pytest.LogCaptureFixture,
):
    """Test that the query function:
    * works correctly for different platforms
    * does not retry a request if the query result file exists
    * creates a query result file on success
    """
    if platform == "openai":

        async def create_with_completion(*_a, **_kwa):
            magicmock = MagicMock(spec=Analysis)
            return magicmock, MagicMock()

    elif platform == "vertexai":

        def create_with_completion(*_a, **_kwa):
            magicmock = MagicMock(spec=Analysis)
            return magicmock, MagicMock()

    no_query.chat.completions.create_with_completion.side_effect = (
        create_with_completion
    )

    main(["--platform", platform, "--papers", "2401.14487"])

    # elif platform == "vertexai":

    #     def create_with_completion(*_a, **_kwa):
    #         magicmock = MagicMock(spec=Analysis)
    #         return magicmock, MagicMock()

    # def from_platform(*_a, **_kwa):
    #     magicmock = MagicMock()
    #     magicmock.chat.completions.create_with_completion.side_effect = (
    #         create_with_completion
    #     )
    #     return magicmock

    # monkeypatch.setattr(paperext.query.instructor, f"from_{platform}", from_platform)

    cfg.dir.queries = tmp_path / "queries"
    main(["--platform", platform, "--papers", "new_1234.12345"])
    assert len(list(cfg.dir.queries.glob(f"{platform}/new_1234.12345_*.json"))) == 1

    for record in filter(lambda r: r.levelname == "ERROR", caplog.records):
        assert "Failed to extract paper information" not in record.message


@pytest.mark.usefixtures("no_query")
def test_query_error_logged(no_query: MagicMock, monkeypatch: pytest.MonkeyPatch):
    """Test that the query function:
    * logs an error when a query fails and continues with the next paper
    """

    async def create_with_completion(*_a, **_kwa):
        raise Exception("Expected exception")

    no_query.chat.completions.create_with_completion.side_effect = (
        create_with_completion
    )

    papers = ["new_1234.12345", "new_1234.23456"]
    logging_error_mock = MagicMock()
    with monkeypatch.context() as m:
        m.setattr(paperext.query.logging, "basicConfig", MagicMock)
        m.setattr(paperext.query.logging, "error", logging_error_mock)
        main(["--platform", "openai", "--papers", *papers])

    error_msg_cnt = 0
    for call_arg in map(lambda c: str(c[0][0]), logging_error_mock.call_args_list):
        if (
            sum(
                text in call_arg
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
