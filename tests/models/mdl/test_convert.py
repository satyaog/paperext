import json

import pytest

from paperext.models.mdl import model, model_v1, model_v2, model_v3
from paperext.models.mdl.convert import _model_dump, convert_model_v1, convert_model_v2


def test_model_dump(cfg):
    """Test that _model_dump produces a valid pydantic model dump."""

    assert model.ExtractionResponse.model_validate_json(
        (cfg.dir.queries / "openai/2401.14487_00.json").read_text()
    ) == model.ExtractionResponse(
        **_model_dump(
            json.loads((cfg.dir.queries / "openai/2401.14487_00.json").read_text())
        )
    )


@pytest.mark.parametrize(
    "query_file,from_version,dest_version",
    [
        ["2401.14487_00", 1, 2],
        ["2401.14487_00", 2, 3],
        ["2402.04821_00", 2, 3],
    ],
)
def test_convert_model(cfg, query_file: str, from_version: int, dest_version: int):
    """Test that the model can be converted from a version to the next version."""

    match from_version:
        case 1:
            convert_model = convert_model_v1
            from_model = model_v1
            dest_model = model_v2
        case 2:
            convert_model = convert_model_v2
            from_model = model_v2
            dest_model = model_v3
        case _:
            raise ValueError(f"Unknown version: {from_version}")

    m = from_model.ExtractionResponse.model_validate_json(
        (cfg.dir.queries / f"v{from_version}/{query_file}.json").read_text()
    )

    assert (
        convert_model(m.extractions)
        == dest_model.ExtractionResponse.model_validate_json(
            (cfg.dir.queries / f"v{dest_version}/{query_file}.json").read_text()
        ).extractions
    )
