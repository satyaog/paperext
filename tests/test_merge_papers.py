import tempfile
from pathlib import Path

import pytest
import yaml

import paperext.merge_papers
from paperext.config import Config
from paperext.merge_papers import _select, _update_progession, _validate_field
from paperext.models.model import PaperExtractions, empty_model
from paperext.models.utils import model_dump_yaml, model_validate_yaml


@pytest.fixture(scope="function", autouse=True)
def tmpdir(cfg: Config, monkeypatch: pytest.MonkeyPatch):
    _tmpdir = tempfile.TemporaryDirectory(dir=str(cfg.dir.root / "tmp"))

    monkeypatch.setattr(paperext.merge_papers, "_TMPDIR", _tmpdir)

    with _tmpdir as _tmpdir:
        yield Path(_tmpdir)


@pytest.fixture(scope="function", autouse=True)
def merged_extractions(cfg: Config):
    merged_file: Path = cfg.dir.merged / "2401.14487.yaml"
    merged_extractions: PaperExtractions = model_validate_yaml(
        PaperExtractions, merged_file.read_text()
    )

    yield merged_extractions, merged_file


@pytest.mark.parametrize(
    ("string", "paper", "expected"),
    [
        ("1234", "4321", False),
        ("1234", "1234", True),
        ("1234", "abcABC1234321", True),
    ],
)
def test_find_in_paper(string: str, paper: str, expected: bool):
    assert bool(list(paperext.merge_papers._find_in_paper(string, paper))) == expected


@pytest.mark.parametrize(
    ("l", "expected"),
    [
        ([], None),
        (list(range(10)), None),
        ([0] * 10, [0]),
        (sum([[x, x, x] for x in range(10)], []), list(range(10))),
    ],
)
def test_remove_duplicates(l: list, expected: list):
    expected = expected or l
    cleaned_list = list(paperext.merge_papers._remove_duplicates(l))
    assert cleaned_list[:1] == l[:1]
    assert cleaned_list == expected


def test_model_dump():
    model_dump = yaml.safe_load(model_dump_yaml(empty_model(PaperExtractions)))

    QUOTE_FLAG = "__QUOTE_FLAG__"

    def _flag_quote(o: dict):
        quote_cnt = 0
        if "quote" in o:
            o["quote"] = QUOTE_FLAG
            quote_cnt += 1
        elif isinstance(o, dict):
            for v in o.values():
                quote_cnt += _flag_quote(v)
        elif isinstance(o, list):
            for v in o:
                quote_cnt += _flag_quote(v)
        return quote_cnt

    quote_cnt = _flag_quote(model_dump)
    assert quote_cnt

    _model_dump = paperext.merge_papers._model_dump(
        "ANYTHING", "", PaperExtractions.model_validate(model_dump)
    )
    assert (
        len(
            [l for l in _model_dump.splitlines() if l.strip().startswith("## WARNING:")]
        )
        == quote_cnt
    )

    _model_dump = paperext.merge_papers._model_dump(
        "ANYTHING",
        QUOTE_FLAG.replace("_", "").lower(),
        PaperExtractions.model_validate(model_dump),
    )
    assert not [
        l for l in _model_dump.splitlines() if l.strip().startswith("## WARNING:")
    ]


@pytest.mark.parametrize(
    ("inputs", "remaining"),
    [
        (["0"], ["1"]),
        (["1"], ["0"]),
        (["NOT_A_CHOICE", "0"], []),
        (["NOT_A_CHOICE"] * 5, []),
    ],
)
def test_input_option(inputs: list, remaining: list, monkeypatch: pytest.MonkeyPatch):
    """Test that _input_option ignores inputs that are not in options until a
    valid input is received"""
    _inputs = inputs + remaining
    monkeypatch.setattr("builtins.input", lambda *_a, **_kwa: _inputs.pop(0))

    options = list(map(str, range(10)))

    try:
        select = paperext.merge_papers._input_option("ANYTHING", options)
        assert select == inputs[-1]
        assert _inputs == remaining

    except IndexError:
        assert sum(_i == "NOT_A_CHOICE" for _i in inputs) == len(inputs)


def test_select(monkeypatch: pytest.MonkeyPatch, tmpdir: Path):
    """Test that _select selects the correct option and"""
    same_options = [{}] + [{}] * 4
    assert same_options[0] is not same_options[1]

    # Test that _select returns right away the first item
    _select("ANYTHING", *same_options, edit=False) is same_options[0]

    assert not list(tmpdir.glob("*"))

    options = list(map(lambda x: str(hash(str(x))), range(5)))
    with monkeypatch.context() as ctx:
        inputs = ["1"]
        ctx.setattr("builtins.input", lambda *_a, **_kwa: inputs.pop(0))
        _select("ANYTHING", *options, edit=False) == options[0]
        assert not inputs

        inputs.extend(["4"])
        _select("ANYTHING", *options, edit=True) == options[3]

        assert list(tmpdir.glob("ANYTHING*"))

    with monkeypatch.context() as ctx:
        ctx.setattr(paperext.merge_papers, "_open_editor", lambda *_a, **_kwa: None)
        inputs = ["e", "y"]
        ctx.setattr("builtins.input", lambda *_a, **_kwa: inputs.pop(0))

        content = _select("ANYTHING", *options, edit=True)

        for o in options:
            assert o in content

        assert list(tmpdir.glob("ANYTHING*"))


def test_validate_field(merged_extractions: tuple[PaperExtractions, Path]):
    empty_extractions: PaperExtractions = empty_model(PaperExtractions)

    merged_extractions, _ = merged_extractions
    merged_extractions: PaperExtractions

    model_dump = yaml.safe_load(model_dump_yaml(merged_extractions))

    for field in model_dump.keys():
        if isinstance(model_dump[field], list):
            setattr(empty_extractions, field, [])

        field_content = yaml.safe_dump(model_dump[field], sort_keys=False).splitlines()
        # Test that commented lines are removed from the yaml
        field_content = (
            field_content[:2]
            + ["   ## Comment line which would break the yaml"]
            + field_content[2:]
        )
        field_content = "\n".join(field_content)
        validated_extractions = _validate_field(
            empty_extractions, None, f"DOES_NOT_EXISTS.{field}", field_content
        )
        assert validated_extractions[field] == model_dump[field]


def test_update_progession(
    cfg: Config, merged_extractions: tuple[Path, PaperExtractions], tmpdir: Path
):
    empty_extractions: PaperExtractions = empty_model(PaperExtractions)
    merged_extractions, merged_file = merged_extractions
    merged_file: Path
    merged_extractions: PaperExtractions

    new_merged_file = cfg.dir.merged / f"new_{merged_file.name}"

    # Make sure the new merged extractions starts with empty data
    assert not new_merged_file.exists()
    new_merged_extractions = _update_progession(empty_extractions, new_merged_file)

    assert new_merged_file.exists()
    assert new_merged_extractions is not empty_extractions
    assert new_merged_extractions == empty_extractions

    model_dump = yaml.safe_load(model_dump_yaml(merged_extractions))

    # Fake merged data and test that only the merged fields are updated
    for field in model_dump.keys():
        assert getattr(new_merged_extractions, field) == getattr(
            empty_extractions, field
        )
        setattr(new_merged_extractions, field, getattr(merged_extractions, field))

        new_merged_extractions = _update_progession(
            new_merged_extractions, new_merged_file
        )
        assert getattr(new_merged_extractions, field) == getattr(
            merged_extractions, field
        )

        # Test that previously edited files are merged into new_merged_file
        field_content = yaml.safe_dump(model_dump[field], sort_keys=False)
        field_file = tmpdir / f"COULD_BE_ANYTHING.{field}.yaml"

        field_file.write_text(field_content)
        _update_progession(empty_extractions, new_merged_file) == new_merged_extractions

    # Make sure the file is updated with the correct data
    assert (
        model_validate_yaml(PaperExtractions, new_merged_file.read_text())
        == merged_extractions
    )
