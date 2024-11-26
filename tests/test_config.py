from pathlib import Path

from paperext.config import Config

from . import CONFIG_FILE


def test_config(monkeypatch):
    monkeypatch.setenv("PAPEREXT_DIR_TESTPATH", "path/to/file")
    monkeypatch.setenv("PAPEREXT_TESTSECTION_TESTOPTION", "updated test value")

    monkeypatch.setenv("PAPEREXT_NOSECTION_OPT", "")
    monkeypatch.setenv("PAPEREXT_SECTION_NOOPT", "")

    config = Config(str(CONFIG_FILE))

    assert config.testsection.testoption == "updated test value"
    assert config.dir.testpath == CONFIG_FILE.parent / "path/to/file"
    assert "nosection" not in config
    assert "noopt" not in config.testsection

    # captured = capsys.readouterr()
    # assert "PAPEREXT_NOSECTION_OPT" in captured.err
    # assert "PAPEREXT_SECTION_NOOPT" in captured.err


def test_resolve():
    config = Config(str(CONFIG_FILE))

    for k in config.dir:
        assert isinstance(config.dir[k], Path)

    assert str(config.dir.testpath.relative_to(CONFIG_FILE.parent)) == "."
