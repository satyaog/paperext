import logging
from pathlib import Path

import pytest

from paperext.config import Config
from paperext.log import logger as main_logger

from . import CONFIG_FILE


def test_global_config():
    """Test that the global config instance is a singleton"""
    global_config = Config.get_global_config()
    assert global_config is Config.get_global_config()

    assert Config(str(CONFIG_FILE)) is not Config(str(CONFIG_FILE))
    assert Config(str(CONFIG_FILE)) == Config(str(CONFIG_FILE))

    config = Config(str(CONFIG_FILE))
    assert config is not global_config

    Config.update_global_config(config)
    assert config == Config.get_global_config()
    assert global_config is Config.get_global_config()


def test_config_setattr():
    """Test setting an existing config attribute"""
    config = Config(str(CONFIG_FILE))
    config.testsection.testoption = "new test value"
    assert config.testsection.testoption == "new test value"

    with pytest.raises(KeyError):
        config.nosection = {}

    with pytest.raises(KeyError):
        config.testsection.noopt = ""


def test_config_env_vars(monkeypatch, caplog):
    """Test that environment variable starting with 'PAPEREXT_' are correctly
    added to the config. Make sure environment variables do not add entry to
    config"""

    monkeypatch.setenv("PAPEREXT_DIR_TESTPATH", "path/to/file")
    monkeypatch.setenv("PAPEREXT_TESTSECTION_TESTOPTION", "updated test value")

    monkeypatch.setenv("PAPEREXT_NOSECTION_OPT", "")
    monkeypatch.setenv("PAPEREXT_SECTION_NOOPT", "")

    config = Config(str(CONFIG_FILE))

    assert config.testsection.testoption == "updated test value"
    assert config.dir.testpath == CONFIG_FILE.parent / "path/to/file"
    assert "nosection" not in config
    assert "noopt" not in config.testsection

    assert any(
        "PAPEREXT_NOSECTION_OPT" in rec.message
        for rec in filter(lambda r: r.levelname == "WARNING", caplog.records)
    )
    assert any(
        "PAPEREXT_SECTION_NOOPT" in rec.message
        for rec in filter(lambda r: r.levelname == "WARNING", caplog.records)
    )


def test_config_dir_section_resolve():
    """Test that the 'dir' section of the config get parsed as Path objects then
    resolved"""

    config = Config(str(CONFIG_FILE))

    for k in config.dir:
        assert isinstance(config.dir[k], Path)

    assert str(config.dir.testpath.relative_to(CONFIG_FILE.parent)) == "."


def test_config_logging_level(monkeypatch):
    """Test that the logging level is correctly set from the config file"""

    monkeypatch.setenv("PAPEREXT_LOGGING_LEVEL", "CRITICAL")
    config = Config(str(CONFIG_FILE))
    assert config.logging.level == "CRITICAL"

    monkeypatch.setenv("PAPEREXT_LOGGING_LEVEL", "DEBUG")
    config = Config(str(CONFIG_FILE))
    assert config.logging.level == "DEBUG"

    main_logger.level = logging.NOTSET
    Config.update_global_config(config)
    assert main_logger.level == logging.DEBUG

    with pytest.raises(ValueError):
        monkeypatch.setenv("PAPEREXT_LOGGING_LEVEL", "NOT A LEVEL")
        Config.update_global_config(Config(str(CONFIG_FILE)))
