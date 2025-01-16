import copy

import pytest

from paperext.config import Config

from . import CONFIG_FILE

Config.apply_global_config(Config(str(CONFIG_FILE)))
_CFG = Config.get_global_config()
_CFG.dir.log.mkdir(exist_ok=True)
_TMPDIR = _CFG.dir.root / "tmp"
_TMPDIR.mkdir(exist_ok=True)


def _clean_up(config: Config = _CFG):
    # Clean up log files
    for _file in _CFG.dir.log.glob(f"*"):
        _file.unlink(missing_ok=True)

    # Clean up tmp files
    for _file in _TMPDIR.glob(f"*/*"):
        _file.unlink(missing_ok=True)

    # Clean up files
    for d in config.dir:
        for _file in config.dir[d].glob(f"new_*"):
            _file.unlink(missing_ok=True)


# Cleanup files
_clean_up()


@pytest.fixture(scope="function", autouse=True)
def cfg():
    Config.apply_global_config(Config(str(CONFIG_FILE)))

    with Config.push() as config:
        yield config

    assert Config.get_global_config() == Config(str(CONFIG_FILE))
