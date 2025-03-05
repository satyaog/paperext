import os
import shutil

import pytest

from . import CONFIG_FILE

os.environ["PAPEREXT_CFG"] = str(CONFIG_FILE)

from paperext.config import Config

with Config.push() as cfg:
    # Correctly load files from project's data directory
    # TODO: use a path from the config file instead of assuming the file is
    # located in the config's data dir
    cfg.dir.data = cfg.dir.root / "../data"
    import paperext.structured_output as _


_CFG = Config.get_global_config()
_CFG.dir.log.mkdir(exist_ok=True)
_TMPDIR = _CFG.dir.root / "tmp"
_TMPDIR.mkdir(exist_ok=True)


def _clean_up(config: Config = _CFG):
    # Clean up log files
    for _file in _CFG.dir.log.glob(f"*"):
        _file.unlink(missing_ok=True)

    # Clean up tmp files
    for _entry in _TMPDIR.glob(f"**/*"):
        if _entry.is_dir():
            shutil.rmtree(_entry)
        else:
            _entry.unlink(missing_ok=True)

    # Clean up new files
    for d in config.dir:
        for _file in config.dir[d].glob(f"new_*"):
            _file.unlink(missing_ok=True)


# Cleanup files
_clean_up()


@pytest.fixture(scope="function", autouse=True)
def cfg():
    with Config.push() as config:
        yield config
