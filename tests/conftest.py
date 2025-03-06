import os
import shutil
from unittest.mock import MagicMock

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

import paperext.query


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


@pytest.fixture()
def no_query(monkeypatch):
    mm = MagicMock(autospec=True)
    create_with_completion: MagicMock = mm.chat.completions.create_with_completion

    def _MagicMock(*_args, **_kwargs):
        def _create_with_completion(*_a, **_kwa):
            return MagicMock(autospec=True), MagicMock(autospec=True)

        create_with_completion.side_effect = (
            create_with_completion.side_effect or _create_with_completion
        )

        if not isinstance(mm.chat.completions.create_with_completion, MagicMock):
            # .chat.completions.create_with_completion has been wrap within a
            # _wrap function. Reset to represent a new object
            mm.chat.completions.create_with_completion = create_with_completion

        return mm

    def _AsyncMagicMock(*_args, **_kwargs):
        async def _create_with_completion(*_a, **_kwa):
            return MagicMock(autospec=True), MagicMock(autospec=True)

        create_with_completion.side_effect = (
            create_with_completion.side_effect or _create_with_completion
        )

        return _MagicMock(*_args, **_kwargs)

    def from_(client, *_args, **_kwargs):
        return client

    monkeypatch.setattr(paperext.query, "AsyncOpenAI", _AsyncMagicMock)
    monkeypatch.setattr(paperext.query, "GenerativeModel", _MagicMock)
    monkeypatch.setattr(paperext.query.instructor, f"from_openai", from_)
    monkeypatch.setattr(paperext.query.instructor, f"from_vertexai", from_)

    yield mm

    create_with_completion.assert_called()
