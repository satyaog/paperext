import configparser
import logging
import os
from pathlib import Path

from paperext.log import logger as paperext_logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

_PREFIX = "PAPEREXT"
_CFG_VARENV = f"{_PREFIX}_CFG"
CONFIG_FILE = os.environ.get(
    _CFG_VARENV, Path(__file__).parent.parent.parent / "config.ini"
)


def config_to_dict(config):
    """Converts a configparser.ConfigParser object to a dictionary."""
    config_dict = {}
    for section in config.sections():
        # Create a dictionary for each section
        section_dict = {}
        for option in config.options(section):
            section_dict[option] = config.get(section, option)
        config_dict[section] = section_dict
    return config_dict


class Config:
    def __init__(self, config_file: str = CONFIG_FILE, config: dict = None) -> None:
        if config:
            self._config = config

        else:
            _config = configparser.ConfigParser()
            _config.read(config_file)

            self._config = config_to_dict(_config)

            self._parse_env_var()
            self._resolve(config_file)

            try:
                paperext_logger.setLevel(self._config["logging"]["level"])
            except KeyError:
                pass

    def __getattribute__(self, name: str):
        try:
            return object.__getattribute__(self, name)

        except AttributeError:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(config=value)
            else:
                return value

    def __getitem__(self, key):
        return self._config[key]

    def __iter__(self):
        return iter(self._config)

    def _parse_env_var(self):
        for varenv, value in os.environ.items():
            if not varenv.startswith(f"{_PREFIX}_") or varenv == _CFG_VARENV:
                continue

            conf_key = varenv.lower().split("_")[1:]

            try:
                section = self._config
                while len(conf_key) > 1:
                    section = section[conf_key.pop(0)]

                option = conf_key.pop()
                # Do not create options
                if option not in section:
                    raise KeyError(option)

                section[option] = value
            except KeyError as err:
                logger.warning(err, exc_info=True)

    def _resolve(self, config_file: str):
        config_file = Path(config_file).resolve().parent

        for k, v in self._config["dir"].items():
            v = Path(v)
            if not v.is_absolute():
                v = config_file / v
            self._config["dir"][k] = v
