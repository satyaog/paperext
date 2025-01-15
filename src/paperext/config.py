import configparser
import copy
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Union

from paperext.log import logger as main_logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

_PREFIX = "PAPEREXT"
_CFG_VARENV = f"{_PREFIX}_CFG"
CONFIG_FILE = os.environ.get(
    _CFG_VARENV, Path(__file__).parent.parent.parent / "config.mdl.ini"
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
    _instance: "Config" = None

    def __init__(self, config_file: str = CONFIG_FILE, config: dict = None) -> None:
        """Create a Config object from a config file or a dictionary."""
        if config:
            self._config = config

        else:
            _config = configparser.ConfigParser(
                interpolation=configparser.ExtendedInterpolation()
            )
            assert _config.read(
                config_file
            ), f"Could not read config file [{config_file}]"

            self._config = config_to_dict(_config)

            self._parse_env_vars()
            self._resolve(config_file)

    def __deepcopy__(self, memo):
        _config = Config(config=copy.deepcopy(self._config, memo))
        memo[id(self)] = _config
        return _config

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Config):
            return value._config == self._config

        elif isinstance(value, dict):
            return value == self._config

        else:
            raise TypeError(f"Cannot compare {type(self)} with {type(value)}")

    def __getattribute__(self, name: str) -> Union["Config", Path]:
        try:
            return object.__getattribute__(self, name)

        except AttributeError:
            return self[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_config":
            return object.__setattr__(self, name, value)

        if isinstance(self._config[name], dict):
            raise ValueError("Can only set values for nested Config objects")

        self._config[name] = value

    def __getitem__(self, key: str) -> Union["Config", Path]:
        value = self._config[key]
        if isinstance(value, dict):
            return Config(config=value)
        else:
            return value

    def __iter__(self):
        return iter(self._config)

    def _parse_env_vars(self):
        for envvar, value in os.environ.items():
            if not envvar.startswith(f"{_PREFIX}_") or envvar == _CFG_VARENV:
                continue

            conf_key = envvar.lower().split("_")[1:]

            try:
                section = self._config
                while len(conf_key) > 1:
                    section = section[conf_key.pop(0)]

                option = conf_key.pop()
                # Do not create options
                if option not in section:
                    raise KeyError(option)

                section[option] = value
            except KeyError:
                logger.warning(
                    f"Could not find env var {envvar} in config", exc_info=True
                )

    def _resolve(self, config_file: str):
        config_parent = Path(config_file).resolve().parent

        for k, v in self._config["dir"].items():
            v = Path(v)
            if not v.is_absolute():
                v = config_parent / v
            self._config["dir"][k] = v

    @staticmethod
    def get_global_config() -> "Config":
        """Returns the global instance of Config."""
        if Config._instance is None:
            Config._instance = Config()
            Config.apply_global_config(
                Config._instance
            )  # Set env vars and logging level
        return copy.deepcopy(Config._instance)

    @staticmethod
    def apply_global_config(config: "Config") -> None:
        """Apply the global instance of Config."""
        Config._instance = config

        try:
            main_logger.setLevel(config._config["logging"]["level"])
        except KeyError:
            pass

        for varenv, val in config._config["env"].items():
            os.environ[varenv.upper()] = val

    @contextmanager
    @staticmethod
    def cfg(config: "Config") -> Generator["Config", None, None]:
        """Context manager to temporarily change the global config."""
        _config = Config._instance
        try:
            Config.apply_global_config(config)
            yield config
        finally:
            Config._instance = _config


class GlobalConfigProxy(Config):
    def __init__(self) -> None:
        pass

    @property
    def _config(self) -> dict:
        assert Config._instance
        return Config._instance._config


CFG = GlobalConfigProxy()
