# SPDX-FileCopyrightText: 2024-present Satya Ortiz-Gagne <satya.ortiz-gagne@mila.quebec>
#
# SPDX-License-Identifier: MIT

from pathlib import Path

from paperext.config import Config

CFG = Config.get_global_config()
Config.update_global_config(CFG)  # Set env vars and logging level

ROOT_DIR: Path = CFG.dir.root
DATA_DIR: Path = CFG.dir.data
CACHE_DIR: Path = CFG.dir.cache
QUERIES_DIR: Path = CFG.dir.queries

LOG_DIR = CFG.dir.log
LOG_DIR.mkdir(exist_ok=True)
