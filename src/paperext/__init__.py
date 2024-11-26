# SPDX-FileCopyrightText: 2024-present Satya Ortiz-Gagne <satya.ortiz-gagne@mila.quebec>
#
# SPDX-License-Identifier: MIT

from pathlib import Path
from paperext.config import Config

ROOT_DIR: Path = Config().dir.root
DATA_DIR: Path = Config().dir.data
CACHE_DIR: Path = Config().dir.cache
QUERIES_DIR: Path = Config().dir.queries

LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
