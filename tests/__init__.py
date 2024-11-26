# SPDX-FileCopyrightText: 2024-present Satya Ortiz-Gagne <satya.ortiz-gagne@mila.quebec>
#
# SPDX-License-Identifier: MIT

from pathlib import Path

from paperext.config import Config


CONFIG_FILE = Path(__file__).with_name("config.ini")
CFG = Config(str(CONFIG_FILE))
