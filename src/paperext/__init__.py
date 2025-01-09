# SPDX-FileCopyrightText: 2024-present Satya Ortiz-Gagne <satya.ortiz-gagne@mila.quebec>
#
# SPDX-License-Identifier: MIT

import os
import pathlib

ROOT_DIR = pathlib.Path(os.environ["PAPEREXT_ROOT"]).resolve()
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
