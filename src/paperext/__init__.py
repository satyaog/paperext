# SPDX-FileCopyrightText: 2024-present Satya Ortiz-Gagne <satya.ortiz-gagne@mila.quebec>
#
# SPDX-License-Identifier: MIT

import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
