import logging
from pathlib import Path
import sys

logging.basicConfig()
logger = logging.getLogger(Path(__file__).parent.name)
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
logger.addHandler(stderr_handler)
