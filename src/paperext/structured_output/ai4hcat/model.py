# This model aims to extract Deep Learning Models, Datasets and Libraries from a
# research paper
# Very bad practice but we will allow it here since it acts like a proxy to the
# latest model version and nothing else is imported
from .model_v1 import *

# * import does not import private variables
# TODO: rename _FIRST_MESSAGE to FIRST_MESSAGE
from .model_v1 import _FIRST_MESSAGE
