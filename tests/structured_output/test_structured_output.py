import pytest

from paperext.structured_output.ai4hcat import model_v1 as ai4hcat_model_v1
from paperext.structured_output.mdl import (
    # model_v1 as mdl_model_v1,
    model_v2 as mdl_model_v2,
    model_v3 as mdl_model_v3,
)


@pytest.mark.parametrize(
    "structured_output_module",
    [ai4hcat_model_v1, mdl_model_v2, mdl_model_v3],
)
def test_empty_model(structured_output_module):
    structured_output_module.empty_model(structured_output_module.PaperExtractions)
