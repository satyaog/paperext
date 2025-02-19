from . import ai4hcat as _ai4hcat
from . import mdl as _mdl
from . import mdl_dom as _mdl_dom


def get_struct_module(struct: str):
    match struct:
        case "ai4hcat":
            return _ai4hcat
        case "mdl":
            return _mdl
        case "mdl_dom":
            return _mdl_dom
