from . import (
    ai4hcat as _ai4hcat,
    mdl as _mdl,
    mdl_dom as _mdl_dom,
    mdl_cat_dom as _mdl_cat_dom,
    mdl_cat_new_dom as _mdl_cat_new_dom,
    mdl_clus_dom as _mdl_clus_dom,
    mdl_find_acr as _mdl_find_acr,
    mdl_sort_dom as _mdl_sort_dom,
)


def get_struct_module(struct: str):
    match struct:
        case "ai4hcat":
            return _ai4hcat
        case "mdl":
            return _mdl
        case "mdl_dom":
            return _mdl_dom
        case "mdl_cat_dom":
            return _mdl_cat_dom
        case "mdl_cat_new_dom":
            return _mdl_cat_new_dom
        case "mdl_clus_dom":
            return _mdl_clus_dom
        case "mdl_find_acr":
            return _mdl_find_acr
        case "mdl_sort_dom":
            return _mdl_sort_dom
        case _:
            raise ValueError(f"Invalid structureed output {struct}")
