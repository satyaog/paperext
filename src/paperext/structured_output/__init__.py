from . import (
    ai4hcat,
    mdl,
    mdl_dom,
    mdl_cat_dom,
    cat_new_mdl_dom,
    cat_new_mdl_mod,
    mdl_clus_dom,
    find_acr_mdl_dom,
    find_acr_mdl_mod,
    papaff,
    parse_doc,
)


def get_struct_module(struct: str):
    match struct:
        case "ai4hcat":
            return ai4hcat
        case "mdl":
            return mdl
        case "mdl_dom":
            return mdl_dom
        case "mdl_cat_dom":
            return mdl_cat_dom
        case "cat_new_mdl_dom":
            return cat_new_mdl_dom
        case "cat_new_mdl_mod":
            return cat_new_mdl_mod
        case "mdl_clus_dom":
            return mdl_clus_dom
        case "find_acr_mdl_dom":
            return find_acr_mdl_dom
        case "find_acr_mdl_mod":
            return find_acr_mdl_mod
        case "papaff":
            return papaff
        case "parse_doc":
            return parse_doc
        case _:
            raise ValueError(f"Invalid structureed output {struct}")
