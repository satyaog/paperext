from pathlib import Path
import typing
import warnings
import pandas as pd
from pydantic import BaseModel, Field, create_model
import yaml

from ..utils import split_entry, str_normalize
from .model import Explained, Model, PaperExtractions


def convert_model_json_to_yaml(model_cls:BaseModel, json_data:str, **kwargs):
    model = model_cls.model_validate_json(json_data, **kwargs)
    yaml_data = model_dump_yaml(model)
    assert model_validate_yaml(model_cls, yaml_data) == model
    return yaml_data


def fix_explained_fields():
    fields = _get_fields(PaperExtractions)
    return create_model(PaperExtractions.__name__, **fields)


def model_dump_yaml(model:BaseModel, **kwargs):
    return yaml.safe_dump(model.model_dump(**kwargs), sort_keys=False, width=120)


def model_validate_yaml(model_cls:BaseModel, yaml_data:str, **kwargs):
    return model_cls.model_validate(yaml.safe_load(yaml_data), **kwargs)


def model2df(model:BaseModel):
    paper_1d_df = {}
    paper_references_df = {k:{} for k in Model.model_fields}

    for k, v in model:
        if isinstance(v, Explained):
            v = v.value

        if k in ("type",):
            v = str_normalize(v.split()[0])

        elif k in ("research_field", "sub_research_field",):
            v = split_entry(v)
            v = map(str_normalize, v)
            v = sorted(set(map(_reasearch_field_alias, v)))

        if k in ("title", "type", "research_field",):
            paper_1d_df[k] = v

        elif k in ("sub_research_field",):
            # This will become a list
            paper_1d_df.setdefault("sub_research_fields", [])
            paper_1d_df["sub_research_fields"] += v

        if k in ("research_field", "sub_research_field",):
            paper_1d_df.setdefault("all_research_fields", [])
            paper_1d_df["all_research_fields"] += v

        elif k in ("models", "datasets", "libraries",):
            for i, entry in enumerate(v):
                for entry_k, entry_v in entry:
                    if isinstance(entry_v, Explained):
                        entry_v = entry_v.value

                    if entry_k in ("name", "type",):
                        entry_v = str_normalize(entry_v)
                    elif entry_k in ("mode",):
                        try:
                            entry_v = _mode_aliases(str_normalize(entry_v.split()[0]))
                        except IndexError:
                            entry_v = None
                    elif entry_k in ("role",):
                        entry_v = str_normalize(entry_v.split()[0])

                    paper_references_df[entry_k][(k, i)] = entry_v

    paper_1d_df["sub_research_fields"] = [pd.Series(paper_1d_df["sub_research_fields"])]
    paper_1d_df["all_research_fields"] = [pd.Series(paper_1d_df["all_research_fields"])]
    paper_1d_df, paper_references_df = (
        pd.DataFrame(paper_1d_df),
        pd.DataFrame(paper_references_df)
    )

    for group in ("models", "datasets", "libraries",):
        try:
            _l = paper_references_df.loc[group]["name"]
            _s = paper_references_df.loc[group]["name"].drop_duplicates()
        except KeyError:
            _l = []
            _s = set()

        if len(_l) != len(_s):
            warnings.warn(f"Possibly duplicated {group} in\n{_l}")

    return paper_1d_df, paper_references_df


def print_model(model_cls:BaseModel, indent = 0):
    for field, info in model_cls.model_fields.items():
        print(" " * indent, field, info)
        if typing.get_origin(info.annotation) == list:
            print_model(info.annotation.__args__[0], indent+2)
            continue

        try:
            print_model(info.annotation, indent+2)
        except AttributeError:
            pass


def _get_fields(model_cls:BaseModel):
    fields = {}
    for field, info in model_cls.model_fields.items():
        if typing.get_origin(info.annotation) == list:
            try:
                sub_fields = _get_fields(info.annotation.__args__[0])
            except AttributeError:
                fields[field] = (info.annotation, Field(description=info.description))
                continue
            fields[field] = (
                typing.List[create_model(field, **sub_fields)],
                Field(description=info.description)
            )
            continue

        try:
            sub_fields = _get_fields(info.annotation)
        except AttributeError:
            fields[field] = (info.annotation, Field(description=info.description))
            continue

        if info.annotation.__base__ == Explained:
            sub_fields = {
                field:(sub_fields["value"][0], Field(description=info.description)),
                **{k:v for k,v in sub_fields.items() if k != "value"}
            }
            cls_name = f"{Explained.__name__}[{field}]"
        else:
            cls_name = info.annotation.__name__
        fields[field] = (create_model(cls_name, **sub_fields), Field(description=info.description))

    return fields
