import random
from pathlib import Path
import re
import sys
import typing
import unicodedata

import pandas as pd
from pydantic import BaseModel, Field, create_model

from .model import Explained, Model, PaperExtractions

ROOT_FOLDER = Path(__file__).resolve().parent.parent
PAPERS_TO_IGNORE={"data/cache/arxiv/2404.09932.txt",}


def build_validation_set(data_dir:Path, seed=42):
    random.seed(seed)

    all_papers = set()
    research_fields = sorted(
        [fn.name.split("_")[0] for fn in data_dir.glob("*_papers.txt")]
    )
    papers_by_field = {}

    for field in research_fields:
        papers_by_field.setdefault(field, set())
        field_papers:set = papers_by_field[field]
        all_field_papers = (data_dir / f"{field}_papers.txt").read_text().splitlines()
        all_field_papers = sorted([p for p in all_field_papers if p])
        while len(field_papers) < 10:
            _field_papers = set(random.sample(all_field_papers, 10 - len(field_papers)))
            field_papers.update(_field_papers - all_papers)
            all_papers.update(_field_papers)
        print(
            f"Selected {len(field_papers)} papers out of {len(all_field_papers)} papers for field {field}",
            file=sys.stderr
        )
        papers_by_field[field] = sorted(field_papers)

    # Try to minimize impact on random selections by filtering the particular
    # papers only at the end
    for field in research_fields:
        papers_by_field.setdefault(field, set())
        field_papers = set(papers_by_field[field])
        field_papers = field_papers - PAPERS_TO_IGNORE
        all_field_papers = (data_dir / f"{field}_papers.txt").read_text().splitlines()
        all_field_papers = sorted([p for p in all_field_papers if p])
        while len(field_papers) < 10:
            _field_papers = set(random.sample(all_field_papers, 10 - len(field_papers)))
            field_papers.update(_field_papers - all_papers - PAPERS_TO_IGNORE)
            all_papers.update(_field_papers)
        papers_by_field[field] = sorted(field_papers)

    validation_set = sum(papers_by_field.values(), [])
    # # Dev validation set
    # validation_set = sum(map(lambda _:random.sample(_, 1), papers_by_field.values()), [])
    return list(map(lambda p:Path(p).absolute(), validation_set))


def fix_explained_fields():
    import pdb ; pdb.set_trace()
    fields = _get_fields(PaperExtractions)
    return create_model(PaperExtractions.__name__, **fields)


def model2df(model:PaperExtractions):
    paper_1d_df = {}
    paper_references_df = {k:{} for k in Model.model_fields}

    for k, v in model:
        if isinstance(v, Explained):
            v = v.value

        if k in ("title", "type", "research_field",):
            paper_1d_df[k] = v

        elif k in ("sub_research_field",):
            # This will become a list
            paper_1d_df.setdefault(k, [])
            paper_1d_df[k].append(v)

        if k in ("research_field", "sub_research_field",):
            paper_1d_df.setdefault("all_research_fields", [])
            paper_1d_df["all_research_fields"].append(v)

        elif k in ("models", "datasets", "libraries",):
            for i, entry in enumerate(v):
                for entry_k, entry_v in entry:
                    if isinstance(entry_v, Explained):
                        entry_v = entry_v.value
                    paper_references_df[entry_k][(k, i)] = entry_v

    paper_1d_df["sub_research_field"] = [pd.Series(paper_1d_df["sub_research_field"])]
    paper_1d_df["all_research_fields"] = [pd.Series(paper_1d_df["all_research_fields"])]
    return pd.DataFrame(paper_1d_df), pd.DataFrame(paper_references_df)


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


def split_entry(string:str, sep_left="[[", sep_right="]]"):
    first, *extra = [_.strip().rstrip(sep_right) for _ in string.split(sep_left)]
    assert len(extra) <= 1
    extra = [_.strip() for _ in extra for _ in _.split(",")]
    return [first, *extra]


def str_eq(string, other):
    return str_normalize(string) == str_normalize(other)


def str_normalize(string):
    string = unicodedata.normalize("NFKC", string).lower()
    string = [_s.split("}}") for _s in string.split("{{")]
    string = sum(string, [])
    exclude = string[1:2]
    string = list(map(
        lambda _s:re.sub(pattern=r"[^a-z0-9]", string=_s, repl=""),
        string[:1] + string[2:]
    ))
    string = "".join(string[:1] + exclude + string[1:])
    return string


def python_module(filename:Path | str):
    return (
        str(Path(filename).relative_to(ROOT_FOLDER).with_suffix(""))
        .replace("/", ".")
    )


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
                List[create_model(field, **sub_fields)],
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
