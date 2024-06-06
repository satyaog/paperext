import os
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import urllib
from time import sleep
from typing import Iterable, List

from pydantic import BaseModel

from . import ROOT_DIR
from .model import ExtractionResponse, PaperExtractions

_STRIP_RE = r"[a-zA-Z0-9].*[a-zA-Z0-9]"
_EDITOR = os.environ.get("VISUAL", os.environ.get("EDITOR", None))
if platform.system() == "Darwin":       # macOS
    _EDITOR = _EDITOR or "open"
else:                                   # linux variants
    _EDITOR = _EDITOR or "xdg-open"


def _open_editor(_f:str):
    p = subprocess.Popen((_EDITOR, _f))
    while p.poll() is None:
        sleep(1)
        continue
    if p.returncode:
        raise subprocess.CalledProcessError(p.returncode, (_EDITOR, _f), p.stdout, p.stderr)
    return p.returncode


def _open(_f:str):
    if platform.system() == "Darwin":       # macOS
        exe = "open"
    else:                                   # linux variants
        exe = "xdg-open"
    p = subprocess.Popen((exe, _f))
    while p.poll() is None:
        sleep(1)
        continue
    if p.returncode:
        raise subprocess.CalledProcessError(p.returncode, (_EDITOR, _f), p.stdout, p.stderr)
    return p.returncode


def _strip(string):
    m = re.search(_STRIP_RE, string.lower())
    return m[0] if m is not None else string


def gen_indents(indent=0, skip_first=False):
    if not skip_first:
        yield ""
    while True:
        yield " " * indent


def get_terminal_width():
    return shutil.get_terminal_size((80, 20)).columns


def check_excerpts(paper_id, paper, obj, prefix="", indents=None):
    if indents is None:
        indents = gen_indents()
    validated = True
    try:
        obj.excerpt
        _indents = gen_indents(2)
        for k, v in obj:
            if k == "value":
                _ = v
            else:
                _ = f"{k}: {v}"
            print(next(indents) + next(_indents), _, sep="")
        excerpt = _strip(obj.excerpt)
        if excerpt not in paper:
            print(next(indents) + next(_indents), f"WARNING: Could not find the exerpt [{excerpt}] justifing the value [{obj.value}] in the paper {paper_id}", sep="")
            validated = False
        return validated
    except AttributeError:
        pass

    if isinstance(obj, str):
        print(next(indents), prefix, obj, sep="")
        return True
    print(next(indents), "[", sep="")
    try:
        prefix = "* "
        _indents = gen_indents(2, skip_first=True)
        padding = 0
        for k, _ in obj:
            padding = max(padding, len(k) + 1)
        for k, v in obj:
            pad = " " * (padding - len(k))
            print(next(indents) + next(_indents), prefix, f"{k}{pad}: ", sep="", end="")
            sub_indents = gen_indents(len(next(indents)) + 2)
            _validated = check_excerpts(paper_id, paper, v, prefix="", indents=sub_indents)
            validated = validated and _validated
    except ValueError:
        try:
            prefix = "- "
            for v in obj:
                sub_indents = gen_indents(len(next(indents)) + 2, skip_first=True)
                _validated = check_excerpts(paper_id, paper, v, prefix=prefix, indents=sub_indents)
                validated = validated and _validated
        except ValueError:
            pass
    print(next(indents), "]", sep="")
    return validated


def _merge_paper_extractions(paper_id, paper, extractions: PaperExtractions, other_extractions: PaperExtractions):
    for (_, v1), (_, v2) in zip(extractions, other_extractions):
        try:
            v2.excerpt
            if not check_excerpts(paper_id, paper, v1) and check_excerpts(paper_id, paper, v2):
                v1.value = v2.value
                v1.excerpt = v2.excerpt
        except AttributeError:
            pass
    for m in other_extractions.models:
        if m not in extractions.models:
            extractions.models.append(m)
    for d in other_extractions.datasets:
        if d not in extractions.datasets:
            extractions.datasets.append(d)
    for f in other_extractions.frameworks:
        if f not in extractions.frameworks:
            extractions.frameworks.append(f)


def _iter(iterable:Iterable):
    lazy_e = None

    try:
        for k, v in iterable.items():
            yield k, v
        return
    except AttributeError as e:
        lazy_e = e

    try:
        for i, v in enumerate(iterable):
            yield i, v
        return
    except ValueError as e:
        lazy_e = e

    raise lazy_e


def _remove_duplicates(l:list):
    last = None if not l else l[0]
    yield last
    for value in l[1:]:
        if value != last:
            yield value
            last = value


def _model_dump_json(paper_id, paper, model:BaseModel):
    _WARNING = f"WARNING: Could not find the quote in the paper {paper_id}"

    model_dump_json = model.model_dump_json(indent=2)

    lines = model_dump_json.splitlines()
    for i, l in enumerate(lines):
        if l.lstrip().startswith('"quote":'):
            lstrip = l[:len(l) - len(l.lstrip())]
            quote = ":".join(l.split(':')[1:]).strip()
            quote = ''.join(['{"quote":', quote, '}'])
            quote = json.loads(quote)["quote"]
            if quote.lower() not in paper:
                lines.insert(i+1, f"{lstrip}// {_WARNING}")
    model_dump_json = "\n".join(lines)
    return model_dump_json


def _model_validate_json(model:BaseModel, model_dump_json:str):
    lines = [
        l
        for l in model_dump_json.splitlines()
        if not l.lstrip().startswith("//")
    ]
    return model.model_validate_json("\n".join(lines))


def _input_option(question:str, options:list):
    select:str = ""
    options = [o.lower() for o in options]
    while select not in options:
        select = input(f"{question} [{','.join(options)}]? ").lower()
    return select


def _select(key:str, *options:List[str], edit=False):
    is_equal = not edit
    for i, v in enumerate(options):
        for o in options[i+1:]:
            if v != o:
                is_equal = False
    if is_equal:
        return options[0]

    short_options = [f"{i+1}" for i, _ in enumerate(options)]
    long_options = short_options[:]
    if edit:
        short_options.append("e")
        long_options.append("edit")
    separator = "=" * max(*map(len, sum([o.splitlines() for o in options], [])))
    separator = separator[:get_terminal_width()]

    for i, option in enumerate(options):
        print()
        prefix = f"== {key} ({i+1}) "
        print(f"{prefix}{separator[len(prefix):]}")
        print(option)
    select = _input_option(f"Select {' or '.join(long_options)}", short_options)
    try:
        return options[int(select) - 1]
    except ValueError:
        pass

    # select == "e"
    return edit_content(key, "\n".join(options))


def edit_content(filename:str, content:str):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir) / f"{filename}.json"
        with tmpfile.open("w+t") as _f:
            _f.write(content)
        _open_editor(str(tmpfile))
        while _input_option("Are you done with the edit", ("y","n")) != "y":
            _open_editor(str(tmpfile))
        with tmpfile.open() as _f:
            _f.seek(0)
            return _f.read()


def merge_paper_extractions(paper_id, paper, extractions:PaperExtractions, *other_extractions: List[PaperExtractions]):
    for keys_values in zip(extractions, *other_extractions):
        values = [v for _, v in keys_values]

        if not [v for v in values[1:] if v != values[0]]:
            # All the values are similar to the first value
            values = values[:1]

        key = keys_values[0][0]
        attribute = f"{extractions.__class__.__name__}.{key}"
        try:
            options:List[BaseModel] = sorted(sum(values, []), key=lambda _:_.name)
            options = list(_remove_duplicates(options))
            options_str = ",\n".join(_model_dump_json(paper_id, paper, entry) for entry in options)
            selection = _select(attribute, options_str, edit=True)
            while True:
                _selection = selection.split("\n},\n")
                _selection = [entry + "\n}" for entry in _selection[:-1]] + _selection[-1:]
                try:
                    selection = [_model_validate_json(options[0], entry) for entry in _selection]
                    break
                except ValueError as e:
                    print(e)
                    print()
                    selection = edit_content(attribute, selection)
            extractions.__dict__[key] = selection
            continue
        except TypeError:
            pass

        try:
            options = [_model_dump_json(paper_id, paper, v) for v in values]
            selection = _select(attribute, *options, edit=True)
            while True:
                try:
                    selection = _model_validate_json(values[0], selection)
                    break
                except ValueError as e:
                    print(e)
                    print("There was an error parsin the json. Please try again")
                    selection = edit_content(attribute, selection)
            extractions.__dict__[key] = selection
            continue
        except AttributeError:
            pass

        extractions.__dict__[key] = _select(attribute, *values, edit=False)


if __name__ == "__main__":
    responses = (ROOT_DIR / "data/queries/").glob("*.json")
    responses = (ExtractionResponse.model_validate_json(_f.read_text()) for _f in responses)

    extractions_tuple = []
    for (_,paper),(_,_),(_,extractions) in responses:
        paper_id = paper
        paper = (ROOT_DIR / "data/cache/arxiv/" / paper_id).read_text().lower().replace("\n", " ")
        extractions_tuple.append((paper_id, paper, extractions))

    extractions_tuple.sort(key=lambda _:_[0])

    extractions_merged = []
    for i, (paper_id, paper, extractions) in enumerate(extractions_tuple):
        if [_paper_id for (_paper_id, _, _) in extractions_merged if _paper_id == paper_id]:
            continue

        other_extractions = [
            _extractions
            for _paper_id, _, _extractions in extractions_tuple[i+1:]
            if _paper_id == paper_id
        ]

        pdf:Path = (ROOT_DIR / "data/cache/arxiv/" / paper_id).with_suffix(".pdf")
        try:
            _open(str(pdf))
        except subprocess.CalledProcessError:
            url = f"https://arxiv.org/pdf/{pdf.stem}"
            urllib.request.urlretrieve(url, str(pdf))
            _open(str(pdf))

        merge_paper_extractions(paper_id, paper, extractions, *other_extractions)
        extractions_merged.append((paper_id, paper, extractions))

    for (paper_id, _, extractions) in extractions_merged:
        f = (ROOT_DIR / "data/merged/") / paper_id
        f = f.with_suffix(".json")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(extractions.model_dump_json(indent=2))
