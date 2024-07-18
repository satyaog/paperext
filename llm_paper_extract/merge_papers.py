from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import urllib.request
from time import sleep
from typing import List, Tuple

from pydantic import BaseModel, ValidationError
from pygments import highlight
from pygments.formatters import TerminalTrueColorFormatter
from pygments.lexers.data import YamlLexer
import yaml

from .models.utils import (
    convert_model_json_to_yaml,
    model_dump_yaml,
    model_validate_yaml,
)

from . import ROOT_DIR
from .models.model import ExtractionResponse, PaperExtractions, empty_model
from .utils import str_normalize

_EDITOR = os.environ.get("VISUAL", os.environ.get("EDITOR", None))
_TMPDIR = tempfile.TemporaryDirectory()

_READER = os.environ.get("READER", None)
if platform.system() == "Darwin":  # macOS
    _READER = _READER or "open"
else:  # linux variants
    _READER = _READER or "xdg-open"


def _open_editor(_f: str):
    if not os.path.exists(_f):
        raise FileNotFoundError(_f)
    p = subprocess.Popen((*_EDITOR.split(" "), _f))
    while p.poll() is None:
        sleep(1)
        continue
    if p.returncode:
        raise subprocess.CalledProcessError(
            p.returncode, (_EDITOR, _f), p.stdout, p.stderr
        )
    return p.returncode


def _open(_f: str):
    if not os.path.exists(_f):
        raise FileNotFoundError(_f)
    p = subprocess.Popen((_READER, _f))
    # Sleep to give a bit of time to reader to spawn
    sleep(1)
    if p.returncode:
        raise subprocess.CalledProcessError(
            p.returncode, (_READER, _f), p.stdout, p.stderr
        )
    return p.returncode


def gen_indents(indent=0, skip_first=False):
    if not skip_first:
        yield ""
    while True:
        yield " " * indent


def get_terminal_width():
    return shutil.get_terminal_size((80, 20)).columns


def _remove_duplicates(l: list):
    if l:
        last = l[0]
        yield last
    for value in l[1:]:
        if value != last:
            yield value
            last = value


def _find_in_paper(string: str, paper: str):
    last_index = -1

    try:
        last_index = paper.index(str_normalize(string), last_index + 1)
        yield string, last_index
        return
    except ValueError:
        pass

    for part in [_s for _s in string.split("...") if _s.strip()]:
        try:
            last_index = paper.index(str_normalize(part), last_index + 1)
            yield part, last_index
        except ValueError:
            return


def _model_dump(paper_id, paper, model: BaseModel):
    _WARNING = f"WARNING: Could not find the quote in the paper {paper_id}"

    model_dump_json = model.model_dump_json(indent=2)
    model_dump_yaml = yaml.safe_dump(
        json.loads(model_dump_json), sort_keys=False, width=120
    )

    lines = model_dump_yaml.splitlines()
    for i, l in enumerate(lines):
        if l.lstrip().startswith("quote:"):
            lstrip = l[: len(l) - len(l.lstrip())]
            end = i + 1
            while end < len(lines):
                len_end_lstrip = len(lines[end]) - len(lines[end].lstrip())
                if lines[end] and len_end_lstrip <= len(lstrip):
                    break
                end += 1
            try:
                quote = yaml.safe_load("\n".join(lines[i:end]))["quote"]
            except (yaml.parser.ParserError, yaml.scanner.ScannerError):
                print(model_dump_yaml)
                print("\n".join(lines[i:end]))
                raise
            if not list(_find_in_paper(quote, paper)):
                lines.insert(end, f"{lstrip}## {_WARNING}")
    model_dump_yaml = "\n".join(lines)
    return model_dump_yaml


def _input_option(question: str, options: list):
    select: str = ""
    options = [o.lower() for o in options]
    while select not in options:
        select = input(f"{question} [{','.join(options)}]? ").lower()
    return select


def _select(key: str, *options: List[str], edit=False):
    is_equal = not edit
    for i, v in enumerate(options):
        for o in options[i + 1 :]:
            if v != o:
                is_equal = False
    if is_equal:
        return options[0]

    short_options = [f"{i+1}" for i, _ in enumerate(options)]
    long_options = short_options[:]
    if edit:
        short_options.append("e")
        long_options.append("edit")
    separator = "=" * max(0, 0, *map(len, sum([o.splitlines() for o in options], [])))
    separator = separator[: get_terminal_width()]

    editable_content = []
    for i, option in enumerate(options):
        prefix = f"## {key} ({i+1}) "

        editable_content.append("")
        editable_content.append(f"{prefix}{separator[len(prefix):]}")
        editable_content.append(option)

    editable_content.append(f"## {key} ")
    for entry in editable_content:
        print(highlight(entry, YamlLexer(), TerminalTrueColorFormatter()), end="")

    selected = _input_option(f"Select {' or '.join(long_options)}", short_options)
    edit = False
    try:
        selected = options[int(selected) - 1]
    except ValueError:
        # selected == "e"
        selected = "\n".join(editable_content)
        edit = True

    return write_content(key, selected, edit=edit)


def write_content(filename: str, content: str, edit=True):
    tmpfile = Path(_TMPDIR.name) / f"{filename}.yaml"

    with tmpfile.open("w+t") as _f:
        _f.write(content)

    while edit:
        _open_editor(str(tmpfile))
        edit = _input_option("Are you done with the edit", ("y", "n")) != "y"

    with tmpfile.open() as _f:
        return _f.read()


def _validate_field(
    model_dump: dict | PaperExtractions,
    model_cls: PaperExtractions.__class__ | None,
    filename: str,
    content: str,
):
    if not isinstance(model_dump, dict):
        model_cls = model_dump.__class__
        model_dump = model_dump.model_dump()
    # PaperExtractions.[sub_research_fields]
    field = ".".join(filename.split(".")[1:])

    try:
        model_dump[field] + []
        # field is a list
        default_empty = "[]"
    except TypeError:
        default_empty = ""

    while True:
        try:
            _content = [
                l
                for l in content.splitlines()
                if l.strip() and not l.lstrip().startswith("##")
            ]
            _content = "\n".join(_content) or default_empty
            model_dump[field] = yaml.safe_load(_content)
            model_cls.model_validate(model_dump)
            return model_dump
        except (yaml.scanner.ScannerError, yaml.parser.ParserError) as e:
            print(e)
            print("There was an error parsing the yaml. Please fix the error")
            content = write_content(filename, content)
        except ValidationError as e:
            print(e)
            print(
                f"There was an error validating the field "
                f"{model_cls.model_fields[field].annotation}. Please "
                f"fix the error"
            )
            content = write_content(filename, content)


def _update_progession(merged_extractions: PaperExtractions, merged_file: Path):
    # Load content of previous field merge in case the user updated the content
    _update = merged_extractions.model_dump()

    fields = list(Path(_TMPDIR.name).glob("*.yaml"))

    if fields:
        fields = subprocess.run(
            [
                "ls",
                "-1t",
                *fields,
            ],
            capture_output=True,
            encoding="utf8",
            check=False,
        ).stdout.splitlines()

        print("Previously edited fields")
        print(*fields, sep="\n")

    for tmpfile in fields:
        tmpfile = Path(tmpfile)
        _update = _validate_field(
            _update, PaperExtractions, tmpfile.stem, tmpfile.read_text()
        )

    _update = merged_extractions.model_validate(_update)
    merged_file.write_text(model_dump_yaml(_update))
    return _update


def _merge_list(
    paper_id: str,
    paper: str,
    attribute: str,
    merged_value: list,
    values: List[BaseModel],
):
    try:
        options: List[BaseModel] = sum(values, [])
    except TypeError:
        return None

    options = sorted(options)

    options = list(_remove_duplicates(options))
    options_str = []
    for _list in (*merged_value, options):
        concat = []
        for entry in _list:
            prefix = "- "
            for l in _model_dump(paper_id, paper, entry).splitlines():
                concat.append(f"{prefix}{l}")
                prefix = "  "
            concat.append("")
        options_str.append("\n".join(concat))

    selection = _select(attribute, *options_str, edit=True) or "[]"
    # write_content(attribute, selection, edit=False)
    # while True:
    #     try:
    #         _selection = yaml.safe_load(selection or "[]")
    #         selection = [options[0].model_validate(entry) for entry in _selection]
    #         break
    #     except (yaml.scanner.ScannerError, yaml.parser.ParserError) as e:
    #         print(e)
    #         print("There was an error parsing the yaml. Please try again")
    #         selection = write_content(attribute, selection)
    #     except ValueError as e:
    #         print(e)
    #         print(f"There was an error validating the model {type(values[0])}. Please try again")
    #         selection = write_content(attribute, selection)

    return selection


def merge_paper_extractions(
    paper_id,
    paper,
    merged_extractions: PaperExtractions,
    *all_extractions: List[PaperExtractions],
):
    f: Path = (ROOT_DIR / "data/merged/") / paper_id
    f = f.with_suffix(".yaml")

    for keys_values in zip(
        empty_model(PaperExtractions), merged_extractions, *all_extractions
    ):
        merged_extractions = _update_progession(merged_extractions, f)

        empty_value, merged_value, *values = [v for _, v in keys_values]

        if not [v for v in values[1:] if v != values[0]]:
            # All the values are similar to the first value
            values = values[:1]

        key = keys_values[0][0]
        attribute = f"{merged_extractions.__class__.__name__}.{key}"

        merged_value = [merged_value] if merged_value != empty_value else []
        selection = _merge_list(paper_id, paper, attribute, merged_value, values)

        if selection is not None:
            # merged_extractions.__dict__[key] = selection
            continue

        try:
            options = [
                _model_dump(paper_id, paper, v) for v in (*merged_value, *values)
            ]
            selection = _select(attribute, *options, edit=True)
            # while True:
            #     try:
            #         selection = values[0].model_validate(yaml.safe_load(selection))
            #         break
            #     except yaml.scanner.ScannerError as e:
            #         print(e)
            #         print("There was an error parsing the yaml. Please try again")
            #         selection = write_content(attribute, selection)
            #     except ValueError as e:
            #         print(e)
            #         print(f"There was an error validating the model {type(values[0])}. Please try again")
            #         selection = write_content(attribute, selection)
            # merged_extractions.__dict__[key] = selection
            continue
        except AttributeError:
            pass

        try:
            options = [v.value for v in (*merged_value, *values)]
            selection = _select(attribute, *options, edit=True)
            # while True:
            #     try:
            #         selection = type(values[0])(selection)
            #         break
            #     except ValueError as e:
            #         print(e)
            #         print("There was an error parsing the value. Please try again")
            #         selection = write_content(attribute, selection)
            # merged_extractions.__dict__[key] = selection
            continue
        except AttributeError:
            pass

        # merged_extractions.__dict__[key] = _select(attribute, *merged_value, *values, edit=True)
        selection = _select(attribute, *merged_value, *values, edit=True)

    return _update_progession(merged_extractions, f)


def get_papers_from_file(
    papers: List[str],
) -> List[Tuple[str, Path, ExtractionResponse]]:
    extractions_tuple = []

    for paper in papers:
        paper_id = paper.strip()
        print("Parsing", paper_id)
        paper = (
            (ROOT_DIR / f"data/cache/arxiv/{paper_id}.txt")
            .read_text()
            .lower()
            .replace("\n", " ")
        )
        responses = list((ROOT_DIR / "data/queries/").glob(f"{paper_id}_[0-9]*.json"))
        if not responses:
            print("No responses found for", paper_id)
            print("Skipping...")
            continue
        responses = (
            ExtractionResponse.model_validate_json(_f.read_text()) for _f in responses
        )
        for (_, paper_id), (_, _), (_, extractions), _ in responses:
            extractions_tuple.append((paper_id, str_normalize(paper), extractions))

    extractions_tuple.sort(key=lambda _: _[0])

    return extractions_tuple


def get_papers_from_folder() -> List[Tuple[str, Path, ExtractionResponse]]:
    responses = (ROOT_DIR / "data/queries/").glob("*.json")

    extractions_tuple = []
    for response_path in responses:
        print("Parsing", response_path)
        try:
            (
                (_, paper),
                (_, _),
                (_, extractions),
                _,
            ) = ExtractionResponse.model_validate_json(response_path.read_text())
        except ValidationError as e:
            print(e)
            print(f"Skipping {response_path}")
            continue
        paper_id = paper
        paper = (
            (ROOT_DIR / "data/cache/arxiv/" / paper_id)
            .read_text()
            .lower()
            .replace("\n", " ")
        )
        extractions_tuple.append((paper_id, paper, extractions))

    extractions_tuple.sort(key=lambda _: _[0])

    return extractions_tuple


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--papers", nargs="*", type=str, default=None, help="Papers to merge"
    )
    parser.add_argument(
        "--input", type=Path, default=None, help="List of papers to merge"
    )
    options = parser.parse_args(argv)

    if options.input:
        with open(options.input, "r") as f:
            papers = get_papers_from_file(f.readlines())
    elif options.papers:
        papers = get_papers_from_file(options.papers)
    else:
        papers = get_papers_from_folder()

    done = []
    for i, (paper_id, paper, _) in enumerate(papers):
        if [_paper_id for (_paper_id, _, _) in done if _paper_id == paper_id]:
            continue

        print("Merging", paper_id)

        f: Path = (ROOT_DIR / "data/merged/") / paper_id
        f = f.with_suffix(".yaml")
        f.parent.mkdir(parents=True, exist_ok=True)

        merged_extractions = empty_model(PaperExtractions)

        if f.exists() or f.with_suffix(".json").exists():
            try:
                merged_extractions = PaperExtractions.model_validate_json(
                    f.with_suffix(".json").read_text()
                )
            except FileNotFoundError:
                merged_extractions = None
            except ValidationError as e:
                print(e)
                print(f"Invalid extraction file... Consider deleting [{f}].")
                continue

            try:
                merged_extractions = model_validate_yaml(
                    PaperExtractions, f.read_text()
                )
            except FileNotFoundError:
                merged_extractions = convert_model_json_to_yaml(
                    PaperExtractions, merged_extractions.model_dump_json()
                )
                f.write_text(merged_extractions)
                merged_extractions = model_validate_yaml(
                    PaperExtractions, merged_extractions
                )
                f.with_suffix(".json").unlink()
            except ValidationError as e:
                print(e)
                print(f"Invalid extraction file... Consider deleting [{f}].")
                continue

            if (
                _input_option(
                    f"The paper {paper_id} has already been merged. Do you wish to "
                    f"redo the merge?",
                    ("y", "n"),
                )
                == "n"
            ):
                done.append((paper_id, paper, merged_extractions))
                continue

        all_extractions = [
            _extractions
            for _paper_id, _, _extractions in papers[i:]
            if _paper_id == paper_id
        ]

        pdf: Path = (ROOT_DIR / "data/cache/arxiv/" / paper_id).with_suffix(".pdf")
        print("Opening", pdf)
        try:
            _open(str(pdf))
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            url = f"https://arxiv.org/pdf/{pdf.stem}"
            print("Downloading from", url, "to", str(pdf))
            urllib.request.urlretrieve(url, str(pdf))
            _open(str(pdf))

        merged_extractions = merge_paper_extractions(
            paper_id, paper, merged_extractions, *all_extractions
        )
        done.append((paper_id, paper, merged_extractions))

        # Clean-up tmp files:
        for tmpfile in Path(_TMPDIR.name).glob("*.yaml"):
            tmpfile.unlink()

        for cmd, check in (
            (["git", "add", f], True),
            (["git", "commit", "-m", f.stem, "--only", f], False),
        ):
            subprocess.run(cmd, check=check)

        print("Merged paper saved to", f)


if __name__ == "__main__":
    main()
