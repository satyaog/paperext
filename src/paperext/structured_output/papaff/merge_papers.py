import argparse
import copy
import json
import os
import platform
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from time import sleep
from typing import List, Tuple

import yaml
from pydantic import BaseModel, ValidationError
from pygments import highlight
from pygments.formatters import TerminalTrueColorFormatter
from pygments.lexers.data import YamlLexer

from paperext import CFG
from paperext.config import Config
from paperext.log import logger
from paperext.merge_papers import (
    _TMPDIR,
    _input_option,
    # _merge_list,
    _model_dump,
    _open,
    _select,
    _update_progession,
    write_content,
)
from paperext.structured_output.papaff.model import (
    AuthorAffiliations,
    Response,
    Analysis,
    empty_model,
)
from paperext.structured_output.parse_doc.model import Response as ParsedDocResponse
from paperext.structured_output.utils import (
    convert_model_json_to_yaml,
    model_dump_yaml,
    model_validate_yaml,
)
from paperext.utils import Paper, str_normalize


def _dump_list_entry(entry_str, comment=False):
    concat = []
    prefix = "- "
    for l in entry_str.splitlines():
        concat.append(f"{'# ' if comment else ''}{prefix}{l}")
        prefix = "  "
    return "\n".join(concat)


def _merge_list(
    paper_id: str,
    paper: str,
    attribute: str,
    empty_value: BaseModel,
    merged_value: list,
    values: list[BaseModel],
    pop_merged_value: bool = True,
):
    try:
        options: List[BaseModel] = sum(values, [])
    except TypeError:
        return None

    options_str = []
    # seen is a list instead of a set as a dirty workaround to avoid
    # TypeError: unhashable type: 'Explained[str]'
    seen = list()
    for _list in (*merged_value, options):
        concat = []
        for entry in _list:
            if entry in seen:
                continue
            concat.append(_dump_list_entry(_model_dump(paper_id, paper, entry)))
            seen.append(entry)
        options_str.extend(concat)

    empty_template = _dump_list_entry(
        _model_dump(paper_id, paper, empty_value), comment=True
    )

    selection = []
    last = False

    while last is not None:
        message = (
            f"Enter the index of the {attribute} you want to store at position"
            f" {len(selection) + 1}"
        )

        if selection:
            print()

        print(
            highlight(
                f"## Current selection of {attribute}",
                YamlLexer(),
                TerminalTrueColorFormatter(),
            ),
            end="",
        )
        for entry in selection:
            print(highlight(entry, YamlLexer(), TerminalTrueColorFormatter()), end="")

        last, _options_str = _select(
            attribute,
            *options_str,
            empty_template=empty_template,
            message=message,
            edit=True,
            stop=True,
        )

        if last is None:
            break

        for entry in yaml.safe_load(last or "[]"):
            entry = empty_value.model_validate(entry)
            yield entry
            selection.append(_dump_list_entry(_model_dump(paper_id, paper, entry)))

        if pop_merged_value:
            options_str = _options_str

    selection, _ = (
        _select(
            attribute,
            "\n".join(selection),
            *(["\n".join(options_str)] * bool(options_str)),
            empty_template=empty_template,
            edit=True,
        )
        or "[]"
    )

    yield selection


def _merge_authors_affiliations(
    paper_id: str,
    paper: str,
    attribute: str,
    empty_value: AuthorAffiliations,
    merged_value: list[AuthorAffiliations],
    values: list[AuthorAffiliations],
    affiliations: list[BaseModel],
):
    for author_affiliations in _merge_list(
        paper_id, paper, attribute, empty_value, merged_value, values
    ):
        author_affiliations: AuthorAffiliations

        if not isinstance(author_affiliations, empty_value.__class__):
            break

        _affiliations = copy.deepcopy(affiliations)

        for i, affiliation in enumerate(affiliations[0]):
            try:
                affiliation_idx = [
                    _a.value for _a in author_affiliations.affiliations
                ].index(affiliation.value)
            except ValueError:
                affiliation_idx = None

            if affiliation_idx is not None:
                _affiliations[0][i] = author_affiliations.affiliations.pop(
                    affiliation_idx
                )

        for author_affiliations_selection in _merge_list(
            paper_id,
            paper,
            # Avoid collision with the whole group of authors_affiliations
            f"_{author_affiliations.author.value.replace(' ', '_')}.affiliations",
            empty_value.affiliations[0],
            [],
            _affiliations + [author_affiliations.affiliations],
            pop_merged_value=False,
        ):
            pass

        author_affiliations.affiliations = list(
            map(
                empty_value.affiliations[0].model_validate,
                yaml.safe_load(author_affiliations_selection),
            )
        )
        yield author_affiliations

    yield author_affiliations


def merge_paper_extractions(
    paper: Paper,
    paper_txt: str,
    merged_extractions: Analysis,
    *all_extractions: List[Analysis],
):
    f: Path = CFG.dir.merged / f"{paper.id}.yaml"

    keys_values_store = {}

    for keys_values in zip(empty_model(Analysis), merged_extractions, *all_extractions):
        keys_values_store[keys_values[0][0]] = [v for _, v in keys_values]

    fields_priority = ["affiliations", "authors_affiliations"]
    fields_priority += [key for key in keys_values_store if key not in fields_priority]

    for key in fields_priority:
        merged_extractions = _update_progession(merged_extractions, f)

        empty_value, merged_value, *values = keys_values_store[key]

        if not [v for v in values[1:] if v != values[0]]:
            # All the values are similar to the first value
            values = values[:1]

        attribute = f"{merged_extractions.__class__.__name__}.{key}"

        merged_value = [merged_value] if merged_value != empty_value else []

        match key:
            case "affiliations":
                # We are only interested in the last selection
                for selection in _merge_list(
                    paper.id, paper_txt, attribute, empty_value[0], merged_value, values
                ):
                    pass
            case "authors_affiliations":
                # We are only interested in the last selection
                for selection in _merge_authors_affiliations(
                    paper.id,
                    paper_txt,
                    attribute,
                    empty_value[0],
                    merged_value,
                    values,
                    [merged_extractions.affiliations],
                ):
                    pass

        if selection is not None:
            continue

    return _update_progession(merged_extractions, f)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paperoni",
        nargs="*",
        type=Path,
        default=[],
        help="Paperoni json report of papers to analyse",
    )
    options = parser.parse_args(argv)

    paperoni = sum([json.loads(_p.read_text()) for _p in options.paperoni], [])

    papers: list[tuple[Paper, str, Analysis]] = []

    for paper in map(Paper, paperoni):
        with Config.push():
            CFG.platform.select = "mistralai"
            CFG.platform.struct = "parse_doc"
            CFG.dir.queries = CFG.dir.data / CFG.platform.struct / "queries"
            parsed_doc = Paper(paper._paper).queries[-1]
            analysis = ParsedDocResponse.model_validate_json(
                parsed_doc.read_text()
            ).analysis
            pdf = "\n---\n".join(analysis.pages_md)

        for response in (
            Response.model_validate_json(_f.read_text()) for _f in sorted(paper.queries)
        ):
            papers.append((paper, str_normalize(pdf), response.analysis))

    done = []
    for i, (paper, paper_txt, _) in enumerate(papers):
        if [_paper for (_paper, _, _) in done if _paper == paper]:
            continue

        logger.info(f"Merging {paper.id}")

        f: Path = CFG.dir.merged / f"{paper.id}.yaml"
        f.parent.mkdir(parents=True, exist_ok=True)

        merged_extractions = empty_model(Analysis)

        if f.exists():
            try:
                merged_extractions = model_validate_yaml(Analysis, f.read_text())
            except ValidationError as e:
                logger.error(e, exc_info=True)
                logger.info(f"Invalid extraction file... Consider deleting [{f}].")
                continue

            if (
                _input_option(
                    f"The paper {paper.id} has already been merged. Do you wish to "
                    f"redo the merge?",
                    ("y", "n"),
                )
                == "n"
            ):
                done.append((paper, paper_txt, merged_extractions))
                continue

        all_extractions = [
            _extractions
            for _paper_id, _, _extractions in papers[i:]
            if _paper_id == paper
        ]

        pdf: Path = paper.pdf.with_suffix(".pdf")
        logger.info(f"Opening {pdf}")
        try:
            _open(str(pdf))
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(e, exc_info=True)
            url = f"https://arxiv.org/pdf/{pdf.stem}"
            logger.info(f"Downloading from {url} to {pdf}")
            urllib.request.urlretrieve(url, str(pdf))
            _open(str(pdf))

        merged_extractions = merge_paper_extractions(
            paper, paper_txt, merged_extractions, *all_extractions
        )
        done.append((paper, paper_txt, merged_extractions))

        # Clean-up tmp files:
        for tmpfile in Path(_TMPDIR.name).glob("*.yaml"):
            tmpfile.unlink()

        for cmd, check in (
            (["git", "add", f], True),
            (["git", "commit", "-m", f.stem, "--only", f], False),
        ):
            subprocess.run(cmd, check=check)

        logger.info(f"Merged paper saved to {f}")


if __name__ == "__main__":
    main()
