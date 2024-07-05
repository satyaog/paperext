import random
from pathlib import Path
import re
import sys
import unicodedata

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