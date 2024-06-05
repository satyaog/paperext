import re

from . import ROOT_DIR
from .model import ExtractionResponse, PaperExtractions

_STRIP_RE = r"[a-zA-Z0-9].*[a-zA-Z0-9]"


def _strip(string):
    m = re.search(_STRIP_RE, string.lower())
    return m[0] if m is not None else string


def gen_indents(indent=0, skip_first=False):
    if not skip_first:
        yield ""
    while True:
        yield " " * indent


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
    except AttributeError as e:
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


def merge_paper_extractions(paper_id, paper, extractions: PaperExtractions, other_extractions: PaperExtractions):
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
    for (paper_id, paper, extractions) in extractions_tuple:
        last = extractions_merged[-1] if extractions_merged else None
        if last is None or last[0] != paper_id:
            extractions_merged.append((paper_id, paper, extractions))
            continue
        merge_paper_extractions(*last, extractions)

    for (paper_id, _, extractions) in extractions_merged:
        f = (ROOT_DIR / "data/merged/") / paper_id
        f = f.with_suffix(".json")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(extractions.model_dump_json(indent=2))

    for (paper_id, paper, extractions) in extractions_merged:
        check_excerpts(paper_id, paper, extractions)