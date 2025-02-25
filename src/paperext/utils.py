import random
import re
import sys
import unicodedata
from pathlib import Path

from paperext import CFG
from paperext.log import logger

ROOT_FOLDER = Path(__file__).resolve().parent.parent
PAPERS_TO_IGNORE = {
    "arxiv/2404.09932.txt",
}


class Paper:
    # Original form of the converted pdf to txt. eg data/cache/*/ARXIV_ID.txt
    LINK_ID_TEMPLATE = "*/{link_id}.txt"
    # Extended form of the converted pdf to txt. eg data/cache/*/PAPER_ID.txt
    PAPER_ID_TEMPLATE = LINK_ID_TEMPLATE.format(link_id="{paper_id}")
    # The the up-to-date form of the converted pdf (by paperoni)
    # eg data/cache/fulltext/PAPER_ID/fulltext.txt
    PAPER_ID_FULLTEXT_TEMPLATE = "fulltext/{paper_id}/fulltext.txt"

    def __init__(self, paper: dict) -> None:
        self._selected_id = None
        self._paper_id = paper["paper_id"]
        self._pdfs = []
        link_ids = [self._paper_id]
        pdfs = []

        for l in paper["links"]:
            link_id = l.get("link", None)
            if link_id and link_id not in link_ids:
                link_ids.append(link_id)

                pdfs += sorted(
                    CFG.dir.cache.glob(self.LINK_ID_TEMPLATE.format(link_id=link_id))
                )

        # Find existing queries and infer the paper id from them
        self._queries = sum(
            [
                sorted(
                    (CFG.dir.queries / CFG.platform.select).glob(f"{link_id}_*.json")
                )
                for link_id in link_ids
            ],
            [],
        )

        _ids = ["_".join(p.stem.split("_")[:-1]) for p in self._queries]

        if _ids:
            if len(set(_ids)) > 1:
                logger.warning(
                    f"Multiple paper queries found for {paper['title']}:\n  "
                    + "\n  ".join(map(str, sorted(set(self._queries))))
                )
            self._selected_id = _ids[0]

        # Try to find the downloaded/converted pdf using the original form of
        # the converted pdf to txt
        if not self._selected_id and pdfs:
            # Favor the first pdf found, it's usually the most relevent and
            # easiest to download / access
            self._selected_id = pdfs[0].stem

        if not self._selected_id and self.pdf:
            self._selected_id = self._paper_id

        self._pdfs = (
            sorted(CFG.dir.cache.glob(self.LINK_ID_TEMPLATE.format(link_id=self.id)))
            + sorted(
                CFG.dir.cache.glob(
                    self.PAPER_ID_TEMPLATE.format(paper_id=self._paper_id)
                )
            )
            + sorted(
                CFG.dir.cache.glob(
                    self.PAPER_ID_FULLTEXT_TEMPLATE.format(paper_id=self._paper_id)
                )
            )
        )

    @property
    def id(self):
        return self._selected_id or self._paper_id

    @property
    def queries(self):
        return self._queries

    @property
    def pdfs(self):
        return self._pdfs

    @property
    def pdf(self):
        return next(
            iter(self.pdfs),
            None,
        )

    def get_link_id_pdf(self):
        link_id_pdf = None

        if self.pdf:
            link_id_pdf = self.pdf.with_stem(self.id)

        if link_id_pdf and not link_id_pdf.exists():
            link_id_pdf.hardlink_to(self.pdf)

        return link_id_pdf


def build_validation_set(seed=42):
    random.seed(seed)

    data_dir = CFG.dir.data

    all_papers = set()
    research_fields = sorted(
        [fn.name.split("_")[0] for fn in data_dir.glob("*_papers.txt")]
    )
    papers_by_field = {}

    for field in research_fields:
        papers_by_field.setdefault(field, set())
        field_papers: set = papers_by_field[field]
        all_field_papers = (data_dir / f"{field}_papers.txt").read_text().splitlines()
        all_field_papers = sorted([p for p in all_field_papers if p])
        while len(field_papers) < 10:
            _field_papers = set(random.sample(all_field_papers, 10 - len(field_papers)))
            field_papers.update(_field_papers - all_papers)
            all_papers.update(_field_papers)
        print(
            f"Selected {len(field_papers)} papers out of {len(all_field_papers)} papers for field {field}",
            file=sys.stderr,
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
    return list(map(lambda p: Path(p).absolute(), validation_set))


def split_entry(string: str, sep_left="[[", sep_right="]]"):
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
    string = list(
        map(
            lambda _s: re.sub(pattern=r"[^a-z0-9]", string=_s, repl=""),
            string[:1] + string[2:],
        )
    )
    string = "".join(string[:1] + exclude + string[1:])
    return string


def python_module(filename: Path | str):
    return str(Path(filename).relative_to(ROOT_FOLDER).with_suffix("")).replace(
        "/", "."
    )
