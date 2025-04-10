import random
import re
import sys
import unicodedata
from pathlib import Path

from paperext import CFG
from paperext.config import Config
from paperext.log import logger

ROOT_FOLDER = Path(__file__).resolve().parent.parent
PAPERS_TO_IGNORE = {
    "arxiv/2404.09932.txt",
}


class PaperBase:
    # Original form of the converted pdf to txt. eg data/cache/*/ARXIV_ID.txt
    LINK_ID_TEMPLATE = "*/{link_id}.txt"
    # Extended form of the converted pdf to txt. eg data/cache/*/PAPER_ID.txt
    PAPER_ID_TEMPLATE = LINK_ID_TEMPLATE.format(link_id="{paper_id}")
    # The the up-to-date form of the converted pdf (by paperoni)
    # eg data/cache/fulltext/PAPER_ID/fulltext.txt
    PAPER_ID_FULLTEXT_TEMPLATE = "fulltext/{paper_id}/fulltext.txt"

    def __init__(self, paper: dict) -> None:
        self._paper = paper
        self._selected_id = None
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

    @property
    def _paper_id(self):
        return self._paper["paper_id"]

    def get_link_id_pdf(self):
        """Return a hardlink, with selected id as name, to the pdf.  Currently,
        the pdf file name is used as an id to check if the query should be done
        or not. As the pdf file name changed with the up-to-date paperoni cache
        structure, a hardlink might be created and returned to avoid redoing the
        query
        """
        link_id_pdf = None

        if self.pdf:
            link_id_pdf = self.pdf.with_stem(self.id)

        if link_id_pdf and not link_id_pdf.exists():
            link_id_pdf.hardlink_to(self.pdf)

        return link_id_pdf


class PaperMD(PaperBase):
    # Original form of the converted pdf to txt. eg data/cache/*/ARXIV_ID.txt
    LINK_ID_TEMPLATE = "*/{link_id}.md"
    # Extended form of the converted pdf to txt. eg data/cache/*/PAPER_ID.txt
    PAPER_ID_TEMPLATE = LINK_ID_TEMPLATE.format(link_id="{paper_id}")
    # The the up-to-date form of the converted pdf (by paperoni)
    # eg data/cache/fulltext/PAPER_ID/fulltext.txt
    PAPER_ID_FULLTEXT_TEMPLATE = "fulltext/{paper_id}/fulltext.md"


class PaperTxt(PaperBase):
    # Original form of the converted pdf to txt. eg data/cache/*/ARXIV_ID.txt
    LINK_ID_TEMPLATE = "*/{link_id}.txt"
    # Extended form of the converted pdf to txt. eg data/cache/*/PAPER_ID.txt
    PAPER_ID_TEMPLATE = LINK_ID_TEMPLATE.format(link_id="{paper_id}")
    # The the up-to-date form of the converted pdf (by paperoni)
    # eg data/cache/fulltext/PAPER_ID/fulltext.txt
    PAPER_ID_FULLTEXT_TEMPLATE = "fulltext/{paper_id}/fulltext.txt"


class Paper(PaperMD):
    def __init__(self, paper: dict):
        super().__init__(paper)
        paper_txt = PaperTxt(paper)

        with Config.push() as cfg:
            cfg.platform.select = "llamaparse"
            cfg.platform.struct = "parse_doc"
            cfg.dir.queries = cfg.dir.data / CFG.platform.struct / "queries"
            paper_md = PaperMD(paper)

        if self._selected_id is None or self._selected_id == self._paper_id:
            self._selected_id = paper_txt.id

        pdf = next(iter(self._pdfs + paper_txt.pdfs), None)

        if pdf is not None:
            pdf = pdf.with_suffix(".md")

            if paper_md.queries and not pdf.exists():
                from paperext.structured_output.parse_doc.model import Response

                markdown = "\n---\n".join(
                    Response.model_validate_json(
                        paper_md.queries[0].read_text()
                    ).analysis.pages_md
                )
                pdf.write_text(markdown)

            if pdf.exists() and pdf not in self._pdfs:
                self._pdfs.append(pdf)

        assert not (set(self._pdfs) & set(paper_txt.pdfs))

        self._pdfs.extend(paper_txt.pdfs)


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
    first, *extra = [_.strip().split(sep_right) for _ in string.split(sep_left)]

    try:
        assert len(first) == 1
        assert len(extra) <= 1
    except AssertionError:
        return [string]

    if extra:
        assert len(extra[0]) <= 2
        extra = extra[0]
    first = "".join([*first, *extra[1:]])
    extra = extra[:1]
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
