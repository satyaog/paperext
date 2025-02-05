import copy
import json
import tempfile
from pathlib import Path

import paperext.utils
from paperext.config import Config
from paperext.utils import (
    Paper,
    build_validation_set,
    split_entry,
    str_eq,
    str_normalize,
)


def test_paper(cfg: Config):
    arxiv_id = "2302.00999"

    def _test(
        paper: Paper, expected_id: str, expected_query_files: list, expected_pdf: Path
    ):
        assert paper.id == expected_id
        assert paper.queries == expected_query_files
        assert paper.pdf == expected_pdf

        for _f in expected_query_files:
            assert _f.exists()

    for p in json.loads(
        (cfg.dir.data / "paperoni-2023-2024-PR_2024-09-30.json").read_text()
    ):
        if list(filter(lambda l: l.get("link", None) == arxiv_id, p["links"])):
            paper = Paper(p)
            break

    assert paper._paper_id == p["paper_id"]

    # When it exists, the filename of an existing a query file should be used to
    # infer the paper id
    expected_id = arxiv_id
    expected_query_files = [
        cfg.dir.queries / cfg.platform.select / f"{expected_id}_00.json"
    ]
    # Paperoni's newer cache structure should be used as the older path
    # structure is not present in test data
    expected_pdf = cfg.dir.cache / f"fulltext/{p['paper_id']}/fulltext.txt"

    _test(paper, expected_id, expected_query_files, expected_pdf)
    assert paper.get_link_id_pdf().exists()
    assert paper.get_link_id_pdf().parent == expected_pdf.parent
    assert paper.get_link_id_pdf().stem == arxiv_id

    paper.get_link_id_pdf().unlink()
    assert paper.get_link_id_pdf().exists()

    p["paper_id"] = "does_not_exist"

    # When it exists, the filename of an existing a query file should be used to
    # infer the paper id
    expected_id = f"query_{arxiv_id}"
    expected_query_files = [
        cfg.dir.queries / cfg.platform.select / f"{expected_id}_00.json"
    ]
    # There is no pdf file for this id, only the query file exists in test data
    expected_pdf = None

    for l in filter(lambda l: l.get("link", None) == arxiv_id, p["links"]):
        l["link"] = expected_id

    _test(Paper(p), expected_id, expected_query_files, expected_pdf)

    # Use the original cache structure when available
    expected_id = f"cache_original_{arxiv_id}"
    # There is no query file for this id, only the pdf file exists in test data
    expected_query_files = []
    expected_pdf = cfg.dir.cache / f"arxiv/{expected_id}.txt"

    for l in filter(lambda l: l.get("link", None).endswith(f"_{arxiv_id}"), p["links"]):
        l["link"] = expected_id

    _test(Paper(p), expected_id, expected_query_files, expected_pdf)

    p["paper_id"] = "cache_fulltext_paper_id"

    # Paperoni's paper id should be used if the older cache structure is not
    # present in test data
    expected_id = p["paper_id"]
    expected_query_files = []
    # Paperoni's newer cache structure should be used as the older path
    # structure is not present in test data
    expected_pdf = cfg.dir.cache / f"fulltext/{expected_id}/fulltext.txt"

    for l in filter(lambda l: l.get("link", None).endswith(f"_{arxiv_id}"), p["links"]):
        l["link"] = f"does_not_exist_{arxiv_id}"

    _test(Paper(p), expected_id, expected_query_files, expected_pdf)


def test_multiple_sources(cfg: Config):
    # Test that multiple downloaded pdf files from multiple sources (e.g. arxiv
    # and/or openreview) do not end up in multiple query files
    arxiv_id = "2304.07193"
    openreview_id = "a68SUt6zFt"

    def _cp(src: Path, dest: Path):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text())

    def _get_papers():
        for p in json.loads(
            (cfg.dir.data / "paperoni-2023-2024-PR_2024-09-30.json").read_text()
        ):
            if list(filter(lambda l: l.get("link", None) == arxiv_id, p["links"])):
                paper = Paper(p)
                with Config.push(arxiv_cfg):
                    arxiv_paper = Paper(p)
                with Config.push(openreview_cfg):
                    openreview_paper = Paper(p)
                break

        return (paper, arxiv_paper, openreview_paper)

    arxiv_cfg = copy.deepcopy(cfg)
    openreview_cfg = copy.deepcopy(cfg)

    for _cfg in (arxiv_cfg, openreview_cfg):
        _cfg.dir.data = Path(
            tempfile.TemporaryDirectory(dir=str(cfg.dir.root / "tmp")).name
        )
        _cfg.dir.cache = _cfg.dir.data / "cache"
        _cfg.dir.merged = _cfg.dir.data / "merged"
        _cfg.dir.queries = _cfg.dir.data / "queries"

        (_cfg.dir.cache / "arxiv/").mkdir(parents=True, exist_ok=True)
        (_cfg.dir.cache / "openreview/").mkdir(parents=True, exist_ok=True)

    # With only a single pdf txt file for each arxiv and openreview
    _cp(
        cfg.dir.cache / f"arxiv/{arxiv_id}.txt",
        arxiv_cfg.dir.cache / f"arxiv/{arxiv_id}.txt",
    )
    _cp(
        cfg.dir.cache / f"openreview/{openreview_id}.txt",
        openreview_cfg.dir.cache / f"openreview/{openreview_id}.txt",
    )

    _, arxiv_paper, openreview_paper = _get_papers()

    assert arxiv_paper.id == arxiv_id
    assert len(arxiv_paper.pdfs) == 1
    assert not arxiv_paper.queries

    assert openreview_paper.id == openreview_id
    assert len(openreview_paper.pdfs) == 1
    assert not openreview_paper.queries

    # With only a single query file of the other source for each arxiv and openreview
    _cp(
        cfg.dir.queries / f"openai/{openreview_id}_00.json",
        arxiv_cfg.dir.queries / f"openai/{openreview_id}_00.json",
    )
    _cp(
        cfg.dir.queries / f"openai/{arxiv_id}_00.json",
        openreview_cfg.dir.queries / f"openai/{arxiv_id}_00.json",
    )

    _, arxiv_paper, openreview_paper = _get_papers()

    # The query file should be priorized when looking for the id to use
    assert arxiv_paper.id == openreview_id
    assert not arxiv_paper.pdfs
    assert len(arxiv_paper.queries) == 1
    assert arxiv_paper.queries[0].name.startswith(f"{openreview_id}_")

    assert openreview_paper.id == arxiv_id
    assert not openreview_paper.pdfs
    assert len(openreview_paper.queries) == 1
    assert openreview_paper.queries[0].name.startswith(f"{arxiv_id}_")

    # With both query files for each arxiv and openreview
    _cp(
        cfg.dir.queries / f"openai/{arxiv_id}_00.json",
        arxiv_cfg.dir.queries / f"openai/{arxiv_id}_00.json",
    )
    _cp(
        cfg.dir.queries / f"openai/{openreview_id}_00.json",
        openreview_cfg.dir.queries / f"openai/{openreview_id}_00.json",
    )

    paper, arxiv_paper, openreview_paper = _get_papers()

    # Both arxiv and openreview contain the same query files so the resulting id
    # should be the same
    assert arxiv_paper.id == openreview_paper.id == paper.id
    assert len(arxiv_paper.pdfs) + len(openreview_paper.pdfs) == 1
    assert len(arxiv_paper.queries) == 2

    assert [_f.name for _f in (arxiv_paper.pdfs or openreview_paper.pdfs)] == [
        _f.name for _f in paper.pdfs
    ]
    assert [_f.name for _f in arxiv_paper.queries] == [
        _f.name for _f in openreview_paper.queries
    ]


def test_build_validation_set(monkeypatch, data_regression, cfg: Config):
    categories = list(range(1000, 1000 + 15))
    files = list(range(15 * 100))

    here = Path(".").resolve()

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg.dir.data = tmp_dir = Path(tmp_dir)
        for cat in categories:
            cat_file = tmp_dir / f"{cat:04}_papers.txt"
            cat_file.write_text(
                "\n".join([f"rel/path/{fi:05}.{fi:05}.txt" for fi in files[:100]])
            )
            files = files[100:]

        val_set = build_validation_set()

        assert len(set(val_set)) == len(categories) * 10
        # Reproducibility
        assert val_set == build_validation_set()
        assert val_set != build_validation_set(42 + 42)

        for p in val_set:
            assert str(p).startswith(f"{here}/")
        rel_val_set = list(map(lambda p: p.relative_to(here), val_set))

        data_regression.check(str(rel_val_set))

        monkeypatch.setattr(paperext.utils, "PAPERS_TO_IGNORE", {str(rel_val_set[0])})

        # There should be only one paper not present and replaced by another
        assert len(set(val_set[1:10]) & set(build_validation_set()[:10])) == 9


def test_split_entry():
    expected_prefix = "prefix"
    prefix, *l = split_entry(f" {expected_prefix} [[l1 , l2 , l3 ]]")

    assert expected_prefix == prefix
    assert l == "l1,l2,l3".split(",")


def test_str_eq():
    assert str_eq("aB 'cDe 1$2#", 'Ab "CdE %1?2')


def test_str_normalize():
    assert str_normalize("aB 'CDe 1$2#") == "abcde12"
    assert str_normalize("aB {{'C}}De 1$2# ") == "ab'cde12"
