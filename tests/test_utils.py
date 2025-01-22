import json
import tempfile
from pathlib import Path

import paperext.utils
from paperext.utils import (
    Paper,
    build_validation_set,
    split_entry,
    str_eq,
    str_normalize,
)
from paperext.config import Config


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

    expected_id = arxiv_id
    expected_query_files = [
        cfg.dir.queries / cfg.platform.select / f"{expected_id}_00.json"
    ]
    expected_pdf = cfg.dir.cache / f"fulltext/{p['paper_id']}/fulltext.txt"

    _test(paper, expected_id, expected_query_files, expected_pdf)
    assert paper.get_link_id_pdf().exists()
    assert paper.get_link_id_pdf().parent == expected_pdf.parent
    assert paper.get_link_id_pdf().stem == arxiv_id

    paper.get_link_id_pdf().unlink()
    assert paper.get_link_id_pdf().exists()

    p["paper_id"] = "does_not_exist"

    expected_id = "query_2302.00999"
    expected_query_files = [
        cfg.dir.queries / cfg.platform.select / f"{expected_id}_00.json"
    ]
    expected_pdf = None

    for l in filter(lambda l: l.get("link", None) == arxiv_id, p["links"]):
        l["link"] = expected_id

    _test(Paper(p), expected_id, expected_query_files, expected_pdf)

    expected_id = "cache_original_2302.00999"
    expected_query_files = []
    expected_pdf = cfg.dir.cache / f"arxiv/{expected_id}.txt"

    for l in filter(lambda l: l.get("link", None).endswith(f"_{arxiv_id}"), p["links"]):
        l["link"] = expected_id

    _test(Paper(p), expected_id, expected_query_files, expected_pdf)

    p["paper_id"] = "cache_fulltext_paper_id"

    expected_id = p["paper_id"]
    expected_query_files = []
    expected_pdf = cfg.dir.cache / f"fulltext/{expected_id}/fulltext.txt"

    for l in filter(lambda l: l.get("link", None).endswith(f"_{arxiv_id}"), p["links"]):
        l["link"] = f"does_not_exist_{arxiv_id}"

    _test(Paper(p), expected_id, expected_query_files, expected_pdf)


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
