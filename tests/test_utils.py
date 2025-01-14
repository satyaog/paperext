import tempfile
from pathlib import Path

import paperext.utils
from paperext.utils import build_validation_set, split_entry, str_eq, str_normalize


def test_build_validation_set(monkeypatch, data_regression):
    categories = list(range(1000, 1000 + 15))
    files = list(range(15 * 100))

    here = Path(".").resolve()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        for cat in categories:
            cat_file = tmp_dir / f"{cat:04}_papers.txt"
            cat_file.write_text(
                "\n".join([f"rel/path/{fi:05}.{fi:05}.txt" for fi in files[:100]])
            )
            files = files[100:]

        val_set = build_validation_set(tmp_dir)

        assert len(set(val_set)) == len(categories) * 10
        # Reproducibility
        assert val_set == build_validation_set(tmp_dir)
        assert val_set != build_validation_set(tmp_dir, 42 + 42)

        for p in val_set:
            assert str(p).startswith(f"{here}/")
        rel_val_set = list(map(lambda p: p.relative_to(here), val_set))

        data_regression.check(str(rel_val_set))

        monkeypatch.setattr(paperext.utils, "PAPERS_TO_IGNORE", {str(rel_val_set[0])})

        # There should be only one paper not present and replaced by another
        assert len(set(val_set[1:10]) & set(build_validation_set(tmp_dir)[:10])) == 9


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
