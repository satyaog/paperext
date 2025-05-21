from dataclasses import dataclass
from paperext.structured_output.papaff.evaluate import (
    StrEntry,
    compare_authors_order,
    compare_affiliations_set,
    compare_author_affiliations,
    update_stats,
    AuthorsOrderDiff,
    AffiliationsSetsDiff,
    AuthorAffiliationsSetsDiff,
    MissingAuthorDiff,
)
from paperext.structured_output.papaff.model import (
    AuthorAffiliations,
    Explained,
    Analysis,
)
from paperext.utils import str_normalize
from collections import Counter


def test_str_entry_init():
    @dataclass
    class TestObj:
        value: str

    # Test with string
    entry = StrEntry("test")
    assert entry.value == "test"
    assert entry.normalize is False
    assert entry.tolerance == 0
    assert entry.log is None

    # Test with object
    other = StrEntry(TestObj("test"))
    assert entry == other

    # Test with non-string
    assert isinstance(StrEntry(123).value, str)
    assert isinstance(StrEntry(None).value, str)


def test_str_entry_eq_with_tolerance():
    tolerance = 2
    entry = StrEntry("1234", tolerance=tolerance)

    for t in range(tolerance + 1):
        other = StrEntry(entry.value[t:])
        assert entry == other
        assert entry == entry.value[t:]

    assert entry != StrEntry(None)


def test_str_entry_eq_with_log():
    tolerance = 2
    log = []

    for t in range(tolerance + 1):
        entry = StrEntry("test", tolerance=t, log=log)
        other = StrEntry(entry.value[t:])
        assert entry == other

        if t == 0:
            assert not log

        if log:
            log_entry = log.pop()
            assert log_entry["value"] == entry.value
            assert log_entry["other"] == other.value
            assert log_entry["distance"] == t

        other = StrEntry(entry.value[t + 1 :])
        assert entry != other
        assert not log


def test_str_entry_eq_with_normalization():
    entry = StrEntry("T E-S_T", normalize=True)
    assert entry == str_normalize(entry.value)


def test_str_entry_hash():
    entry = StrEntry("T E-S_T")
    assert hash(entry) == hash(entry.value)

    entry = StrEntry("T E-S_T", normalize=True)
    assert hash(entry) != hash(entry.value)
    assert hash(entry) == hash(str_normalize(entry.value))


def test_str_entry_str():
    entry = StrEntry("T E-S_T", normalize=True)
    assert str(entry) == entry.value


def test_compare_authors_order():
    make_author_affiliations = lambda author: AuthorAffiliations(
        author=Explained[str](value=author, reasoning="", quote=""), affiliations=[]
    )

    # Test exact match
    authors1 = [
        make_author_affiliations("John Doe"),
        make_author_affiliations("Jane Smith"),
    ]
    authors2 = authors1[:]
    result, diff, warnings = compare_authors_order(authors1, authors2)
    assert result is True
    assert diff is None
    assert warnings == []

    # Test different order
    authors1 = [
        make_author_affiliations("John Doe"),
        make_author_affiliations("Jane Smith"),
    ]
    authors2 = list(reversed(authors1))
    result, diff, warnings = compare_authors_order(authors1, authors2)
    assert result is False
    assert isinstance(diff, AuthorsOrderDiff)
    assert diff.message == "Authors order mismatch"
    assert diff.validated == [a.author.value for a in authors1]
    assert diff.predicted == [a.author.value for a in authors2]
    assert warnings is None

    # Test with typos
    authors1 = [
        make_author_affiliations("John Doe"),
        make_author_affiliations("Jane Smith"),
    ]
    authors2 = [
        make_author_affiliations("Jon Doe"),
        make_author_affiliations("Jane Smith"),
    ]
    result, diff, warnings = compare_authors_order(authors1, authors2)
    assert result is True
    assert diff is None
    assert len(warnings) == 1
    assert warnings[0]["type"] == "author_name"
    assert warnings[0]["index"] == 0
    assert warnings[0]["validated"] == "John Doe"
    assert warnings[0]["predicted"] == "Jon Doe"
    assert warnings[0]["distance"] == 1

    # Test with multiple typos
    authors1 = [
        make_author_affiliations("John Doe"),
        make_author_affiliations("Jane Smith"),
    ]
    authors2 = [
        make_author_affiliations("Jon Doe"),
        make_author_affiliations("Jne Smith"),
    ]
    result, diff, warnings = compare_authors_order(authors1, authors2)
    assert result is True
    assert diff is None
    assert len(warnings) == 2
    assert all(w["type"] == "author_name" for w in warnings)
    assert {w["validated"] for w in warnings} == {a.author.value for a in authors1}
    assert {w["predicted"] for w in warnings} == {a.author.value for a in authors2}
    assert all(w["distance"] == 1 for w in warnings)

    # Test with case differences (normalization)
    authors1 = [
        make_author_affiliations("John Doe"),
        make_author_affiliations("Jane Smith"),
    ]
    authors2 = [make_author_affiliations(a.author.value.upper()) for a in authors1]
    result, diff, warnings = compare_authors_order(authors1, authors2)
    assert result is True
    assert diff is None
    assert warnings == []

    # Test with different length lists
    authors1 = [make_author_affiliations("John Doe")]
    authors2 = [
        make_author_affiliations("John Doe"),
        make_author_affiliations("Jane Smith"),
    ]
    result, diff, warnings = compare_authors_order(authors1, authors2)
    assert result is False
    assert isinstance(diff, AuthorsOrderDiff)
    assert diff.message == "Authors order mismatch"
    assert len(diff.validated) == 1
    assert len(diff.predicted) == 2
    assert warnings is None


def test_compare_affiliations_set():
    make_affiliation = lambda name: Explained[str](value=name, reasoning="", quote="")

    # Test exact match
    affs1 = [
        make_affiliation("University of California"),
        make_affiliation("Stanford University"),
    ]
    affs2 = affs1[:]
    result, diff, warnings = compare_affiliations_set(affs1, affs2)
    assert result is True
    assert diff is None
    assert warnings == []

    # Test different sets
    affs1 = [
        make_affiliation("University of California"),
        make_affiliation("Stanford University"),
    ]
    affs2 = [make_affiliation("MIT"), make_affiliation("Stanford University")]
    result, diff, warnings = compare_affiliations_set(affs1, affs2)
    assert result is False
    assert isinstance(diff, AffiliationsSetsDiff)
    assert diff.message == "Affiliations sets mismatch"
    assert set(map(str, diff.validated)) == set(map(StrEntry, affs1))
    assert set(map(str, diff.predicted)) == set(map(StrEntry, affs2))
    assert set(map(str, diff.missing)) == {StrEntry("University of California")}
    assert set(map(str, diff.extra)) == {StrEntry("MIT")}
    assert warnings is None

    # Test with typos
    affs1 = [
        make_affiliation("University of California"),
        make_affiliation("Stanford University"),
    ]
    affs2 = [
        make_affiliation("University of Califonia"),
        make_affiliation("Stanford University"),
    ]
    result, diff, warnings = compare_affiliations_set(affs1, affs2)
    assert result is True
    assert diff is None
    assert len(warnings) == 1
    assert warnings[0]["type"] == "affiliation_name"
    assert warnings[0]["index"] is None
    assert warnings[0]["validated"] == "University of California"
    assert warnings[0]["predicted"] == "University of Califonia"
    assert warnings[0]["distance"] == 1

    # Test with multiple typos
    affs1 = [
        make_affiliation("University of California"),
        make_affiliation("Stanford University"),
    ]
    affs2 = [
        make_affiliation("University of Califonia"),
        make_affiliation("Standford University"),
    ]
    result, diff, warnings = compare_affiliations_set(affs1, affs2)
    assert result is True
    assert diff is None
    assert len(warnings) == 2
    assert all(w["type"] == "affiliation_name" for w in warnings)
    assert {w["validated"] for w in warnings} == {StrEntry(a) for a in affs1}
    assert {w["predicted"] for w in warnings} == {StrEntry(a) for a in affs2}
    assert all(w["distance"] == 1 for w in warnings)

    # Test with case differences (normalization)
    affs1 = [
        make_affiliation("University of California"),
        make_affiliation("Stanford University"),
    ]
    affs2 = [make_affiliation(aff.value.upper()) for aff in affs1]
    result, diff, warnings = compare_affiliations_set(affs1, affs2)
    assert result is True
    assert diff is None
    assert warnings == []

    # Test with different length lists
    affs1 = [make_affiliation("University of California")]
    affs2 = [
        make_affiliation("University of California"),
        make_affiliation("Stanford University"),
    ]
    result, diff, warnings = compare_affiliations_set(affs1, affs2)
    assert result is False
    assert isinstance(diff, AffiliationsSetsDiff)
    assert diff.message == "Affiliations sets mismatch"
    assert set(map(str, diff.validated)) == set(map(StrEntry, affs1))
    assert set(map(str, diff.predicted)) == set(map(StrEntry, affs2))
    assert not diff.missing
    assert set(map(str, diff.extra)) == {StrEntry("Stanford University")}
    assert warnings is None


def test_compare_author_affiliations():
    make_author_affiliations = lambda author, affs: AuthorAffiliations(
        author=Explained[str](value=author, reasoning="", quote=""),
        affiliations=[
            Explained[str](value=aff, reasoning="", quote="") for aff in affs
        ],
    )

    # Test exact match
    authors1 = [
        make_author_affiliations("John Doe", ["University of California"]),
        make_author_affiliations("Jane Smith", ["Stanford University"]),
    ]
    authors2 = authors1[:]
    result, diff, warnings = compare_author_affiliations(authors1, authors2)
    assert result is True
    assert diff is None
    assert warnings == []

    # Test missing author
    authors1 = [
        make_author_affiliations("John Doe", ["University of California"]),
        make_author_affiliations("Jane Smith", ["Stanford University"]),
    ]
    authors2 = [authors1[0]]
    result, diff, warnings = compare_author_affiliations(authors1, authors2)
    assert result is False
    assert len(diff) == 1
    assert isinstance(diff[0], MissingAuthorDiff)
    assert diff[0].index == 1
    assert diff[0].validated == "Jane Smith"
    assert diff[0].predicted is None
    assert diff[0].message == "Missing author"
    assert warnings is None

    # Test with typos in authors names
    authors1 = [
        make_author_affiliations("John Doe", []),
        make_author_affiliations("Jane Smith", []),
    ]
    authors2 = [
        make_author_affiliations("Jane Smith", []),
        make_author_affiliations("Jon Doe", []),
    ]
    result, diff, warnings = compare_author_affiliations(authors1, authors2)
    assert result is True
    assert diff is None
    assert len(warnings) == 1
    assert warnings[0]["type"] == "author_name"
    assert warnings[0]["validated"] == "John Doe"
    assert warnings[0]["predicted"] == "Jon Doe"
    assert warnings[0]["distance"] == 1

    # Test different affiliations
    authors1 = [
        make_author_affiliations("John Doe", ["University of California"]),
        make_author_affiliations("Jane Smith", ["Stanford University"]),
    ]
    authors2 = [
        make_author_affiliations("John Doe", ["MIT"]),
        make_author_affiliations("Jane Smith", ["Stanford University"]),
    ]
    result, diff, warnings = compare_author_affiliations(authors1, authors2)
    assert result is False
    assert len(diff) == 1
    assert isinstance(diff[0], AuthorAffiliationsSetsDiff)
    assert diff[0].author == "John Doe"
    assert diff[0].message == "Author affiliations sets mismatch"
    assert set(diff[0].validated) == {StrEntry("University of California")}
    assert set(diff[0].predicted) == {StrEntry("MIT")}
    assert set(diff[0].missing) == {StrEntry("University of California")}
    assert set(diff[0].extra) == {StrEntry("MIT")}
    assert warnings is None

    # Test with typos in affiliations
    authors1 = [
        make_author_affiliations("John Doe", ["University of California"]),
        make_author_affiliations("Jane Smith", ["Stanford University"]),
    ]
    authors2 = [
        make_author_affiliations("John Doe", ["University of Califonia"]),
        make_author_affiliations("Jane Smith", ["Stanford University"]),
    ]
    result, diff, warnings = compare_author_affiliations(authors1, authors2)
    assert result is True
    assert diff is None
    assert len(warnings) == 1
    assert warnings[0]["type"] == "author_affiliation_name"
    assert warnings[0]["author"] == "John Doe"
    assert warnings[0]["index"] is None
    assert warnings[0]["validated"] == "University of California"
    assert warnings[0]["predicted"] == "University of Califonia"
    assert warnings[0]["distance"] == 1

    # Test with multiple typos
    authors1 = [
        make_author_affiliations("John Doe", ["University of California"]),
        make_author_affiliations("Jane Smith", ["Stanford University"]),
    ]
    authors2 = [
        make_author_affiliations("John Doe", ["University of Califonia"]),
        make_author_affiliations("Jane Smith", ["Standford University"]),
    ]
    result, diff, warnings = compare_author_affiliations(authors1, authors2)
    assert result is True
    assert diff is None
    assert len(warnings) == 2
    assert all(w["type"] == "author_affiliation_name" for w in warnings)
    assert {w["author"] for w in warnings} == {"John Doe", "Jane Smith"}
    assert {w["validated"] for w in warnings} == {
        "University of California",
        "Stanford University",
    }
    assert {w["predicted"] for w in warnings} == {
        "University of Califonia",
        "Standford University",
    }
    assert all(w["distance"] == 1 for w in warnings)

    # Test with case differences (normalization)
    authors1 = [
        make_author_affiliations("John Doe", ["University of California"]),
        make_author_affiliations("Jane Smith", ["Stanford University"]),
    ]
    authors2 = [
        make_author_affiliations("John Doe", ["UNIVERSITY OF CALIFORNIA"]),
        make_author_affiliations("Jane Smith", ["STANFORD UNIVERSITY"]),
    ]
    result, diff, warnings = compare_author_affiliations(authors1, authors2)
    assert result is True
    assert diff is None
    assert warnings == []

    # Test with multiple affiliations
    authors1 = [
        make_author_affiliations(
            "John Doe", ["University of California", "Lawrence Berkeley Lab"]
        ),
        make_author_affiliations("Jane Smith", ["Stanford University"]),
    ]
    authors2 = [
        make_author_affiliations(
            "John Doe", ["University of California", "Lawrence Berkeley Lab"]
        ),
        make_author_affiliations("Jane Smith", ["Stanford University"]),
    ]
    result, diff, warnings = compare_author_affiliations(authors1, authors2)
    assert result is True
    assert diff is None
    assert warnings == []


def test_update_stats():
    # Helper function to create test data
    make_author_affiliations = lambda author, affs: AuthorAffiliations(
        author=Explained[str](value=author, reasoning="", quote=""),
        affiliations=[
            Explained[str](value=aff, reasoning="", quote="") for aff in affs
        ],
    )

    # Test basic stats update
    stats = Counter()
    validated = Analysis(
        authors_affiliations=[
            make_author_affiliations("John Doe", ["University A", "University B"]),
            make_author_affiliations("Jane Smith", ["University C"]),
        ],
        affiliations=["University A", "University B", "University C"],
    )

    update_stats(
        stats,
        validated,
        authors_order=True,
        affiliations=True,
        author_affiliations=True,
    )

    assert stats["total"] == 1
    assert stats["authors_total"] == 2
    assert stats["affiliations_total"] == 3
    assert stats["author_affiliations_total"] == 3
    assert stats["authors_order_pass"] == 1
    assert stats["authors_order_fail"] == 0
    assert stats["affiliations_pass"] == 1
    assert stats["affiliations_fail"] == 0
    assert stats["author_affiliations_pass"] == 1
    assert stats["author_affiliations_fail"] == 0

    # Test with failures
    update_stats(
        stats,
        validated,
        authors_order=False,
        affiliations=False,
        author_affiliations=False,
    )

    assert stats["total"] == 2
    assert stats["authors_order_pass"] == 1
    assert stats["authors_order_fail"] == 1
    assert stats["affiliations_pass"] == 1
    assert stats["affiliations_fail"] == 1
    assert stats["author_affiliations_pass"] == 1
    assert stats["author_affiliations_fail"] == 1

    # Test with author_affiliations_diff
    author_affiliations_diff = [
        MissingAuthorDiff(
            index=0,
            validated="John Doe",
            predicted=None,
        ),
        AuthorAffiliationsSetsDiff(
            author="Jane Smith",
            validated=["University C"],
            predicted=["University D"],
            missing=["University C"],
            extra=["University D"],
        ),
    ]

    update_stats(
        stats,
        validated,
        authors_order=True,
        affiliations=True,
        author_affiliations=False,
        author_affiliations_diff=author_affiliations_diff,
    )

    assert stats["missing_authors"] == 1
    assert stats["missing_author_affiliations"] == 1
    assert stats["extra_author_affiliations"] == 1
    assert stats["wrong_author_affiliations"] == 0

    # Test with affiliations_diff
    affiliations_diff = AffiliationsSetsDiff(
        validated=["University A", "University B"],
        predicted=["University D", "University E"],
        missing=["University A", "University B"],
        extra=["University D", "University E"],
    )

    update_stats(
        stats,
        validated,
        authors_order=True,
        affiliations=False,
        author_affiliations=True,
        affiliations_diff=affiliations_diff,
    )

    assert stats["missing_affiliations"] == 2
    assert stats["extra_affiliations"] == 2
    assert stats["wrong_affiliations"] == 0

    # Test with both diffs and some matches
    affiliations_diff = AffiliationsSetsDiff(
        validated=["University A", "University B"],
        predicted=["University A", "University D"],
        missing=["University B"],
        extra=["University D"],
    )

    update_stats(
        stats,
        validated,
        authors_order=True,
        affiliations=False,
        author_affiliations=True,
        affiliations_diff=affiliations_diff,
    )

    assert stats["missing_affiliations"] == 3  # 2 + 1 (from previous match)
    assert stats["extra_affiliations"] == 3  # 2 + 1 (from previous match)
    assert stats["wrong_affiliations"] == 1  # University A matches

    # Test with empty validated data
    empty_stats = Counter()
    empty_validated = Analysis(authors_affiliations=[], affiliations=[])

    update_stats(
        empty_stats,
        empty_validated,
        authors_order=True,
        affiliations=True,
        author_affiliations=True,
    )

    assert empty_stats["total"] == 1
    assert empty_stats["authors_total"] == 0
    assert empty_stats["affiliations_total"] == 0
    assert empty_stats["author_affiliations_total"] == 0
