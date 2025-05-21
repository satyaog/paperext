import argparse
from dataclasses import dataclass
import functools
import json
import csv
from pathlib import Path
import sys
import tempfile
import yaml
from collections import Counter
from typing import List, Optional, Tuple, Dict, Any, Union

from rich.console import Console
from rich.table import Table
import Levenshtein

from paperext import CFG
from paperext.config import Config
from paperext.log import logger
from paperext.structured_output.papaff.model import Analysis
from paperext.structured_output.parse_doc.model import Response as ParsedDocResponse
from paperext.utils import Paper, str_normalize


@dataclass
class ReportEntry:
    paper: str
    title: str
    authors_order: str
    affiliations: str
    author_affiliations: str
    llm_file: str
    val_file: str
    pdf_file: str
    pdf_txt_file: str
    authors_order_diff: Optional[Dict] = None
    affiliations_diff: Optional[Dict] = None
    author_affiliations_diff: Optional[Dict] = None
    typo_warnings: Optional[List[Dict]] = None

    def to_dict(self):
        return vars(self)


@dataclass
class StrEntry:
    """String entry with optional normalization and tolerance for typos."""

    value: str
    normalize: bool = False
    tolerance: int = 0
    log: list[dict] = None

    def __init__(self, value, normalize=False, tolerance=0, log=None):
        if hasattr(value, "value"):
            value = value.value
        if not isinstance(value, str):
            value = str(value)

        self.value = value
        self.normalize = normalize
        self.tolerance = tolerance
        self.log = log

    def has_typo(self, other):
        if not isinstance(other, StrEntry):
            other = StrEntry(other, normalize=self.normalize, tolerance=self.tolerance)

        if self.normalize:
            self_str = str_normalize(self.value)
            other_str = str_normalize(other.value)
        else:
            self_str = self.value
            other_str = other.value

        return has_typo(self_str, other_str, self.tolerance)

    def __eq__(self, other):
        if not isinstance(other, StrEntry):
            other = StrEntry(other, normalize=self.normalize, tolerance=self.tolerance)

        has_typo, distance = self.has_typo(other)
        if self.log is not None and has_typo and distance:
            self.log.append(
                {
                    "value": str(self),
                    "other": str(other),
                    "distance": distance,
                }
            )
        return has_typo

    def __hash__(self):
        if self.normalize:
            self_str = str_normalize(self.value)
        else:
            self_str = self.value

        return hash(self_str)

    def __str__(self):
        return self.value

    def __repr__(self):
        if self.normalize:
            return f"{str_normalize(self.value)} ({self.value})"
        else:
            return f"{self.value}"


@dataclass
class CompareDiff:
    validated: Any
    predicted: Any

    @property
    def message(self):
        return None


@dataclass
class ListDiff(CompareDiff):
    validated: List[StrEntry]
    predicted: List[StrEntry]
    missing: List[StrEntry]
    extra: List[StrEntry]


@dataclass
class AuthorsOrderDiff(ListDiff):
    missing: List[StrEntry] = None
    extra: List[StrEntry] = None

    @property
    def message(self):
        return "Authors order mismatch"


@dataclass
class AffiliationsSetsDiff(ListDiff):
    @property
    def message(self):
        return "Affiliations sets mismatch"


@dataclass
class AuthorAffiliationsSetsDiff(AffiliationsSetsDiff):
    author: str

    @property
    def message(self):
        return "Author affiliations sets mismatch"


@dataclass
class MissingAuthorDiff(CompareDiff):
    index: int
    validated: str
    predicted: Optional[str]

    @property
    def message(self):
        return "Missing author"


def paper_md(paper: Paper) -> Path:
    with Config.push():
        CFG.platform.select = "mistralai"
        CFG.platform.struct = "parse_doc"
        CFG.dir.queries = CFG.dir.data / CFG.platform.struct / "queries"

        paper_md = ParsedDocResponse.model_validate_json(
            Paper(paper._paper).queries[-1].read_text()
        ).analysis

    # write md to temporary file
    with tempfile.NamedTemporaryFile(mode="wt", delete=False, suffix=".md") as f:
        f.write("\n---\n".join(paper_md.pages_md))

    return Path(f.name)


def load_analysis(file_path: Path) -> Analysis:
    """Load and parse an Analysis object from a JSON or YAML file.

    Args:
        file_path: Path to the file containing analysis data

    Returns:
        Analysis object with parsed data

    Raises:
        ValueError: If the file type is not supported
    """
    if file_path.suffix == ".json":
        with file_path.open() as f:
            data = json.load(f)
    elif file_path.suffix in {".yaml", ".yml"}:
        with file_path.open() as f:
            data = yaml.safe_load(f)

    # LLM output: nested under 'analysis'
    if "analysis" in data:
        return Analysis.parse_obj(data["analysis"])
    else:
        return Analysis.parse_obj(data)


def has_typo(s1: str, s2: str, max_distance: int = 2) -> Tuple[bool, int]:
    """Check if two strings differ by at most max_distance characters.

    Args:
        s1: First string
        s2: Second string
        max_distance: Maximum allowed Levenshtein distance

    Returns:
        Tuple of (bool, int): True if strings differ by at most max_distance characters,
        and the actual distance.
    """
    if max_distance == 0:
        return s1 == s2, 0
    else:
        distance = Levenshtein.distance(s1, s2)
        return distance <= max_distance, distance


def compare_authors_order(
    authors_list: list, other_list: list
) -> Tuple[bool, Optional[AuthorsOrderDiff], Optional[List[Dict]]]:
    """Compare if authors appear in the same order in both lists.

    Args:
        authors_list: First list of author affiliations
        other_list: Second list of author affiliations

    Returns:
        Tuple of (bool, dict, list):
        - True if authors are in the same order, False otherwise
        - If False, dict contains the differences
        - List of typo warnings if any
    """
    _str_entry = functools.partial(StrEntry, normalize=True, tolerance=2)
    log = []
    l1 = [_str_entry(aa.author, log=log) for aa in authors_list]
    l2 = [_str_entry(aa.author) for aa in other_list]

    # Check for matches
    if l1 != l2:
        return (
            False,
            AuthorsOrderDiff(validated=l1, predicted=l2),
            None,
        )

    # Report typos
    warnings = []
    while log:
        log_entry = log.pop()
        index = l1.index(log_entry["value"])
        warnings.append(
            {
                "type": "author_name",
                "index": index,
                "validated": log_entry["value"],
                "predicted": log_entry["other"],
                "distance": log_entry["distance"],
            }
        )

    return True, None, warnings


def compare_affiliations_set(
    affiliations_list: list, other_list: list
) -> Tuple[bool, Optional[AffiliationsSetsDiff], Optional[List[Dict]]]:
    """Compare if both lists contain the same set of affiliations.

    Args:
        affiliations_list: First list of affiliations
        other_list: Second list of affiliations

    Returns:
        Tuple of (bool, dict, list):
        - True if both contain the same affiliations, False otherwise
        - If False, dict contains the differences
        - List of typo warnings if any
    """
    _str_entry = functools.partial(StrEntry, normalize=True, tolerance=2)
    log = []
    l1 = [_str_entry(aff, log=log) for aff in affiliations_list]
    l2 = [_str_entry(aff) for aff in other_list]

    missing = l1[:]
    extra = []
    for other in l2:
        try:
            index = missing.index(other)
            missing.pop(index)
        except ValueError:
            extra.append(other)

    if missing or extra:
        return (
            False,
            AffiliationsSetsDiff(
                validated=l1, predicted=l2, missing=missing, extra=extra
            ),
            None,
        )

    # Check for typos in missing/extra affiliations
    warnings = []
    while log:
        log_entry = log.pop()
        warnings.append(
            {
                "type": "affiliation_name",
                "index": None,
                "validated": log_entry["value"],
                "predicted": log_entry["other"],
                "distance": log_entry["distance"],
            }
        )

    return True, None, warnings


def compare_author_affiliations(authors_list: list, other_list: list) -> Tuple[
    bool,
    Optional[List[Union[AuthorAffiliationsSetsDiff, MissingAuthorDiff]]],
    Optional[List[Dict]],
]:
    """Compare if authors have the same affiliations in both lists.

    Args:
        authors_list: First list of author affiliations
        other_list: Second list of author affiliations

    Returns:
        Tuple of (bool, dict, list):
        - True if authors have the same affiliations, False otherwise
        - If False, dict contains the differences
        - List of typo warnings if any
    """
    differences = []
    warnings = []
    for i, aa1 in enumerate(authors_list):
        author1 = StrEntry(aa1.author, normalize=True, tolerance=2, log=[])
        aa2 = next(filter(lambda x: author1 == x.author, other_list), None)

        if aa2 is None:
            differences.append(
                MissingAuthorDiff(index=i, validated=str(author1), predicted=None)
            )
            continue

        while author1.log:
            log_entry = author1.log.pop()
            warnings.append(
                {
                    "type": "author_name",
                    "index": i,
                    "validated": log_entry["value"],
                    "predicted": log_entry["other"],
                    "distance": log_entry["distance"],
                }
            )

        is_same, diff, aff_warnings = compare_affiliations_set(
            aa1.affiliations, aa2.affiliations
        )

        if not is_same:
            differences.append(
                AuthorAffiliationsSetsDiff(author=str(author1), **vars(diff))
            )
            continue

        warnings.extend(
            [
                {**w, "type": f"author_{w['type']}", "author": str(author1)}
                for w in aff_warnings
            ]
        )

    if differences:
        return False, differences, warnings

    return True, None, warnings


def update_stats(
    stats: Counter,
    validated: Analysis,
    authors_order: bool,
    affiliations: bool,
    author_affiliations: bool,
    author_affiliations_diff: Optional[
        List[Union[AuthorAffiliationsSetsDiff, MissingAuthorDiff]]
    ] = None,
    affiliations_diff: Optional[AffiliationsSetsDiff] = None,
) -> None:
    # Update statistics
    stats["total"] += 1
    stats["authors_total"] += len(validated.authors_affiliations)
    stats["affiliations_total"] += len(validated.affiliations)
    stats["author_affiliations_total"] += sum(
        map(lambda x: len(x.affiliations), validated.authors_affiliations)
    )

    stats["authors_order_pass"] += authors_order
    stats["authors_order_fail"] += not authors_order
    stats["affiliations_pass"] += affiliations
    stats["affiliations_fail"] += not affiliations
    stats["author_affiliations_pass"] += author_affiliations
    stats["author_affiliations_fail"] += not author_affiliations

    # Track missing/extra author and affiliations counts
    if author_affiliations_diff:
        stats["missing_authors"] += len(
            [d for d in author_affiliations_diff if isinstance(d, MissingAuthorDiff)]
        )
        for diff in author_affiliations_diff:
            if isinstance(diff, AuthorAffiliationsSetsDiff):
                _missing = len(diff.missing)
                _extra = len(diff.extra)
                _wrong = min(_missing, _extra)
                stats["missing_author_affiliations"] += _missing - _wrong
                stats["extra_author_affiliations"] += _extra - _wrong
                stats["wrong_author_affiliations"] += _wrong

    # Track missing/extra affiliations counts
    if affiliations_diff:
        _missing = len(affiliations_diff.missing)
        _extra = len(affiliations_diff.extra)
        _wrong = min(_missing, _extra)
        stats["missing_affiliations"] += _missing - _wrong
        stats["extra_affiliations"] += _extra - _wrong
        stats["wrong_affiliations"] += _wrong


def format_differences(diff_data: ReportEntry) -> None:
    """Format and print differences using rich formatting.

    Args:
        diff_data: ReportEntry object containing difference information
    """
    console = Console(force_terminal=True, stderr=True)

    if (
        diff_data.authors_order_diff
        or diff_data.affiliations_diff
        or diff_data.author_affiliations_diff
        or diff_data.typo_warnings
    ):
        console.print(f"\n[bold red]Differences for {diff_data.title}[/bold red]")
        console.print(f"{diff_data.pdf_file=}")
        console.print(f"{diff_data.pdf_txt_file=}")
        console.print(f"{diff_data.llm_file=}")
        console.print(f"{diff_data.val_file=}")

    # Print typo warnings first
    if diff_data.typo_warnings:
        console.print("\n[bold yellow]Typo Warnings:[/bold yellow]")
        for warning in diff_data.typo_warnings:
            if warning["type"] == "author_name":
                console.print(f"  Author name typo (distance={warning['distance']}):")
                console.print(f"    Validated: {warning['validated']}")
                console.print(f"    Predicted: {warning['predicted']}")
            elif warning["type"] == "author_affiliation_name":
                console.print(
                    f"  Affiliation typo for author {warning['author']} (distance={warning['distance']}):"
                )
                console.print(f"    Validated: {warning['validated']}")
                console.print(f"    Predicted: {warning['predicted']}")
            elif warning["type"] == "affiliation_name":
                console.print(f"  Affiliation typo (distance={warning['distance']}):")
                console.print(f"    Validated: {warning['validated']}")
                console.print(f"    Predicted: {warning['predicted']}")

    if diff_data.authors_order_diff:
        console.print(
            f"\n[bold magenta]{diff_data.authors_order_diff.message}[/bold magenta]"
        )
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("Validated Order")
        table.add_column("LLM Order")

        validated = diff_data.authors_order_diff.validated
        predicted = diff_data.authors_order_diff.predicted

        # Find the differences and highlight them
        for v, l in zip(
            validated + [""] * len(predicted), predicted + [""] * len(validated)
        ):
            if not v and not l:
                continue
            if v != l:
                table.add_row(f"[red]{v}[/red]", f"[red]{l}[/red]")
            else:
                table.add_row(str(v), str(l))

        console.print(table)

    if diff_data.affiliations_diff:
        console.print(
            f"\n[bold magenta]{diff_data.affiliations_diff.message}[/bold magenta]"
        )

        # Missing affiliations
        if diff_data.affiliations_diff.missing:
            console.print("\n[bold yellow]Missing affiliations:[/bold yellow]")
            for aff in diff_data.affiliations_diff.missing:
                console.print(f"  [red]- {str(aff)}[/red]")

        # Extra affiliations
        if diff_data.affiliations_diff.extra:
            console.print("\n[bold yellow]Extra affiliations:[/bold yellow]")
            for aff in diff_data.affiliations_diff.extra:
                console.print(f"  [green]+ {str(aff)}[/green]")

    if diff_data.author_affiliations_diff:
        console.print(f"\n[bold magenta]Author-Affiliation Differences[/bold magenta]")

        for diff in diff_data.author_affiliations_diff:
            if isinstance(diff, AuthorAffiliationsSetsDiff):
                console.print(f"\n[bold]Author: {diff.author}[/bold]")

                if diff.missing:
                    console.print("  [bold yellow]Missing affiliations:[/bold yellow]")
                    for aff in diff.missing:
                        console.print(f"    [red]- {aff}[/red]")

                if diff.extra:
                    console.print("  [bold yellow]Extra affiliations:[/bold yellow]")
                    for aff in diff.extra:
                        console.print(f"    [green]+ {aff}[/green]")

            elif isinstance(diff, MissingAuthorDiff):
                console.print(f"\n[bold yellow]{diff.message}:[/bold yellow]")
                table = Table(show_header=False, box=None)
                table.add_column("Type")
                table.add_column("Index")
                table.add_column("Name", style="red")
                table.add_row("Validated", str(diff.index), diff.validated)
                table.add_row("Predicted", "", diff.predicted)
                console.print(table)


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Evaluate LLM author/affiliation extraction."
    )
    parser.add_argument(
        "--paperoni",
        nargs="*",
        type=Path,
        default=[],
        help="Paperoni json reports of papers to evaluate",
    )
    parser.add_argument(
        "--format", choices=["csv", "json"], default="csv", help="Output format"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output file (default: stdout)"
    )
    parser.add_argument(
        "--show-differences",
        action="store_true",
        help="Show detailed differences when there are mismatches",
    )
    options = parser.parse_args(argv)

    # Load paperoni papers
    paperoni = sum([json.loads(_p.read_text()) for _p in options.paperoni], [])
    logger.info(f"Found {len(paperoni)} papers in paperoni reports")

    # Create Paper objects for all paperoni papers
    papers = [Paper(_p) for _p in paperoni]

    # Find all validated YAMLs
    results = []
    stats = Counter()

    for paper_obj in papers:
        val_file = CFG.dir.validated / f"{paper_obj.id}.yaml"

        if not val_file.exists():
            continue

        # Get LLM query files for this paper
        llm_file = paper_obj.queries[-1]

        try:
            # validated = clean_model_dict(load_analysis(val_file))
            # llm = clean_model_dict(load_analysis(llm_file))
            validated = load_analysis(val_file)
            llm = load_analysis(llm_file)
        except Exception as e:
            logger.error(f"Error processing paper {paper_obj.id}: {e}")
            results.append(
                ReportEntry(
                    paper=paper_obj.id,
                    title=paper_obj._paper["title"],
                    authors_order=f"ERROR: {e}",
                    affiliations=f"ERROR: {e}",
                    author_affiliations=f"ERROR: {e}",
                    llm_file=str(llm_file),
                    val_file=str(val_file),
                    pdf_file=str(paper_obj.pdf.with_suffix(".pdf")),
                    pdf_txt_file=str(paper_obj.pdf),
                ).to_dict()
            )
            stats["total"] += 1
            stats["error"] += 1
            continue

        authors_order, authors_order_diff, authors_order_warnings = (
            compare_authors_order(
                validated.authors_affiliations, llm.authors_affiliations
            )
        )
        affiliations, affiliations_diff, affiliations_warnings = (
            compare_affiliations_set(validated.affiliations, llm.affiliations)
        )
        author_affiliations, author_affiliations_diff, author_affiliations_warnings = (
            compare_author_affiliations(
                validated.authors_affiliations, llm.authors_affiliations
            )
        )

        # Collect all warnings
        typo_warnings = []
        if authors_order_warnings:
            typo_warnings.extend(authors_order_warnings)
        if affiliations_warnings:
            typo_warnings.extend(affiliations_warnings)
        if author_affiliations_warnings:
            typo_warnings.extend(author_affiliations_warnings)

        update_stats(
            stats,
            validated,
            authors_order,
            affiliations,
            author_affiliations,
            author_affiliations_diff,
            affiliations_diff,
        )

        report_entry = ReportEntry(
            paper=paper_obj.id,
            title=paper_obj._paper["title"],
            authors_order="PASS" if authors_order else "FAIL",
            affiliations="PASS" if affiliations else "FAIL",
            author_affiliations="PASS" if author_affiliations else "FAIL",
            llm_file=str(llm_file),
            val_file=str(val_file),
            pdf_file=str(paper_obj.pdf.with_suffix(".pdf")),
            pdf_txt_file=str(paper_md(paper_obj)),
        )

        if options.show_differences:
            if not authors_order:
                assert authors_order_diff
                report_entry.authors_order_diff = authors_order_diff
            if not affiliations:
                assert affiliations_diff
                report_entry.affiliations_diff = affiliations_diff
            if not author_affiliations:
                assert author_affiliations_diff
                report_entry.author_affiliations_diff = author_affiliations_diff
            if typo_warnings:
                report_entry.typo_warnings = typo_warnings

            format_differences(report_entry)

        results.append(report_entry.to_dict())

    # No results case
    if not results:
        logger.warning("No papers were found for evaluation.")
        return

    # Calculate summary statistics
    authors_total = stats["authors_total"]
    affiliations_total = stats["affiliations_total"]
    author_affiliations_total = stats["author_affiliations_total"]
    missing_authors = stats["missing_authors"]
    wrong_affiliations = stats["wrong_affiliations"]
    missing_affiliations = stats["missing_affiliations"]
    extra_affiliations = stats["extra_affiliations"]
    wrong_author_affiliations = stats["wrong_author_affiliations"]
    missing_author_affiliations = stats["missing_author_affiliations"]
    extra_author_affiliations = stats["extra_author_affiliations"]
    summary = {
        "total_papers": stats["total"],
        "authors_order_pass_rate": (
            f"{stats['authors_order_pass'] / (stats['authors_order_pass'] + stats['authors_order_fail']):.4f}"
            if (stats["authors_order_pass"] + stats["authors_order_fail"]) > 0
            else "N/A"
        ),
        "affiliations_pass_rate": (
            f"{stats['affiliations_pass'] / (stats['affiliations_pass'] + stats['affiliations_fail']):.4f}"
            if (stats["affiliations_pass"] + stats["affiliations_fail"]) > 0
            else "N/A"
        ),
        "author_affiliations_pass_rate": (
            f"{stats['author_affiliations_pass'] / (stats['author_affiliations_pass'] + stats['author_affiliations_fail']):.4f}"
            if (stats["author_affiliations_pass"] + stats["author_affiliations_fail"])
            > 0
            else "N/A"
        ),
        "avg_missing_authors": f"{missing_authors / authors_total:.4f}",
        "avg_wrong_affiliations": f"{wrong_affiliations / affiliations_total:.4f}",
        "avg_missing_affiliations": f"{missing_affiliations / affiliations_total:.4f}",
        "avg_extra_affiliations": f"{extra_affiliations / affiliations_total:.4f}",
        "avg_wrong_author_affiliations": f"{wrong_author_affiliations / author_affiliations_total:.4f}",
        "avg_missing_author_affiliations": f"{missing_author_affiliations / author_affiliations_total:.4f}",
        "avg_extra_author_affiliations": f"{extra_author_affiliations / author_affiliations_total:.4f}",
        "error_files": stats["error"],
        **{k: stats[k] for k in sorted(stats.keys())},
    }

    # Output
    if options.format == "csv":
        fieldnames = ReportEntry.__dataclass_fields__.keys()

        if options.output:
            with options.output.open("w") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

                # Write summary statistics
                print("\nSUMMARY", file=f)
                for key, value in summary.items():
                    print(f"{key},{value}", file=f)

        else:
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

            # Print summary statistics
            print("\nSUMMARY")
            for key, value in summary.items():
                print(f"{key},{value}")

    else:  # json
        output_data = {"results": results, "summary": summary}

        if options.output:
            with options.output.open("w") as f:
                json.dump(output_data, f, indent=2)
        else:
            json.dump(output_data, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
