#!/usr/bin/env python3

import argparse
import copy
import csv
import json
import math
import pickle
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import Levenshtein
import numpy as np
import pandas as pd
import tqdm

from paperext.config import CFG, Config
from paperext.structured_output import get_struct_module
from paperext.structured_output.compfore_cat_dom.compfore_cat_dom_emb import (
    build_cluster_tree,
    cluster_domains,
    get_domains_embeddings,
    process_papers_context,
    compute_paper_embeddings,
)
from paperext.structured_output.compfore_cat_dom.stats import (
    analyse_domains_categorization,
    build_categorization_map,
)
from paperext.utils import Paper, str_normalize


def load_professors_csv(csv_path: Path) -> Dict[str, str]:
    """Load professors from CSV file.

    Args:
        csv_path: Path to CSV file with professors

    Returns:
        Dictionary mapping email to display name
    """
    professors = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            email = row["email"].strip().lower()
            name = row["Display Name"].strip()
            professors[email] = name
    return professors


def load_students_csv(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """Load students from CSV file.

    Args:
        csv_path: Path to CSV file with students

    Returns:
        Dictionary mapping student email to student info (name, prof, prof_email)
    """
    students = defaultdict(
        lambda: {"name": None, "profs": [], "profs_email": [], "years": []}
    )
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            email = row["student_email"].strip().lower()
            students[email]["name"] = row["student_name"].strip()
            students[email]["profs"].append(row["professor_name"].strip())
            students[email]["profs_email"].append(
                row["professor_email"].strip().lower()
            )
            students[email]["years"].append(int(row["year"].strip()))
    return students


def extract_year_from_paper(paper: dict) -> int:
    """Extract publication year from paper.

    Args:
        paper: Paper dictionary

    Returns:
        Publication year as integer, or 0 if not found
    """
    year = 0

    for release in paper.get("releases", []):
        venue = release.get("venue", {})
        date = venue.get("date", {})
        _year = None
        if "timestamp" in date:
            _year = datetime.fromtimestamp(date["timestamp"]).year
        elif "text" in date:
            # Try to extract year from text date
            text_date = date["text"]
            try:
                # Handle formats like "2025-02-01", "2025-05", "2025"
                _year = int(text_date.split("-")[0])
            except (ValueError, IndexError):
                continue

        if release.get("peer_reviewed", False):
            return _year
        else:
            year = year or _year

    return year


def identify_authors_roles(
    paper: dict, professors: Dict[str, str], students: Dict[str, Dict[str, str]]
) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
    """Identify professors and students in a paper's authors.

    Args:
        paper: Paper dictionary
        professors: Dictionary of professor emails to names
        students: Dictionary of student emails to student info

    Returns:
        Tuple of (professor_emails, student_emails)
    """
    professor_emails = []
    student_emails = []

    # Since students don't always have a mila email, we first find all
    # professors to then be able to match potential students names with authors
    # names
    for author_info in paper.get("authors", []):
        author = author_info.get("author", {})
        links = author.get("links", [])

        for link in links:
            if link.get("type") == "email.mila" or link.get(
                "link", ""
            ).strip().lower().endswith("@mila.quebec"):
                email = link.get("link", "").strip().lower()
                if email in professors:
                    professor_emails.append(email)
                elif email in students:
                    student_emails.append(email)
                break

    # List all students of the paper's MILA professors
    potential_students_emails = sorted(
        email
        for email, student_info in students.items()
        if set(student_info["profs_email"]) & set(professor_emails)
    )

    # Compute the distances between author names and potential students names
    authors_distances: dict[str, list[tuple[int, str]]] = defaultdict(list)

    for author_info in paper.get("authors", []):
        author = author_info.get("author", {})
        links = author.get("links", [])

        for link in links:
            if link.get("type") == "email.mila" or link.get(
                "link", ""
            ).strip().lower().endswith("@mila.quebec"):
                break

        else:
            for email in potential_students_emails:
                if author["name"].lower() == students[email]["name"].lower():
                    student_emails.append(email)
                    break
            else:
                # If the distance between the normalized author name and a
                # student is less or equal to 2, we consider it a match
                for email in potential_students_emails:
                    distance = Levenshtein.distance(
                        str_normalize(author["name"]),
                        str_normalize(students[email]["name"]),
                    )
                    if distance <= 2:
                        authors_distances[author["name"]].append((distance, email))

    sorted_distances = {
        author_name: sorted(distances)
        for author_name, distances in authors_distances.items()
    }
    for distances in sorted_distances.values():
        student_emails.append(distances[0][1])

    return professor_emails, student_emails


def get_paper_domains_embeddings(
    paper: dict, papers_embeddings: dict = None
) -> List[Tuple[str, np.ndarray]]:
    """Extract domains and their embeddings from a paper's analysis.

    Args:
        paper: Paper dictionary
        papers_embeddings: Optional pre-computed embeddings dictionary

    Returns:
        List of (domain_name, embedding) tuples
    """
    # paper_obj = Paper(paper)
    domains_embeddings = []
    paper_id = paper["paper_id"]

    # if not paper_obj.queries:
    if paper_id not in papers_embeddings:
        return domains_embeddings

    try:
        # Get the latest query/analysis
        # query_file = paper_obj.queries[-1]
        # analysis = (
        #     get_struct_module(CFG.platform.struct)
        #     .model.Response.model_validate_json(query_file.read_text())
        #     .analysis
        # )

        # If we have precomputed embeddings, use them
        # if papers_embeddings and paper_id in papers_embeddings:
        paper_embedding_data = papers_embeddings[paper_id]
        # for analysis_data in paper_embedding_data.get("analyses", []):
        analysis_data = next(iter(paper_embedding_data["analyses"][-1:]), {})
        for domain, embeddings in analysis_data.get("domains", {}).items():
            # embeddings is a list of numpy arrays, we need individual embeddings
            for embedding in embeddings:
                domains_embeddings.append((domain, embedding))
        # else:
        #     # If no precomputed embeddings, just return domain names with None embeddings
        #     for research_field in [
        #         analysis.primary_research_field,
        #         *analysis.sub_research_fields,
        #     ]:
        #         domains_embeddings.append((research_field.name.value, None))

    except Exception as e:
        print(f"Error processing paper {paper_id}: {e}")

    return domains_embeddings


def compute_papers_embeddings(papers: List[dict], model_name: str = None) -> dict:
    """Compute embeddings for all papers using the existing pipeline.

    Args:
        papers: List of paper dictionaries
        model_name: Ollama model name for embeddings

    Returns:
        Dictionary of paper embeddings
    """
    print("Loading papers context...")
    papers_context = process_papers_context(
        tqdm.tqdm(papers, desc="Processing papers context"), n_queries=0
    )

    print("Computing paper embeddings...")
    papers_embeddings = {}
    for paper_id, paper_context in tqdm.tqdm(
        papers_context.items(), desc="Generating paper embeddings"
    ):
        try:
            paper_embeddings = compute_paper_embeddings(paper_context, model_name)
            papers_embeddings[paper_id] = paper_embeddings
        except Exception as e:
            print(f"Error computing embeddings for paper {paper_id}: {e}")
            continue

    return papers_embeddings


def analyze_papers_professors_students(
    papers: List[dict],
    professors: Dict[str, str],
    students: Dict[str, Dict[str, str]],
    domain_to_embedding: Dict[str, np.ndarray] = None,
    categorization_map: Dict[str, dict] = None,
    papers_embeddings: dict = None,
) -> Dict:
    """Analyze papers by professors and students.

    Args:
        papers: List of paper dictionaries
        professors: Dictionary of professor emails to names
        students: Dictionary of student emails to student info
        domain_to_embedding: Optional domain embeddings for categorization analysis
        categorization_map: Optional categorization mapping
        papers_embeddings: Optional precomputed paper embeddings

    Returns:
        Dictionary containing analysis results
    """
    results = {
        "papers_per_year_per_prof": defaultdict(lambda: defaultdict(int)),
        "students_per_prof_per_paper": defaultdict(list),
        "paper_details": [],
        "domain_categorization_per_paper": {},
        "professor_stats": defaultdict(
            lambda: {
                "name": "",
                "total_papers": 0,
                "years_active": set(),
                "total_students": 0,
                "unique_students": set(),
                "papers_by_year": defaultdict(list),
            }
        ),
        "student_stats": defaultdict(
            lambda: {
                "total_papers": 0,
                "professors_worked_with": set(),
                "years_active": set(),
                "assigned_profs": set(),
                "assigned_profs_email": set(),
            }
        ),
    }

    print(f"Analyzing {len(papers)} papers...")

    for paper in tqdm.tqdm(papers, desc="Processing papers"):
        paper_id = paper["paper_id"]
        title = paper["title"]
        year = extract_year_from_paper(paper)

        prof_emails, student_emails = identify_authors_roles(
            paper, professors, students
        )

        # Skip papers without professors
        if not prof_emails:
            continue

        paper_details = {
            "paper_id": paper_id,
            "title": title,
            "year": year,
            "professors": [professors[email] for email in prof_emails],
            "professor_emails": prof_emails,
            "students": [students[email]["name"] for email in student_emails],
            "student_emails": student_emails,
        }

        results["paper_details"].append(paper_details)

        # Update professor stats
        for prof_email in prof_emails:
            prof_name = professors[prof_email]
            if year > 0:
                # Not sure if we need
                # results["papers_per_year_per_prof"][prof_email][year] if we
                # have
                # results["professor_stats"][prof_email]["papers_by_year"][year]
                results["papers_per_year_per_prof"][prof_email][year] += 1
                results["professor_stats"][prof_email]["name"] = prof_name
                results["professor_stats"][prof_email]["papers_by_year"][year].append(
                    paper_id
                )
                results["professor_stats"][prof_email]["years_active"].add(year)

            results["professor_stats"][prof_email]["total_papers"] += 1

            # Only count students who are assigned to this specific professor
            students_for_this_prof = [
                student_email
                for student_email in student_emails
                if prof_email in students[student_email]["profs_email"]
            ]

            results["professor_stats"][prof_email]["total_students"] += len(
                students_for_this_prof
            )
            results["professor_stats"][prof_email]["unique_students"].update(
                students_for_this_prof
            )

            # Track students per professor per paper (only students assigned to this prof)
            results["students_per_prof_per_paper"][prof_email].append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "year": year,
                    "students": [
                        students[email]["name"] for email in students_for_this_prof
                    ],
                    "student_emails": students_for_this_prof,
                    "num_students": len(students_for_this_prof),
                }
            )

        # Update student stats
        for student_email in student_emails:
            results["student_stats"][student_email]["total_papers"] += 1
            results["student_stats"][student_email]["professors_worked_with"].update(
                prof_emails
            )
            results["student_stats"][student_email]["assigned_profs"].update(
                students[student_email]["profs"]
            )
            results["student_stats"][student_email]["assigned_profs_email"].update(
                students[student_email]["profs_email"]
            )
            if year > 0:
                results["student_stats"][student_email]["years_active"].add(year)

        # Domain categorization analysis (if embeddings and categorization are available)
        if domain_to_embedding and categorization_map:
            domains_embeddings = get_paper_domains_embeddings(paper, papers_embeddings)
            try:
                categorization_analysis = analyse_domains_categorization(
                    domains_embeddings, domain_to_embedding, categorization_map
                )

                sorted_domains = sorted(
                    categorization_analysis.items(),
                    key=lambda x: x[1]["count"],
                    reverse=True,
                )

                # scale in exp space the counts and normalize
                paper_domains = {
                    domain: {
                        **domain_info,
                        "scale_normalized": math.exp(domain_info["count"]),
                    }
                    for domain, domain_info in sorted_domains
                    if domain_info["count"] > 0
                    # and domain
                    # in [
                    #     "natural language processing",
                    #     "causal representation learning - representation learning",
                    #     "deep reinforcement learning - reinforcement learning",
                    #     "computer vision - image classification",
                    #     "model optimization - optimization in deep learning",
                    #     "model-based reinforcement learning",
                    #     "transfer learning",
                    #     "interpretable machine learning",
                    #     "generative modeling - generative models",
                    #     "language modeling - language models",
                    #     "graph neural networks - graph neural networks explainability",
                    #     "multi-agent reinforcement learning - multi-agent systems",
                    #     "geometric deep learning",
                    # ]
                }
                paper_domains = {
                    domain: {
                        **domain_info,
                        "scale_normalized": domain_info["scale_normalized"]
                        / sum(
                            domain_info["scale_normalized"]
                            for _, domain_info in paper_domains.items()
                        ),
                    }
                    for domain, domain_info in paper_domains.items()
                }

                results["domain_categorization_per_paper"][paper_id] = paper_domains
            except Exception as e:
                print(f"Error analyzing domains for paper {paper_id}: {e}")

    # Convert sets to lists for JSON serialization
    for prof_email, stats in results["professor_stats"].items():
        stats["years_active"] = sorted(list(stats["years_active"]))
        stats["unique_students"] = list(stats["unique_students"])

    for student_email, stats in results["student_stats"].items():
        stats["professors_worked_with"] = list(stats["professors_worked_with"])
        stats["years_active"] = sorted(list(stats["years_active"]))

    # sorted entries by professor name
    results["papers_per_year_per_prof"] = {
        k: v
        for k, v in sorted(
            results["papers_per_year_per_prof"].items(), key=lambda x: professors[x[0]]
        )
    }
    # sorted entries by student name
    results["students_per_prof_per_paper"] = {
        k: v
        for k, v in sorted(
            results["students_per_prof_per_paper"].items(),
            key=lambda x: professors[x[0]],
        )
    }
    # sorted entries by professor name
    results["professor_stats"] = {
        k: v
        for k, v in sorted(
            results["professor_stats"].items(), key=lambda x: x[1]["name"]
        )
    }

    return results


def print_summary_stats(results: Dict, professors: Dict[str, str]):
    """Print summary statistics from the analysis."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Professor stats
    prof_stats = results["professor_stats"]
    print(f"\nTotal Professors: {len(prof_stats)}")

    print(f"\nTop 10 Professors by Total Papers:")
    top_profs = sorted(
        prof_stats.items(), key=lambda x: x[1]["total_papers"], reverse=True
    )[:10]
    for prof_email, stats in top_profs:
        prof_name = professors[prof_email]
        years_range = (
            f"{min(stats['years_active'])}-{max(stats['years_active'])}"
            if stats["years_active"]
            else "N/A"
        )
        print(
            f"  {prof_name}: {stats['total_papers']} papers, {len(stats['unique_students'])} unique students, active {years_range}"
        )

    # Analyze top research domains per top professor per year
    print(f"\nTop 10 Research Domains per Top Professors per Year:")
    print("-" * 50)

    # Collect domain data by professor and year
    prof_year_domains = defaultdict(lambda: defaultdict(Counter))

    for paper in results["paper_details"]:
        paper_id = paper["paper_id"]
        year = paper["year"]

        if year > 0 and paper_id in results["domain_categorization_per_paper"]:
            categorization = results["domain_categorization_per_paper"][paper_id]

            # Add to each professor's domain count for this year
            for prof_email in paper["professor_emails"]:
                for domain, domain_info in categorization.items():
                    prof_year_domains[prof_email][year][domain] += domain_info[
                        "scale_normalized"
                    ]

            prof_year_domains[prof_email][year]["total_count"] += 1

        else:
            print(f"No domain categorization for paper {paper_id}")

    # Display top 10 domains per top professor per year
    for prof_email in sorted(
        prof_year_domains.keys(),
        key=lambda x: prof_stats[x]["total_papers"],
        reverse=True,
    )[:10]:
        prof_name = professors[prof_email]
        print(f"\n{prof_name} ({prof_stats[prof_email]['total_papers']}):")
        year_data = prof_year_domains[prof_email]

        left_pad = ""

        for year in sorted(year_data.keys()):
            domain_counts = copy.deepcopy(year_data[year])
            top_domains = domain_counts.most_common(10)

            left_pad = f"  {year} ({year_data[year]['total_count']:02.0f}): "

            for domain, count in top_domains:
                if domain == "total_count":
                    continue
                print(f"{left_pad}{domain} ({count:.2f})")
                left_pad = " " * len(left_pad)

            if not top_domains:
                print(f"{left_pad}No domains categorized")

    # Student stats
    student_stats = results["student_stats"]
    print(f"\nTotal Students: {len(student_stats)}")

    print(f"\nTop 10 Most Productive Students:")
    top_students = sorted(
        student_stats.items(), key=lambda x: x[1]["total_papers"], reverse=True
    )[:10]
    for student_email, stats in top_students:
        years_range = (
            f"{min(stats['years_active'])}-{max(stats['years_active'])}"
            if stats["years_active"]
            else "N/A"
        )
        assigned_profs = stats.get("assigned_profs", [])
        print(
            f"  {student_email}: {stats['total_papers']} papers, assigned to {assigned_profs}, worked with {len(stats['professors_worked_with'])} professors, active {years_range}"
        )

    # Professor-Student collaboration analysis
    print(f"\nProfessor-Student Assignment Analysis:")
    prof_student_collab = defaultdict(set)
    for student_email, stats in student_stats.items():
        for prof_email in stats.get("assigned_profs_email", []):
            prof_student_collab[prof_email].add(student_email)

    print(f"Professors with most assigned students who published papers:")
    top_prof_students = sorted(
        prof_student_collab.items(), key=lambda x: len(x[1]), reverse=True
    )[:5]
    for prof_email, student_set in top_prof_students:
        prof_name = professors[prof_email]
        print(f"  {prof_name}: {len(student_set)} assigned students with publications")

    # Cross-professor collaboration (students working with professors other than their assigned one)
    cross_collab_students = []
    for student_email, stats in student_stats.items():
        assigned_prof_email = stats.get("assigned_prof_email")
        profs_worked_with = stats["professors_worked_with"]

        # Check if student worked with professors other than their assigned one
        other_profs = [
            email for email in profs_worked_with if email != assigned_prof_email
        ]
        if other_profs:
            cross_collab_students.append((student_email, stats, other_profs))

    print(
        f"\nStudents collaborating outside their assignment: {len(cross_collab_students)}"
    )
    if cross_collab_students:
        print("Top cross-collaborative students:")
        cross_collab_students.sort(key=lambda x: len(x[2]), reverse=True)
        for student_email, stats, other_profs in cross_collab_students[:5]:
            assigned_profs = stats.get("assigned_profs", [])
            other_prof_names = [professors.get(email, email) for email in other_profs]
            print(
                f"  {student_email}: assigned to {assigned_profs}, also worked with {', '.join(other_prof_names)}"
            )

    # Year analysis
    all_years = set()
    for stats in prof_stats.values():
        all_years.update(stats["years_active"])

    if all_years:
        print(f"\nYear Range: {min(all_years)} - {max(all_years)}")

        # Papers per year
        papers_per_year = Counter()
        for prof_stats in prof_stats.values():
            for year, paper_ids in prof_stats["papers_by_year"].items():
                papers_per_year[year] += len(paper_ids)

        print(f"\nPapers per Year:")
        for year in sorted(papers_per_year.keys()):
            print(f"  {year}: {papers_per_year[year]} papers")

    print(f"\nTotal Papers Analyzed: {len(results['paper_details'])}")
    if results["domain_categorization_per_paper"]:
        print(
            f"Papers with Domain Categorization: {len(results['domain_categorization_per_paper'])}"
        )

    else:
        print("No domain categorization data available")


def save_results_to_files(results: Dict, output_dir: Path):
    """Save analysis results to various output files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # # Save complete results as JSON
    # json_output = output_dir / "complete_analysis.json"
    # with open(json_output, "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    # print(f"Complete analysis saved to {json_output}")

    # Save professor stats as CSV
    prof_csv = output_dir / "professor_stats.csv"
    with open(prof_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Professor",
                "Professor Email",
                "Total Papers",
                "Unique Students",
                "Year",
            ]
        )

        for prof_email, stats in results["professor_stats"].items():
            prof_name = stats["name"]

            for year, paper_ids in sorted(stats["papers_by_year"].items()):
                writer.writerow(
                    [
                        prof_name,
                        prof_email,
                        len(paper_ids),
                        len(stats["unique_students"]),
                        year,
                    ]
                )
    print(f"Professor statistics saved to {prof_csv}")

    # Save student stats as CSV
    student_csv = output_dir / "student_stats.csv"
    with open(student_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Student Email",
                "Total Papers",
                "Assigned Professor",
                "Assigned Prof Email",
                "Professors Worked With",
                "Years Active",
                "First Year",
                "Last Year",
            ]
        )

        for student_email, stats in results["student_stats"].items():
            years = stats["years_active"]
            first_year = min(years) if years else "N/A"
            last_year = max(years) if years else "N/A"
            years_str = ", ".join(map(str, years))
            profs_worked_with = ", ".join(stats["professors_worked_with"])

            writer.writerow(
                [
                    student_email,
                    stats["total_papers"],
                    ", ".join(stats["assigned_profs"]),
                    ", ".join(stats["assigned_profs_email"]),
                    profs_worked_with,
                    years_str,
                    first_year,
                    last_year,
                ]
            )
    print(f"Student statistics saved to {student_csv}")

    # Save paper details as CSV
    papers_csv = output_dir / "paper_details.csv"
    with open(papers_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Paper ID",
                "Title",
                "Year",
                "Num Professors",
                "Num Students",
                # "Professor Name",
                # "Professor Email",
                # "Student Name",
                # "Student Email",
                "domain",
                "normalized",
                "per_prof_ratio",
                "per_student_ratio",
            ]
        )

        for paper in sorted(
            results["paper_details"], key=lambda x: (x["year"], x["paper_id"])
        ):
            # # Each line contains a single professor email and a single student
            # # email. The student must be assigned to the professor.
            # for prof_name, prof_email in zip(
            #     paper["professors"], paper["professor_emails"]
            # ):
            #     for student_name, student_email in zip(
            #         paper["students"] or [""], paper["student_emails"] or [""]
            #     ):
            #         if student_email in (
            #             results["student_stats"][student_email]["assigned_prof_email"]
            #             or [""]
            #         ):
            #             writer.writerow(
            #                 [
            #                     paper["paper_id"],
            #                     paper["title"],
            #                     paper["year"],
            #                     len(paper["professor_emails"]),
            #                     len(paper["student_emails"]),
            #                     prof_name,
            #                     prof_email,
            #                     student_name,
            #                     student_email,
            #                 ]
            #             )

            categorization = results["domain_categorization_per_paper"][
                paper["paper_id"]
            ]

            for domain, domain_info in categorization.items():
                writer.writerow(
                    [
                        paper["paper_id"],
                        paper["title"],
                        paper["year"],
                        len(paper["professor_emails"]),
                        len(paper["student_emails"]),
                        domain,
                        domain_info["scale_normalized"],
                        domain_info["scale_normalized"]
                        / len(paper["professor_emails"]),
                        (
                            domain_info["scale_normalized"]
                            / len(paper["student_emails"])
                            if len(paper["student_emails"]) > 0
                            else 0
                        ),
                    ]
                )

    print(f"Paper details saved to {papers_csv}")

    papers_details = pd.read_csv(str(papers_csv))

    # Group by domain and year for professor ratios
    domains_prof_ratio_by_year = pd.concat(
        [
            papers_details.groupby(["domain"])["per_prof_ratio"].sum(),
            papers_details.groupby(["domain", "Year"])["per_prof_ratio"]
            .sum()
            .reset_index()
            .pivot(index="domain", columns="Year", values="per_prof_ratio")
            .fillna(0),
        ],
        axis=1,
    )
    # Rename columns to add prefix
    domains_prof_ratio_by_year.columns = [
        f"per prof ratio - {col}" if isinstance(col, int) or str.isdigit(col) else col
        for col in domains_prof_ratio_by_year.columns
    ]

    # Group by domain and year for student ratios
    domains_student_ratio_by_year = pd.concat(
        [
            papers_details.groupby(["domain"])["per_student_ratio"].sum(),
            papers_details.groupby(["domain", "Year"])["per_student_ratio"]
            .sum()
            .reset_index()
            .pivot(index="domain", columns="Year", values="per_student_ratio")
            .fillna(0),
        ],
        axis=1,
    )
    # Rename columns to add prefix
    domains_student_ratio_by_year.columns = [
        (
            f"per student ratio - {col}"
            if isinstance(col, int) or str.isdigit(col)
            else col
        )
        for col in domains_student_ratio_by_year.columns
    ]

    pd.concat(
        [domains_prof_ratio_by_year, domains_student_ratio_by_year], axis=1
    ).to_csv(output_dir / "domains_ratio_by_year.csv")

    # Save papers per year per professor as CSV
    papers_per_year_csv = output_dir / "papers_per_year_per_professor.csv"
    with open(papers_per_year_csv, "w", newline="", encoding="utf-8") as f:
        # Collect all years
        all_years = set()
        for prof_data in results["papers_per_year_per_prof"].values():
            all_years.update(prof_data.keys())
        all_years = sorted(all_years)

        writer = csv.writer(f)
        writer.writerow(
            ["Professor", "Professor Email"]
            + [str(year) for year in all_years]
            + ["Total"]
        )

        for prof_email, year_data in results["papers_per_year_per_prof"].items():
            row = [prof_name, prof_email]
            total = 0
            for year in all_years:
                normalized_similarity = year_data.get(year, 0)
                row.append(normalized_similarity)
                total += normalized_similarity
            row.append(total)
            writer.writerow(row)
    print(f"Papers per year per professor saved to {papers_per_year_csv}")


def main(argv: List[str] = None):
    parser = argparse.ArgumentParser(
        description="Analyze papers by professors and students with domain categorization"
    )
    parser.add_argument(
        "--professors-csv",
        type=Path,
        required=True,
        help="CSV file with professors (email, Display Name columns)",
    )
    parser.add_argument(
        "--students-csv",
        type=Path,
        required=True,
        help="CSV file with students (email, Display Name, prof, prof email columns)",
    )
    parser.add_argument(
        "--paperoni",
        nargs="*",
        type=Path,
        help="Paperoni json report of papers to analyse",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_output"),
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--embeddings-cache",
        type=Path,
        help="Path to cached embeddings pickle file (optional)",
    )
    # parser.add_argument(
    #     "--categorization-json",
    #     type=Path,
    #     help="Path to domain categorization JSON file (optional)",
    # )
    parser.add_argument(
        "--compute-embeddings",
        action="store_true",
        help="Compute domain embeddings for categorization analysis",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model name for embeddings (if computing embeddings)",
    )
    parser.add_argument(
        "--save-embeddings",
        type=Path,
        help="Path to save computed embeddings (optional)",
    )

    options = parser.parse_args(argv)

    # Load professors
    print("Loading professors...")
    professors = load_professors_csv(options.professors_csv)
    print(f"Loaded {len(professors)} professors")

    # Load students
    print("Loading students...")
    students = load_students_csv(options.students_csv)
    print(f"Loaded {len(students)} students")

    # Load papers
    print("Loading papers...")
    # Load papers
    papers = []
    for papers_json_path in options.paperoni:
        papers.extend(json.loads(Path(papers_json_path).read_text()))
    print(f"Loaded {len(papers)} papers")

    # Configure paperext for analysis
    with Config.push():
        CFG.platform.select = "openai"
        CFG.platform.struct = "mdl"
        CFG.dir.queries = CFG.dir.data / CFG.platform.struct / "queries"

        # Setup domain analysis if requested
        domain_to_embedding = None
        categorization_map = None
        papers_embeddings = None

        if (
            options.compute_embeddings
            or options.embeddings_cache
            # or options.categorization_json
        ):
            print("Setting up domain analysis...")

            # Load or compute embeddings
            if options.embeddings_cache and options.embeddings_cache.exists():
                print(f"Loading cached embeddings from {options.embeddings_cache}")
                with open(options.embeddings_cache, "rb") as f:
                    papers_embeddings = pickle.load(f)
            elif options.compute_embeddings:
                print("Computing domain embeddings (this may take a while)...")
                papers_embeddings = compute_papers_embeddings(papers, options.model)

                # Save embeddings if requested
                if options.save_embeddings:
                    print(f"Saving embeddings to {options.save_embeddings}")
                    with open(options.save_embeddings, "wb") as f:
                        pickle.dump(papers_embeddings, f)

            # Compute domain embeddings for categorization
            if papers_embeddings:
                print("Computing domain embeddings for categorization...")
                domain_to_embedding = get_domains_embeddings(
                    papers_embeddings, context_type="justification"
                )
                print(f"Computed embeddings for {len(domain_to_embedding)} domains")

            # # Load categorization if available
            # if options.categorization_json and options.categorization_json.exists():
            #     print(f"Loading categorization from {options.categorization_json}")
            #     with open(options.categorization_json, "r", encoding="utf-8") as f:
            #         categorization = json.load(f)
            #     categorization_map = build_categorization_map(categorization)
            #     print(
            #         f"Built categorization map with {len(categorization_map)} entries"
            #     )
            # elif domain_to_embedding:
            print("Computing categorization using clustering...")
            # Compute categorization using clustering
            clusterer = cluster_domains(domain_to_embedding, metric="euclidean")
            categorization = build_cluster_tree(
                clusterer, list(domain_to_embedding.keys())
            )
            categorization_map = build_categorization_map(categorization)
            print(f"Built categorization map with {len(categorization_map)} entries")

            # # Save computed categorization
            # categorization_file = options.output_dir / "computed_categorization.json"
            # categorization_file.parent.mkdir(parents=True, exist_ok=True)
            # with open(categorization_file, "w", encoding="utf-8") as f:
            #     json.dump(categorization, f, indent=2, ensure_ascii=False)
            # print(f"Saved computed categorization to {categorization_file}")

        # Perform analysis
        results = analyze_papers_professors_students(
            papers,
            professors,
            students,
            domain_to_embedding,
            categorization_map,
            papers_embeddings,
        )

    # Print summary
    print_summary_stats(results, professors)

    # Save results
    save_results_to_files(results, options.output_dir)


if __name__ == "__main__":
    main()
