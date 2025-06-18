import copy
import json
import csv
from collections import Counter
import math
from pathlib import Path
import pickle
import pprint
from typing import Generator, Iterable

import numpy as np
import tqdm

from paperext.config import CFG
from paperext.sanitize_categorization import _flatten_dict, _update_sanitized_map
from paperext.structured_output import get_struct_module
from paperext.structured_output.compfore_cat_dom.compfore_cat_dom_emb import (
    build_cluster_tree,
    cluster_domains,
    get_domains_embeddings,
    process_papers_context,
)
from paperext.utils import Paper


def iter_domains(
    papers: list[dict | Paper], last_query_only=False, include_aliases=False
) -> Generator[str, None, None]:
    for paper in papers:
        if not isinstance(paper, Paper):
            paper = Paper(paper)

        for query in paper.queries[-1:] if last_query_only else paper.queries:
            analysis = (
                get_struct_module(CFG.platform.struct)
                .model.Response.model_validate_json(query.read_text())
                .analysis
            )

            for research_field in (
                analysis.primary_research_field,
                *analysis.sub_research_fields,
            ):
                aliases = set()
                if include_aliases:
                    aliases.update(research_field.aliases)
                yield [research_field.name.value] + sorted(
                    aliases - {research_field.name.value}
                )


def reduce_domains(paper_domains: Iterable[list[str]]) -> list[str]:
    """
    Merge each sets that shares at least one domain then from each merged set,
    keep the domain with the highest occurrence rate.

    Args:
        paper_domains: The list of domains

    Returns:
        The reduced list of domains
    """
    paper_domains = list(paper_domains)

    domain_counter = Counter()
    merged_domains: dict[str, set[str]] = {}

    for domains in paper_domains:
        domain_counter.update(domains)
        for domain in domains:
            merged_domains.setdefault(domain, set())
            merged_domains[domain].update(domains)

    for domain, domains in merged_domains.items():
        for other_domain, other_domains in merged_domains.items():
            if domains & other_domains:
                domains.update(other_domains)
                merged_domains[other_domain] = domains

    return [
        sorted(domains, key=lambda x: domain_counter[x], reverse=True)[0]
        for _ in paper_domains
        for domains in merged_domains[_[0]]
    ]


def json_to_txt(d: dict, indent: int = 0) -> str:
    txt = []
    for key, value in sorted(d.items()):
        _open = f"- {key}"
        if isinstance(value, dict):
            _value = json_to_txt(value, indent + 1)
        else:
            _value = value

        if not _value:
            txt.append(f"{'  ' * indent}{_open}")
        else:
            txt.extend([f"{'  ' * indent}{entry}" for entry in (f"{_open}:", _value)])
    return "\n".join([entry for entry in txt if entry.strip()])


def analyze_category_locations(responses: list[dict]):
    """Analyze statistics about the location of selected and parent categories."""
    selected_locations = Counter()
    parent_locations = Counter()

    for response in responses:
        analysis = response["analysis"]

        selected_domain = analysis["selected_domain"]
        if isinstance(selected_domain, dict):
            selected_domain = selected_domain["value"]

        parent_category = analysis["parent_category"]
        if isinstance(parent_category, dict):
            parent_category = parent_category["value"]

        categorization = response["query_data"]["categorization"]
        categorization_lines = json_to_txt(categorization).splitlines()
        categorization_lines = [
            line.strip() for line in categorization_lines if line.strip()
        ]

        domains = response["query_data"]["domains"]

        if parent_category == "ignore":
            parent_locations[-1] += 1
        else:
            for i, line in enumerate(categorization_lines):
                line10 = int(i / 20)

                if line.strip().rstrip(":") == f"- {parent_category}":
                    parent_locations[line10] += 1
                    break
            else:
                assert parent_category not in set(_flatten_dict(categorization))

        for i, domain in enumerate(domains):
            line10 = int(i / 10)

            if domain == selected_domain:
                selected_locations[line10] += 1
                break
        else:
            assert selected_domain not in domains

    return selected_locations, parent_locations


# def analyse_domains_categorization(domains: list[str], categorization_map: dict):
#     # Counter for each domain including subcategories
#     domain_counter = Counter()
#     for domain in domains:
#         domain_counter[domain] += 1

#     def subcategories_counter(domain: str):
#         return domain_counter[domain] + sum(
#             subcategories_counter(subdomain) for subdomain in categorization_map[domain]
#         )

#     def add_counter_to_categorization_map(cat_map: dict):
#         result = {}
#         for domain, value in cat_map.items():
#             categorization = add_counter_to_categorization_map(value)
#             count = subcategories_counter(domain)
#             result[domain] = {
#                 "categorization": categorization,
#                 "count": count,
#                 "percentage": (
#                     count / sum(domain_counter.values())
#                     if sum(domain_counter.values())
#                     else 0
#                 ),
#             }

#         return result

#     return add_counter_to_categorization_map(categorization_map)


def analyse_domains_categorization(
    domains_embeddings: list[tuple[str, np.ndarray]],
    domain_to_embedding: dict[str, np.ndarray],
    categorization_map: dict[str, dict],
):
    """Analyze the categorization of domains. Add a count and percentage to each
    entry in categorization_map. Count is a sum of cosine distance
    between the domain embedding and the average embedding of the domain.

    For each domain, take the 2 domains from domain_to_embedding for which the
    cosine distance is the smallest. For these 2 domains, add distance to
    the count of the domain in categorization_map.

    Args:
        domains_embeddings: The list of domains with their corresponding embeddings
        domain_to_embedding: The dictionary of domains to their average embeddings
        categorization_map: The categorization map

    Returns:
        The categorization analysis.
    """

    def _cosine_distance(embedding: np.ndarray, other: np.ndarray) -> float:
        return 1 - np.dot(embedding, other) / (
            np.linalg.norm(embedding) * np.linalg.norm(other)
        )

    # 45 deg
    MIN_SIMILARITY = math.cos(math.pi / 4)
    # 67.5 deg
    LOWER_SCALING_BOUND = math.cos(math.pi * 3 / 8)

    domain_counter = Counter()

    for _, embedding in domains_embeddings:
        min_distance_domains = sorted(
            domain_to_embedding.items(),
            key=lambda x: _cosine_distance(embedding, x[1]),
        )

        # upper_scaling_bound = 1 - _cosine_distance(
        #     embedding, min_distance_domains[0][1]
        # )
        # lower_scaling_bound = MIN_SIMILARITY - (1 - upper_scaling_bound)

        for other_domain, other_embedding in min_distance_domains:
            similarity = 1 - _cosine_distance(embedding, other_embedding)
            if similarity >= MIN_SIMILARITY:
                # Add the scaled similarity
                domain_counter[other_domain] += (similarity - LOWER_SCALING_BOUND) / (
                    1 - LOWER_SCALING_BOUND
                )
            else:
                break

    def subcategories_counter(domain: str):
        return domain_counter[domain] + sum(
            subcategories_counter(subdomain) for subdomain in categorization_map[domain]
        )

    def add_counter_to_categorization_map(cat_map: dict):
        result = {}
        for domain, value in cat_map.items():
            categorization = add_counter_to_categorization_map(value)
            # count = subcategories_counter(domain)
            count = domain_counter[domain]
            result[domain] = {
                "categorization": categorization,
                "count": count,
                "percentage": (
                    count / sum(domain_counter.values())
                    if sum(domain_counter.values())
                    else 0
                ),
            }

        return result

    return add_counter_to_categorization_map(categorization_map)


def build_categorization_map(categorization: dict):
    """
    Build a map of domains to their subcategorization dictionary.

    Args:
        categorization: The categorization dictionary

    Returns:
        A dictionary mapping domains to their subcategorization dictionary
    """
    categorization_map = {}
    for parent, children in categorization.items():
        assert parent not in categorization_map

        categorization_map[parent] = children
        children_map = build_categorization_map(children)
        # make sure we're not adding existing entries to the categorization map
        assert not (
            set(children_map) & set(categorization_map)
        ), f"Existing entries in {children_map}: {set(children_map) & set(categorization_map)}"

        categorization_map = {**categorization_map, **children_map}

    return categorization_map


def main():
    responses = []
    response_dir = Path("data/compfore_cat_dom/queries/openai/xml_01")
    for response_json in response_dir.glob("*.json"):
        response = json.loads(response_json.read_text())
        responses.append(response)

    args_hash = "043bbea3ca955e478eaff0a60a7814ba7eed72ac7f7fad63ce357d5dd7aa41c4"
    papers_embeddings = Path(f"papers_embeddings_{args_hash}.pkl")
    with papers_embeddings.open("rb") as f:
        papers_embeddings = pickle.load(f)

    domain_to_embedding = get_domains_embeddings(
        papers_embeddings, context_type="justification"
    )

    clusterer = cluster_domains(domain_to_embedding, metric="euclidean")
    categorization = build_cluster_tree(clusterer, list(domain_to_embedding.keys()))

    # categorization = json.loads(Path("data/mdl/categorized_domains.json").read_text())
    categorization_map = build_categorization_map(categorization)

    # paperoni = Path("data/paperoni-2022-01-01-2025-01-01-PR_2025-02-05.json")
    paperoni = Path("data/paperoni-2023-2024-PR_2024-07-05.json")
    paperoni = json.loads(paperoni.read_text())

    # papers_context = process_papers_context(
    #     tqdm.tqdm(paperoni, desc=f"Processing papers", unit="paper(s)"), n_queries=1
    # )

    # domains = sum(
    #     iter_domains(
    #         tqdm.tqdm(
    #             paperoni,
    #             total=len(paperoni),
    #             desc="Domains",
    #         ),
    #         last_query_only=True,
    #     ),
    #     [],
    # )

    # sanitized_map = {}
    # _update_sanitized_map(sanitized_map, *_flatten_dict(categorization))
    # domains = _update_sanitized_map(sanitized_map, *domains)

    domains_embeddings = sum(
        [
            [(domain, embedding) for embedding in embeddings]
            for paper_embedding in papers_embeddings.values()
            for analysis in paper_embedding["analyses"][-1:]
            for domain, embeddings in analysis["domains"].items()
        ],
        [],
    )

    # Analyze category locations
    selected_locations, parent_locations = analyze_category_locations(responses)

    selected_stats = {
        k: v / sum(selected_locations.values()) * 100
        for k, v in selected_locations.items()
    }
    parent_stats = {
        k: v / sum(parent_locations.values()) * 100 for k, v in parent_locations.items()
    }

    print(f"Response directory: {response_dir.name}")
    print(f"Selected Category Locations ({sum(selected_locations.values())}):")
    print("-------------------------")
    for line10 in sorted(selected_stats.keys()):
        percentage = selected_stats[line10]
        print(f"{line10:03}: {percentage:.2f}%")

    print(f"\nParent Category Locations ({sum(parent_locations.values())}):")
    print("------------------------")
    for line10 in sorted(parent_stats.keys()):
        percentage = parent_stats[line10]
        print(f"{line10:03}: {percentage:.2f}%")

    categorization_analysis = analyse_domains_categorization(
        domains_embeddings, domain_to_embedding, categorization_map
    )

    def sort_categorization_analysis(
        cat: dict, sort_by: str = None, filter_percentage: float = -1.0
    ):
        """
        Sort the categorization analysis
        Args:
            cat: The categorization map
            sort_by: The field to sort by
            filter_percentage: The percentage to filter by

        Returns:
            A dictionary of the categorization analysis
        """
        if sort_by:
            cat = {
                k: v
                for k, v in sorted(
                    cat.items(),
                    key=lambda x: categorization_analysis[x[0]][sort_by],
                    reverse=True,
                )
            }

        return {
            k: sort_categorization_analysis(v, sort_by, filter_percentage)
            for k, v in cat.items()
            if categorization_analysis[k]["percentage"] > filter_percentage
            or len(categorization_analysis[k]["categorization"])
        }

    def format_categorization_analysis(cat: dict):
        """
        Format the categorization analysis
        Args:
            cat: The categorization map
            sort_by: The field to sort by
            filter_percentage: The percentage to filter by

        Returns:
            A dictionary of the categorization analysis
        """

        def format_domain(domain_value: tuple[str, dict]):
            domain, value = domain_value
            return (
                f"{domain} ({categorization_analysis[domain]['count']} / {categorization_analysis[domain]['percentage'] * 100:.2f}%)",
                format_categorization_analysis(value),
            )

        return {domain: value for domain, value in map(format_domain, cat.items())}

    # Print to file the categorization analysis
    with Path("cat.out").open("wt") as _f:
        pprint.pprint(
            format_categorization_analysis(categorization),
            _f,
        )

    # Print the categorization analysis sorted by count
    with Path("cat.out.sorted").open("wt") as _f:
        pprint.pprint(
            format_categorization_analysis(
                sort_categorization_analysis(categorization, "count")
            ),
            _f,
            sort_dicts=False,
        )

    # Write a CSV that contains the sorted list of domains categorization paths, count and percentage filtered by percentage > 1%
    def write_domains_csv(csv_writer: csv.writer, parents: list[str], cat: dict):
        """
        Write a CSV that contains the sorted list of domains categorization
        paths, count and percentage filtered by percentage > 1%

        Args:
            csv_writer: The CSV writer
            parents: The list of parents
            cat: The categorization map
        """
        for domain, value in cat.items():
            domain_path = parents + [domain]
            csv_writer.writerow(
                [
                    ".".join(domain_path),
                    categorization_analysis[domain]["count"],
                    categorization_analysis[domain]["percentage"],
                ]
            )
            write_domains_csv(csv_writer, domain_path, value)

    with Path("domains.csv").open("wt") as _f:
        writer = csv.writer(_f)
        writer.writerow(["domain", "count", "percentage"])
        write_domains_csv(
            writer, [], sort_categorization_analysis(categorization, "count", 0.01)
        )


if __name__ == "__main__":
    main()
