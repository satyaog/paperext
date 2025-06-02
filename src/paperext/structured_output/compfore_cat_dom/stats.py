import json
import csv
from collections import Counter
from pathlib import Path
from statistics import mean

from paperext.config import CFG, Config
from paperext.sanitize_categorization import _flatten_dict
from paperext.utils import Paper


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


if __name__ == "__main__":
    responses = []
    response_dir = Path("data/compfore_cat_dom/queries/")
    for response_json in response_dir.glob("*.json"):
        response = json.loads(response_json.read_text())
        responses.append(response)

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
