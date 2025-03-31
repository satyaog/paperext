import pandas as pd

from paperext.config import CFG
from paperext.structured_output.utils import model_validate_yaml
from paperext.structured_output.ai4hcat.model import (
    CATEGORISATION_TREE,
    Category,
    PaperExtractions,
    SubCategory,
    get_sub_categories,
)


pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", 1024)  # No line width limit
pd.set_option("display.max_colwidth", None)  # Show full content of each cell
pd.set_option("display.float_format", "{:.1f}".format)


def main():
    validation_set = CFG.dir.data / "ai4hcat/validation_set_pr.txt"

    objectives = {}
    for cat in CATEGORISATION_TREE:
        subcats = get_sub_categories(cat)

        objectives[(cat, cat)] = {"objective": max(10, 3 * len(subcats)), "cnt": 0}

        for subcat in subcats:
            objectives[(cat, subcat)] = {
                "objective": objectives[(cat, cat)]["objective"] / len(subcats),
                "cnt": 0,
            }

    objectives.pop((Category("N/A").value, SubCategory("N/A").value))
    objectives[(Category("N/A").value, SubCategory("N/A").value)] = {
        "objective": 100,
        "cnt": 0,
    }

    validation_files_objective = sum(
        objectives[cat, cat]["objective"] for cat in CATEGORISATION_TREE
    )
    validation_files = set()
    for validation_file in validation_set.read_text().splitlines():
        if not validation_file.strip():
            continue

        if validation_file in validation_files:
            continue

        validation_files.add(validation_file)

        validation_file = CFG.dir.merged / f"{validation_file}.yaml"

        validation: PaperExtractions = model_validate_yaml(
            PaperExtractions, validation_file.read_text()
        )

        categories = set()
        categories_pairs = set()
        for cat in [validation.primary_category, *validation.secondary_categories]:
            cat = cat.value
            categories.add(cat.value)

            for subcat in [
                validation.primary_sub_category,
                *validation.secondary_sub_categories,
            ]:
                subcat = subcat.value

                if subcat == SubCategory("N/A"):
                    continue

                if (cat.value, subcat.value) in objectives:
                    categories_pairs.add((cat.value, subcat.value))

        if len(categories) >= 2 and Category("N/A").value in categories:
            categories.remove(Category("N/A").value)

        for cat in categories:
            objectives[(cat, cat)]["cnt"] += 1

        for pair in categories_pairs:
            objectives[pair]["cnt"] += 1

    report = pd.DataFrame(objectives).transpose()
    report = pd.concat(
        [
            report,
            pd.DataFrame(
                {
                    ("validation_files_cnt", ""): {
                        "objective": validation_files_objective,
                        "cnt": len(validation_files),
                    }
                }
            ).transpose(),
        ]
    )

    report = report.assign(missing=report.objective - report.cnt)
    report.loc[report["missing"] < 0, "missing"] = 0.0

    print(report)


if __name__ == "__main__":
    main()
