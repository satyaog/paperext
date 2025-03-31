import argparse
from pathlib import Path

try:
    import numpy as np
    from mlcm import mlcm

    np.int
except AttributeError:
    # Fix AttributeError in mlcm.mlcm
    # *** AttributeError: module 'numpy' has no attribute 'int'.
    # `np.int` was a deprecated alias for the builtin `int`. To avoid this error
    # in existing code, use `int` by itself. Doing this will not modify any
    # behavior and is safe. When replacing `np.int`, you may wish to use e.g.
    # `np.int64` or `np.int32` to specify the precision. If you wish to review
    # your current use, check the release note link for additional information.
    # The aliases was originally deprecated in NumPy 1.20; for more details and
    # guidance see the original release note at:
    #     https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations. Did you mean: 'inf'?
    np.int = int
import pandas as pd
from sklearn.metrics import confusion_matrix

from paperext import CFG
from paperext.log import logger
from paperext.structured_output.ai4hcat.model import (
    Category,
    Response,
    PaperExtractions,
    SubCategory,
    get_categories,
    get_sub_categories,
)
from paperext.structured_output.utils import model_validate_yaml
from paperext.utils import build_validation_set

PROG = f"{Path(__file__).stem.replace('_', '-')}"

DESCRIPTION = """
Utility to analyses Chat-GPT responses on papers

Confidence and multi-label confidence matrices will be dumped into data/analysis
"""

EPILOG = f"""
Example:
  $ {PROG} --input data/validation_set.txt
"""


# Change display settings to show the entire table
pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", 1024)  # No line width limit
pd.set_option("display.max_colwidth", None)  # Show full content of each cell


def _csv_fn(stem: str, index: int) -> str:
    collapse_tag = "_coll" if CFG.evaluation.collapse_cat else ""
    match stem:
        case "research_fields_categories":
            stem = f"{stem}_{CFG.evaluation.dom_cat}{collapse_tag}"
        case "models.category":
            stem = f"{stem}_{CFG.evaluation.mod_cat}{collapse_tag}"
        case _:
            pass

    return stem + f"_{index:02}.csv"


def _cm(
    annotations: pd.DataFrame, predictions: pd.DataFrame, classes: pd.DataFrame = None
):
    if classes is None:
        classes = pd.concat([annotations, predictions])

    classes = classes.sort_values(ignore_index=True)
    classes.drop_duplicates(inplace=True, ignore_index=True)

    return confusion_matrix(annotations, predictions, labels=classes), classes


def _mlcm(
    annotations: pd.DataFrame, predictions: pd.DataFrame, classes: pd.DataFrame = None
):
    if classes is None:
        classes = pd.concat(list(annotations) + list(predictions))

    classes = classes.sort_values(ignore_index=True)
    classes.drop_duplicates(inplace=True, ignore_index=True)

    _ann, _pred = (
        [classes.isin(arr).astype(int) for arr in annotations],
        [classes.isin(arr).astype(int) for arr in predictions],
    )

    return mlcm.cm(_ann, _pred), classes


def _evaluate_precision(papers: list):
    """Analyse the performance of the LLM on the given papers though confusion
    matrices and multi-label confusion matrices"""
    annotated = {"category": [], "subcategory": []}
    predictions = {"category": [], "subcategory": []}

    for f in papers:
        stage = {
            "ann": [[], []],
            "pred": [[], []],
        }
        logger.info(f"Fetching data from {f}")
        model: PaperExtractions = model_validate_yaml(PaperExtractions, f.read_text())

        stage["ann"][0:2] = [model.primary_category.value.value], [
            model.primary_sub_category.value.value
        ]

        queries_dir = CFG.dir.queries / CFG.platform.select
        for i, query_f in enumerate(sorted(queries_dir.glob(f"{f.stem}*.json"))):
            logger.info(f"Fetching data from {query_f}")
            model = Response.model_validate_json(query_f.read_text()).extractions

            cat_choices = [
                model.primary_category.value.value,
                *map(lambda x: x.value.value, model.secondary_categories),
            ]
            cat_choices[1:] = [
                cat for cat in cat_choices if cat != Category("N/A").value
            ][1:]
            stage["pred"][0].append(cat_choices)

            subcat_choices = [
                model.primary_sub_category.value.value,
                *map(lambda x: x.value.value, model.secondary_sub_categories),
            ]
            subcat_choices[1:] = [
                subcat
                for subcat in subcat_choices
                if subcat != SubCategory("N/A").value
            ][1:]
            stage["pred"][1].append(subcat_choices)

        annotated["category"].append(pd.Series(stage["ann"][0]).drop_duplicates())
        annotated["subcategory"].append(pd.Series(stage["ann"][1]).drop_duplicates())
        predictions["category"].append(pd.Series(stage["pred"][0][0]).drop_duplicates())
        predictions["subcategory"].append(
            pd.Series(stage["pred"][1][0]).drop_duplicates()
        )

    annotated = pd.DataFrame(annotated)
    predictions = pd.DataFrame(predictions)

    _analysis_dir = CFG.dir.evaluation / CFG.platform.select
    _analysis_dir.mkdir(parents=True, exist_ok=True)

    for label, classes in (
        ("category", pd.DataFrame(get_categories())[0]),
        (
            "subcategory",
            pd.DataFrame(
                sum([get_sub_categories(cat) for cat in get_categories()], [])
            )[0],
        ),
    ):
        mat, classes = _cm(
            annotated[label].apply(lambda x: x[0]),
            predictions[label].apply(lambda x: x[0]),
            classes,
        )

        df = pd.DataFrame(mat, index=classes, columns=classes)
        na_row = df.loc[["N/A"]]
        df.drop("N/A", inplace=True)
        df = pd.concat([df, na_row], axis=0)

        na_col = df.pop("N/A")
        df["N/A"] = na_col

        (_analysis_dir / _csv_fn(label, i)).write_text(df.to_csv())

    for label, classes in (
        ("category", pd.DataFrame(get_categories())[0]),
        (
            "subcategory",
            pd.DataFrame(
                sum([get_sub_categories(cat) for cat in get_categories()], [])
            )[0],
        ),
    ):
        (conf_mat, normal_conf_mat), classes = _mlcm(
            annotated[label], predictions[label], classes
        )

        df = pd.DataFrame(
            conf_mat,
            index=[*classes, "No True Label"],
            columns=[*classes, "No Predicted Label"],
        )
        na_plus_row = df.loc[["N/A", "No True Label"]]
        df.drop(["N/A", "No True Label"], inplace=True)
        df = pd.concat([df, na_plus_row], axis=0)

        na_col = df.pop("N/A")
        np_col = df.pop("No Predicted Label")
        df["N/A"] = na_col
        df["No Predicted Label"] = np_col

        (_analysis_dir / _csv_fn(f"{label}_mlcm", i)).write_text(df.to_csv())
        logger.debug(
            "\n".join(
                [
                    f"{label}:",
                    "Raw confusion Matrix:",
                    str(conf_mat),
                    "Normalized confusion Matrix (%):",
                    str(normal_conf_mat),
                ]
            )
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--papers", nargs="*", type=str, default=None, help="Papers to analyse"
    )
    parser.add_argument(
        "--input",
        metavar="TXT",
        type=Path,
        default=None,
        help="List of papers to analyse",
    )
    options = parser.parse_args(argv)

    if options.input:
        with open(options.input, "r") as f:
            papers = list(map(Path, [l.strip() for l in f.readlines() if l.strip()]))
    elif options.papers:
        papers = list(map(Path, options.papers))
    else:
        papers = [
            CFG.dir.merged / paper.with_suffix(".yaml").name
            for paper in build_validation_set()
        ]

    if not any(map(lambda p: p.exists(), papers)):
        papers = [CFG.dir.merged / f"{paper}.yaml" for paper in papers]

    assert all(map(lambda p: p.exists(), papers))

    _evaluate_precision(papers)


if __name__ == "__main__":
    main()
