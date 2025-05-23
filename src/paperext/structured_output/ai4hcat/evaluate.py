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
pd.set_option(
    "display.float_format", "{:.2f}".format
)  # Format floats with 2 decimal places


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


def _calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate TP, TN, FP, FN, Precision, and Recall for each class."""
    metrics = pd.DataFrame(index=df.index)

    for idx in df.index:
        # Get the row and column for this class
        row = df.loc[idx]
        if idx == "No True Label":
            col = df["No Predicted Label"]
        else:
            col = df[idx]

        # Calculate metrics
        tp = col[idx]  # True positives are on the diagonal
        fp = col.sum() - tp  # False positives are sum of column minus TP
        fn = row.sum() - tp  # False negatives are sum of row minus TP
        tn = df.values.sum() - (tp + fp + fn)  # True negatives are all other cells

        # Calculate precision and recall
        if idx == "No True Label":
            precision = ""
            recall = ""
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else ""
            recall = tp / (tp + fn) if (tp + fn) > 0 else ""

        # Add to metrics DataFrame
        metrics.loc[idx, "TP"] = tp
        metrics.loc[idx, "TN"] = tn
        metrics.loc[idx, "FP"] = fp
        metrics.loc[idx, "FN"] = fn

        metrics.loc[idx, "Precision"] = precision
        metrics.loc[idx, "Recall"] = recall

    # Set proper dtypes in metrics dataframe
    metrics["TP"] = metrics["TP"].astype("int32")
    metrics["TN"] = metrics["TN"].astype("int32")
    metrics["FP"] = metrics["FP"].astype("int32")
    metrics["FN"] = metrics["FN"].astype("int32")

    return metrics


def reorder_special_labels(
    df: pd.DataFrame, indices: list[str], columns: list[str]
) -> pd.DataFrame:
    """Reorder special labels (like 'N/A' and 'No * Label') to the end of the DataFrame.

    Args:
        df: DataFrame to reorder
        indices: List of indices to move to the end
        columns: List of columns to move to the end

    Returns:
        DataFrame with special labels at the end
    """
    special_rows = df.loc[indices]
    df = df.drop(indices)
    df = pd.concat([df, special_rows], axis=0)

    for col in columns:
        special_col = df.pop(col)
        df[col] = special_col

    return df


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

            cat_choices = list(
                map(
                    lambda x: x.value.value,
                    [model.primary_category] + model.secondary_categories,
                ),
            )
            cat_choices[1:] = [
                cat for cat in cat_choices if cat != Category("N/A").value
            ][1:]
            stage["pred"][0].append(cat_choices)

            subcat_choices = list(
                map(
                    lambda x: x.value.value,
                    [model.primary_sub_category] + model.secondary_sub_categories,
                ),
            )
            subcat_choices[1:] = [
                subcat
                for subcat in subcat_choices
                if subcat != SubCategory("N/A").value
            ][1:]
            stage["pred"][1].append(subcat_choices)

        annotated["category"].append(pd.Series(stage["ann"][0]).drop_duplicates())
        annotated["subcategory"].append(pd.Series(stage["ann"][1]).drop_duplicates())
        predictions["category"].append(
            pd.Series(stage["pred"][0][-1]).drop_duplicates()
        )
        predictions["subcategory"].append(
            pd.Series(stage["pred"][1][-1]).drop_duplicates()
        )

        # Check for mismatches and log them
        if stage["ann"][0][0] not in stage["pred"][0][-1]:
            logger.warning(
                f"Category mismatch for paper\t{f.stem}\n"
                + f"  Expected:\t{stage['ann'][0]}\n"
                + f"  Got:     \t{stage['pred'][0][-1]}"
            )

        if stage["ann"][1][0] not in stage["pred"][1][-1]:
            logger.warning(
                f"Subcategory mismatch for paper\t{f.stem}\n"
                + f"  Expected:\t{stage['ann'][1]}\n"
                + f"  Got:     \t{stage['pred'][1][-1]}"
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
        # Top 1
        mat, classes = _cm(
            annotated[label].apply(lambda x: x[0]),
            predictions[label].apply(lambda x: x[0]),
            classes,
        )

        # One of predictions is correct
        ann_pred = pd.concat(
            [annotated[label], predictions[label]], keys=["ann", "pred"], axis=1
        )
        pred_one_of = ann_pred.apply(
            lambda x: (
                x["ann"][0] if (x["pred"] == x["ann"][0]).any() else x["pred"][0]
            ),
            axis=1,
        )
        mat_one_of, classes = _cm(
            annotated[label].apply(lambda x: x[0]),
            pred_one_of,
            classes,
        )

        for m, name in zip([mat, mat_one_of], ["top1", "oneof"]):
            df = pd.DataFrame(m, index=classes, columns=classes)
            df = reorder_special_labels(df, ["N/A"], ["N/A"])

            # Calculate and append metrics
            metrics = _calculate_metrics(df)
            df = pd.concat([df, metrics], axis=1)

            print(label, i)
            print(metrics)

            (_analysis_dir / _csv_fn(f"{label}_{name}", i)).write_text(df.to_csv())

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
        df = reorder_special_labels(
            df, ["N/A", "No True Label"], ["N/A", "No Predicted Label"]
        )

        # Calculate and append metrics
        metrics = _calculate_metrics(df)
        df = pd.concat([df, metrics], axis=1)

        print(f"{label}_mlcm", i)
        print(metrics)

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
