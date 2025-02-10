import argparse
from pathlib import Path
from typing import List

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
from paperext.structured_output.mdl.model import ExtractionResponse, PaperExtractions
from paperext.structured_output.utils import model2df, model_validate_yaml
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


def _append_left_indices(df: pd.DataFrame, indices: List[tuple]):
    df = df.copy(True)
    if df.empty:
        return df

    try:
        index_names = list(range(len(df.index.levels)))
        df.index.names = index_names
        df.reset_index(inplace=True)
    except AttributeError:
        index_names = list(range(df.index.nlevels - 1))
        if index_names:
            df.index.names = index_names

    inds = [ind for ind, _ in indices]
    for ind, val in indices:
        df.loc[:, ind] = val

    df.set_index([*inds, *index_names], inplace=True)
    df.index.names = [None] * (len(inds) + len(index_names))

    return df


def _cm(annotations: pd.DataFrame, predictions: pd.DataFrame):
    classes = pd.concat([annotations, predictions])
    classes.sort_values(inplace=True, ignore_index=True)
    classes.drop_duplicates(inplace=True, ignore_index=True)

    return confusion_matrix(annotations, predictions, labels=classes), classes


def _mlcm(annotations: pd.DataFrame, predictions: pd.DataFrame):
    classes = pd.concat(list(annotations) + list(predictions))
    classes.sort_values(inplace=True, ignore_index=True)
    classes.drop_duplicates(inplace=True, ignore_index=True)

    _ann, _pred = (
        [classes.isin(arr).astype(int) for arr in annotations],
        [classes.isin(arr).astype(int) for arr in predictions],
    )

    return mlcm.cm(_ann, _pred), classes


def _measure_precision(papers: list):
    """Analyse the performance of the LLM on the given papers though confusion
    matrices and multi-label confusion matrices"""
    annotated = [[], []]
    predictions = [[], []]

    for f in papers:
        if not f.exists():
            continue

        stage = {
            "ann": [[], []],
            "pred": [[], []],
        }
        logger.info(f"Fetching data from {f}")
        model = model_validate_yaml(PaperExtractions, f.read_text())
        paper_attr, paper_refs = map(
            lambda m: _append_left_indices(m, [("paper_id", f.stem)]),
            model2df(model),
        )

        stage["ann"][0].append(paper_attr)
        stage["ann"][1].append(paper_refs)

        queries_dir = CFG.dir.queries / CFG.platform.select
        for i, query_f in enumerate(sorted(queries_dir.glob(f"{f.stem}*.json"))):
            logger.info(f"Fetching data from {query_f}")
            model = ExtractionResponse.model_validate_json(
                query_f.read_text()
            ).extractions

            paper_attr, paper_refs = map(
                lambda m: _append_left_indices(
                    m, [("paper_id", f.stem), ("attempt", i)]
                ),
                model2df(model),
            )

            stage["pred"][0].append(paper_attr)
            stage["pred"][1].append(paper_refs)

        if stage["pred"][0]:
            annotated[0].extend(stage["ann"][0])
            annotated[1].extend(stage["ann"][1])
            predictions[0].extend(stage["pred"][0])
            predictions[1].extend(stage["pred"][1])

    annotated[0] = pd.concat(annotated[0])
    annotated[1] = pd.concat(annotated[1])

    predictions[0] = pd.concat(predictions[0])
    predictions[1] = pd.concat(predictions[1])

    _analysis_dir = CFG.dir.measure / CFG.platform.select
    _analysis_dir.mkdir(parents=True, exist_ok=True)

    max_attempt = max(
        predictions[0]
        .reset_index(
            [
                0,
            ],
            drop=True,
        )
        .index
    )

    for label in ("title", "type", "primary_research_field"):
        for i in range(max_attempt + 1):
            mat, classes = _cm(
                annotated[0].loc[:, label],
                predictions[0].loc[:, i, :].loc[:, label],
            )
            # Title confusion matrix should be the identity
            # if label == "title":
            #     assert (_mat == np.identity(_mat.shape[0])).all()

            (_analysis_dir / f"{label}_{i:02}.csv").write_text(
                pd.DataFrame(mat, index=classes, columns=classes).to_csv()
            )

    for label in (
        "sub_research_fields",
        "all_research_fields",
        "research_fields_categories",
    ):
        for i in range(max_attempt + 1):
            ann, pred = (
                annotated[0].loc[:, label],
                predictions[0].loc[:, i, :].loc[:, label],
            )

            (conf_mat, normal_conf_mat), classes = _mlcm(
                [_ann.drop_duplicates() for _ann in ann],
                [_pred.drop_duplicates() for _pred in pred],
            )

            (_analysis_dir / f"{label}_{i:02}.csv").write_text(
                pd.DataFrame(
                    conf_mat,
                    index=[*classes, "No True Label"],
                    columns=[*classes, "No Predicted Label"],
                ).to_csv()
            )
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

    for group in (
        "models",
        "datasets",
        "libraries",
    ):
        for i in range(max_attempt + 1):
            # [1] indexes the references (models, datasets, libraries)
            ann, pred = (
                annotated[1].loc[:, group, :],
                predictions[1].loc[:, i, group, :],
            )
            ann: pd.DataFrame
            pred: pd.DataFrame

            papers = (
                ann.index.get_level_values(0)  # paper id
                .union(pred.index.get_level_values(0))  # matching paper id
                .sort_values()
                .drop_duplicates()
            )
            ann_per_paper, pred_per_paper = (
                [ann[ann.index.isin([p], level=0)] for p in papers],
                [pred[pred.index.isin([p], level=0)] for p in papers],
            )

            col = "name"
            (conf_mat, normal_conf_mat), names = _mlcm(
                [_ann[col] for _ann in ann_per_paper],
                [_pred[col] for _pred in pred_per_paper],
            )

            (_analysis_dir / f"{group}.{col}_{i:02}.csv").write_text(
                pd.DataFrame(
                    conf_mat,
                    index=[*names, "No True Label"],
                    columns=[*names, "No Predicted Label"],
                ).to_csv()
            )

            logger.debug(
                "\n".join(
                    [
                        f"{group}.{col}:",
                        "Raw confusion Matrix:",
                        str(conf_mat),
                        "Normalized confusion Matrix (%):",
                        str(normal_conf_mat),
                    ]
                )
            )

            col = "category"
            (conf_mat, normal_conf_mat), names = _mlcm(
                [_ann[col].drop_duplicates() for _ann in ann_per_paper],
                [_pred[col].drop_duplicates() for _pred in pred_per_paper],
            )

            (_analysis_dir / f"{group}.{col}_{i:02}.csv").write_text(
                pd.DataFrame(
                    conf_mat,
                    index=[*names, "No True Label"],
                    columns=[*names, "No Predicted Label"],
                ).to_csv()
            )

            logger.debug(
                "\n".join(
                    [
                        f"{group}.{col}:",
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
            papers = list(map(Path, [l.strip() for l in f.readlines()]))
    elif options.papers:
        papers = list(map(Path, options.papers))
    else:
        papers = [
            CFG.dir.merged / paper.with_suffix(".yaml").name
            for paper in build_validation_set()
        ]

    if not any(map(lambda p: p.exists(), papers)):
        papers = [CFG.dir.merged / f"{paper}.yaml" for paper in papers]

    assert any(map(lambda p: p.exists(), papers))

    _measure_precision(papers)


if __name__ == "__main__":
    main()
