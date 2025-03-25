import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from paperext.config import CFG, Config
from paperext.sanitize_categorization import (
    _flatten_dict,
    _make_sanitized_map,
    _update_sanitized_map,
    sanitize_categories,
)
from paperext.structured_output.find_acr_el.query import (
    AcronymsData,
    identify_terms_acronyms,
)
from paperext.structured_output.mdl.stats.stats import load_analysis
from paperext.structured_output.find_acr_mdl_mod.state import State


@dataclass
class ModelAcronymsData(AcronymsData):
    def iter_categorized_terms(self):
        yield from _flatten_dict(self.categorized_terms["algorithms"])
        yield from _flatten_dict(self.categorized_terms["classic_ml"])
        yield from _flatten_dict(self.categorized_terms["neural networks"])
        yield from _flatten_dict(self.categorized_terms["others"])

    def list_paper_terms(self) -> list[str]:
        return pd.concat(
            [
                self.papers_data["models"]["name"],
                self.papers_data["models"]["aliases"].explode().dropna(),
            ]
        )

    def concurrent_terms(self, terms) -> np.ndarray:
        papers_name_exploded = self.papers_data["models"]
        papers_aliases_exploded = self.papers_data["models"].explode("aliases")
        titles = pd.concat(
            [
                papers_name_exploded[papers_name_exploded["name"].isin(terms)]["title"],
                papers_aliases_exploded[papers_aliases_exploded["aliases"].isin(terms)][
                    "title"
                ],
            ],
        ).unique()

        if not len(titles):
            return pd.Series().unique()

        related_papers = self.papers_data["models"][
            self.papers_data["models"]["title"].isin(titles)
        ]

        return pd.concat(
            [
                related_papers["name"],
                related_papers["aliases"].explode().dropna(),
            ]
        ).unique()


def main(argv: list = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paperoni",
        nargs="*",
        type=Path,
        help="Paperoni json report of papers to analyse",
    )
    parser.add_argument(
        "--categorized-models",
        type=Path,
        help="Path to categorized terms",
    )
    options = parser.parse_args(argv)

    papers = []
    for papers_json_path in options.paperoni:
        papers.extend(json.loads(Path(papers_json_path).read_text()))

    acronyms_data = ModelAcronymsData(
        json.loads(options.categorized_models.read_text()),
        load_analysis(papers, CFG.dir.queries / CFG.platform.select)[0],
    )

    sanitized_map = _make_sanitized_map(acronyms_data.list_paper_terms())

    acronyms_data.papers_data["models"]["name"][:] = list(
        map(lambda x: sanitized_map[x], acronyms_data.papers_data["models"]["name"])
    )
    for model_aliases in acronyms_data.papers_data["models"]["aliases"]:
        model_aliases[:] = map(lambda x: sanitized_map[x], model_aliases)

    _update_sanitized_map(
        sanitized_map,
        *set(acronyms_data.iter_categorized_terms()),
    )

    acronyms_data.categorized_terms = sanitize_categories(
        acronyms_data.categorized_terms, None, sanitized_map
    )
    acronyms_data.categorized_terms["classic_ml"] = acronyms_data.categorized_terms.pop(
        "classic ml"
    )

    with Config.push():
        CFG.platform.struct = Path(__file__).parent.name

        acronyms, left_overs = identify_terms_acronyms(
            acronyms_data=acronyms_data,
            sanitized_map=sanitized_map,
            state_cls=State,
        )

    if options.categorized_models.with_stem(
        f"{options.categorized_models.stem}_acronyms"
    ).exists():
        acronyms = {
            **json.loads(
                options.categorized_models.with_stem(
                    f"{options.categorized_models.stem}_acronyms"
                ).read_text()
            ),
            **acronyms,
        }

    options.categorized_models.with_stem(
        f"{options.categorized_models.stem}_acronyms"
    ).write_text(
        json.dumps(
            {k: v[0][1] for k, v in acronyms.items()},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
