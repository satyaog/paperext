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
from paperext.structured_output import get_struct_module
from paperext.structured_output.find_acr_el.query import (
    AcronymsData,
    identify_terms_acronyms,
)
from paperext.structured_output.mdl.stats.stats import load_analysis
from paperext.structured_output.find_acr_mdl_dom.state import State
from paperext.utils import Paper


def list_domains(papers: list[dict | Paper]):
    for paper in papers:
        if not isinstance(paper, Paper):
            paper = Paper(paper)

        for query in paper.queries:
            extractions = (
                get_struct_module(CFG.platform.struct)
                .model.Response.model_validate_json(query.read_text())
                .analysis
            )

            for research_field in (
                extractions.primary_research_field,
                *extractions.sub_research_fields,
            ):
                yield research_field.name.value
                yield from research_field.aliases


@dataclass
class DomainAcronymsData(AcronymsData):
    def iter_categorized_terms(self):
        yield from _flatten_dict(self.categorized_terms["abstract_research_topics"])
        yield from _flatten_dict(self.categorized_terms["application_domains"])

    def list_paper_terms(self) -> list[str]:
        return self.papers_data["attrs"]["research_fields"].explode()

    def concurrent_terms(self, terms) -> np.ndarray:
        papers_exploded = self.papers_data["attrs"].explode("research_fields")
        titles = list(
            papers_exploded[papers_exploded["research_fields"].isin(terms)][
                "title"
            ].unique()
        )

        if not titles:
            return pd.DataFrame()

        related_papers = self.papers_data["attrs"][
            self.papers_data["attrs"]["title"].isin(titles)
        ]

        return related_papers["research_fields"].explode().unique()


def main(argv: list = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paperoni",
        nargs="*",
        type=Path,
        help="Paperoni json report of papers to analyse",
    )
    parser.add_argument(
        "--categorized-domains",
        type=Path,
        help="Path to categorized terms",
    )
    options = parser.parse_args(argv)

    papers = []
    for papers_json_path in options.paperoni:
        papers.extend(json.loads(Path(papers_json_path).read_text()))

    acronyms_data = DomainAcronymsData(
        json.loads(options.categorized_domains.read_text()),
        load_analysis(papers, CFG.dir.queries / CFG.platform.select)[0],
    )

    sanitized_map = _make_sanitized_map(acronyms_data.list_paper_terms())

    for research_fields in acronyms_data.papers_data["attrs"]["research_fields"]:
        research_fields[:] = map(lambda x: sanitized_map[x], research_fields)

    _update_sanitized_map(
        sanitized_map,
        *set(acronyms_data.iter_categorized_terms()),
    )

    acronyms_data.categorized_terms = {
        k.replace(" ", "_"): v
        for k, v in sanitize_categories(
            acronyms_data.categorized_terms, None, sanitized_map
        ).items()
    }

    with Config.push():
        CFG.platform.struct = Path(__file__).parent.name

        acronyms, left_overs = identify_terms_acronyms(
            acronyms_data=acronyms_data,
            sanitized_map=sanitized_map,
            state_cls=State,
        )

    if options.categorized_domains.with_stem(
        f"{options.categorized_domains.stem}_acronyms"
    ).exists():
        acronyms = {
            **json.loads(
                options.categorized_domains.with_stem(
                    f"{options.categorized_domains.stem}_acronyms"
                ).read_text()
            ),
            **acronyms,
        }

    options.categorized_domains.with_stem(
        f"{options.categorized_domains.stem}_acronyms"
    ).write_text(
        json.dumps(
            {k: v[0][1] for k, v in acronyms.items()},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
