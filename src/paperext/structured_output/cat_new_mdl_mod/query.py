import argparse
import json
from pathlib import Path

from paperext.config import CFG, Config
from paperext.sanitize_categorization import (
    _flatten_dict,
    _make_sanitized_map,
    _update_sanitized_map,
    sanitize_categories,
)

# from paperext.structured_output import get_struct_module
from paperext.structured_output.cat_new_el.query import categorise_new_element
from paperext.structured_output.mdl.stats.stats import load_analysis
from paperext.structured_output.cat_new_mdl_mod.state import State
from paperext.structured_output.cat_new_mdl_mod.model import Response
from paperext.structured_output.find_acr_mdl_mod.query import ModelAcronymsData


def parse_response(response: Response):
    return (
        [domain.value for domain in response.analysis.semantically_equivalent_models],
        response.analysis.closest_parent_model.value,
        response.analysis.closest_child_model.value,
        response.analysis.closest_sibling_model.value,
    )


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
        help="Path to categorized models",
    )
    parser.add_argument(
        "--accronyms",
        type=Path,
        help="Path to categorized models",
    )

    options = parser.parse_args(argv)

    papers = []
    for papers_json_path in options.paperoni:
        papers.extend(json.loads(Path(papers_json_path).read_text()))

    models_data = ModelAcronymsData(
        json.loads(options.categorized_models.read_text()),
        load_analysis(papers, CFG.dir.queries / CFG.platform.select)[0],
    )

    if options.accronyms:
        accronyms_map = json.loads(options.accronyms.read_text().lower())
    else:
        accronyms_map = {}

    sanitized_map = _make_sanitized_map(models_data.list_paper_terms())

    models_data.papers_data["models"]["name"][:] = list(
        map(lambda x: sanitized_map[x], models_data.papers_data["models"]["name"])
    )
    for model_aliases in models_data.papers_data["models"]["aliases"]:
        model_aliases[:] = map(lambda x: sanitized_map[x], model_aliases)

    _update_sanitized_map(
        sanitized_map,
        *(
            set(models_data.iter_categorized_terms())
            | set(accronyms_map.keys())
            | set(accronyms_map.values())
        ),
    )

    models_data.categorized_terms = sanitize_categories(
        models_data.categorized_terms, accronyms_map, sanitized_map
    )
    models_data.categorized_terms["classic_ml"] = models_data.categorized_terms.pop(
        "classic ml"
    )

    all_models = set(models_data.list_paper_terms())

    Path(options.categorized_models).with_suffix(".tmp").write_text(
        json.dumps(
            models_data.categorized_terms, indent=2, sort_keys=True, ensure_ascii=False
        )
    )

    with Config.push():
        CFG.platform.struct = Path(__file__).parent.name

        make_state = (
            lambda subject, propositions, sanitized_map, *args, **kwargs: State(
                *args,
                **kwargs,
                model=subject,
                other_models=propositions,
                sanitized_map=sanitized_map,
            )
        )
        for categorized_models in categorise_new_element(
            categorized_elements=models_data.categorized_terms,
            all_elements=all_models,
            sanitized_map=sanitized_map,
            make_state=make_state,
            parse_response=parse_response,
            excludes=set(
                ["algorithms", "classic_ml", "ignore", "neural networks", "others"]
            ),
        ):
            Path(options.categorized_models).with_suffix(".tmp").write_text(
                json.dumps(
                    categorized_models, indent=2, sort_keys=True, ensure_ascii=False
                )
            )

    options.categorized_models.with_suffix(".tmp").rename(options.categorized_models)


if __name__ == "__main__":
    main()
