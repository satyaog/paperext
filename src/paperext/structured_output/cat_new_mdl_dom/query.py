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
from paperext.structured_output.cat_new_mdl_dom.state import State
from paperext.structured_output.cat_new_mdl_dom.model import Response
from paperext.structured_output.find_acr_mdl_dom.query import list_domains


def parse_response(response: Response):
    return (
        [domain.value for domain in response.analysis.semantically_equivalent_domains],
        response.analysis.closest_parent_domain.value,
        response.analysis.closest_child_domain.value,
        response.analysis.closest_sibling_domain.value,
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
        "--categorized-domains",
        type=Path,
        default=CFG.dir.data / "mdl/categorized_domains.json",
        help="Path to categorized domains",
    )
    parser.add_argument(
        "--accronyms",
        type=Path,
        help="Path to categorized domains",
    )

    options = parser.parse_args(argv)

    papers = []
    for papers_json_path in options.paperoni:
        papers.extend(json.loads(Path(papers_json_path).read_text()))

    categorized_domains = json.loads(options.categorized_domains.read_text())

    if options.accronyms:
        accronyms_map = json.loads(options.accronyms.read_text().lower())
    else:
        accronyms_map = {}

    analysis, _ = load_analysis(papers, CFG.dir.queries / CFG.platform.select)
    sanitized_map = _make_sanitized_map(set(list_domains(papers)))
    _update_sanitized_map(
        sanitized_map,
        *(
            set(_flatten_dict(categorized_domains["abstract_research_topics"]))
            | set(_flatten_dict(categorized_domains["application_domains"]))
            | set(accronyms_map.keys())
            | set(accronyms_map.values())
            | set(analysis["attrs"]["research_fields"].explode())
        ),
    )

    accronyms_map = {
        sanitized_map[k]: sanitized_map[v] for k, v in accronyms_map.items()
    }
    sanitized_map = {
        k: accronyms_map.get(sanitized_map[k], v) for k, v in sanitized_map.items()
    }
    categorized_domains = {
        k.replace(" ", "_"): v
        for k, v in sanitize_categories(
            categorized_domains, accronyms_map, sanitized_map
        ).items()
    }

    for research_fields in analysis["attrs"]["research_fields"]:
        research_fields[:] = map(lambda x: sanitized_map[x], research_fields)

    all_domains = set(analysis["attrs"]["research_fields"].explode())

    Path(options.categorized_domains).with_suffix(".tmp").write_text(
        json.dumps(categorized_domains, indent=2, sort_keys=True)
    )

    with Config.push():
        CFG.platform.struct = Path(__file__).parent.name

        make_state = (
            lambda subject, propositions, sanitized_map, *args, **kwargs: State(
                *args,
                **kwargs,
                domain=subject,
                other_domains=propositions,
                sanitized_map=sanitized_map,
            )
        )
        for categorized_domains in categorise_new_element(
            categorized_elements=categorized_domains,
            all_elements=all_domains,
            sanitized_map=sanitized_map,
            make_state=make_state,
            parse_response=parse_response,
            excludes=set(["abstract_research_topics", "application_domains"]),
        ):
            Path(options.categorized_domains).with_suffix(".tmp").write_text(
                json.dumps(categorized_domains, indent=2, sort_keys=True)
            )

    options.categorized_domains.with_suffix(".tmp").rename(options.categorized_domains)


if __name__ == "__main__":
    main()
