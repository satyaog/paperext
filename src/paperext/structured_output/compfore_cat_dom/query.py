import argparse
import asyncio
import copy
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path

from instructor.exceptions import InstructorRetryException
from sentence_transformers import SentenceTransformer
import tqdm

from paperext.config import CFG, Config
from paperext.query import PLATFORMS, PROG, batch_queries
from paperext.sanitize_categorization import (
    _dict_heads,
    _flatten_dict,
    _update_sanitized_map,
)

# from paperext.structured_output import get_struct_module
from paperext.structured_output import get_struct_module
from paperext.structured_output.compfore_cat_dom.state import State
from paperext.structured_output.compfore_cat_dom.model import Response
from paperext.structured_output.mdl_clus_dom.state import _sort_categories
from paperext.utils import Paper


def iter_domains(papers: list[dict | Paper]):
    for paper in papers:
        if not isinstance(paper, Paper):
            paper = Paper(paper)

        for query in paper.queries:
            analysis = (
                get_struct_module(CFG.platform.struct)
                .model.Response.model_validate_json(query.read_text())
                .analysis
            )

            for research_field in (
                analysis.primary_research_field,
                *analysis.sub_research_fields,
            ):
                yield research_field.name.value
                yield from research_field.aliases


def insert_in_categories(selected: str, parent: str, categories: dict) -> dict:
    """Insert a selected domain under its parent category in the hierarchical structure.

    Args:
        selected: The domain to insert
        parent: The parent category where the domain should be inserted
        categories: The hierarchical structure to modify

    Returns:
        The modified hierarchical structure with the domain inserted
    """
    # Create a deep copy to avoid modifying the original
    result = copy.deepcopy(categories)

    def _insert_recursive(cats: dict) -> bool:
        """Recursively search for the parent category and insert the domain.

        Args:
            cats: The current level of the category structure

        Returns:
            True if the domain was inserted, False otherwise
        """
        if parent in cats:
            cats[parent] = {selected: {}}
            return True

        for value in cats.values():
            if _insert_recursive(value):
                return True

        return False

    # Try to insert the domain
    if not _insert_recursive(result):
        raise ValueError(f"Parent category '{parent}' not found in categories")

    return result


def main(argv: list = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paperoni",
        nargs="*",
        type=Path,
        help="Paperoni json report of papers to analyse",
    )
    parser.add_argument(
        "--categorization",
        metavar="PATH",
        type=Path,
        help="Path to categorization JSON file",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        type=Path,
        default=None,
        help="Path to sanitized categorization JSON file (defaults to `categorization`)",
    )
    options = parser.parse_args(argv)
    options.out = options.out or options.categorization

    papers = []
    for papers_json_path in options.paperoni:
        papers.extend(json.loads(Path(papers_json_path).read_text()))

    categories = json.loads(options.categorization.read_text())

    with Config.push():
        CFG.platform.select = "openai"
        CFG.platform.struct = "mdl"
        CFG.dir.queries = CFG.dir.data / CFG.platform.struct / "queries"
        domains = list(iter_domains(papers))

    sanitized_map = {}
    domains = set(
        _update_sanitized_map(sanitized_map, *domains, *_flatten_dict(categories))
    )
    categories["ignore"] = {k: {} for k in _flatten_dict(categories["ignore"])}

    def _(d: dict):
        return {sanitized_map[k]: _(v) for k, v in d.items()}

    categories = _(_dict_heads(categories, 0, 2))
    domains = domains - set(_flatten_dict(categories))

    model = SentenceTransformer("all-MiniLM-L6-v2")
    domains = _sort_categories(model, domains)

    with Config.push():
        CFG.platform.struct = Path(__file__).parent.name
        CFG.dir.queries = CFG.dir.data / CFG.platform.struct / "queries"

        LOG_FILE = CFG.dir.log / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logging.basicConfig(
            filename=LOG_FILE.with_suffix(f".{PROG}.{CFG.platform.struct}.dbg"),
            level=logging.DEBUG,
            force=True,
        )

        client = PLATFORMS[CFG.platform.select]()

        for _ in tqdm.tqdm(domains, desc="Categorizing"):
            _filename_prefix = "".join(
                sorted(set(_term[0] for _term in sorted(domains)))
            )
            _filename = "_".join(
                [
                    _filename_prefix,
                    hashlib.sha256("".join(sorted(domains)).encode()).hexdigest(),
                ]
            )

            make_state = lambda *args, **kwargs: State(
                *args, **kwargs, categorization=categories, domains=domains
            )
            while True:
                try:
                    responses: list[Response] = asyncio.run(
                        batch_queries(
                            client,
                            [(None, Path(_filename))],
                            destination=CFG.dir.queries / CFG.platform.select,
                            state_cls=make_state,
                        )
                    )
                    break

                except InstructorRetryException:
                    continue

            selected = responses[-1].analysis.selected_domain  # .value
            parent = responses[-1].analysis.parent_category  # .value

            categories = insert_in_categories(selected, parent, categories)
            domains.remove(selected)

    (options.out.write_text if str(options.out) != "-" else print)(
        json.dumps(categories, indent=2, sort_keys=True, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
