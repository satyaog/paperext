import argparse
import copy
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import random
import sys

from instructor.exceptions import InstructorRetryException
from bs4 import BeautifulSoup
from openai import OpenAI

from sentence_transformers import SentenceTransformer
import tqdm

from paperext.config import CFG, Config
from paperext.log import logger
from paperext.query import PROG
from paperext.sanitize_categorization import (
    _dict_heads,
    _flatten_dict,
    _update_sanitized_map,
)
from paperext.structured_output import get_struct_module
from paperext.structured_output.compfore_cat_dom.model_v1 import (
    FIRST_MESSAGE,
    SYSTEM_MESSAGE,
    Analysis,
    Explained,
    Response,
    RETRY_MESSAGE_PARENT,
    RETRY_MESSAGE_SELECTED,
)
from paperext.structured_output.mdl_clus_dom.state import _sort_categories
from paperext.utils import Paper


PROMPT = """\
Sanitize the institution names in the following XML. Each author
is enclosed in <author> tags, and the institution is
enclosed in <institution> tags. The institution names
should be sanitized, and the authors should be
grouped by institution. The output should be a list of
<institution> tags, each containing a <institution-name> tag with the
sanitized institution name, and a list of <author> tags
with the authors' names in <author-name>. Authors may be affiliated with
multiple institutions but the input XML will only contain
one institution name per author. If the institution name contains
multiple institutions, please split them and add the authors
to the corresponding institution. The same institution may have
multiple names, so please use the most common name. 

Explain concisely the reasoning behind your choice of institution name in a 
first <explanation> tag. Follow with the XML output enclosed in <institutions> tags.

Here is the XML:
{xml}

Now provide your answer:
<explanation>"""


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
            assert (
                selected not in cats[parent]
            ), f"Domain {selected} already in {parent}: {sorted(cats[parent].keys())}"
            cats[parent] = {**cats[parent], selected: {}}
            return True

        for value in cats.values():
            if _insert_recursive(value):
                return True

        return False

    # Try to insert the domain
    if not _insert_recursive(result):
        raise ValueError(f"Parent category '{parent}' not found in categories")

    return result


def prompt(client: OpenAI, prompt_input: str):
    # Get response from LLM
    response = client.responses.create(model=CFG.openai.model, input=prompt_input)

    # Parse the XML response
    answer = response.output_text
    soup = BeautifulSoup(answer, features="lxml")

    # Extract the components
    explanation = "\n".join([e.text for e in soup.find_all("explanation")]).strip()
    selected_domain = "\n".join(
        [d.text for d in soup.find_all("selected_domain")]
    ).strip()
    parent_category = "\n".join(
        [p.text for p in soup.find_all("parent_category")]
    ).strip()

    logger.info(f"Explanation: {explanation}")
    logger.info(f"Selected domain: {selected_domain}")
    logger.info(f"Parent category: {parent_category}")

    return explanation, selected_domain, parent_category, response.usage


def query(client: OpenAI, paper_id: str, categorization: dict, domains: list):
    """Query the LLM to select and categorize a domain from the list.

    Args:
        client: OpenAI client instance
        categorization: Dictionary containing the hierarchical structure
        domains: List of domains to categorize

    Returns:
        Response object containing the analysis with selected domain and parent category
    """

    # Convert categorization dict to TXT format
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
                txt.extend(
                    [f"{'  ' * indent}{entry}" for entry in (f"{_open}:", _value)]
                )
        return "\n".join([entry for entry in txt if entry.strip()])

    categorization_txt = json_to_txt(
        categorization
    )  # json.dumps(categorization, sort_keys=True, indent=2)
    selected_domain = None
    parent_category = None

    # Format the prompt
    prompt_input = (
        SYSTEM_MESSAGE.format(categorization_txt)
        + "\n"
        + FIRST_MESSAGE.format("\n".join([f"- {d}" for d in domains]))
        + "\n"
        + """Format your output in XML with the following structure:
<explanation>
Your detailed explanation of why you chose this domain and parent category
</explanation>
<selected_domain>
The exact domain name from the provided list
</selected_domain>
<parent_category>
The exact parent category name from the hierarchical structure
</parent_category>
"""
    )

    i = 0
    while selected_domain is None or parent_category is None:
        response_file = CFG.dir.queries / f"{paper_id}_{i:02}.json"

        query_data = {
            "categorization": categorization,
            "domains": domains,
            "selected": selected_domain,
            "parent": parent_category,
        }

        if response_file.exists():
            response = Response.model_validate_json(response_file.read_text())
            selected_domain, parent_category = (
                response.analysis.selected_domain.value,
                response.analysis.parent_category.value,
            )

        else:
            explanation, selected_domain, parent_category, usage = prompt(
                client, prompt_input
            )

            response = Response(
                paper=paper_id,
                words=0,
                analysis=Analysis(
                    selected_domain=Explained[str](
                        value=selected_domain, reasoning=explanation
                    ),
                    parent_category=Explained[str](
                        value=parent_category, reasoning=explanation
                    ),
                ),
                usage=usage.model_dump(),
                query_data=query_data,
            )
            response_file.write_text(response.model_dump_json(indent=2))

        yield response

        if parent_category not in set(_flatten_dict(categorization)):
            prompt_input = (
                SYSTEM_MESSAGE.format(categorization_txt)
                + "\n"
                + RETRY_MESSAGE_PARENT.format(
                    parent_category, "\n".join([f"- {d}" for d in domains])
                )
                + "\n"
                + """Format your output in XML with the following structure:
<explanation>
Your detailed explanation of why you chose this domain and parent category
</explanation>
<selected_domain>
The exact domain name from the provided list
</selected_domain>
<parent_category>
The exact parent category name from the hierarchical structure
</parent_category>
"""
            )
            parent_category = None

        if selected_domain not in domains:
            prompt_input = (
                SYSTEM_MESSAGE.format(categorization_txt)
                + "\n"
                + RETRY_MESSAGE_SELECTED.format(
                    selected_domain, "\n".join([f"- {d}" for d in domains])
                )
                + "\n"
                + """Format your output in XML with the following structure:
<explanation>
Your detailed explanation of why you chose this domain and parent category
</explanation>
<selected_domain>
The exact domain name from the provided list
</selected_domain>
<parent_category>
The exact parent category name from the hierarchical structure
</parent_category>
"""
            )
            selected_domain = None

        i += 1

        if i > 10:
            raise ValueError(
                f"Failed to select a domain and parent category after {i} attempts"
            )


def main(argv: list[str] = None):
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
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

        client = OpenAI()
        # Something messes with the seed, so we pre-generated seeds
        random.seed(options.seed)
        sample_seeds = [random.randint(0, sys.maxsize) for _ in range(len(domains) * 2)]

        for _ in tqdm.tqdm(range(len(domains)), desc="Categorizing"):
            _categories = _dict_heads(categories, 0, 4)
            _categories["ignore"] = {}

            random.seed(sample_seeds.pop())
            _domains = random.sample(domains, min(len(domains), 50))

            _filename_prefix = "".join(
                sorted(set(_term[0] for _term in sorted(_domains)))
            )
            _filename = "_".join(
                [
                    _filename_prefix,
                    hashlib.sha256("".join(sorted(_domains)).encode()).hexdigest(),
                ]
            )

            while True:
                try:
                    responses: list[Response] = list(
                        query(client, _filename, _categories, _domains)
                    )
                    break

                except InstructorRetryException:
                    continue

            selected = responses[-1].analysis.selected_domain.value
            parent = responses[-1].analysis.parent_category.value

            categories = insert_in_categories(selected, parent, categories)
            domains.remove(selected)

    (options.out.write_text if str(options.out) != "-" else print)(
        json.dumps(categories, indent=2, sort_keys=True, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
