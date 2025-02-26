import argparse
import asyncio
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
from pprint import pprint

from instructor.exceptions import InstructorRetryException
from sentence_transformers import SentenceTransformer
import tqdm
from paperext.config import CFG, Config
from paperext.query import PLATFORMS, PROG, batch_queries
from paperext.sanitize_categorization import _flatten_dict
from paperext.structured_output.mdl_clus_dom.model import Response
from paperext.structured_output.mdl_clus_dom.state import (
    State,
    _find_min_max_threshold,
    _sort_categories,
    cluster_categories,
)


def main(argv: list = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categorized-domains",
        type=Path,
        default=CFG.dir.data / "mdl/categorized_domains.json",
        help="Path to categorized domains",
    )
    options = parser.parse_args(argv)

    domains = json.loads(options.categorized_domains.read_text().lower())
    del domains["ignore"]
    domains = sorted(
        set(
            sum(
                [list(_flatten_dict(domains[key])) for key in domains],
                [],
            )
        )
    )

    model = SentenceTransformer("all-MiniLM-L6-v2")
    categories = {d: {} for d in _sort_categories(model, sorted(domains))}

    min_tolerace, max_tolerance = _find_min_max_threshold(model, sorted(categories))
    tolerance = max_tolerance

    with Config.push():
        # CFG.platform.select = "openai"
        CFG.platform.select = "ollama"
        CFG.platform.struct = Path(__file__).parent.name
        CFG.ollama.model = "deepseek-r1:32b"
        # CFG.ollama.model = "deepseek-r1:14b"

        LOG_FILE = CFG.dir.log / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logging.basicConfig(
            filename=LOG_FILE.with_suffix(f".{PROG}.{CFG.platform.struct}.dbg"),
            level=logging.DEBUG,
            force=True,
        )

        client = PLATFORMS[CFG.platform.select]()

        while tolerance >= min_tolerace and len(categories) >= 2:
            tolerance_step = max(0.01, (tolerance - min_tolerace) / 2)
            tolerance = tolerance - tolerance_step

            pre_categorisation = cluster_categories(
                model, sorted(categories), tolerance=tolerance
            )

            _categories = {}

            for cat, sub_cats in tqdm.tqdm(
                pre_categorisation.items(),
                desc="Identifying generic domains amoung clusters",
                unit="cluster",
            ):
                sub_cats.setdefault(cat, {})

                if len(sub_cats) > 1:
                    _sub_cats = sorted(sub_cats)
                    _filename = hashlib.sha256("".join(_sub_cats).encode()).hexdigest()

                    print(f"Identifying the most generic domain among {_sub_cats}")

                    make_state = lambda *args, **kwargs: State(
                        *args, **kwargs, domains=_sub_cats
                    )
                    while True:
                        try:
                            responses: list[Response] = asyncio.run(
                                batch_queries(
                                    client,
                                    [(None, Path(_filename))],
                                    destination=CFG.dir.data
                                    / CFG.platform.struct
                                    / "queries"
                                    / CFG.platform.select,
                                    state_cls=make_state,
                                )
                            )
                            break

                        except InstructorRetryException:
                            continue

                    cat = responses[-1].extractions.generic_domain.value.lower()
                    matches = (
                        [d for d in _sub_cats if d == cat]
                        or [
                            d
                            for d in _sub_cats
                            if d.replace(" ", "") == cat.replace(" ", "")
                        ]
                        or [
                            d
                            for d in _sub_cats
                            if d.replace("-", "").replace(" ", "")
                            == cat.replace("-", "").replace(" ", "")
                        ]
                    )
                    assert len(matches) == 1
                    cat = matches[0]

                    print(
                        f"Identified [{cat}] as the most generic domain among {_sub_cats}"
                    )

                sub_cats.pop(cat.lower())

                for sub_cat in sub_cats:
                    sub_cats[sub_cat] = categories[sub_cat]
                sub_cats[cat] = categories[cat]
                _categories[cat] = sub_cats

            categories = _categories

    pprint(categories)


if __name__ == "__main__":
    main()
