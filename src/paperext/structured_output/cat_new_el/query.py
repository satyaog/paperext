import asyncio
from datetime import datetime
import hashlib
import logging
import math
from pathlib import Path

from instructor.exceptions import InstructorRetryException
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import tqdm

from paperext.config import CFG
from paperext.query import PLATFORMS, PROG, batch_queries
from paperext.sanitize_categorization import (
    _update_sanitized_map,
)


def _update_sorted_propositions(
    last_propositions: list,
    similarities,
    skipped,
    k=10,
):
    new_domain = last_propositions[0][1]

    if new_domain in skipped:
        return last_propositions[1:]

    propositions = []

    for proposition in last_propositions[1:]:
        remaining = proposition[1]
        distances = sorted(
            [
                (1 - similarities[remaining, domain], domain)
                for domain in proposition[2:] + [new_domain]
            ]
        )
        propositions.append(
            [distances[0][0], remaining] + [d[1] for d in distances[:k]]
        )

    return sorted(propositions)


def get_proposition(
    remainings,
    pool: set,
    similarities: dict[tuple[str, str], float],
    k=10,
):
    propositions = []
    for remaining in remainings:
        distances = sorted(
            [(1 - similarities[remaining, domain], domain) for domain in pool]
        )
        propositions.append(
            [distances[0][0], remaining] + [d[1] for d in distances[1 : k + 1]]
        )

    return sorted(propositions)


def build_dataframe(categorized_elements):
    def build_tree(categorized_elements) -> list:
        rows = []
        for key in categorized_elements:
            rows.append({"category": key})
            if categorized_elements[key]:
                subcats = build_tree(categorized_elements[key])
                for subcat in subcats:
                    subcat[f"parent{len(subcat)}"] = key
                rows.extend(subcats)

        return rows

    return pd.DataFrame(build_tree(categorized_elements))


def get_category(selection, categorized_elements, df):
    try:
        domain_row = df[df["category"] == selection].iloc[0]
    except:
        import pdb

        pdb.set_trace()
    for parent in domain_row[::-1]:
        if pd.isnull(parent):
            continue
        categorized_elements = categorized_elements[parent]

    return categorized_elements


def any_remainings(categorized_df, all_elements: set, skipped):
    return sorted((all_elements - set(categorized_df["category"])) - set(skipped))


def categorise_new_element(
    categorized_elements: dict,
    all_elements: set,
    sanitized_map: dict[str, str],
    make_state: callable,
    parse_response: callable,
    excludes: set = None,
):
    excludes = excludes or set()

    skipped = []
    df = build_dataframe(categorized_elements)
    remainings = any_remainings(df, all_elements, skipped)

    pool = set((d for d in all_elements if d.strip())) - excludes - set(remainings)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    entries = sorted((set(remainings) | all_elements) - excludes)
    embeddings = model.encode(entries)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = model.similarity(embeddings, embeddings)
    similarities = {
        (domain, other): similarities[i, j]
        for i, domain in enumerate(entries)
        for j, other in enumerate(entries)
    }

    LOG_FILE = CFG.dir.log / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        filename=LOG_FILE.with_suffix(f".{PROG}.{CFG.platform.struct}.dbg"),
        level=logging.DEBUG,
        force=True,
    )

    all_distances = get_proposition(
        remainings=remainings,
        pool=pool,
        similarities=similarities,
        k=20,
    )

    client = PLATFORMS[CFG.platform.select]()

    for _ in tqdm.tqdm(list(range(len(all_distances))), desc="Categorizing elements"):
        subject, *propositions = all_distances[0][1:]

        _filename_prefix = "".join(
            [subject[0]] + sorted(set(domain[0] for domain in sorted(propositions)))
        )
        _filename = "_".join(
            [
                _filename_prefix,
                hashlib.sha256(
                    "".join([subject] + sorted(propositions)).encode()
                ).hexdigest(),
            ]
        )

        _make_state = lambda *args, **kwargs: make_state(
            subject,
            propositions,
            sanitized_map,
            *args,
            **kwargs,
        )

        while True:
            try:
                responses = asyncio.run(
                    batch_queries(
                        client,
                        [(None, Path(_filename))],
                        destination=CFG.dir.data
                        / CFG.platform.struct
                        / "queries"
                        / CFG.platform.select,
                        state_cls=_make_state,
                    )
                )
                break

            except InstructorRetryException:
                continue

        for r in responses:
            semantically_equivalents, closest_parent, closest_child, closest_sibling = (
                parse_response(r)
            )

            list(
                _update_sanitized_map(
                    sanitized_map,
                    *(
                        semantically_equivalents
                        + [
                            closest_parent,
                            closest_child,
                            closest_sibling,
                        ]
                    ),
                )
            )

        _propositions_map = {
            sanitized_map[domain]: (i, domain) for i, domain in enumerate(propositions)
        }

        selection = None
        equivalent_matches = []
        parent_matches = []
        child_matches = []
        sibling_matches = []

        for r in responses:
            semantically_equivalents, closest_parent, closest_child, closest_sibling = (
                parse_response(r)
            )

            _equivalent_matches = (
                sorted(
                    _propositions_map[sanitized_map[_el]]
                    for _el in semantically_equivalents
                    if sanitized_map[_el] in _propositions_map
                )
                if len(semantically_equivalents) / len(propositions) <= 0.2
                else []
            )

            if _equivalent_matches:
                equivalent_matches.extend(_equivalent_matches)
                break

            if (
                _parent_match := _propositions_map.get(
                    sanitized_map[closest_parent], None
                )
            ) is not None:
                parent_matches.append(_parent_match)

            if (
                _child_match := _propositions_map.get(
                    sanitized_map[closest_child], None
                )
            ) is not None:
                child_matches.append(_child_match)

            if (
                _sibling_match := _propositions_map.get(
                    sanitized_map[closest_sibling], None
                )
            ) is not None:
                sibling_matches.append(_sibling_match)

        equivalent_match = next(iter(sorted(equivalent_matches)), (math.inf, None))
        parent_match = next(iter(sorted(parent_matches)), (math.inf, None))
        child_match = next(iter(sorted(child_matches)), (math.inf, None))
        sibling_match = next(iter(sorted(sibling_matches)), (math.inf, None))

        if equivalent_match[1]:
            selection = equivalent_match
        else:
            selection = sorted((parent_match, child_match, sibling_match))[0]

        if selection[1] is None:
            skipped.append(subject)
        elif selection in (equivalent_match, parent_match):
            get_category(selection[1], categorized_elements, df)[subject] = {}
        elif selection is child_match:
            parent = df[df["category"] == selection[1]]["parent1"].iloc[0]
            parent_group = get_category(parent, categorized_elements, df)
            chosen_group = parent_group.pop(selection[1])
            parent_group[subject] = {selection[1]: chosen_group}
        elif selection is sibling_match:
            parent = df[df["category"] == selection[1]]["parent1"].iloc[0]
            get_category(parent, categorized_elements, df)[subject] = {}

        df = build_dataframe(categorized_elements)
        remainings = any_remainings(df, all_elements, skipped)

        yield categorized_elements

        pool = set(
            sum(
                (domain_distances[2:] for domain_distances in all_distances[1:]),
                [],
            )
            + ([subject] if subject not in skipped else [])
        )

        all_distances = _update_sorted_propositions(
            all_distances, similarities, skipped, k=20
        )
