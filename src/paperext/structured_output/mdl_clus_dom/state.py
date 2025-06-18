import json
import math
from pathlib import Path
from typing import Generator

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from paperext.config import CFG
from paperext.sanitize_categorization import _dict_heads, _flatten_dict
from paperext.utils import Paper
from paperext.structured_output.mdl_clus_dom.model import (
    FIRST_MESSAGE,
    RETRY_MESSAGE,
    SYSTEM_MESSAGE,
    GenericDomain,
    Response,
)


def _list_children_indices(
    node_index: int, clustering_model: AgglomerativeClustering, pool: set
):
    if node_index >= len(clustering_model.labels_):
        node_index = node_index - len(clustering_model.labels_)

    if node_index not in pool:
        return

    pool.remove(node_index)

    for child_index in clustering_model.children_[node_index]:
        if child_index >= len(clustering_model.labels_):
            yield from _list_children_indices(child_index, clustering_model, pool)

        else:
            yield child_index


def _sort_categories(model: SentenceTransformer, categories: list, **kwargs):
    categories = sorted(set(categories))

    embeddings = model.encode(categories, show_progress_bar=False)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    kwargs = {
        # Default values
        "compute_full_tree": True,
        "linkage": "complete",
        **kwargs,
        # Forced values
        "n_clusters": None,
        "metric": "cosine",
        "distance_threshold": 0,
    }

    # Perform agglomerative clustering

    clustering_model = AgglomerativeClustering(**kwargs)
    clustering_model.fit(embeddings)

    return [
        categories[index]
        for index in _list_children_indices(
            len(clustering_model.children_) - 1,
            clustering_model,
            set(range(len(categories))),
        )
    ]


def _find_min_max_threshold(model: SentenceTransformer, categories: list, **kwargs):
    embeddings = model.encode(categories, show_progress_bar=False)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    kwargs = {
        # Default values
        "compute_full_tree": True,
        "linkage": "complete",
        **kwargs,
        # Forced values
        "n_clusters": None,
        "metric": "cosine",
        "distance_threshold": 0,
    }

    # Perform agglomerative clustering

    clustering_model = AgglomerativeClustering(**kwargs)
    clustering_model.fit(embeddings)

    min_tolerance, max_tolerance = 0.0, 1.0

    for tolerance in (i / 100 for i in range(100, 0, -1)):
        distance_threshold = 1 - math.cos(math.pi * (1 - tolerance))
        count = len(
            clustering_model.distances_[
                clustering_model.distances_ < distance_threshold
            ]
        )

        if count >= len(clustering_model.distances_) - 1:
            min_tolerance = tolerance
            break
        elif count == 0:
            max_tolerance = tolerance

    return min_tolerance, max_tolerance


def cluster_categories(
    model: SentenceTransformer, categories: dict[str:dict], tolerance=0.75, **kwargs
):
    sorted_categories = sorted(categories)
    all_categories = sorted(set(_flatten_dict(categories)))

    embeddings = model.encode(sorted_categories, show_progress_bar=False)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    all_embeddings = model.encode(all_categories, show_progress_bar=False)
    all_embeddings = all_embeddings / np.linalg.norm(
        all_embeddings, axis=1, keepdims=True
    )
    similarities = model.similarity(all_embeddings, all_embeddings)
    similarities = {
        (domain, other): similarities[i, j]
        for i, domain in enumerate(all_categories)
        for j, other in enumerate(all_categories)
    }

    distance_threshold = 1 - math.cos(math.pi * (1 - tolerance))

    kwargs = {
        # Default values
        "compute_full_tree": True,
        "linkage": "complete",
        **kwargs,
        # Forced values
        "n_clusters": None,
        "metric": "cosine",
        "distance_threshold": distance_threshold,
    }

    # Perform agglomerative clustering

    clustering_model = AgglomerativeClustering(**kwargs)
    clustering_model.fit(embeddings)

    cluster_assignment = clustering_model.labels_

    clusters = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clusters:
            clusters[cluster_id] = []

        clusters[cluster_id].append(sorted_categories[sentence_id])

    categorisation = {}

    for cluster in clusters.values():
        scores = []
        for element in cluster:
            score = [
                similarities[(element, other)] for other in cluster if element != other
            ]
            score = sum(score) / (len(score) or 1)
            # for level in range(2):
            #     context_score = [
            #         similarities[(element, other)]
            #         for other in _dict_heads(
            #             categories[element], start_level=level, level=level + 1
            #         )
            #         if element != other
            #     ]
            #     context_score = sum(context_score) / (
            #         len(cluster) * (level + 1) * (len(context_score) or 1)
            #     )
            #     score += context_score
            scores.append((score, element))
        scores.sort(reverse=True)
        general_domain = scores[0][1]
        categorisation[general_domain] = {s[1]: {} for s in scores[1:]}

    return categorisation


class State:
    def __init__(self, paper: Paper, pdf_txt: Path, domains: list):
        self._paper = paper
        self._pdf_txt = pdf_txt
        self._domains = domains

        self.responses: list[Response] = []

    @property
    def categories_refs(self):
        return self._categories_refs

    def format_messages(self) -> Generator[list[dict[str:str]], None, None]:
        sanitized_domains = [
            d.lower().replace("-", " ").replace(" ", "") for d in self._domains
        ]

        _messages = [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": FIRST_MESSAGE.format(
                    "\n".join([f'"{d}"' for d in self._domains])
                ),
            },
        ]

        success = False

        while not success:
            yield _messages[:]
            category = (
                self.responses[-1]
                .extractions.generic_domain.value.lower()
                .replace("-", " ")
                .replace(" ", "")
            )
            rejected = [
                d.lower().replace("-", " ").replace(" ", "")
                for d in self.responses[-1].extractions.rejected_domains
            ]

            if category not in sanitized_domains:
                _messages[1]["content"] = (
                    RETRY_MESSAGE.format(
                        category, "\n".join([f'"{d}"' for d in self._domains])
                    ),
                )
                continue

            _messages[1]["content"] = (
                FIRST_MESSAGE.format("\n".join([f'"{d}"' for d in self._domains])),
            )

            if set(sanitized_domains) - set([category, *rejected]):
                # the whole list of domains should be considered
                continue

            if (
                category in rejected
                and (
                    sum(category == d for d in sanitized_domains)
                    - sum(category == d for d in rejected)
                )
                > 1
            ):
                # category can not be rejected and selected at the same time
                continue

            success = True

    def push_response(self, response: Response):
        self.responses.append(response)

    def get_response_cls(self):
        return Response

    def get_response_model(self):
        return GenericDomain
