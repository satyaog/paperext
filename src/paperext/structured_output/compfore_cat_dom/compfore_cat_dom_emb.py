import argparse
from collections import Counter
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import tempfile
import pickle
import pprint
from typing import Callable, Iterable, Any, Literal, Union

import hdbscan
from matplotlib import pyplot as plt
import numpy as np
import ollama
import openai
from openai import OpenAI
from openai.types.responses.response import Response
from pydantic import BaseModel
from sklearn.manifold import TSNE
import tqdm
import umap

from packaging.version import Version
from paperext.config import CFG, Config
from paperext.sanitize_categorization import (
    _flatten_dict,
    _update_sanitized_map,
    sanitize_categories,
)
from paperext.structured_output import get_struct_module
from paperext.utils import Paper


DOMAINS_TO_IGNORE = [
    "machine learning",
    "deep learning - deep learning theory",
    # "natural language processing",
    "machine learning theory - theoretical machine learning",
    # "deep reinforcement learning - reinforcement learning",
    "neural network training - neural networks",
    # "computational neuroscience",
    # "causal representation learning - representation learning",
    # "multi-modal learning",
    # "multi-task learning",
    # "model optimization - optimization in deep learning",
    # "computer vision - image classification",
    # "meta-learning",
    # "bioinformatics - computational biology",
    # "transfer learning",
    # "model-based reinforcement learning",
    "cognitive neuroscience - neuroscience",
    # "interpretable machine learning",
    "optimization methods",
    "optimization - optimization algorithms",
    # "language modeling - language models",
    # "self-supervised learning",
    # "generative modeling - generative models",
    "artificial intelligence",
    # "neuroimaging",
    "neuroscience-inspired ai",
    "benchmarking - model evaluation",
    # "neural network optimization",
    "markov decision processes",
    "software engineering",
    # "graph neural networks - graph neural networks explainability",
    # "explainable ai - explainable artificial intelligence",
    # "contrastive learning - equilibrium-based methods",
    # "geometric deep learning",
    # "multi-agent reinforcement learning - multi-agent systems",
    # "neurology",
    # "genomics",
    # "adversarial machine learning",
]


@dataclass
class Message:
    type: Literal["system", "user", "assistant"]
    prompt: str
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        return self.prompt.format(*self.args, **self.kwargs)

    def format_message(self) -> dict:
        return {
            "role": self.type,
            "content": self.content,
        }


def openai_prompt(
    client: OpenAI,
    messages: list[Message],
    structured_model: BaseModel = None,
    structured_version: Version = Version("0.0.0"),
    no_parse: bool = False,
    cache_dir: Path = None,
    max_attemps: int = 1,
    check: Callable[[Response], bool] = lambda _: True,
):
    """Generate a prompt for a list of messages and cache the response. If the
    response can be loaded from cache, return it.

    Args:
        client: OpenAI client
        messages: List of messages
        structured_model: Pydantic model to parse the response
        structured_version: Version of the structured model
        no_parse: If True, do not parse the response
        cache_path: Path to cache the response
        max_attemps: Maximum number of attempts
        check: Function to check if the response is valid

    Returns:
        Response from OpenAI
    """

    no_parse = no_parse or not structured_model
    attempt = 0

    # Generate a hash of the messages
    messages_hash = hashlib.sha256(
        json.dumps(
            [(m.prompt, m.args, m.kwargs) for m in messages], sort_keys=True
        ).encode()
    ).hexdigest()

    # If cache_dir is not provided, use the tmp dir
    cache_dir = cache_dir or Path(tempfile.gettempdir()) / "paperext"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check if the response is in cache
    cache_filename = f"{CFG.openai.model}_{messages_hash}"

    if structured_model:
        cache_filename += f"_{str(structured_version)}"

    response = None
    while attempt < max_attemps and (response is None or not check(response)):
        cache_file = cache_dir / f"{cache_filename}_{attempt:02d}.json"
        attempt += 1

        if cache_file.exists():
            response = Response.model_validate_json(cache_file.read_text())
            continue

        # Generate the response
        if no_parse:
            response = client.responses.create(
                model=CFG.openai.model,
                input=[m.format_message() for m in messages],
            )
        else:
            response = client.responses.parse(
                model=CFG.openai.model,
                input=[m.format_message() for m in messages],
                response_model=structured_model,
            )

        # Save the response to cache
        cache_file.write_text(response.model_dump_json())

    # assert check(response), f"Response is not valid: {response}"

    return response


def process_papers_context(
    papers: Iterable[Union[dict, "Paper"]],
    n_queries: int = 1,
) -> dict[str, dict[str, str | list[dict]]]:
    """Process papers to extract and sanitize domains with their justifications.

    Args:
        papers: Iterable of paper dictionaries or Paper objects
        n_queries: Number of queries to use for each paper. If 0, uses all queries.

    Returns:
        Dictionary of paper id to context
    """
    # First collect all paper-abstract-domain-justifications
    papers_context: dict[str, dict[str, str | list[dict]]] = {}

    for paper in papers:
        if not isinstance(paper, Paper):
            paper = Paper(paper)

        assert (
            paper._paper_id not in papers_context
        ), f"Paper {paper._paper_id} already processed"

        papers_context[paper._paper_id] = {
            "abstract": paper._paper["abstract"].strip() or None,
            "analyses": [],
        }

        for query in paper.queries[-n_queries:]:
            analysis = (
                get_struct_module(CFG.platform.struct)
                .model.Response.model_validate_json(query.read_text())
                .analysis
            )

            _domains = [
                (domain.name.value, domain.name.justification)
                for domain in [
                    analysis.primary_research_field,
                    *analysis.sub_research_fields,
                ]
            ]

            papers_context[paper._paper_id]["analyses"].append({"domains": _domains})

    sanitized_map = {}
    _update_sanitized_map(
        sanitized_map,
        *[
            domain
            for paper_context in papers_context.values()
            for analysis in paper_context["analyses"]
            for domain, _ in analysis["domains"]
        ],
    )

    # Build map of sanitized domain to list of justifications
    for analysis in [
        analysis
        for paper_context in papers_context.values()
        for analysis in paper_context["analyses"]
    ]:
        domains_context = analysis["domains"]
        analysis["domains"] = {}
        for domain, justification in domains_context:
            analysis["domains"].setdefault(sanitized_map[domain], set()).add(
                justification
            )

    return papers_context


def compute_paper_embeddings(
    paper_context: dict[str, str | list[dict]], model_name: str = None
) -> dict[str, np.ndarray | list[dict[str, list[np.ndarray]]]]:
    """Get embeddings for a paper context.

    Args:
        paper_context: Dictionary of paper context
        model_name: Name of the ollama model to use for embeddings. If None, uses the model from config.

    Returns:
        Dictionary of paper id to embeddings
    """
    # Configure ollama client
    client = ollama.Client(CFG.ollama.url)
    model_name = model_name or CFG.ollama.model

    paper_embeddings = {"abstract": None, "analyses": []}

    # Generate embeddings using ollama
    paper_embeddings["abstract"] = (
        np.array(
            client.embeddings(model=model_name, prompt=paper_context["abstract"])[
                "embedding"
            ]
        )
        if paper_context["abstract"]
        else None
    )

    for analysis in paper_context["analyses"]:
        _domains = {}

        for domain, justifications in analysis["domains"].items():
            justifications = [
                np.array(client.embeddings(model=model_name, prompt=j)["embedding"])
                for j in justifications
            ]
            assert (
                domain not in _domains
            ), f"Domain {domain} already processed for paper {paper_context['paper_id']}"
            _domains[domain] = justifications

        paper_embeddings["analyses"].append({"domains": _domains})

    return paper_embeddings


def process_domain_context(
    papers: Iterable[Union[dict, "Paper"]],
    n_queries: int = 1,
    context_type: Literal["justification", "abstract"] = "justification",
) -> dict[str, list[str]]:
    """Process papers to extract and sanitize domains with their justifications.

    Args:
        papers: Iterable of paper dictionaries or Paper objects
        n_queries: Number of queries to use for each paper. If 0, uses all queries.

    Returns:
        Dictionary of domain to justifications
    """
    # First collect all domain-justification pairs
    domains_context = []

    # Extract domains and their justifications
    for paper in papers:
        if not isinstance(paper, Paper):
            paper = Paper(paper)

        for query in paper.queries[-n_queries:]:
            analysis = (
                get_struct_module(CFG.platform.struct)
                .model.Response.model_validate_json(query.read_text())
                .analysis
            )

            for domain in [
                analysis.primary_research_field,
                *analysis.sub_research_fields,
            ]:
                match context_type:
                    case "justification":
                        context = domain.name.justification
                    case "abstract":
                        context = paper._paper["abstract"]
                context = context.strip()
                if not context:
                    continue
                domains_context.append((domain.name.value, context))

    # Sanitize domain names
    sanitized_map = {}
    domains = _update_sanitized_map(sanitized_map, *[d for d, _ in domains_context])

    # Build map of sanitized domain to list of justifications
    domain_to_context: dict[str, set[str]] = {}
    for domain, context in zip(domains, [context for _, context in domains_context]):
        domain_to_context.setdefault(domain, set()).add(context)

    return domain_to_context


def merge_similar_domains(
    domain_to_embeddings: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    """Merge similar domains using cosine similarity.

    Args:
        domain_to_embeddings: Dictionary of domain to embeddings

    Returns:
        Dictionary of domain to embeddings
    """

    domain_to_embeddings = copy.deepcopy(domain_to_embeddings)

    def _cosine_distance(embedding: np.ndarray, other: np.ndarray) -> float:
        return 1 - np.dot(embedding, other) / (
            np.linalg.norm(embedding) * np.linalg.norm(other)
        )

    cosine_distances = {
        domain: sorted(
            (
                _cosine_distance(
                    np.mean(
                        embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True),
                        axis=0,
                    ),
                    np.mean(
                        other_embeddings
                        / np.linalg.norm(other_embeddings, axis=1, keepdims=True),
                        axis=0,
                    ),
                ),
                other_domain,
            )
            for other_domain, other_embeddings in domain_to_embeddings.items()
        )[1:]
        for domain, embeddings in tqdm.tqdm(
            domain_to_embeddings.items(), desc="Computing cosine distances"
        )
    }

    MAX_DISTANCE = 1 - math.cos(math.pi * 30 / 180)

    for domain, distances in cosine_distances.items():
        # Skip domains that have been merged
        if domain not in domain_to_embeddings:
            continue

        distance, other = distances[0]

        # Skip domains that are too far apart
        if distance > MAX_DISTANCE:
            continue

        # Skip domains for which there is no other domain closer than the two
        # domains
        if domain != cosine_distances[other][0][1]:
            continue

        # Merge domains
        domain_embeddings = domain_to_embeddings.pop(domain)
        other_embeddings = domain_to_embeddings.pop(other)
        domain_to_embeddings[" - ".join(sorted([domain, other]))] = np.mean(
            [
                (domain_embeddings + other_embeddings)
                / np.linalg.norm(
                    (domain_embeddings + other_embeddings), axis=1, keepdims=True
                )
            ],
            axis=0,
        )

    return domain_to_embeddings


def get_domains_embeddings(
    papers_embeddings: dict[str, dict[str, str | list[dict]]],
    context_type: Literal["justification", "abstract"] = "justification",
) -> dict[str, np.ndarray]:
    """Extract domains and their embeddings from papers embeddings.

    Args:
        papers_embeddings: Dictionary of paper id to embeddings

    Returns:
        Dictionary of domain to embeddings
    """
    # Generate embeddings using ollama
    domain_to_embeddings: dict[str, list[np.ndarray]] = {}
    for paper_embeddings in papers_embeddings.values():
        for domain, justifications in [
            (domain, justifications)
            for analysis in paper_embeddings["analyses"]
            for domain, justifications in analysis["domains"].items()
        ]:
            match context_type:
                case "justification":
                    context = justifications
                case "abstract":
                    context = [paper_embeddings["abstract"]]

            context = [el for el in context if el is not None]

            domain_to_embeddings.setdefault(domain, []).extend(context)

    # merge similar domains
    domain_to_embeddings = merge_similar_domains(domain_to_embeddings)

    # merge each domain embeddings using the mean of the normalized embeddings
    domain_to_embedding: dict[str, np.ndarray] = {
        domain: np.mean(
            embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True), axis=0
        )
        for domain, embeddings in domain_to_embeddings.items()
    }

    domain_to_embedding = {
        domain: embedding
        for domain, embedding in domain_to_embedding.items()
        if domain not in DOMAINS_TO_IGNORE
    }

    return domain_to_embedding


def build_cluster_tree(
    clusterer: hdbscan.HDBSCAN,
    domains: list[str],
) -> dict[str, Any]:
    """Build a hierarchical tree structure from HDBSCAN clustering results.

    Args:
        clusterer: HDBSCAN clustering model
        domains: List of domain names

    Returns:
        Dictionary representing the hierarchical tree structure
    """
    # Get the clusters graph edges
    graph_edges = clusterer.condensed_tree_.to_pandas()

    # Build the tree structure
    tree = {}

    def format_node_name(
        node_name: str, lambda_val: float = -1.0, lambda_mean: float = -1.0
    ) -> str:
        return node_name

    def format_node_lambda(
        node_name: str, lambda_val: float = -1.0, lambda_mean: float = -1.0
    ) -> str:
        lambda_annotation = " \ ".join(
            [
                f"lambda={lambda_val:.2f}",
                *([f"{lambda_mean:.2f}"] if lambda_mean >= 0 else []),
            ]
        )
        return f"{node_name} ({lambda_annotation})"

    # Helper function to recursively build the tree
    def _build_tree(node_id: int, lambda_val: float = -1.0) -> dict[str, dict]:
        # If this is a leaf node
        if node_id < len(domains):
            return {format_node_name(domains[node_id], lambda_val): {}}

        # Get children of this node
        children = graph_edges[graph_edges["parent"] == node_id][
            ["child", "lambda_val"]
        ]

        node = {}

        for child, lambda_val in children.values.tolist():
            child_node = _build_tree(int(child), lambda_val)

            # assert that the child node is not already in the parent node
            assert not (set(child_node.keys()) & set(node))

            node.update(child_node)

        return {
            format_node_name(
                f"cluster_{node_id - len(domains)}",
                lambda_val,
                children["lambda_val"].mean(),
            ): node
        }

    # Find the root node
    root_id = len(domains)
    tree = _build_tree(root_id)

    return tree


def identify_cluster_labels(tree: dict[str, Any]) -> dict[str, str]:
    """Generate labels for each cluster in the tree.

    Args:
        tree: Dictionary representing the hierarchical tree structure

    Returns:
        Dictionary of cluster to label
    """
    # Starting from the leafs, iteratively generate labels for each cluster
    tree = copy.deepcopy(tree)

    # Helper function to recursively generate labels for each cluster
    def _generate_cluster_label(tree: dict[str, Any]) -> dict[str, str]:
        for node_id, children in sorted(tree.items()):
            label = _generate_cluster_label(children)
            if not label:
                continue
            assert node_id.startswith("cluster_")
            tree[label] = tree.pop(node_id)

        if tree:
            # Query OpenAI to generate a label for the cluster
            client = OpenAI()
            messages = [
                #                 Message(
                #                     type="system",
                #                     prompt="""You are a Deep Learning Research Domain ontology expert that identifies the most generic Deep Learning Research Domain which encompasses the best a cluster of Deep Learning Research Domains.
                # - The Deep Learning Research Domain must exists in the provided cluster of Deep Learning Research Domains.
                # - The Deep Learning Research Domain must not be too generic like 'Deep Learning', 'Machine Learning', 'Artificial Intelligence', 'Intelligent Systems', 'Cognitive Computing' or 'Cognitive Systems'.
                # Return only the Deep Learning Research Domain.""",
                #                 ),
                #                 Message(
                #                     type="system",
                #                     prompt="""You are a Deep Learning Research Domain ontology expert that identifies the most generic Deep Learning Research Domain name from a cluster of Deep Learning Research Domains or generates one if none is found.
                # - The Deep Learning Research Domain name should not contain 'Deep Learning', 'Machine Learning', 'Intelligent Systems', 'Cognitive Computing', 'Cognitive Systems' or 'and'.
                # - The Deep Learning Research Domain name should not contain acronyms or abbreviations.
                # - The Deep Learning Research Domain name should be generic enough to encompass the cluster, but not too generic.
                # - The Deep Learning Research Domain name should be in English.
                # Return only the Deep Learning Research Domain name.""",
                #                 ),
                Message(
                    type="system",
                    prompt="""You are a Deep Learning Research Domain ontology expert that generates the most generic Deep Learning Research Domain name for a cluster of Deep Learning Research Domains.

- The Deep Learning Research Domain name should not contain 'Deep Learning', 'Machine Learning', 'Intelligent Systems', 'Cognitive Computing', 'Cognitive Systems' or 'and'.
- The Deep Learning Research Domain name should not contain acronyms or abbreviations.
- The Deep Learning Research Domain name should be generic enough to encompass the cluster, but not too generic.

Return only the Deep Learning Research Domain name.""",
                ),
                Message(
                    type="user",
                    prompt="The cluster of Deep Learning Research Domains:\n\n{}",
                    args=("\n".join([f"- {k}" for k in sorted(tree.keys())]),),
                ),
            ]
            response = openai_prompt(
                client,
                messages,
                max_attemps=3,
                # check=lambda r: (
                #     r.output_text.lower().strip().strip("*\"'").strip()
                #     in [k.lower() for k in tree]
                # ),
                check=lambda r: (
                    r.output_text.lower().strip().strip("*\"'").strip()
                    not in [k.lower() for k in tree]
                ),
            )
            return response.output_text.lower().strip().strip("*\"'").strip()

        return None

    with Config.push():
        CFG.openai.model = "gpt-4o"
        _generate_cluster_label(tree)

    return tree


def cluster_domains(
    domain_to_embedding: dict[str, np.ndarray], **kwargs
) -> hdbscan.HDBSCAN:
    """Perform HDBSCAN clustering on domain embeddings.

    Args:
        domain_to_embeddings: Dictionary of domain to embeddings

    Returns:
        HDBSCAN clustering model
    """
    embeddings = np.array(list(domain_to_embedding.values()))
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    reducer = umap.UMAP(
        # n_neighbors=15,
        n_neighbors=len(domain_to_embedding) // 100,
        # min_dist=0.0,
        # n_components=50,
        n_components=min(len(domain_to_embedding) // 10, 100),
        metric="euclidean",
        random_state=42,
    )
    embeddings = reducer.fit_transform(embeddings)
    kwargs = {
        "metric": "euclidean",
        "cluster_selection_method": "leaf",
        # "min_cluster_size": len(domain_to_embeddings) // 100,
        **kwargs,
    }
    clusterer = hdbscan.HDBSCAN(**kwargs)
    clusterer.fit(embeddings)
    return clusterer


def plot_clusters(
    clusterer: hdbscan.HDBSCAN,
    domain_to_embeddings: dict[str, np.ndarray],
) -> None:
    """Plot cluster visualizations using HDBSCAN condensed tree and TSNE.

    Args:
        clusterer: HDBSCAN clustering model
        domain_to_embeddings: Dictionary of domain to embeddings
    """
    # Enable interactive mode
    plt.ion()

    # Plot HDBSCAN condensed tree
    plt.figure(figsize=(12, 8))
    clusterer.condensed_tree_.plot(select_clusters=True)
    plt.title("HDBSCAN Condensed Tree")
    plt.draw()

    # Create TSNE visualization
    embeddings = np.array(list(domain_to_embeddings.values()))
    labels = clusterer.labels_
    probabilities = clusterer.probabilities_

    # Apply TSNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot TSNE results
    plt.figure(figsize=(12, 8))

    # Get unique labels and count clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Use viridis colormap
    cmap = plt.cm.get_cmap("viridis", n_clusters + 1)

    # Create a color list for each unique label
    colors = []
    for label in labels:
        if label == -1:  # Noise points in grey
            colors.append("grey")
        else:
            colors.append(cmap(label))

    # Plot points with cluster colors
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=colors,
        alpha=0.6,
        s=100,
        edgecolors="white",
        linewidth=0.5,
    )

    # Add a second scatter plot for probability visualization
    # Points with high probability will be more opaque
    plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=colors,
        alpha=probabilities * 0.3,  # Scale probabilities for better visibility
        s=100,
        marker="o",
        edgecolor="none",
    )

    plt.colorbar(scatter, label="Cluster")
    plt.title(
        f"T-SNE Visualization of Domain Clusters ({n_clusters} clusters)\n(Opacity indicates cluster probability)"
    )
    plt.draw()

    # Keep plots open until user closes them
    plt.show(block=True)


def main(argv: list[str] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paperoni",
        nargs="*",
        type=Path,
        help="Paperoni json report of papers to analyse",
    )
    parser.add_argument(
        "--context-type",
        choices=["justification", "abstract"],
        default="justification",
        help="Type of context to use for domain extraction (justification or abstract)",
    )
    parser.add_argument(
        "--model",
        default=CFG.ollama.model,
        help="Name of the ollama model to use for embeddings (overrides config)",
    )
    parser.add_argument(
        "--clustering-algorithm",
        choices=["hdbscan"],
        default="hdbscan",
        help="Clustering algorithm to use",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        type=Path,
        default="-",
        help="Path to output JSON file with clustering results",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot cluster visualizations",
    )
    options = parser.parse_args(argv)

    # args hash excluding output options
    def _hash(option: Any) -> str:
        if isinstance(option, str):
            return option
        # if dict, hash each value
        elif isinstance(option, dict):
            return {k: _hash(v) for k, v in option.items()}
        # if iterable, hash each item
        elif isinstance(option, Iterable):
            return tuple(map(_hash, option))
        else:
            return str(option)

    args_hash = hashlib.sha256(
        json.dumps(
            {
                k: _hash(v)
                for k, v in vars(options).items()
                if k not in ["context_type", "out", "plot"]
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    # domain_to_embeddings_cache = Path(f"domain_to_embeddings_{args_hash}.pkl")
    papers_embeddings_cache = Path(f"papers_embeddings_{args_hash}.pkl")

    # Load papers
    papers = []
    for papers_json_path in options.paperoni:
        papers.extend(json.loads(Path(papers_json_path).read_text()))

    # reload papers_embeddings pickle file if it exists
    if papers_embeddings_cache.exists():
        with papers_embeddings_cache.open("rb") as f:
            papers_embeddings = pickle.load(f)

    else:
        # Compute papers domains embeddings
        papers_context = process_papers_context(
            tqdm.tqdm(papers, desc=f"Processing papers", unit="paper(s)"),
            n_queries=0,
        )

        papers_embeddings = {
            paper_id: compute_paper_embeddings(paper_context, options.model)
            for paper_id, paper_context in tqdm.tqdm(
                papers_context.items(),
                desc="Generating papers embeddings",
                unit="paper(s)",
            )
        }

        # save papers_embeddings to pickle file
        with papers_embeddings_cache.open("wb") as f:
            pickle.dump(papers_embeddings, f)

    domain_to_embedding = get_domains_embeddings(
        papers_embeddings, context_type=options.context_type
    )

    # for metric in hdbscan.dist_metrics.METRIC_MAPPING.keys():
    # for alpha in np.linspace(0.0, 1.0, 10):
    for metric in ["euclidean"]:
        try:
            # Perform clustering
            clusterer = cluster_domains(domain_to_embedding, metric=metric)
        except (TypeError, ValueError):
            continue

        # Plot cluster visualizations
        if options.plot:
            plot_clusters(clusterer, domain_to_embedding)

    # Build cluster tree
    cluster_tree = build_cluster_tree(clusterer, list(domain_to_embedding.keys()))

    # Generate label for each clusters in tree
    # cluster_tree = identify_cluster_labels(cluster_tree)

    # # count the number of occurences of each key in the cluster tree
    # key_counter = Counter()
    # for key in _flatten_dict(cluster_tree):
    #     key_counter[key] += 1

    assert len(cluster_tree) == 1
    cluster_tree = cluster_tree.pop(list(cluster_tree.keys())[0])

    cluster_tree = sanitize_categories(cluster_tree)

    # save cluster tree to json file
    (options.out.write_text if str(options.out) != "-" else print)(
        json.dumps(cluster_tree, indent=2, sort_keys=True, ensure_ascii=False)
    )

    # # Prepare results
    # results = {
    #     "domains": list(domain_to_embeddings.keys()),
    #     "labels": clusterer.labels_.tolist(),
    #     "probabilities": clusterer.probabilities_.tolist(),
    #     "cluster_persistence": (
    #         clusterer.cluster_persistence_.tolist()
    #         if hasattr(clusterer, "cluster_persistence_")
    #         else None
    #     ),
    #     "cluster_tree": cluster_tree,
    # }

    # # Save results
    # if options.out:
    #     options.out.write_text(json.dumps(results, indent=2))
    # else:
    #     print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
