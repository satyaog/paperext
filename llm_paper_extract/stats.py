from __future__ import annotations

import argparse
import json
import os
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Callable

import pandas as pd
import plotly.graph_objects as go

# Load json of papers to select
# Load analysis of papers based on the json
# Log paper analysis missing

# Compute stats:
# - Number of papers
# - Papers per conference
# - Papers per research domain
# - Papers per models
# - Papers per library


DEFAULT_KEYS = (
    "pdf",
    "arxiv",
    "openreview",
    "pubmed",
    "acl",
    "mag",
    "html",
    "doi.abstract",
    "semantic_scholar.abstract",
)


def do_we_have(papers, analysis, title):
    def clean_title(title):
        return title.lower().replace(" ", "")

    title = title.lower().replace(" ", "")
    if any(clean_title(paper["title"]) == title for paper in papers):
        print("Paper in json")
    else:
        print("Paper not in json")

    if (analysis["attrs"]["title"].str.lower().str.replace(" ", "") == title).any():
        print("Analysis found")
    else:
        print("Analysis not found")


def get_counts_per_source(df, keys=DEFAULT_KEYS):
    df["strlinks"] = df["links"].astype(str)
    rest = df
    articles = {}
    rest_mask = df["links"].str.len() > 0

    for key in keys:
        mask = df["strlinks"].str.contains(f"'type': '{key}")
        articles[key] = df[rest_mask & mask]
        rest_mask = rest_mask & ~mask
        rest = df[~rest_mask]

    return articles, rest


# See if we can map models to sizes.

types = defaultdict(int)
links_found = defaultdict(list)


def normalize_paper_type(paper_type):
    paper_type = paper_type.lower()
    paper_type = (
        paper_type.replace("study", "").replace("research", "").strip("_").strip(" ")
    )

    return paper_type


def normalize_role(role):
    return role.lower()


def normalize_model_name(model):
    model = model.lower()
    replacements = [
        ("resnet18", "resnet-18"),
        ("resnet50", "resnet-50"),
        ("(cnn)", ""),
        ("(ppo)", ""),
        ("(gnn)", ""),
        ("(svm)", ""),
        ("(vision transformer)", ""),
        ("(vit)", ""),
        ("convolutional neural network", "cnn"),
        ("proximal policy optimization", "ppo"),
        ("graph neural network", "gnn"),
        ("vision transformer", "vit"),
        ("support vector machine", "svm"),
    ]
    for old, new in replacements:
        model = model.replace(old, new).strip(" ")
    return model


def normalize_dataset_name(dataset):
    dataset = dataset.lower()
    replacements = [
        ("cifar10", "cifar-10"),
        ("cifar100", "cifar-100"),
    ]
    for old, new in replacements:
        dataset = dataset.replace(old, new)
    return dataset


def normalize_library_name(library):
    library = library.lower()
    replacements = [
        (" (pyg)", ""),
        ("pyg", "pytorch geometric"),
        ("huggingface transformers", "transformers"),
    ]
    for old, new in replacements:
        library = library.replace(old, new)
    return library


def find_analysis(paper, folder: Path, selector: Callable) -> Path | None:
    for link in paper["links"]:
        paper_path = folder / f"{link['link']}_00.json"
        if (
            os.path.exists(paper_path)
            and paper_path not in links_found[paper["paper_id"]]
        ):
            if selector(paper_path):
                types[link["type"]] += 1
                links_found[paper["paper_id"]].append(paper_path)
                # return paper_path

    paper_path = folder / f"{paper['paper_id']}_00.json"
    if os.path.exists(paper_path) and paper_path not in links_found[paper["paper_id"]]:
        if selector(paper_path):
            links_found[paper["paper_id"]].append(paper_path)

    if links_found[paper["paper_id"]]:
        return links_found[paper["paper_id"]][0]

    return None


def load_single_analysis(paper, folder: Path, selector: Callable):
    paper_path = find_analysis(paper, folder, selector)
    if paper_path is None:
        return None

    paper_analysis = json.load(open(paper_path))
    extraction = paper_analysis["extractions"]

    # TODO: Normalize some fields here.

    rval = {}

    rval["attrs"] = [
        {
            "paper": paper_analysis["paper"],
            "paper_id": paper["paper_id"],
            "title": extraction["title"]["value"],
            "type": normalize_paper_type(extraction["type"]["value"]),
            "primary_research_field": extraction["primary_research_field"]["name"][
                "value"
            ],
            "research_fields": [extraction["primary_research_field"]["name"]["value"]]
            + sum(
                [
                    [field["name"]["value"]] + field["aliases"]
                    for field in extraction["sub_research_fields"]
                ],
                [],
            ),
        }
    ]
    rval["models"] = [
        {
            "name": normalize_model_name(model["name"]["value"]),
            "aliases": [normalize_model_name(alias) for alias in model["aliases"]],
            "is_contributed": model["is_contributed"]["value"],
            "is_executed": model["is_executed"]["value"],
            "is_compared": model["is_compared"]["value"],
            "referenced_paper_title": model["referenced_paper_title"]["value"],
            "referenced_paper_justification": model["referenced_paper_title"][
                "justification"
            ],
        }
        for model in extraction["models"]
    ]
    rval["datasets"] = [
        {
            "name": normalize_dataset_name(dataset["name"]["value"]),
            "aliases": [normalize_dataset_name(alias) for alias in dataset["aliases"]],
            "role": normalize_role(dataset["role"]),
            "referenced_paper_title": dataset["referenced_paper_title"]["value"],
            "referenced_paper_justification": dataset["referenced_paper_title"][
                "justification"
            ],
        }
        for dataset in extraction["datasets"]
    ]
    rval["libraries"] = [
        {
            "name": normalize_library_name(library["name"]["value"]),
            "aliases": [normalize_library_name(alias) for alias in library["aliases"]],
            "role": normalize_role(library["role"]),
        }
        for library in extraction["libraries"]
    ]

    for key in ["models", "datasets", "libraries"]:
        for item in rval[key]:
            assert not (set(item.keys()) & set(rval.keys()))
            item.update(rval["attrs"][0])

    return rval


def load_analysis(papers, folder: Path, selector: Callable | None = None):
    if selector is None:
        selector = lambda x: True
    missing_papers = []
    papers_analysis = defaultdict(list)
    for paper in papers:
        extraction = load_single_analysis(paper, folder, selector)
        if extraction is None:
            print("missing", paper["paper_id"])
            missing_papers.append(paper)
            continue

        for key in extraction:
            papers_analysis[key].extend(extraction[key])

    papers_analysis = {
        key: pd.DataFrame(papers_analysis[key]) for key in papers_analysis
    }

    return papers_analysis, pd.DataFrame(missing_papers)


def find_item(tree, item):
    if item in tree:
        return tree[item]

    for key in tree:
        mapping = find_item(tree[key], item)
        if mapping is not None:
            return mapping


def build_subtree_mapping(tree, item, _root_level=True):

    if item is not None:
        subtree = find_item(tree, item)
    else:
        subtree = tree

    if subtree is None:
        return {}

    mapping = {}
    if _root_level:
        mapping[item] = f"{item} (Other)"

    for key in subtree:
        mapping[key] = key
        if subtree[key]:
            submapping = build_subtree_mapping(subtree, key, _root_level=False)
            assert submapping
            submapping = {k: key for k in submapping}
            duplicates = set(mapping.keys()) & set(submapping.keys())
            if duplicates:
                print("warning, duplicates", duplicates)
            # assert not duplicates, duplicates
            mapping.update(submapping)

    return mapping


def get_papers_stats(analysis, category_trees, categories_selected, k=None):

    selectors = {
        "multi-domains": get_papers_multi_domain,
        "domains": get_papers_per_domain,
        "models": get_papers_per_models,
    }
    selected_papers = analysis
    for i, (info_type, category) in enumerate(categories_selected):
        selected_papers, dropped_papers = selectors[info_type](
            selected_papers,
            category_trees[info_type.replace("multi-", "")],
            category,
            replace_labels=i + 1 == len(categories_selected),
        )

    stats = {"domains": get_paper_domains_stats, "models": get_paper_model_stats}[
        info_type.replace("multi-", "")
    ](analysis, selected_papers, dropped_papers)

    if k is not None:
        kstats = stats[-k:]
        kstats.loc["Other"] = stats[:-k].sum(axis=0)
        stats = kstats

    return stats


def get_paper_domains_stats(analysis, selected_papers, dropped_papers):

    selected_papers["attrs"] = (
        selected_papers["attrs"]
        .explode("research_fields")
        .rename(columns={"research_fields": "name"})
    )

    return _get_paper_stats(
        analysis,
        selected_papers,
        dropped_papers,
        "attrs",
    )


def get_paper_model_stats(analysis, selected_papers, dropped_papers):

    return _get_paper_stats(analysis, selected_papers, dropped_papers, "models")


def _get_paper_stats(analysis, selected_papers, dropped_papers, info_type):

    counts = (selected_papers[info_type].groupby("name")["title"].count()).to_frame(
        "counts"
    )

    counts.sort_values("counts", ascending=True, inplace=True)

    all_papers_in_parent_domain = (
        selected_papers["attrs"]["paper_id"].nunique()
        + dropped_papers["attrs"]["paper_id"].nunique()
    )
    counts.loc["Other"] = [dropped_papers[info_type]["paper_id"].nunique()]
    # assert counts["counts"].sum() == all_papers_in_parent_domain, (
    #     counts["counts"].sum(),
    #     all_papers_in_parent_domain,
    # )

    counts["global_prop"] = counts["counts"] / analysis["attrs"].shape[0]
    counts["domain_prop"] = counts["counts"] / all_papers_in_parent_domain

    return counts


def get_papers_multi_domain(analysis, domains_tree, domains, replace_labels=True):

    multi_domain_papers = analysis
    single_domain_papers = {}
    for domain in domains:
        selected_papers, dropped_papers = get_papers_per_domain(
            analysis, domains_tree, domain, replace_labels=replace_labels
        )
        multi_domain_papers = intersect_papers(multi_domain_papers, selected_papers)
        single_domain_papers = join_papers(
            analysis, single_domain_papers, dropped_papers
        )

    return multi_domain_papers, single_domain_papers


def get_papers_per_domain(analysis, domains, domain, replace_labels=True):
    domains_mapping = build_subtree_mapping(domains, domain)

    attrs = (
        analysis["attrs"]
        .explode("research_fields")
        .drop_duplicates(["paper_id", "research_fields"])
    )
    attrs["research_fields"] = attrs["research_fields"].map(domains_mapping)
    attrs.dropna(subset=["research_fields"], inplace=True)
    attrs.drop_duplicates(["paper_id", "research_fields"], inplace=True)

    selected_papers, dropped_papers = select_papers(analysis, attrs["paper_id"])
    if replace_labels:
        if not attrs.empty:
            aggregate = {str(name): "first" for name in attrs.columns}
            aggregate["research_fields"] = list
            selected_papers["attrs"] = attrs.groupby(["paper_id"]).agg(aggregate)
        else:
            selected_papers["attrs"] = attrs

    # return analysis.groupby("paper_id").first().sort_values(by=["paper_id"])
    return selected_papers, dropped_papers


def get_papers_per_models(analysis, models, model, replace_labels=True):
    models_mapping = build_subtree_mapping(models, model)

    models_mapped = analysis["models"].copy()
    models_mapped["name"] = analysis["models"]["name"].map(models_mapping)
    models_mapped.dropna(subset=["name"], inplace=True)
    models_mapped.drop_duplicates(subset=["paper_id", "name"], inplace=True)

    selected_papers, dropped_papers = select_papers(analysis, models_mapped["paper_id"])
    if replace_labels:
        selected_papers["models"] = models_mapped

    return selected_papers, dropped_papers


def select_papers(analysis, ids):
    selected_papers = {}
    dropped_papers = {}
    for key in analysis:
        is_in = analysis[key]["paper_id"].isin(ids)
        selected_papers[key] = analysis[key][is_in]
        dropped_papers[key] = analysis[key][~is_in]

        assert not (
            set(selected_papers[key]["paper_id"]) & set(dropped_papers[key]["paper_id"])
        )

    return selected_papers, dropped_papers


def print_titles(analysis, ids):
    print(analysis["attrs"][analysis["attrs"]["paper_id"].isin(ids)]["title"])


def join_papers(analysis, papers1, papers2):
    if papers1:
        papers1_ids = set(papers1["attrs"]["paper_id"].unique())
    else:
        papers1_ids = set()

    if papers2:
        papers2_ids = set(papers2["attrs"]["paper_id"].unique())
    else:
        papers2_ids = set()

    paper_ids = papers1_ids | papers2_ids
    return select_papers(analysis, paper_ids)[0]


def intersect_papers(papers1, papers2):
    paper_ids = set(papers1["attrs"]["paper_id"].unique()) & set(
        papers2["attrs"]["paper_id"].unique()
    )
    return select_papers(papers1, paper_ids)[0]


def build_graph(
    analysis,
    category_trees,
    base_abstract_topics,
    application_domains,
    models,
    n_titles=20,
):
    # get papers per domain
    # then map them to application_domains
    # then map them to models
    nodes = (
        application_domains
        + ["Other applications", "No applications"]
        + base_abstract_topics
        + ["Other domains", "No domains"]
        + models
        + ["Other models", "No models"]
    )
    nodes_custom_data = {k: k for k in nodes}
    links = []

    def build_link(node1, node2, value, label, customdata="", reversed=False):
        return {
            "source": nodes.index(node1 if not reversed else node2),
            "target": nodes.index(node2 if not reversed else node1),
            "value": value,
            "label": label,
            "customdata": customdata,
        }

    def build_subgraph(
        root,
        other_label,
        no_label,
        selected_papers,
        tree,
        function,
        categories,
        reversed=False,
    ):
        mapping = build_subtree_mapping(tree, None)
        mapping = {category: mapping[category] for category in categories}

        subgraph_links = []
        remaining_papers_from_subgraph = selected_papers
        for category in categories:
            papers_per_category, dropped_papers_from_category = function(
                selected_papers, tree, category, replace_labels=False
            )
            if papers_per_category["attrs"].shape[0]:
                subgraph_links.append(
                    build_link(
                        root,
                        category,
                        papers_per_category["attrs"].shape[0],
                        "",
                        "<br />".join(papers_per_category["attrs"]["title"][:n_titles]),
                        reversed=reversed,
                    )
                )
            remaining_papers_from_subgraph = intersect_papers(
                remaining_papers_from_subgraph,
                dropped_papers_from_category,
            )

        if remaining_papers_from_subgraph["attrs"].shape[0]:
            unlabeled_papers = remaining_papers_from_subgraph
            labeled_remaining_papers = dict()

            for category in set(mapping.values()):
                papers_per_category, dropped_papers_from_category = function(
                    remaining_papers_from_subgraph, tree, category, replace_labels=False
                )

                unlabeled_papers = intersect_papers(
                    unlabeled_papers, dropped_papers_from_category
                )

                labeled_remaining_papers = join_papers(
                    remaining_papers_from_subgraph,
                    labeled_remaining_papers,
                    papers_per_category,
                )

            if unlabeled_papers["attrs"].shape[0]:
                subgraph_links.append(
                    build_link(
                        root,
                        no_label,
                        unlabeled_papers["attrs"].shape[0],
                        "",
                        "<br />".join(unlabeled_papers["attrs"]["title"][:n_titles]),
                        reversed=reversed,
                    )
                )

            if labeled_remaining_papers["attrs"].shape[0]:
                subgraph_links.append(
                    build_link(
                        root,
                        other_label,
                        labeled_remaining_papers["attrs"].shape[0],
                        "",
                        "<br />".join(
                            labeled_remaining_papers["attrs"]["title"][:n_titles]
                        ),
                        reversed=reversed,
                    )
                )

        return subgraph_links

    def build_node_customdata(selected_papers, category):
        domain_stats = get_papers_stats(
            selected_papers,
            category_trees,
            category,
            k=20,
        )
        return "<br />".join(
            [
                f"{count: 3d}: {name}"
                for (name, count) in domain_stats[["counts"]].itertuples()
            ]
        )

    remaining_papers = analysis
    for base_abstract_topic in base_abstract_topics:

        selected_papers, dropped_papers = get_papers_per_domain(
            analysis,
            category_trees["domains"],
            base_abstract_topic,
            replace_labels=False,
        )

        nodes_custom_data[base_abstract_topic] = build_node_customdata(
            selected_papers, [("domains", base_abstract_topic)]
        )

        links += build_subgraph(
            base_abstract_topic,
            "Other applications",
            "No applications",
            selected_papers,
            category_trees["domains"],
            get_papers_per_domain,
            application_domains,
            reversed=True,
        )
        links += build_subgraph(
            base_abstract_topic,
            "Other models",
            "No models",
            selected_papers,
            category_trees["models"],
            get_papers_per_models,
            models,
        )

        # remaining_papers_from_application = dict()
        # for application_domain in application_domains:
        #     papers_per_application_domain, dropped_papers_from_application = (
        #         get_papers_per_domain(
        #             selected_papers, category_trees["domains"], application_domain
        #         )
        #     )
        #     links.append(
        #         {
        #             "source": nodes.index(application_domain),
        #             "target": nodes.index(base_abstract_topic),
        #             "value": papers_per_application_domain["attrs"].shape[0],
        #             "label": "",
        #         }
        #     )
        #     remaining_papers_from_application = intersect_papers(
        #         remaining_papers_from_application,
        #         dropped_papers_from_application,
        #     )

        # links.append(
        #     {
        #         "source": nodes.index("Other applications"),
        #         "target": nodes.index(base_abstract_topic),
        #         "value": remaining_papers_from_application["attrs"].shape[0],
        #         "label": "",
        #     }
        # )

        # for model in models:
        #     papers_per_model, dropped_papers_from_model = get_papers_per_models(
        #         papers_per_application_domain, category_trees["domains"], model
        #     )

        #     links.append(
        #         {
        #             "source": nodes.index(base_abstract_topic),
        #             "target": nodes.index(model),
        #             "value": papers_per_model["attrs"].shape[0],
        #             "label": "",
        #         }
        #     )
        #     remaining_papers_from_models = intersect_papers(
        #         remaining_papers_from_models,
        #         dropped_papers_from_model,
        #     )

        # links.append(
        #     {
        #         "source": nodes.index(base_abstract_topic),
        #         "target": nodes.index("Other models"),
        #         "value": remaining_papers_from_models["attrs"].shape[0],
        #         "label": "",
        #     }
        # )

        remaining_papers = intersect_papers(remaining_papers, dropped_papers)

    links += build_subgraph(
        "Other domains",
        "Other applications",
        "No applications",
        remaining_papers,
        category_trees["domains"],
        get_papers_per_domain,
        application_domains,
        reversed=True,
    )

    links += build_subgraph(
        "Other domains",
        "Other models",
        "No models",
        remaining_papers,
        category_trees["models"],
        get_papers_per_models,
        models,
    )

    nodes_custom_data["Other domains"] = build_node_customdata(
        remaining_papers, [("domains", "abstract_research_topics")]
    )

    remaining_papers = analysis
    for model in models:
        selected_papers, dropped_papers = get_papers_per_models(
            analysis, category_trees["models"], model, replace_labels=False
        )
        nodes_custom_data[model] = build_node_customdata(
            selected_papers, [("models", model)]
        )
        remaining_papers = intersect_papers(remaining_papers, dropped_papers)

    nodes_custom_data["Other models"] = build_node_customdata(
        remaining_papers, [("models", "neural networks")]
    )

    # TODO: make counts of remaining papers per domain to add in hover.

    nodes = [{"label": node, "customdata": nodes_custom_data[node]} for node in nodes]

    return nodes, links


def trimm_overgeneric_labels(
    analysis, category_trees, base_abstract_topics, overgeneric_labels=None
):

    if overgeneric_labels is None:
        overgeneric_labels = [
            "Deep Learning and Neural Networks",
            "Learning Paradigms",
            "Evaluation",
            "Probabilistic Models and Inference",
            "Optimization and Algorithms",
        ]
    mapping = build_subtree_mapping(
        category_trees["domains"], "abstract_research_topics"
    )

    mapping.update(
        {
            key: value
            for key, value in build_subtree_mapping(
                category_trees["domains"]["abstract_research_topics"],
                "Deep Learning and Neural Networks",
            ).items()
            if key == "Graph Neural Networks"
        }
    )

    pprint(set(mapping.values()))

    overgeneric_labels = set(overgeneric_labels)
    base_abstract_topics = set(base_abstract_topics)

    def foo(row):
        have_some_abstract_topics = any(
            mapping.get(r, None) in base_abstract_topics for r in row["research_fields"]
        )
        row["research_fields"] = [
            r
            for r in row["research_fields"]
            if mapping.get(r, None) not in overgeneric_labels
            or not have_some_abstract_topics
        ]
        return row

    analysis["attrs"] = analysis["attrs"].apply(foo, axis=1, result_type="expand")

    return analysis


def sankey(analysis, category_trees):

    # TODO: Update research fields based on model/algorithms

    base_abstract_topics = [
        "Computer Vision",
        "Natural Language Processing",
        "Reinforcement Learning and Decision Making",
        "Adversarial Machine Learning",
        "Time-Series",
        "Generative Models",
        "Graph-based",
    ]

    multi_modal = {
        "Vision + NLP": ["Computer Vision", "Natural Language Processing"],
        "Vision + RL": [
            "Computer Vision",
            "Reinforcement Learning and Decision Making",
        ],
    }

    analysis = analysis.copy()

    print(analysis["attrs"].index)

    print(
        get_papers_stats(
            analysis,
            category_trees,
            [("domains", "abstract_research_topics")],
        )
    )

    print(
        get_papers_stats(
            analysis,
            category_trees,
            [("multi-domains", ["Computer Vision", "Natural Language Processing"])],
        )
    )

    analysis = trimm_overgeneric_labels(analysis, category_trees, base_abstract_topics)

    print(
        get_papers_stats(
            analysis,
            category_trees,
            [("domains", "abstract_research_topics")],
        )
    )

    print(
        get_papers_stats(
            analysis,
            category_trees,
            [("domains", "Optimization"), ("domains", "abstract_research_topics")],
        )
    )

    print(
        get_papers_stats(
            analysis,
            category_trees,
            [("domains", "Deep Learning and Neural Networks")],
        )
    )

    # TODO: Find a way to drop 'Deep Learning and Neural Networks' from the research fields
    #      when the papers are already selected for another abstract topic.

    # selected_papers = get_papers_per_domain(
    #     analysis, category_trees["domains"], "abstract_research_topics"
    # )[0]

    # # need to use mapping

    # # Drop 'Deep Learning and Neural Networks'
    # research_fields = selected_papers["attrs"].explode("research_fields")
    # selected_papers = research_fields[
    #     research_fields["research_fields"].isin(base_abstract_topics)
    # ]["paper_id"].unique()
    # dl_to_drop = (
    #     research_fields["research_fields"] == "Deep Learning and Neural Networks"
    # ) & (research_fields["paper_id"].isin(selected_papers))
    # aggregate = {str(name): "first" for name in research_fields.columns}
    # aggregate["research_fields"] = list
    # aggregate.pop("paper_id")
    # analysis["attrs"] = (
    #     research_fields[~dl_to_drop].groupby("paper_id").agg(aggregate).reset_index()
    # )

    # print(research_fields[~dl_to_drop].shape)
    # print(research_fields.shape)
    # print(selected_papers.shape)
    # print(analysis["attrs"])

    # print(
    #     get_papers_stats(
    #         analysis,
    #         category_trees,
    #         [("domains", "abstract_research_topics")],
    #     )
    # )

    # return

    # there should be a category (other) counting all papers not falling in any of these categories

    models = [
        "autoencoder",
        "graph neural network",
        "generative flow networks",
        "transformer",
        "multi layer perceptron",
        "convolutional neural network",
        "recurrent neural network",
        "normalizing flow",
        "generative adversarial network",
        "diffusion model",
        "bayesian network",
        "classic_ml",
    ]

    application_domains = [
        "Biomedical and Healthcare",
        "Computational Biology and Chemistry",
        "Engineering and Robotics",
        "Neuroscience",
        "Astronomy & Astrophysics",
        "AI Ethics and Governance",
        # "Reinforcement Learning and Decision Making",
        # "Natural Language Processing",
        # "Computer Vision",
    ]

    nodes, links = build_graph(
        analysis, category_trees, base_abstract_topics, application_domains, models
    )

    # url = "https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/mocks/sankey_energy.json"
    # response = urllib.request.urlopen(url)
    # data = json.loads(response.read())

    # pprint(data)

    fig = go.Figure(
        data=[
            go.Sankey(
                # valueformat=".0f",
                # valuesuffix="TWh",
                node=dict(
                    pad=15,
                    thickness=15,
                    line=dict(color="black", width=0.5),
                    label=[node["label"] for node in nodes],
                    color=["blue", "red", "green", "yellow"],
                    customdata=[node["customdata"] for node in nodes],
                    hovertemplate="%{label}<br /><br />%{customdata}<extra></extra>",
                ),
                link=dict(
                    source=[link["source"] for link in links],
                    target=[link["target"] for link in links],
                    value=[link["value"] for link in links],
                    label=[link["label"] for link in links],
                    customdata=[link["customdata"] for link in links],
                    hovertemplate="%{source.label}-%{target.label}<br /><br />%{customdata}<extra></extra>",
                ),
            )
        ]
    )

    fig.update_layout(
        hovermode="x",
        title="Literature Analysis, Mila, 2023.",
        font=dict(size=10, color="white"),
        plot_bgcolor="black",
        paper_bgcolor="black",
    )

    fig.show()


def sanity_checks(analysis, papers):

    # How many papers have reinforcement learning algorithms labeled but not the research domain, and vice-versa
    # Hom many papers have graph neural networks labels but not the research domain, and vice-versa
    # TODO

    # How many papers have convolutional neural networks labels but do not have a research domain falling under computer vision.
    return


def update_research_fields_based_on_datasets(analysis, category_trees):
    print(
        get_papers_stats(
            analysis,
            category_trees,
            [("domains", "abstract_research_topics")],
        )
    )

    pprint(category_trees["datasets"])
    mapping = build_subtree_mapping(category_trees["datasets"], None)
    pprint(mapping)

    analysis["datasets"]["name"] = analysis["datasets"]["name"].map(mapping)

    print(
        analysis["datasets"].dropna(subset=["name"]).groupby("paper_id").first().shape
    )

    dataset_mappings_to_research_fields = {
        "Vision + Physics-based": ["Computer Vision", "Physics-based"],
        "Vision + RL": ["Computer Vision", "Reinforcement Learning"],
        "Vision + NLP": ["Computer Vision", "Natural Language Processing"],
        "Reinforcement Learning Environment": ["Reinforcement Learning"],
        "Large Multi-Modal Benchmarks": [
            "Computer Vision",
            "Natural Language Processing",
            "Audio",
            "Tabular",
            "Graph-based",
        ],
        "GeoSpatial": ["Computer Vision", "Geospatial"],
    }

    def map_datasets_to_research_fields(datasets):
        research_fields = []
        for dataset in datasets:
            research_fields += dataset_mappings_to_research_fields.get(
                dataset, [dataset]
            )

        return research_fields

    def foo(row):
        paper_datasets = analysis["datasets"][
            analysis["datasets"]["paper_id"] == row["paper_id"]
        ]["name"].unique()

        row["research_fields"] = list(
            set(row["research_fields"])
            | set(map_datasets_to_research_fields(paper_datasets))
        )

        return row

    analysis["attrs"] = analysis["attrs"].apply(foo, axis=1, result_type="expand")

    def _get_matching_count(data, ref):
        data["attrs"] = data["attrs"].explode("research_fields")
        print(data["attrs"][data["attrs"]["research_fields"] == ref])
        return (
            data["attrs"][data["attrs"]["research_fields"] == ref]
            .groupby("paper_id")["paper_id"]
            .first()
            .nunique()
        )

    def get_mapping_counts(analysis, dataset_category, research_field):
        paper_ids = analysis["datasets"][
            analysis["datasets"]["name"] == dataset_category
        ]["paper_id"].unique()
        wdataset, wodataset = select_papers(analysis, paper_ids)
        n_d = wdataset["attrs"].shape[0]
        n_wod = wodataset["attrs"].shape[0]
        assert n_d + n_wod == analysis["attrs"].shape[0]
        n_fd = _get_matching_count(wdataset, research_field)
        n_wofd = n_d - n_fd
        n_fwod = _get_matching_count(wodataset, research_field)
        n_wofwod = n_wod - n_fwod

        print("          | dataset | not dataset |")
        print("field     | {: 7d} | {: 11d} |".format(n_fd, n_fwod))
        print("not field | {: 7d} | {: 11d} |".format(n_wofd, n_wofwod))

    get_mapping_counts(
        select_papers(
            analysis,
            analysis["datasets"]
            .dropna(subset=["name"])
            .groupby("paper_id")["paper_id"]
            .first()
            .unique(),
        )[0],
        "Computer Vision",
        "Computer Vision",
    )

    get_mapping_counts(
        select_papers(
            analysis,
            analysis["datasets"]
            .dropna(subset=["name"])
            .groupby("paper_id")["paper_id"]
            .first()
            .unique(),
        )[1],
        "Computer Vision",
        "Computer Vision",
    )

    print(
        get_papers_stats(
            analysis,
            category_trees,
            [("domains", "abstract_research_topics")],
        )
    )
    return


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--papers",
        nargs="*",
        type=Path,
        default=[],
        help="Paperoni Json files of papers to analyse",
    )
    parser.add_argument(
        "--analysis-folder",
        type=Path,
        default=Path("data/queries"),
        help="Folder containing the annotations of the papers",
    )
    parser.add_argument(
        "--categorized-domains",
        type=Path,
        default=Path("data/categorized_domains.json"),
        help="Path to categorized domains",
    )
    parser.add_argument(
        "--categorized-models",
        type=Path,
        default=Path("data/categorized_models.json"),
        help="Path to categorized models",
    )
    parser.add_argument(
        "--categorized-datasets",
        type=Path,
        default=Path("data/categorized_datasets.json"),
        help="Path to categorized datasets",
    )
    options = parser.parse_args(argv)

    category_trees = {
        "domains": json.load(open(options.categorized_domains)),
        "models": json.load(open(options.categorized_models)),
        "datasets": json.load(open(options.categorized_datasets)),
    }

    papers = []
    for papers_json_path in options.papers:
        papers.extend(json.load(open(papers_json_path)))

    def select_first_execution(paper_path):
        return datetime.fromtimestamp(os.path.getmtime(paper_path)) < datetime(
            2024, 7, 12
        )

    analysis, missing_analysis = load_analysis(
        papers, options.analysis_folder, selector=lambda paper_path: True
    )

    print(analysis["attrs"].shape[0], "paper analysis found")
    print(missing_analysis.shape[0], "paper analysis missing")

    print(
        analysis["attrs"]["research_fields"]
        .str.len()
        .value_counts()
        .reset_index()
        .sort_values(by="research_fields")
    )

    link_df = pd.DataFrame(links_found.items(), columns=["paper_id", "links"])
    link_df["n_links"] = link_df["links"].str.len()
    print(link_df.groupby("n_links").count())
    pd.options.display.max_colwidth = 300
    print(link_df[link_df["links"].str.len() > 1]["links"])

    missing_papers_per_sources = get_counts_per_source(missing_analysis)[0]
    for key in DEFAULT_KEYS:
        print(key, missing_papers_per_sources[key].shape[0])

    # Divide application domains from research topics

    # First, filter out theoretical papers (and invalid types)
    # Print number of models, datasets, and libraries for these types

    analysis["models"]["is_executed"] = (
        analysis["models"]["is_executed"] | analysis["models"]["is_contributed"]
    )
    analysis["models"]["is_compared"] = (
        analysis["models"]["is_executed"] | analysis["models"]["is_compared"]
    )

    print(analysis["attrs"].groupby("type").count()["paper"])

    # analysis, dropped = select_papers(
    #     analysis,
    #     analysis["attrs"][analysis["attrs"]["type"] == "empirical"]["paper_id"],
    # )

    model_select_basis = "is_compared"
    models = (
        analysis["models"]
        .groupby("paper_id")
        .agg({"paper_id": "first", model_select_basis: "sum"})
    )
    pd.options.display.max_colwidth = 300
    print(models.groupby(model_select_basis).count().sort_values(model_select_basis))
    print_titles(analysis, models[models[model_select_basis] == 1]["paper_id"][:10])

    update_research_fields_based_on_datasets(analysis, category_trees)

    # TODO: Select papers with at least one model executed or at least one dataset.

    analysis, dropped = select_papers(
        analysis,
        models[models[model_select_basis] > 0]["paper_id"],
    )

    print(
        analysis["attrs"].groupby("type").count()["paper"],
        "papers with executed models.",
    )
    print(
        dropped["attrs"].groupby("type").count()["paper"],
        "papers without executed models.",
    )

    # models = (
    #     analysis["models"]
    #     .groupby("paper_id")
    #     .agg({"paper_id": "first", model_select_basis: "sum"})
    # )

    print_titles(
        dropped,
        dropped["attrs"][dropped["attrs"]["type"] == "empirical"]["paper_id"][:10],
    )

    # models["is_executed"] = models["is_executed"] > 0

    # Then, filter out papers that do not have executed models
    # Print number of models, datasets, and libraries for these types

    # Print distribution of research topics
    # get_papers_per_domain(analysis["attrs"], category_trees, "abstract_research_topics")

    print(get_papers_stats(analysis, category_trees, [("domains", None)]))

    print(
        get_papers_stats(
            analysis,
            category_trees,
            [("domains", "abstract_research_topics")],
        )
    )

    print(
        get_papers_stats(
            analysis,
            category_trees,
            [("domains", "application_domains")],
        )
    )

    # TODO: Categorize datasets and adjust research fields based on them.
    #       EX: A paper with MNIST should have a Computer Vision label.

    # analysis = select_papers(
    #     analysis,
    #     analysis["datasets"]
    #     .dropna(subset=["name"])
    #     .groupby("paper_id")["paper_id"]
    #     .first()
    #     .unique(),
    # )[0]
    return sankey(analysis, category_trees)

    abstract_domains_of_interest = [
        "Reinforcement Learning and Decision Making",
        "Deep Learning and Neural Networks",
        "Natural Language Processing",
        "Computer Vision",
        "Generative Models",
        "AI Ethics and Governance",
    ]

    for abstract_domain in abstract_domains_of_interest:
        print(abstract_domain)
        print(
            get_papers_stats(analysis, category_trees, [("domains", abstract_domain)])
        )
        print(
            get_papers_stats(
                analysis,
                category_trees,
                [("domains", abstract_domain), ("model", "neural networks")],
            )
        )
        print(
            get_papers_stats(
                analysis,
                category_trees,
                [("domains", abstract_domain), ("model", "reinforcement learning")],
            )
        )
        return

    print(
        get_papers_stats(analysis, category_trees, [("domains", "application_domains")])
    )

    application_domains_of_interest = [
        "Biomedical and Healthcare",
        "Computational Biology and Chemistry",
        "Engineering and Robotics",
        "Neuroscience",
    ]

    for application_domain in application_domains_of_interest:
        print(application_domain)
        print(
            get_papers_stats(
                analysis, category_trees, [("domains", application_domain)]
            )
        )

    print(
        get_papers_stats(
            analysis,
            category_trees,
            [
                ("domains", "Reinforcement Learning and Decision Making"),
                ("domains", "application_domains"),
            ],
        )
    )

    print(
        get_papers_stats(
            analysis,
            category_trees,
            [
                ("domains", "Biomedical and Healthcare"),
                ("domains", "abstract_research_topics"),
            ],
        )
    )

    print(
        get_papers_stats(
            analysis,
            category_trees,
            [
                ("domains", "Computational Biology and Chemistry"),
                ("domains", "abstract_research_topics"),
            ],
        )
    )

    print(
        get_papers_stats(
            analysis,
            category_trees,
            [
                ("domains", "Computational Biology and Chemistry"),
                ("domains", "Deep Learning and Neural Networks"),
            ],
        )
    )

    print(
        get_papers_stats(
            analysis,
            category_trees,
            [
                ("domains", "Graph Neural Networks"),
                ("domains", "application_domains"),
            ],
        )
    )

    # TODO: Compute number of papers not covered by a selection of domains and models.

    sankey()

    return

    print(
        papers_per_domain["attrs"]
        .groupby("research_fields")
        .count()
        .sort_values("title")
    )

    papers_per_domain = get_papers_per_domain(
        analysis, category_trees, "AI Ethics and Governance"
    )[0]

    print(
        papers_per_domain["attrs"]
        .groupby("research_fields")
        .count()
        .sort_values("title")
    )

    papers_per_domain = get_papers_per_domain(
        analysis, category_trees, "Natural Language Processing"
    )[0]

    print(
        papers_per_domain["attrs"]
        .groupby("research_fields")
        .count()
        .sort_values("title")
    )

    papers_per_domain = get_papers_per_domain(
        analysis, category_trees, "Computer Vision"
    )[0]

    print(
        papers_per_domain["attrs"]
        .groupby("research_fields")
        .count()
        .sort_values("title")
    )

    papers_per_domain = get_papers_per_domain(
        analysis, category_trees, "application_domains"
    )[0]

    print(
        papers_per_domain["attrs"]
        .groupby("research_fields")
        .count()
        .sort_values("title")
    )

    papers_per_domain = get_papers_per_domain(
        analysis, category_trees, "Engineering and Robotics"
    )[0]

    print(
        papers_per_domain["attrs"]
        .groupby("research_fields")
        .count()
        .sort_values("title")
    )

    # print_titles(papers_per_domain, papers_per_domain["attrs"][papers_per_domain["attrs"]["research_fields"] == 'Software Engineering']['paper_id'])

    papers_per_domain = get_papers_per_domain(analysis, category_trees, "Robotics")[0]

    print(
        papers_per_domain["attrs"]
        .groupby("research_fields")
        .count()
        .sort_values("title")
    )

    papers_per_domain = get_papers_per_domain(
        analysis, category_trees, "Biomedical and Healthcare"
    )[0]

    print(
        papers_per_domain["attrs"]
        .groupby("research_fields")
        .count()
        .sort_values("title")
    )

    papers_per_domain = get_papers_per_domain(
        analysis, category_trees, "Computational Biology and Chemistry"
    )[0]

    print(
        papers_per_domain["attrs"]
        .groupby("research_fields")
        .count()
        .sort_values("title")
    )

    papers_per_domain = get_papers_per_domain(
        analysis, category_trees, "Computational Chemistry"
    )[0]

    print(
        papers_per_domain["attrs"]
        .groupby("research_fields")
        .count()
        .sort_values("title")
    )

    papers_per_model, dropped_models = get_papers_per_models(
        analysis, models_tree, "neural networks"
    )

    print(
        papers_per_model["models"].groupby("name").count().sort_values("title")
        / analysis["attrs"].shape[0]
    )

    print(
        dropped_models["attrs"]
        .explode("research_fields")
        .groupby("research_fields")
        .count()
        .sort_values("title")
        / dropped_models["attrs"].shape[0]
    )

    print(
        papers_per_model["attrs"]
        .explode("research_fields")
        .groupby("research_fields")
        .count()
        .sort_values("title")
        / papers_per_model["attrs"].shape[0]
    )

    papers_per_model = get_papers_per_models(analysis, models_tree, None)[0]

    print(
        papers_per_model["models"].groupby("name").count().sort_values("title")
        / analysis["attrs"].shape[0]
    )

    papers_per_model = get_papers_per_models(
        analysis, models_tree, "reinforcement learning"
    )[0]

    print(
        papers_per_model["models"].groupby("name").count().sort_values("title")
        / analysis["attrs"].shape[0]
    )

    papers_per_model = get_papers_per_models(analysis, models_tree, "classic_ml")[0]

    print(
        papers_per_model["models"].groupby("name").count().sort_values("title")
        / analysis["attrs"].shape[0]
    )

    # I want to know who are the professors most publishing in the top domains.
    # Are they publishing in these domains specifically?

    import pdb

    pdb.set_trace()

    # What top domains
    # For top domains
    # What are the applications? (and which proportion without any application)
    # What are the models? (and which proportion without any model)
    # For those without any models, what are the algorithms?
    # What are the datasets? (and which proportion without any dataset)


if __name__ == "__main__":
    main()
