import json
import logging
import re
import typing
import unicodedata
import warnings
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, Field, create_model

from paperext import ROOT_DIR
from paperext.models.model import Explained, PaperExtractions

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_MODE_ALIASES = {
    "trained": ["training", "evaluation", "pretraining"],
    "inference": [],
    "finetuned": [],
}

_MODE_ALIASES = {alias: k for k, v in _MODE_ALIASES.items() for alias in {k, *v}}

_RESEARCH_FIELDS_ALIASES = {
    "3dreconstruction": [],
    "3dvision": [],
    "accentclassification": [],
    "aiconsciousness": [],
    "aiethics": [],
    "aiethicsandhumancomputerinteractionhci": [],
    "aiforhumanity": [],
    "anomalydetection": [],
    "artificialgeneralintelligence": ["agi"],
    "artificialintelligence": ["ai"],
    "attentionmechanisms": [],
    "autonomousvehiclesystems": [],
    "bayesianinferenceandgenerativemodels": [],
    "combinatorialoptimization": [],
    "computationalbiology": [],
    "computervision": ["cv"],
    "continuallearning": ["cl"],
    "crosslingualtransferlearning": [],
    "deeplearning": ["dl"],
    "deepreinforcementlearning": ["drl"],
    "differentiableprogramming": [],
    "efficientinference": [],
    "energymanagementinroboticsystems": [],
    "evaluationmetrics": [],
    "fairnessinai": [],
    "fairnessinrecommendersystems": [],
    "generativemodels": ["generativemodeling"],
    "goalconditionedreinforcementlearning": [],
    "graphneuralnetwork": [
        "gnn",
        "gnns",
        "graphneuralnetworks",
        "graphneuralnetworksgnns",
    ],
    "humancomputerinteraction": ["hci"],
    "humanintheloopreinforcementlearning": [],
    "interpretability": [],
    "interpretablemachinelearning": [],
    "longtermmemory": [],
    "machinelearning": ["ml"],
    "mathematics": [],
    "medical": [],
    "medicalimageanalysis": [],
    "medicalimagesegmentation": [],
    "medicalimaging": [],
    "microscopyimageanalysis": [],
    "modelcompressionsparsetrainingpruning": [],
    "modeloptimization": [],
    "modelriskmanagement": [],
    "modelsafetyethicsinai": [],
    "molecularpropertyprediction": [],
    "multiagentreinforcementlearning": [],
    "multilingualandlowresourcelanguageprocessing": [],
    "multilingualdatasetsandlargelanguagemodels": [],
    "multilingualnlp": [],
    "musicrecommendationsystems": [],
    "naturallanguageprocessing": ["nlp"],
    "navigationagents": [],
    "neuraldifferentialequations": [],
    "neuralnetworkarchitectures": [],
    "neuralnetworkoptimization": [],
    "neuralsymboliclearning": [],
    "optimizationandmetaheuristics": [],
    "optimizationandtraining": [],
    "optimizationindeeplearning": [],
    "outofdistribution": [],
    "proteinstructureprediction": [],
    "recommendersystems": [],
    "reinforcementlearning": ["rl"],
    "representationlearning": [],
    "roboticphotography": [],
    "roboticplanningandcontrol": [],
    "robotics": [],
    "sampleefficientreinforcementlearning": [],
    "sceneunderstanding": [],
    "scientificmachinelearning": [],
    "speechprocessing": [],
    "speechrecognition": [],
    "textclassification": [],
    "theoremproving": [],
    "timeseriesanomalydetection": [],
    "timeseriesforecasting": [],
    "trajectoryprediction": [],
    "transitnetworkdesigngraphlearning": [],
    "ultrasoundimaging": [],
    "visualcomputing": [],
    "visualquestionanswering": ["vqa"],
    "weatherforecast": [],
}

_RESEARCH_FIELDS_ALIASES = {
    alias: k
    for k, v in _RESEARCH_FIELDS_ALIASES.items()
    for alias in {k, *v, *([f"{k}{v[0]}"] if v else [])}
}


def str_normalize(string):
    string = unicodedata.normalize("NFKC", string).lower()
    string = re.sub(pattern=r"[\s/_\.\(\),\[\]\{\}-]", string=string, repl="")
    return string


def _refs_category_map(categoried_refs_file: Path, fields: list):
    fields = [".".join(map(str_normalize, field.split("."))) for field in fields]

    categoried_refs = json.loads(categoried_refs_file.read_text())

    def list_refs(categories: dict):
        for name, sub_cat in categories.items():
            name = str_normalize(name)
            for sub_name in list_refs(sub_cat):
                yield f"{name}.{sub_name}"

            yield name

    for ref in list_refs(categoried_refs):
        for field in fields:
            if ref.startswith(field + ".") or ref == field:
                yield (ref.split(".")[-1], field)
                break


def _domains_category_map():
    yield from _refs_category_map(
        ROOT_DIR / "data/categorized_domains.json",
        [
            "abstract_research_topics.computer vision",
            "abstract_research_topics.graph-based",
            "abstract_research_topics.natural language processing",
            "abstract_research_topics.reinforcement learning and decision making.reinforcement learning",
            "abstract_research_topics",
            "application_domains",
            "ignore",
        ],
    )


def _models_category_map():
    yield from _refs_category_map(
        ROOT_DIR / "data/categorized_models.json",
        [
            # "algorithms.optimizer",
            # "algorithms.other algorithms",
            # "algorithms.reinforcement learning",
            "algorithms",
            "classic_ml",
            "ignore",
            # "neural networks.attention network",
            # "neural networks.autoencoder",
            # "neural networks.bayesian network",
            # "neural networks.convolutional neural network.ResNet",
            # "neural networks.convolutional neural network.Very Deep Convolutional Networks",
            "neural networks.convolutional neural network",
            "neural networks.diffusion model",
            # "neural networks.generative adversarial network",
            "neural networks.generative flow networks",
            "neural networks.graph neural network",
            "neural networks.multi layer perceptron",
            # "neural networks.normalizing flow",
            "neural networks.recurrent neural network",
            # "neural networks.transformer.Generative Pre-trained Transformer",
            # "neural networks.transformer.Vision Transformer",
            # "neural networks.transformer.bert",
            "neural networks.transformer",
            "neural networks",
        ],
    )


_DOMAINS_CATEGORY_MAP = {
    domain: category for domain, category in _domains_category_map()
}
_MODELS_CATEGORY_MAP = {model: category for model, category in _models_category_map()}

# assert sorted(_MODELS_FIELD_MAP) == sorted([m for m, _ in _models_field_map()])


def _mode_aliases(mode):
    return _MODE_ALIASES.get(mode, mode)


def _reasearch_field_alias(field):
    return _RESEARCH_FIELDS_ALIASES.get(field, field)


def convert_model_json_to_yaml(model_cls: BaseModel, json_data: str, **kwargs):
    model = model_cls.model_validate_json(json_data, **kwargs)
    yaml_data = model_dump_yaml(model)
    assert model_validate_yaml(model_cls, yaml_data) == model
    return yaml_data


def fix_explained_fields():
    fields = _get_fields(PaperExtractions)
    return create_model(PaperExtractions.__name__, **fields)


def model_dump_yaml(model: BaseModel, **kwargs):
    return yaml.safe_dump(
        model.model_dump(**kwargs, mode="json"), sort_keys=False, width=120
    )


def model_validate_yaml(model_cls: BaseModel, yaml_data: str, **kwargs):
    return model_cls.model_validate(yaml.safe_load(yaml_data), **kwargs)


def _get_value(entry: Explained):
    if not isinstance(entry, Explained):
        return entry

    return entry.value


def _aliases(entry: dict):
    ALIASES_FIELDS = {"name", "aliases"}

    if not isinstance(entry, dict) or (
        (set(entry.keys()) & ALIASES_FIELDS) != ALIASES_FIELDS
    ):
        return entry

    entry["name"] = str_normalize(entry["name"])
    return entry


def _mode_and_role(entry: dict):
    MODE_AND_ROLE_FIELDS = {"is_contributed", "is_executed", "is_compared"}

    if not isinstance(entry, dict) or (
        (set(entry.keys()) & MODE_AND_ROLE_FIELDS) != MODE_AND_ROLE_FIELDS
    ):
        return entry

    entry["mode_and_role"] = []
    for k in MODE_AND_ROLE_FIELDS:
        entry["mode_and_role"].append(entry[k])
        del entry[k]

    return entry


def _model_dump(model: BaseModel | typing.Any):
    model = _get_value(model)

    if isinstance(model, BaseModel):
        model = {k: v for k, v in map(lambda f: (f[0], _model_dump(f[1])), model)}
        model = _mode_and_role(model)
        model = _aliases(model)

    elif isinstance(model, list):
        return list(map(_model_dump, model))

    return model


def model2df(model: BaseModel):
    paper_1d_df = {"all_research_fields": []}
    paper_references_df = {}

    for k, v in _model_dump(model).items():
        if k in ("type",):
            v = str_normalize(v.split()[0])
        elif k in ("primary_research_field",):
            v = v["name"]
        elif k in ("sub_research_fields",):
            v = [srf["name"] for srf in v]

        if k in (
            "title",
            "type",
            "primary_research_field",
            "sub_research_fields",
        ):
            paper_1d_df[k] = v

        if k in ("primary_research_field",):
            paper_1d_df["all_research_fields"].append(v)

        elif k in ("sub_research_fields",):
            paper_1d_df["all_research_fields"].extend(v)

        elif k in (
            "models",
            "datasets",
            "libraries",
        ):
            for i, entry in enumerate(v):
                for entry_k, entry_v in entry.items():
                    if entry_k in (
                        "aliases",
                        "referenced_paper_title",
                    ):
                        continue

                    paper_references_df.setdefault(entry_k, {})

                    if entry_k in ("role",):
                        entry_v = str_normalize(entry_v.split()[0])

                    paper_references_df[entry_k][(k, i)] = entry_v

    for domain in paper_1d_df["all_research_fields"]:
        paper_1d_df.setdefault("research_fields_categories", [])

        try:
            category: str = _DOMAINS_CATEGORY_MAP[domain]
        except KeyError as e:
            map_error = e
            logger.error(map_error, exc_info=True)
            continue

        if not category.startswith("abstractresearchtopics."):
            category = "ignore"

        paper_1d_df["research_fields_categories"].append(category)

    map_error = None
    for group in (
        "models",
        "datasets",
    ):
        if "name" not in paper_references_df:
            continue

        if group == "models":
            _map = _MODELS_CATEGORY_MAP
            _check = lambda cat: cat.startswith("neuralnetworks.")
        else:
            continue

        paper_references_df.setdefault("category", {})

        categories = []
        name_df = paper_references_df["name"]
        for (k, i), category in name_df.items():
            if k != group:
                continue

            try:
                _category = category
                category: str = _map[category]
            except KeyError as e:
                map_error = e
                logger.error(map_error, exc_info=True)
                continue

            if not _check(category):
                category = "ignore"

            categories.append(category)
            paper_references_df["category"][(k, i)] = category

    if map_error:
        raise map_error

    paper_1d_df["sub_research_fields"] = [pd.Series(paper_1d_df["sub_research_fields"])]
    paper_1d_df["all_research_fields"] = [pd.Series(paper_1d_df["all_research_fields"])]
    paper_1d_df["research_fields_categories"] = [
        pd.Series(paper_1d_df["research_fields_categories"])
    ]
    paper_1d_df, paper_references_df = (
        pd.DataFrame(paper_1d_df),
        pd.DataFrame(paper_references_df),
    )

    for group in (
        "models",
        "datasets",
        "libraries",
    ):
        try:
            _l = paper_references_df.loc[group]["name"]
            _s = paper_references_df.loc[group]["name"].drop_duplicates()
        except KeyError:
            _l = []
            _s = set()

        if len(_l) != len(_s):
            warnings.warn(f"Possibly duplicated {group} in\n{_l}")

    return paper_1d_df, paper_references_df


def fix_explained_fields():
    fields = _get_fields(PaperExtractions)
    return create_model(PaperExtractions.__name__, **fields)


def print_model(model_cls: BaseModel, indent=0):
    for field, info in model_cls.model_fields.items():
        print(" " * indent, field, info)
        if typing.get_origin(info.annotation) == list:
            print_model(info.annotation.__args__[0], indent + 2)
            continue

        try:
            print_model(info.annotation, indent + 2)
        except AttributeError:
            pass


def _get_fields(model_cls: BaseModel):
    fields = {}
    for field, info in model_cls.model_fields.items():
        if typing.get_origin(info.annotation) == list:
            try:
                sub_fields = _get_fields(info.annotation.__args__[0])
            except AttributeError:
                fields[field] = (info.annotation, Field(description=info.description))
                continue
            fields[field] = (
                typing.List[create_model(field, **sub_fields)],
                Field(description=info.description),
            )
            continue

        try:
            sub_fields = _get_fields(info.annotation)
        except AttributeError:
            fields[field] = (info.annotation, Field(description=info.description))
            continue

        if info.annotation.__base__ == Explained:
            sub_fields = {
                field: (sub_fields["value"][0], Field(description=info.description)),
                **{k: v for k, v in sub_fields.items() if k != "value"},
            }
            cls_name = f"{Explained.__name__}[{field}]"
        else:
            cls_name = info.annotation.__name__
        fields[field] = (
            create_model(cls_name, **sub_fields),
            Field(description=info.description),
        )

    return fields
