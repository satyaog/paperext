from __future__ import annotations

import csv
import enum
import typing
from typing import Any, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

from paperext import CFG
from paperext.log import logger
from paperext.utils import str_normalize

_APPLICATIONS_KEY = "applications"
_PAPERS_EXAMPLES_KEY = "papers_examples"


def load_categorisation_tree():
    categorisation_tree = {
        "N/A": {
            "N/A": {
                _APPLICATIONS_KEY: [],
                _PAPERS_EXAMPLES_KEY: [],
            }
        }
    }

    _category, _sub_category, _application, _example = None, None, None, None

    with (CFG.dir.data / "ai4h_categorization.csv").open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            _category = row["Category"].strip() or _category
            _sub_category = row["Sub-Category"].strip() or _sub_category
            _application = row["AI APPLICATION"].strip()
            _example = (row["Paper Example"] or "").strip()

            category = categorisation_tree.setdefault(_category, {})
            sub_category = category.setdefault(_sub_category, {})
            applications = sub_category.setdefault(_APPLICATIONS_KEY, [])
            exemples = sub_category.setdefault(_PAPERS_EXAMPLES_KEY, [])
            if _application and _application not in ("TBD",):
                applications.append(_application)
            if _example:
                exemples.append(_example)

    return categorisation_tree


CATEGORISATION_TREE = load_categorisation_tree()


def get_categories():
    return sorted(CATEGORISATION_TREE.keys())


def get_sub_categories(category: str):
    return sorted(CATEGORISATION_TREE.get(category, {}).keys())


def get_applications(category: str, sub_category: str):
    return sorted(
        CATEGORISATION_TREE.get(category, {})
        .get(sub_category, {})
        .get(_APPLICATIONS_KEY, [])
    )


SYSTEM_MESSAGE = (
    """Your role is to analyze a Deep Learning scientific paper. Your primary goals are to:

1. **Assign an eco-responsible category** and **sub-category** that are a central topic in the paper.
2. If applicable, **extract any eco-responsible applications** discussed in the paper.
3. Additionally, try to identify **secondary categories** and **sub-categories** related to the paper.

Your classification should be based on the predefined hierarchical list below. If you cannot find a suitable match, feel free to suggest a new category or select 'N/A' if the paper is not relevant to eco-responsible AI.

Below is a hierarchical list of eco-responsible categories, sub-categories, and applications to guide your classification:

"""
    + "\n".join(
        f"* {category}:\n"
        + "\n".join(
            f"  * {sub_category}"
            + (":\n" if get_applications(category, sub_category) else "")
            + "\n".join(
                f"    * {application}"
                for application in get_applications(category, sub_category)
            )
            for sub_category in get_sub_categories(category)
        )
        for category in CATEGORISATION_TREE
        if category != "N/A"
    )
    + """

If the paper does not fit into any predefined category, please select the most appropriate option or suggest a new category or sub-category.

If the paper is not directly related to eco-responsible AI, please choose 'N/A' as the category and sub-category."""
)

FIRST_MESSAGE = """The paper to analyze is:

{}"""

_EMPTY_FLAG = "__EMPTY__"


class Category(str, enum.Enum):
    CLIMATE_CHANGE_MITIGATION = "Climate Change Mitigation"
    CLIMATE_CHANGE_ADAPTATION = "Climate Change Adaptation"
    CLIMATE_SCIENCE = "Climate Science"
    NATURAL_SYSTEMS_PROTECTION = "Natural Systems Protection"
    POLLUTION = "Pollution"
    SUSTAINABLE_FINANCE = "Sustainable Finance"
    SUPPORT_TO_GLOBAL_SOUTH = "Support to Global South"
    NA = "N/A"


class SubCategory(str, enum.Enum):
    RENEWABLE_ENERGY_AND_GRID_OPTIMIZATION = "Renewable Energy and Grid Optimization"
    BATTERY_ENERGY_STORAGE_SYSTEM_BESS = "Battery Energy Storage System (BESS)"
    ENERGY_EFFICIENCY_IN_BUILDINGS = "Energy Efficiency in Buildings"
    WASTE = "Waste"
    TRANSPORTATION = "Transportation"
    INDUSTRY = "Industry"
    AGRICULTURE = "Agriculture"
    CARBON_REMOVAL = "Carbon Removal"
    CARBON_CAPTURE_AND_STORAGE_CCS = "Carbon Capture and Storage (CCS)"
    MONITORING_REPORTING_AND_VERIFICATION = "Monitoring, Reporting, and Verification"
    CLIMATE_RISKS_MODELING = "Climate Risks Modeling"
    FOOD_SECURITY = "Food Security"
    RELIEF_EFFORTS = "Relief Efforts"
    FARMERS_SUPPORT = "Farmers Support"
    MIGRATION_SUPPORT = "Migration Support"
    GHG_MEASUREMENT_AND_TRACKING = "GHG Measurement and Tracking"
    CLIMATE_MODELING_AND_PREDICTIONS = "Climate Modeling and Predictions"
    FOREST_MANAGEMENT = "Forest Management"
    BIODIVERSITY = "Biodiversity"
    PEATLANDS = "Peatlands"
    OCEAN_PROTECTION = "Ocean Protection"
    AIR_POLLUTION = "Air Pollution"
    CHEMICAL_POLLUTION = "Chemical Pollution"
    # WASTE = "Waste"
    ESG_AND_DISCLOSURES = "ESG and Disclosures"
    CLIMATE_FINANCE_IMPACT_AND_THEMATIC_INVESTMENTS = (
        "Climate Finance (impact and thematic investments)"
    )
    CLIMATE_DATA = "Climate Data"
    POLICY_ADVICE_AND_KNOWLEDGE_SHARING = "Policy Advice and Knowledge Sharing"
    SUPPORT_RE_DEPLOYMENT = "Support RE Deployment"
    NA = "N/A"


T = TypeVar("T")


class Explained(BaseModel, Generic[T]):
    value: T
    justification: str = Field(
        description="Explain why this value was chosen",
    )
    quote: str = Field(
        description="Direct quote from paper that best supports this value",
    )

    def __eq__(self, other: "Explained"):
        return str_normalize(str(self.value)) == str_normalize(str(other.value))

    def __lt__(self, other: "Explained"):
        if isinstance(self.value, bool):
            return not self.value < other.value
        return str_normalize(str(self.value)) < str_normalize(str(other.value))


class PaperExtractions(BaseModel):
    title: Explained[str] = Field(
        description="Title of the paper",
    )
    description: str = Field(
        description="Short description of the paper",
    )
    primary_category: Explained[Category] = Field(
        description="Primary eco-responsible category of the paper",
    )
    secondary_categories: List[Explained[Category]] = Field(
        description="List of secondary eco-responsible categories of the paper",
    )
    primary_sub_category: Explained[SubCategory] = Field(
        description="Primary eco-responsible sub-category of the paper",
    )
    secondary_sub_categories: List[Explained[SubCategory]] = Field(
        description="List of secondary eco-responsible sub-category of the paper",
    )
    applications: List[Explained[str]] = Field(
        description="List of eco-responsible applications discussed in the paper",
    )
    new_primary_category: Explained[str] = Field(
        description="New eco-responsible category if none of the listed categories fit the paper",
    )
    new_primary_sub_category: Explained[str] = Field(
        description="New eco-responsible sub-category if none of the listed sub-categories fit the paper",
    )


class Response(BaseModel):
    paper: str
    words: int
    extractions: PaperExtractions
    usage: Optional[Any]


def _is_base(cls, other):
    try:
        return cls.__base__ == other
    except AttributeError as e:
        logger.debug(f"{cls} is not based on {other}: {e}", exc_info=True)
        return False


def _empty_fields(model_cls: BaseModel):
    try:
        iter_fields = model_cls.model_fields.items()
    except AttributeError:
        if typing.get_origin(model_cls) == list:
            return [_empty_fields(model_cls.__args__[0])]
        else:
            return _EMPTY_FLAG

    if _is_base(model_cls, Explained):
        fields = {k: (_empty_fields(v) if k == "value" else "") for k, v in iter_fields}
    else:
        fields = {}
        for k, field in iter_fields:
            fields[k] = _empty_fields(field.annotation)

    return fields


def empty_model(model_cls):
    empty_fields = _empty_fields(model_cls)
    empty_fields["primary_category"]["value"] = Category.NA.value
    empty_fields["seconday_categories"][0]["value"] = Category.NA.value
    empty_fields["primary_sub_category"]["value"] = SubCategory.NA.value
    empty_fields["secondary_sub_categories"][0]["value"] = SubCategory.NA.value

    return model_cls(**empty_fields)
