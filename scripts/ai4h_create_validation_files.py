import json
from pathlib import Path

from paperext.config import CFG
from paperext.structured_output.utils import model_dump_yaml
from paperext.utils import Paper
from paperext.structured_output.ai4hcat.model import Category, Response, SubCategory


if __name__ == "__main__":
    paperoni = Path(CFG.dir.data / "paperoni-2022-01-01-2023-01-01-PR_2025-02-28.json")
    _file = Path("ai4h_cat - validation_02.csv")

    data = {}
    for line in _file.read_text().splitlines():
        cat, cat_verif, subcat, subcat_verif, paper_id, *_ = line.split(",")
        assert not _

        if cat == "N/A" and subcat == "N/A":
            continue

        if cat_verif != "OK" and cat_verif != cat:
            cat = "N/A"
        if subcat_verif != "OK" and subcat_verif != subcat:
            subcat = "N/A"

        data[paper_id] = (cat, subcat)

    CFG.dir.merged.mkdir(exist_ok=True)

    for p in json.loads(paperoni.read_text()):
        paper = Paper(p)

        if paper._paper_id not in data:
            continue

        assert paper.queries

        cat, subcat = data.pop(paper._paper_id)
        response = paper.queries[0]
        analysis = Response.model_validate_json(response.read_text()).extractions

        if analysis.sustainable_development_is_central.value != (
            cat != "N/A" or subcat != "N/A"
        ):
            analysis.sustainable_development_is_central.value = (
                not analysis.sustainable_development_is_central.value
            )
            analysis.sustainable_development_is_central.justification = ""
            analysis.sustainable_development_is_central.quote = ""

        if analysis.primary_category.value != Category(cat):
            analysis.primary_category.value = Category(cat)
            analysis.primary_category.justification = ""
            analysis.primary_category.quote = ""
        if analysis.primary_sub_category.value != SubCategory(subcat):
            analysis.primary_sub_category.value = SubCategory(subcat)
            analysis.primary_sub_category.justification = ""
            analysis.primary_sub_category.quote = ""

        analysis.secondary_categories = []
        analysis.secondary_sub_categories = []
        analysis.applications = []
        analysis.new_primary_category.value = ""
        analysis.new_primary_category.justification = ""
        analysis.new_primary_category.quote = ""
        analysis.new_primary_sub_category.value = ""
        analysis.new_primary_sub_category.justification = ""
        analysis.new_primary_sub_category.quote = ""

        (
            CFG.dir.merged
            / response.with_stem("_".join(response.stem.split("_")[:-1]))
            .with_suffix(".yaml")
            .name
        ).write_text(model_dump_yaml(analysis))

    assert not data
