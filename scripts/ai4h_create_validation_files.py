import csv
import json
from pathlib import Path

from paperext.config import CFG
from paperext.structured_output.utils import model_dump_yaml
from paperext.utils import Paper
from paperext.structured_output.ai4hcat.model import (
    Category,
    Explained,
    Response,
    SubCategory,
)


def iter_categories(categories: list[str]):
    pool = set()

    for cat in categories:
        try:
            cat = Category(cat)

        except ValueError:
            assert cat.lower() in ("", "ko/check")
            cat = Category("N/A")

        if cat not in pool:
            yield cat
            pool.add(cat)

    if not pool:
        yield Category("N/A")


def iter_subcategories(subcategories: list[str]):
    pool = set()

    for subcat in subcategories:
        try:
            subcat = SubCategory(subcat)

        except ValueError:
            assert subcat.lower() in ("", "ko/check")
            subcat = SubCategory("N/A")

        if subcat not in pool:
            yield subcat
            pool.add(subcat)

    if not pool:
        yield SubCategory("N/A")


def main():
    paperoni = Path(CFG.dir.data / "paperoni-2022-01-01-2023-01-01-PR_2025-02-28.json")
    _file = Path(CFG.dir.data / "ai4hcat/export_03.csv")

    data: dict[str:dict] = {}
    lines = list(csv.reader(_file.read_text().splitlines()))
    while lines:
        line = lines.pop(0)

        if not lines[0][5]:
            lines.pop(0)

        _title, cat, cat_verif, subcat, subcat_verif, _paper_id, *_ = line
        assert not _

        if cat == "N/A" and subcat == "N/A":
            continue

        if _paper_id:
            paper_id = _paper_id

        if cat_verif == "OK":
            cat_verif = cat
        if subcat_verif == "OK":
            subcat_verif = subcat

        data.setdefault(paper_id, {"cat": [], "subcat": []})
        if cat != "N/A" or not data[paper_id]["cat"]:
            data[paper_id]["cat"].append(cat_verif)
        if subcat != "N/A" or not data[paper_id]["subcat"]:
            data[paper_id]["subcat"].append(subcat_verif)

    CFG.dir.merged.mkdir(exist_ok=True)

    paper_ids = set()
    for p in json.loads(paperoni.read_text()):
        paper = Paper(p)

        if paper._paper_id not in data:
            continue

        assert paper.queries

        paper_data = data.pop(paper._paper_id)
        categories = list(iter_categories(paper_data["cat"]))
        subcategories = list(iter_subcategories(paper_data["subcat"]))

        response = paper.queries[0]
        analysis = Response.model_validate_json(response.read_text()).extractions

        if analysis.sustainable_development_is_central.value != (
            any(c != Category("N/A") for c in categories)
            or any(sc != SubCategory("N/A") for sc in subcategories)
        ):
            analysis.sustainable_development_is_central.value = (
                not analysis.sustainable_development_is_central.value
            )
            analysis.sustainable_development_is_central.justification = ""
            analysis.sustainable_development_is_central.quote = ""

        if analysis.primary_category.value != categories[0]:
            analysis.primary_category.value = categories[0]
            analysis.primary_category.justification = ""
            analysis.primary_category.quote = ""
        if analysis.primary_sub_category.value != subcategories[0]:
            analysis.primary_sub_category.value = subcategories[0]
            analysis.primary_sub_category.justification = ""
            analysis.primary_sub_category.quote = ""

        analysis.secondary_categories += [
            Explained(value=Category("N/A"), justification="", quote="")
        ] * (len(categories[1:]) - len(analysis.secondary_categories))
        analysis.secondary_sub_categories += [
            Explained(value=SubCategory("N/A"), justification="", quote="")
        ] * (len(subcategories[1:]) - len(analysis.secondary_sub_categories))

        i = -1
        for i, (prediction, valid) in enumerate(
            zip(analysis.secondary_categories, categories[1:])
        ):
            if prediction.value != valid:
                prediction.value = valid
                prediction.justification = ""
                prediction.quote = ""
        analysis.secondary_categories[i + 1 :] = []

        i = -1
        for i, (prediction, valid) in enumerate(
            zip(analysis.secondary_sub_categories, subcategories[1:])
        ):
            if prediction.value != valid:
                prediction.value = valid
                prediction.justification = ""
                prediction.quote = ""
        analysis.secondary_sub_categories[i + 1 :] = []

        analysis.applications = []
        analysis.new_primary_category.value = ""
        analysis.new_primary_category.justification = ""
        analysis.new_primary_category.quote = ""
        analysis.new_primary_sub_category.value = ""
        analysis.new_primary_sub_category.justification = ""
        analysis.new_primary_sub_category.quote = ""

        paper_id = "_".join(response.stem.split("_")[:-1])
        (
            CFG.dir.merged / response.with_stem(paper_id).with_suffix(".yaml").name
        ).write_text(model_dump_yaml(analysis))

        paper_ids.add(paper_id)

    assert not data

    print(*sorted(paper_ids), sep="\n")


if __name__ == "__main__":
    main()
