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
            assert cat.lower() in ("", "ko/check"), f"Invalid [{cat}] category"
            cat = Category.NA

        if cat != Category.NA and cat not in pool:
            yield cat
            pool.add(cat)

    if not pool:
        yield Category.NA


def iter_subcategories(subcategories: list[str]):
    pool = set()

    for subcat in subcategories:
        try:
            subcat = SubCategory(subcat)

        except ValueError:
            assert subcat.lower() in (
                "",
                "ko/check",
            ), f"Invalid [{subcat}] category"
            subcat = SubCategory.NA

        if subcat != SubCategory.NA and subcat not in pool:
            yield subcat
            pool.add(subcat)

    if not pool:
        yield SubCategory.NA


def main():
    paperoni = (
        None
        # Path(CFG.dir.data / "paperoni-2022-01-01-2023-01-01-PR_2025-02-28.json")
        # Path(CFG.dir.data / "paperoni-2022-01-01-2023-01-01_2025-03-01.json")
    )
    _file = Path(CFG.dir.data / "ai4hcat/export_05.csv")

    data: dict[str:dict] = {}
    lines = list(csv.reader(_file.read_text().splitlines()))
    while lines:
        line = lines.pop(0)

        try:
            next_cat_verif = next(iter_categories([lines[0][2]]))
        except AssertionError:
            next_cat_verif = Category.NA

        try:
            next_subcat_verif = next(iter_subcategories([lines[0][4]]))
        except AssertionError:
            next_subcat_verif = Category.NA

        if (
            not lines[0][5]
            and next_cat_verif == Category.NA
            and next_subcat_verif == SubCategory.NA
        ):
            lines.pop(0)

        _title, cat, cat_verif, subcat, subcat_verif, _paper_id, *_ = line
        assert not _

        if _paper_id:
            paper_id = _paper_id

        if cat_verif == "OK":
            cat_verif = cat
        if subcat_verif == "OK":
            subcat_verif = subcat

        cat, cat_verif = map(lambda x: next(iter_categories([x])), (cat, cat_verif))
        subcat, subcat_verif = map(
            lambda x: next(iter_subcategories([x])), (subcat, subcat_verif)
        )

        if not (
            set((cat, cat_verif, subcat, subcat_verif))
            - set((Category.NA, SubCategory.NA))
        ):
            continue

        data.setdefault(paper_id, {"cat": [], "subcat": []})
        if (
            next(iter_categories([cat_verif])) != Category.NA
            or not data[paper_id]["cat"]
        ):
            data[paper_id]["cat"].append(cat_verif)
        if (
            next(iter_subcategories([subcat_verif])) != SubCategory.NA
            or not data[paper_id]["subcat"]
        ):
            data[paper_id]["subcat"].append(subcat_verif)

    CFG.dir.merged.mkdir(exist_ok=True)

    paper_ids = set()

    papers_queries = []
    if paperoni:
        for p in json.loads(paperoni.read_text()):
            paper = Paper(p)

            if paper._paper_id not in data:
                continue

            assert paper.queries

            papers_queries.append((paper._paper_id, paper.queries[0]))

    else:
        for paper_id in data:
            query_file = sorted(
                (CFG.dir.queries / CFG.platform.select).glob(f"{paper_id}_*.json")
            )[0]

            papers_queries.append((paper_id, query_file))

    for paper_id, response in papers_queries:
        paper_data = data.pop(paper_id)
        categories = list(iter_categories(paper_data["cat"]))
        subcategories = list(iter_subcategories(paper_data["subcat"]))

        analysis = Response.model_validate_json(response.read_text()).extractions

        if analysis.sustainable_development_is_central.value != (
            any(c != Category.NA for c in categories)
            or any(sc != SubCategory.NA for sc in subcategories)
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
            Explained(value=Category.NA, justification="", quote="")
        ] * (len(categories[1:]) - len(analysis.secondary_categories))
        analysis.secondary_sub_categories += [
            Explained(value=SubCategory.NA, justification="", quote="")
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
