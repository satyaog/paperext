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


NA_FILES = """
800c64c426191fb8c64d64d1abef0220
801b1948440f8d32e6a0f07f77a6244c
8030e031298afd71781d18dbac1936cf
80612da3712ecbdb63097830f0f45716
8078611b46ac8434ddffc3e716f5aa8e
80e8fc01ed0d5d0a91ed463bd30ff38b
8141cd943245fda9531e413caae7c965
8150a24a83456f23173dcbd371d6e49b
81e27400c7f954f3a17bfada8dd7432d
81e7ee7cb4315a3d3fc59eee750b7f5d
81f80c0bd41d7ee648de71a0f98c3dd2
821cf905c323dc067e4dd6a7b45e356c
821fb4397b45ae7f2242011574bfc807
8234cb64c7fc5e81831c08dd0c162e85
82383802ddd1ebe932f89b2a83681ed5
8253261ee0c1ce451a1b79fd0bde4f46
8278240614e76b0a674455d3c3769c34
82944eb09743ec0a34d854bd3ec318b1
8347206bebbcbb10be6412b1f20f316f
838cb1104c619004d4207e57ec075430
83bcfb73e4ed01e38be74f757196fd38
842f97fe9b2b9cdb8e734db777647332
8438d99675ffe3cd61f947878a3f85f6
84d155b4537139753e36c711717ba40c
851fa17af8345c2e2066f32cea870202
85b331c203251771ba650e9e8eba1ba5
85c9b8d4b8958f50e83285e7820dcfe1
8674745f1cf9e89de6bff9e4d85a80ee
8677014f1420563b32b185937275a330
8692afce6cb14effd90b46fd9a36a20e
86c1f057a680058424bac6ce1042523c
86fbb4ae61b347c1f2f88c714e646aad
871beb289fa93a4e1d03aeb93923d8de
875dd4b6f05e1a232b850149661badfb
879bd798379c2445d70c7b1a229eae37
87b55166dec55a745c61ac243d5332bc
87cc0062308ab80b8b6138d93209fa2a
88054901fa80690fd920340ffbbfd31f
88158acd03d95b65644c1402f536bfce
882b320559cbdf75b02c4b9c21443ca0
8858dbee4574118b9d520004f41063e1
888ad1859cdf10e5e21b5e2b69988739
88d0fbb713fd626755ce2bf67d495053
88e350746c925089515ceba236bda55d
893687d5bdc8f3ec4c3642ad321d6957
f00954a5c603790fbef8315e51434126
""".splitlines()

NA_FILES = set(na_file for na_file in NA_FILES if na_file.strip())


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

        if paper_id not in NA_FILES and not (
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

        if analysis.climate_change_is_central.value != (
            any(c != Category.NA for c in categories)
            or any(sc != SubCategory.NA for sc in subcategories)
        ):
            analysis.climate_change_is_central.value = (
                not analysis.climate_change_is_central.value
            )
            analysis.climate_change_is_central.justification = ""
            analysis.climate_change_is_central.quote = ""

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
