# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "paperext @ git+https://github.com/satyaog/paperext",
# ]
# ///

import argparse
import json
from pathlib import Path

from paperext.config import CFG
from paperext.structured_output import get_struct_module
from paperext.log import logger
from paperext.utils import Paper


def iter_pdf_urls(paper: dict):
    links = paper["links"][:]

    while links:
        l: dict = links.pop(0)

        if l["type"] == "semantic_scholar.abstract":
            yield l["url"]

        for _if, pdf_link in (
            # Favor arxiv links if available
            # If the link is an arxiv link, the arxiv id is in the `link` field.
            # The arxiv id can be used to build an url and download the pdf file.
            (
                l["type"].lower().startswith("arxiv"),
                f"https://arxiv.org/pdf/{l['link']}",
            ),
            # If `url` is available, use it to download the pdf. The `link`
            # field should contain the id for the pdf file.
            (
                "url" in l,
                l.get("url", None),
            ),
            # If none of the above worked, try to download the pdf from the `link`
            (True, l["link"]),
        ):
            if not _if or "pdf" not in pdf_link:
                continue

            yield pdf_link


def format_reponses(response):
    data = {}
    data["description"] = response.extractions.description

    data["sustainable_development_is_central"] = (
        response.extractions.sustainable_development_is_central.value
    )
    data["sustainable_development_is_central_justification"] = (
        response.extractions.sustainable_development_is_central.justification
    )
    data["category"] = response.extractions.primary_category.value.value
    data["category_justification"] = response.extractions.primary_category.justification
    data["sub-category"] = response.extractions.primary_sub_category.value.value
    data["sub-category_justification"] = (
        response.extractions.primary_sub_category.justification
    )

    data["secondary_categories"] = [
        (category.value.value, category.justification)
        for category in response.extractions.secondary_categories
    ]

    data["secondary_sub-categories"] = [
        (sub_category.value.value, sub_category.justification)
        for sub_category in response.extractions.secondary_sub_categories
    ]

    data["applications"] = [
        (application.value, application.justification)
        for application in response.extractions.applications
    ]

    data["new_category"] = response.extractions.new_primary_category.value
    data["new_category_justification"] = (
        response.extractions.new_primary_category.justification
    )
    data["new_primary_sub-category"] = (
        response.extractions.new_primary_sub_category.value
    )
    data["new_primary_sub-category_justification"] = (
        response.extractions.new_primary_sub_category.justification
    )

    return data


parser = argparse.ArgumentParser()
parser.add_argument(
    "--paperoni",
    metavar="JSON",
    nargs="+",
    default=[],
    type=Path,
)
parser.add_argument(
    "--papers",
    nargs="+",
    default=[],
    type=Path,
)
options = parser.parse_args()

data = {}

for response_file in map(Path, options.papers):
    response = get_struct_module(
        CFG.platform.struct
    ).model.Response.model_validate_json(response_file.read_text())
    id = "_".join(response_file.stem.split("_")[:-1])
    parsed_response = format_reponses(response)
    parsed_response["id"] = id
    parsed_response["urls"] = [f"https://arxiv.org/pdf/{id}"]
    data[response.extractions.title.value] = parsed_response

for p in sum([json.loads(paperoni.read_text()) for paperoni in options.paperoni], []):
    paper = Paper(p)

    for response in map(
        lambda q: get_struct_module(
            CFG.platform.struct
        ).model.Response.model_validate_json(q.read_text()),
        paper.queries,
    ):
        parsed_response = format_reponses(response)
        parsed_response["id"] = p["paper_id"]
        parsed_response["urls"] = list(iter_pdf_urls(p))
        data[p["title"]] = parsed_response

header = [
    "paper title",
    "sustainable development is central",
    *"category;sub-category;ai application;examples".split(";")[:-1],
    "id",
    "urls",
]
lines = []
print(*header, sep=";")
for title, paper_data in sorted(
    data.items(),
    key=(lambda x: (x[1]["category"], x[1]["sub-category"], x[0])),
):
    line1 = (
        title,
        str(paper_data["sustainable_development_is_central"]),
        paper_data["category"],
        paper_data["sub-category"],
        "",
        paper_data["id"],
        *paper_data["urls"],
    )
    line2 = (
        paper_data["description"].replace(";", " _ "),
        paper_data["sustainable_development_is_central_justification"],
        paper_data["category_justification"].replace(";", " _ "),
        paper_data["sub-category_justification"].replace(";", " _ "),
        "",
        "",
        "",
    )
    print(*line1, sep=";")
    print(*line2, sep=";")
    for i in range(
        max(
            map(
                len,
                (
                    paper_data["secondary_categories"],
                    paper_data["secondary_sub-categories"],
                    paper_data["applications"],
                ),
            )
        )
    ):
        category = (
            paper_data["secondary_categories"][i][0]
            if i < len(paper_data["secondary_categories"])
            else ""
        )
        sub_category = (
            paper_data["secondary_sub-categories"][i][0]
            if i < len(paper_data["secondary_sub-categories"])
            else ""
        )
        application = (
            paper_data["applications"][i][0]
            if i < len(paper_data["applications"])
            else ""
        )
        line1 = (
            " " * len(title),
            " " * len(str(paper_data["sustainable_development_is_central"])),
            category.replace(";", " _ "),
            sub_category.replace(";", " _ "),
            application.replace(";", " _ "),
            "",
            "",
        )
        line2 = (
            " " * len(title),
            " " * len(str(paper_data["sustainable_development_is_central"])),
            (
                paper_data["secondary_categories"][i][1].replace(";", " _ ")
                if category
                else ""
            ),
            (
                paper_data["secondary_sub-categories"][i][1].replace(";", " _ ")
                if sub_category
                else ""
            ),
            paper_data["applications"][i][1].replace(";", " _ ") if application else "",
            "",
            "",
        )
        print(*line1, sep=";")
        print(*line2, sep=";")
