# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "paperext @ git+https://github.com/satyaog/paperext",
# ]
# ///

import argparse
import json
from pathlib import Path
import sys

from paperext.config import CFG
from paperext.structured_output import get_struct_module
import paperext.structured_output.mdl.model as structured_output
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


parser = argparse.ArgumentParser()
parser.add_argument(
    "paperoni",
    metavar="JSON",
    nargs="+",
    type=Path,
)
options = parser.parse_args()

data = {}
for p in sum([json.loads(paperoni.read_text()) for paperoni in options.paperoni], []):
    paper = Paper(p)
    for response in map(
        lambda q: get_struct_module(
            CFG.platform.struct
        ).model.Response.model_validate_json(q.read_text()),
        paper.queries,
    ):
        data.setdefault(p["title"], {})
        data[p["title"]]["description"] = response.extractions.description

        data[p["title"]]["category"] = response.extractions.primary_category.value.value
        data[p["title"]][
            "category_justification"
        ] = response.extractions.primary_category.justification
        data[p["title"]][
            "sub-category"
        ] = response.extractions.primary_sub_category.value.value
        data[p["title"]][
            "sub-category_justification"
        ] = response.extractions.primary_sub_category.justification

        data[p["title"]]["secondary_categories"] = [
            (category.value.value, category.justification)
            for category in response.extractions.secondary_categories
        ]

        data[p["title"]]["secondary_sub-categories"] = [
            (sub_category.value.value, sub_category.justification)
            for sub_category in response.extractions.secondary_sub_categories
        ]

        data[p["title"]]["applications"] = [
            (application.value, application.justification)
            for application in response.extractions.applications
        ]

        data[p["title"]][
            "new_category"
        ] = response.extractions.new_primary_category.value
        data[p["title"]][
            "new_category_justification"
        ] = response.extractions.new_primary_category.justification
        data[p["title"]][
            "new_primary_sub-category"
        ] = response.extractions.new_primary_sub_category.value
        data[p["title"]][
            "new_primary_sub-category_justification"
        ] = response.extractions.new_primary_sub_category.justification

        data[p["title"]]["urls"] = list(iter_pdf_urls(p))


header = [
    "paper title",
    *"category;sub-category;ai application;examples".split(";")[:-1],
    "urls",
]
lines = []
print(*header, sep=";")
for title, paper_data in sorted(data.items()):
    line1 = (
        title,
        paper_data["category"],
        paper_data["sub-category"],
        "",
        *paper_data["urls"],
    )
    line2 = (
        paper_data["description"],
        paper_data["category_justification"],
        paper_data["sub-category_justification"],
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
            category,
            sub_category,
            application,
            "",
        )
        line2 = (
            " " * len(title),
            paper_data["secondary_categories"][i][1] if category else "",
            paper_data["secondary_sub-categories"][i][1] if sub_category else "",
            paper_data["applications"][i][1] if application else "",
            "",
        )
        print(*line1, sep=";")
        print(*line2, sep=";")
