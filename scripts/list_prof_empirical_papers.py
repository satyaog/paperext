# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "paperext @ git+https://github.com/satyaog/paperext",
# ]
# ///

import argparse
import json
from pathlib import Path

import paperext.structured_output.mdl.model as structured_output
from paperext.log import logger
from paperext.utils import Paper

parser = argparse.ArgumentParser()
parser.add_argument(
    "profs",
    metavar="CSV",
    type=Path,
)
parser.add_argument(
    "paperoni",
    metavar="JSON",
    type=Path,
)
options = parser.parse_args()


data = {}


def get_mila_email(author: dict):
    for link in author.get("links", []):
        if link["type"] == "email.mila":
            return link["link"]

    return None


header = []
profs = {}
for line in options.profs.read_text().splitlines():
    if not header:
        header = [label.lower().strip().replace(" ", "_") for label in line.split(",")]
        last = header.pop()
        while not last:
            last = header.pop()
        header.append(last)
        continue

    prof = dict(zip(header, line.split(",")))
    try:
        prof["core_prof"] = int(prof["core_prof"])
    except ValueError:
        prof["core_prof"] = 0

    assert prof["email"] not in profs or profs[prof["email"]] == prof

    profs[prof["email"]] = prof


profs = {k: v for k, v in profs.items() if v["core_prof"] == 1}


for p in json.loads(options.paperoni.read_text()):
    paper = Paper(p)
    if not paper.queries:
        logger.warning(f"No queries found for {p['title']}")
        continue

    response = structured_output.ExtractionResponse.model_validate_json(
        paper.queries[0].read_text()
    )

    if response.extractions.type.value != structured_output.ResearchType.EMPIRICAL:
        continue

    for author in p["authors"]:
        mila_email = get_mila_email(author["author"])
        if not mila_email:
            logger.debug(f"No mila email found for {author['author']['name']}")
            continue

        if not mila_email in profs:
            logger.debug(f"Author {mila_email} is not a mila prof")
            continue

        data.setdefault((mila_email, author["author"]["name"]), [])
        data[(mila_email, author["author"]["name"])].append(
            (
                p["title"],
                [
                    link.get("url", None) or link["link"]
                    for link in p["links"]
                    if "pdf" in link["type"].lower().split(".")
                ],
            )
        )

print("email", "name", "index", "title", "urls", sep=",")
for (email, name), papers in sorted(data.items()):
    for index, (title, urls) in enumerate(sorted(papers[:20])):
        title = title.strip()
        for url in urls:
            print(email, name, index, title, url, sep=",")
            email = " " * len(email)
            name = " " * len(name)
            index = " "
            title = " " * len(title)
