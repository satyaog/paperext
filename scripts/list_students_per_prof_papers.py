# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "paperext @ git+https://github.com/satyaog/paperext",
# ]
# ///

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import paperext.structured_output.mdl.model as structured_output
from paperext.log import logger
from paperext.paperoni.report import date_type
from paperext.paperoni.report import main as paperoni_report
from paperext.utils import Paper

parser = argparse.ArgumentParser()
parser.add_argument(
    "profs",
    metavar="CSV",
    type=Path,
)
parser.add_argument(
    "--paperoni",
    metavar="JSON",
    type=Path,
)
parser.add_argument(
    "--start",
    metavar="YYYY-MM-DD",
    default=None,
    type=date_type,
)
parser.add_argument(
    "--end",
    metavar="YYYY-MM-DD",
    default=None,
    type=date_type,
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


if not options.paperoni:
    with NamedTemporaryFile("+wt") as tmpfile:
        paperoni_report(
            [
                "--output",
                tmpfile.file.name,
                *(
                    ["--start", options.start.strftime("%Y-%m-%d")]
                    if options.start
                    else []
                ),
                *(["--end", options.end.strftime("%Y-%m-%d")] if options.end else []),
            ]
        )
        papers = json.loads(tmpfile.read())
else:
    papers = json.loads(options.paperoni.read_text())


for p in papers:
    paper = Paper(p)
    if not paper.queries:
        logger.warning(f"No queries found for {p['title']}")
        continue

    response = structured_output.ExtractionResponse.model_validate_json(
        paper.queries[0].read_text()
    )

    mila_authors = set()
    for author in p["authors"]:
        mila_email = get_mila_email(author["author"])
        if not mila_email:
            logger.debug(f"No mila email found for {author['author']['name']}")
            continue

        mila_authors.add((mila_email, author["author"]["name"]))

    for email, name in (
        (email, name) for email, name in mila_authors if email in profs
    ):
        data.setdefault((email, name), set())
        students = [(email, name) for email, name in mila_authors if email not in profs]
        if not students:
            logger.warning(
                f"No students found for prof {(email, name)} in paper {p['title']}"
            )
            continue
        data[(email, name)].update(
            ((email, name) for email, name in mila_authors if email not in profs)
        )

print("prof_email", "prof", "index", "student_email", "student", sep=",")
for (prof_email, prof_name), students in sorted(data.items()):
    for index, (student_email, student_name) in enumerate(sorted(students)):
        print(prof_email, prof_name, index, student_email, student_name, sep=",")
        prof_email = " " * len(prof_email)
        prof_name = " " * len(prof_name)
        index = " " * len(str(index))
