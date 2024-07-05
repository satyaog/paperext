import argparse
import ast
from pathlib import Path
import sys
from typing import List

from .utils import python_module

PROG=f"python3 -m {python_module(__file__)}"

DESCRIPTION="""
Utility to parse and gather stats on query logs
"""

EPILOG=f"""
Example:
  $ {PROG} query.out
    Per paper errors
    ================
    Failures count for paper: 1
    ---------------------------
    primary_research_field.aliases      1 ~1
    primary_research_field.name         1 ~1
    sub_research_fields.*.aliases       2 ~1
    sub_research_fields.*.name          2 ~1

    [...]
    Failures count for paper: 2
    ---------------------------
    datasets                            1 ~1
    models.*                           13 ~1
    sub_research_fields.*.aliases       1 ~1
    sub_research_fields.*.name          1 ~1

    libraries.*.aliases                 1 ~1
    libraries.*.referenced_paper_title  1 ~1
    libraries.*.role                    1 ~1

    [...]
    Failures count for paper: 2
    ---------------------------
    Generic Error: 
      Invalid JSON: control character (\\u0000-\\u001F) found while parsing a string at line 1 column 2885 [...]
        For further information visit https://errors.pydantic.dev/2.7/v/json_invalid,)

    Generic Error: 
      Invalid JSON: control character (\\u0000-\\u001F) found while parsing a string at line 1 column 2885 [...]
        For further information visit https://errors.pydantic.dev/2.7/v/json_invalid,)

    [...]
    Error type involved in a paper validation
    =========================================
    Generic Error                       2 /62
    datasets                            5 /62
    libraries.*.aliases                10 /62
    libraries.*.referenced_paper_title  9 /62
    libraries.*.role                    9 /62
    models.*                            3 /62
    primary_research_field.aliases     40 /62
    primary_research_field.name        42 /62
    sub_research_fields.*.aliases      32 /62
    sub_research_fields.*.name         37 /62

    Queries Stats
    =============
    Requests cnt                          204
    Errors cnt                            62
    1 failure(s)                          46
    2 failure(s)                          8
"""


def parse_pydantic_error_lines(error_lines:List[str]):
    errors = {
        "field": [],
        "generic": [],
    }
    current_error = {"type":"generic", "details":[]}
    for l in error_lines:
        if l.startswith("  "):
            current_error["details"].append(l)
        else:
            if current_error["details"]:
                err = current_error.copy()
                err["details"] = "\n".join(err["details"])
                errors[current_error["type"]].append(err)
            current_error["type"] = "field"
            current_error["name"] = l
    if current_error["details"]:
        err = current_error.copy()
        err["details"] = "\n".join(err["details"])
        errors[current_error["type"]].append(err)
    return errors


def parse_request_output(request_output:str):
    while request_output:
        request_output = request_output[request_output.index(":")+1:]
        try:
            request_output = ast.literal_eval(request_output)
        except SyntaxError:
            continue
        break

    assert sorted(request_output.keys()) == ['files', 'json_data', 'method', 'url']

    return request_output


def request_id(request_output:dict):
    message_content = request_output["json_data"]["messages"][1]["content"]
    message_content_lines = message_content.split("\n")
    message_paper_lines = message_content_lines[1:]
    id = "\n".join(message_paper_lines[:40])

    assert len(id) >= 100

    return id[:200]


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input",
        metavar="TXT",
        type=Path,
        help="query stdout file to parse"
    )
    options = parser.parse_args(argv)

    lines = options.input.read_text().splitlines()

    checks = {
        "error_begin": {
            "line_idx": None,
            "cnt": 0,
            "check": lambda l:l.startswith("Message: 'Error response:"),
            "invalidate": ["pydantic_error_stop"]
        },
        "pydantic_error_begin": {
            "line_idx": None,
            "cnt": 0,
            "check": lambda l:l.startswith("Arguments:"),
            "invalidate": []
        },
        "request_output": {
            "line_idx": None,
            "cnt": 0,
            "check": lambda l:l.startswith("DEBUG:openai._base_client:Request options:"),
            "invalidate": []
        },
        "pydantic_error_stop": {
            "line_idx": None,
            "cnt": 0,
            "check": lambda l:l.startswith("DEBUG:") or "Failed to extract paper informations from" in l or l.startswith("DEBUG:") or not l.strip(),
            "invalidate": ["error_begin", "pydantic_error_begin"]
        },
    }

    error_begin = lambda:checks["error_begin"]["line_idx"]
    pydantic_error_begin = lambda:checks["pydantic_error_begin"]["line_idx"]
    request_output = lambda:checks["request_output"]["line_idx"]
    pydantic_error_stop = lambda:checks["pydantic_error_stop"]["line_idx"]

    errors = {}
    error_lines = []

    for i, l in enumerate(lines):
        print(i, l[:100], sep="\t", file=sys.stderr)

        for v in checks.values():
            if v["check"](l):
                v["line_idx"] = i
                v["cnt"] += 1
                for k in v["invalidate"]:
                    checks[k]["line_idx"] = None
                break

        if pydantic_error_stop():
            if error_lines:
                request_output_dict = lines[request_output()]
                request_output_dict = parse_request_output(request_output_dict)
                id = request_id(request_output_dict)
                new_errors = parse_pydantic_error_lines(error_lines[1:])
                for f in new_errors["field"]:
                    name = f["name"]
                    name = [
                        "*" if part.isdigit() else part
                        for part in name.split(".")
                    ]
                    f["name"] = ".".join(name)
                errors.setdefault(id, [])
                errors[id].append(new_errors)
            error_lines = []

        if error_begin() and pydantic_error_begin():
            error_lines.append(l)

    stats = {
        "errors": {"Generic Error": 0},
        "failures": {}
    }
    errors_stats = stats["errors"]
    cols = []
    cols.append(("", "", "", "Per paper errors"))
    cols.append(("", "", "", "================"))
    for paper_errors in sum(map(lambda entry:[{"failures_cnt":len(entry)}] + entry, errors.values()), []):
        if paper_errors.get("failures_cnt", 0):
            failures_cnt = paper_errors['failures_cnt']
            stats["failures"].setdefault(failures_cnt, 0)
            stats["failures"][failures_cnt] += 1
            cols.append(("", "", "", f"Failures count for paper: {failures_cnt}"))
            cols.append(("", "", "",  "---------------------------"))
            continue

        fields = paper_errors.get("field", [])
        fields_stats = {}
        for field in sorted(set([f["name"] for f in fields])):
            errors_stats.setdefault(field, 0)
            fields_stats[field] = sum(field == f['name'] for f in fields)
            errors_stats[field] += 1
            cols.append((field, str(fields_stats[field]), "~1"))

        for generic in paper_errors.get("generic", []):
            errors_stats["Generic Error"] += 1
            cols.append(("", "", "", "Generic Error:", "\n" + generic["details"]))

        cols.append(("", "", ""))

    cols.append(("", "", "", "Error type involved in a paper validation"))
    cols.append(("", "", "", "========================================="))
    for field in sorted(errors_stats.keys()):
        cols.append((field, str(errors_stats[field]), f"/{checks['error_begin']['cnt']}"))

    cols.append(("", "", ""))
    cols.append(("", "", "", "Queries Stats"))
    cols.append(("", "", "", "============="))
    cols.append(("Requests cnt", "", str(checks["request_output"]["cnt"])))
    cols.append(("Errors cnt", "", str(checks["error_begin"]["cnt"])))
    for failures_cnt in sorted(stats["failures"]):
        cols.append((f"{failures_cnt} failure(s)", "", str(stats["failures"][failures_cnt])))

    max_cols = (
        max([len(col[0]) for col in cols]),
        max([len(col[1]) for col in cols]),
        max([len(col[2]) for col in cols]),
    )
    for col in cols:
        try:
            col[3]
            print(*col[3:])
            continue
        except IndexError:
            pass

        col = (
            col[0] + " " * (max_cols[0] - len(col[0])),
            " " * (max_cols[1] - len(col[1])) + col[1],
            col[2],
        )
        print(*col)


if __name__ == "__main__":
    main()
