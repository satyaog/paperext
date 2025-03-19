import argparse
from datetime import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from time import sleep
from typing import Any
import urllib.request
from multiprocessing.pool import ThreadPool
from pathlib import Path
from xml.etree import ElementTree

import yaml

from paperext import CFG
from paperext.log import logger
from paperext.utils import Paper

PROG = f"{Path(__file__).stem.replace('_', '-')}"

DESCRIPTION = """
Utility to download and convert a list of papers' pdfs -> "txts.

stdout will contain the list of files successfully converted separated by '\\n'.
"""

EPILOG = f"""
Example:
  $ PAPEREXT_LOGGING_LEVEL=INFO {PROG} --paperoni paperoni-2024-07-04.json
    [DEBUG]
    data/cache/arxiv/1901.07186.txt
    data/cache/arxiv/1906.05433.txt
    ...
    data/cache/html/874f823e6462acbbb07cc57d32e09217.txt
    data/cache/html/8a46fcbc0c34ea85102920cba7039290.txt
    ...
    data/cache/openreview/0k_DN90uWF.txt
    data/cache/openreview/2Q8TZWAHv4.txt
    ...
    data/cache/pdf/80c62591b54231aa42e4418fe3d45e8f.txt
    data/cache/pdf/81709b4783324a59fd2632ee694e9071.txt
    ...
    Successfully downloaded and converted 587 out of 867 papers
    arxiv:455/455
    html:3/3
    openreview:52/52
    pdf:77/145
  $ PAPEREXT_LOGGING_LEVEL=INFO {PROG} --paperoni paperoni-2024-07-04.json > data/query_set.txt
    [DEBUG]
    Successfully downloaded and converted 587 out of 867 papers
    arxiv:455/455
    html:3/3
    openreview:52/52
    pdf:77/145
"""


def which_hatch() -> str:
    for hatch in subprocess.run(
        [
            "which",
            "-a",
            "hatch",
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.splitlines():
        try:
            subprocess.run(
                [hatch, "--help"],
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.debug(
                f"{hatch} failed with error: {e}",
                exc_info=True,
            )
            continue
        return hatch

    else:
        return None


_HATCH = which_hatch()


def paperoni_download(paper_data: dict, cache_dir: Path):
    paper = Paper(paper_data)

    if paper.pdfs:
        return paper_data["paper_id"], paper.get_link_id_pdf(), ["EXISTING"]

    for filename in [CFG.env.paperoni_config, os.environ["PAPERONI_CONFIG"]]:
        config_filename = Path(filename).resolve()
        config = yaml.safe_load(config_filename.read_text())
        assert (
            "fulltext" in config["paperoni"]["paths"]
        ), "The paperoni configuration file internal structure seams to have changed or is invalid"
        config["paperoni"]["paths"]["fulltext"] = str(cache_dir / "fulltext")
        break

    else:
        raise FileNotFoundError(
            "paperoni config not found. Cannot download using paperoni"
        )

    try:
        # Use a temporary yaml config file with a modified fulltext path
        with tempfile.NamedTemporaryFile(
            "w+",
            prefix=f"{config_filename.stem}_",
            suffix=".yaml",
            dir=str(config_filename.parent),
        ) as _f:
            yaml.dump(config, _f)

            subprocess.run(
                [
                    *(
                        (_HATCH, "run", "paperoni:paperoni")
                        if _HATCH
                        else ("paperoni",)
                    ),
                    "download",
                    "--config",
                    _f.name,
                    "--title",
                    paper_data["title"],
                ],
                stdout=sys.stderr.fileno(),
                check=True,
            )

        paper = Paper(paper_data)

        if not paper.pdfs:
            raise FileNotFoundError(f"Could not find converted file")

        link_types = ["_PAPERONI"]

    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.error(
            f"Failed to download or convert using paperoni {paper_data['paper_id']}:{paper_data['title']}: {e}",
            exc_info=True,
        )
        link_types = sorted(set([l["type"].split(".")[0] for l in paper_data["links"]]))

    return paper_data["paper_id"], paper.get_link_id_pdf(), link_types


def convert_pdf(pdf, text, pdf_link):
    pdf.parent.mkdir(parents=True, exist_ok=True)

    if not pdf.exists():
        logger.info(f"Downloading from {pdf_link} to {pdf}")

        try:
            urllib.request.urlretrieve(pdf_link, str(pdf))

        except (urllib.error.HTTPError, ValueError) as e:
            logger.error(f"Failed to download {pdf_link}: {e}", exc_info=True)
            pdf.unlink(missing_ok=True)
            return None

    if not text.exists():
        # pdftotext comes from https://poppler.freedesktop.org/
        try:
            cmd = ["pdftotext", str(pdf), str(text)]
            # Redirect stderr to stdout to then redirect the combined stdout and
            # stderr to stderr
            p = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            logger.info(p.stdout)

            if p.returncode:
                raise subprocess.CalledProcessError(
                    p.returncode, cmd, p.stdout, p.stderr
                )

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to convert {pdf} to {text}: {e}", exc_info=True)
            pdf.unlink(missing_ok=True)
            text.unlink(missing_ok=True)
            return None

    return text


def download_and_convert_paper(
    paper_id: str, links: list, cache_dir: Path, check_only=False
):
    text = None
    link_types = []

    while links:
        l: dict = links.pop(0)
        link_type = l["type"].split(".")[0]

        for _if, pdf, pdf_link in (
            # Favor arxiv links if available
            # If the link is an arxiv link, the arxiv id is in the `link` field.
            # The arxiv id can be used to build an url and download the pdf file.
            (
                l["type"].lower().startswith("arxiv"),
                cache_dir / f"arxiv/{l['link']}.pdf",
                f"https://arxiv.org/pdf/{l['link']}",
            ),
            # If `url` is available, use it to download the pdf. The `link`
            # field should contain the id for the pdf file.
            (
                "url" in l,
                cache_dir / link_type / f"{l['link']}.pdf",
                l.get("url", None),
            ),
            # If none of the above worked, try to download the pdf from the `link`
            (True, cache_dir / link_type / f"{paper_id}.pdf", l["link"]),
        ):
            if not _if:
                continue

            text = pdf.with_suffix(".txt")
            link_types.append(link_type)

            if text.exists():
                logger.info(f"Found existing {text}")
                links[:] = []
                break

            if check_only:
                continue

            if convert_pdf(pdf, text, pdf_link) is not None:
                links[:] = []
                break

            logger.warning("retrying...")

        else:
            text = None

    if text is not None:
        link_types = link_types[-1:]

    return text, sorted(set(link_types))


def _iter_list(key: str, dictionary: dict[str, Any]):
    unique_tag = key
    index = 1
    while unique_tag in dictionary:
        yield dictionary[unique_tag]
        unique_tag = f"{key}:{index}"
        index += 1


def _unique_key(key: str, dictionary: dict[str, Any]):
    unique_tag = key
    index = -1
    for index, _ in enumerate(_iter_list(key, dictionary)):
        pass
    if index + 1:
        unique_tag = f"{key}:{index + 1}"
    return unique_tag


def parse_element(element: ElementTree.Element):
    parsed = {}
    passed = set()

    for field in element.iter():
        if field == element or field in passed:
            continue

        passed.add(field)

        clean_tag = field.tag.split("}")[-1]
        unique_tag = _unique_key(clean_tag, parsed)

        text = (field.text or "").strip()
        text = " ".join(
            [split for split in text.replace("\n", " ").split(" ") if split.strip()]
        )

        sub_element, _passed = parse_element(field)
        passed.update(_passed)

        sub_element = {**sub_element, **field.attrib}

        assert not text or not sub_element
        parsed[unique_tag] = text or sub_element

    for k in list(parsed):
        if len(_list := list(_iter_list(k, parsed))) > 1:
            assert f"{k}:list" not in parsed
            parsed[f"{k}:list"] = _list

    return parsed, passed


def date_type(string: str):
    return datetime.strptime(string, "%Y-%m-%d")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--paperoni",
        metavar="JSON",
        nargs="+",
        type=Path,
        help="Paperoni json output of papers to download and convert pdfs -> txts",
    )
    parser.add_argument(
        "--arxiv",
        metavar="STR",
        nargs="+",
        default=tuple(),
        help="List of arXiv ids use to download and convert pdfs -> txts",
    )
    parser.add_argument(
        "--url",
        metavar="STR",
        nargs="+",
        default=tuple(),
        help="List of urls to download and convert pdfs -> txts",
    )
    parser.add_argument(
        "--cache-dir",
        metavar="DIR",
        type=Path,
        default=CFG.dir.cache,
        help="Directory to store downloaded and converted pdfs -> txts",
    )
    parser.add_argument(
        "--no-sort",
        dest="sort",
        default=True,
        action="store_false",
        help="arXiv max number of papers to fetch",
    )

    # Create a subparser for "arxiv"
    subparsers = parser.add_subparsers(dest="src", help="arXiv commands")

    # arXiv subparser
    arxiv_parser = subparsers.add_parser(
        "arxiv", help="Options for arXiv-related operations"
    )

    arxiv_parser.add_argument(
        "--query",
        metavar="STR",
        help="arXiv query to fetch papers",
    )
    arxiv_parser.add_argument(
        "--cat",
        metavar="STR",
        help="arXiv category to filter papers",
    )
    arxiv_parser.add_argument(
        "--au",
        metavar="STR",
        help="arXiv author to filter papers",
    )
    arxiv_parser.add_argument(
        "--start",
        metavar="YYYY-MM-DD",
        default=datetime.fromtimestamp(0),
        type=date_type,
        help="arXiv start date to filter papers",
    )
    arxiv_parser.add_argument(
        "--end",
        metavar="YYYY-MM-DD",
        default=datetime(datetime.now().year + 100, 1, 1),
        type=date_type,
        help="arXiv end date to filter papers",
    )
    arxiv_parser.add_argument(
        "--max-result",
        metavar="INT",
        default=100,
        type=int,
        help="arXiv max number of papers to fetch",
    )
    options = parser.parse_args(argv)

    options.cache_dir.mkdir(parents=True, exist_ok=True)

    completed = []
    failed = []

    papers = sum(
        [json.loads(paperoni.read_text()) for paperoni in (options.paperoni or [])], []
    )

    with ThreadPool(processes=8) as pool:
        for paper_id, text_file, link_types in pool.starmap(
            paperoni_download,
            ((paper, options.cache_dir) for paper in papers),
        ):
            if text_file:
                completed.append((paper_id, text_file, link_types))
            else:
                failed.append((paper_id, text_file, link_types))

    entries = []
    entry_elements = None
    start = 0
    search_query = " AND ".join(
        [
            *([f"cat:{options.cat}"] if options.cat else []),
            *([f"au:{options.au}"] if options.au else []),
            *([f"all:{options.query}"] if options.query else []),
        ]
    )
    while (
        (entry_elements is None or entry_elements)
        and search_query
        and len(entries) <= options.max_result
        and start <= 1000
    ):
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": 100,
            "sortBy": "relevance",
        }
        encoded_params = urllib.parse.urlencode(params)
        url = f"http://export.arxiv.org/api/query?{encoded_params}"
        data = urllib.request.urlopen(url)
        data = data.read().decode("utf-8")

        entry_elements = ElementTree.fromstring(data).findall(
            "{http://www.w3.org/2005/Atom}entry"
        )

        for entry in entry_elements:
            start += 1
            parsed, _ = parse_element(entry)

            if not (
                options.start
                <= datetime.strptime(parsed["updated"], "%Y-%m-%dT%H:%M:%SZ")
                <= options.end
            ):
                continue

            if [p for p in entries if p["id"] == parsed["id"]]:
                break

            entries.append(parsed)

        sleep(5)

    urls = [
        (
            f"https://arxiv.org/pdf/{arxiv_id}",
            options.cache_dir / f"arxiv/{arxiv_id}.pdf",
            "arxiv",
        )
        for arxiv_id in [entry["id"].split("/")[-1].split("v")[0] for entry in entries][
            : options.max_result
        ]
        + list(options.arxiv)
    ]

    for url, pdf_file, link_type in urls + [
        (url, None, "rawurl") for url in options.url
    ]:
        match link_type:
            case "arxiv":
                pass
            case "rawurl":
                domain = (
                    # Remove scheme ending with "//"
                    "//".join(url.split("//")[-1:])
                    # Keep only the host
                    .split("/")[0]
                )
                hash_object = hashlib.sha256()
                hash_object.update(url.encode())
                pdf_file = (
                    options.cache_dir
                    / f"{link_type}_{domain}/{hash_object.hexdigest()}.pdf"
                )

        text_file = convert_pdf(pdf_file, pdf_file.with_suffix(".txt"), url)

        if text_file is not None:
            completed.append((url, text_file, [link_type]))

        else:
            failed.append((url, text_file, [link_type]))

    completed_list = [str(text_file) for _, text_file, _ in completed]
    if options.sort:
        completed_list.sort()

    print(*completed_list, sep="\n")

    logger.info(
        f"Successfully downloaded and converted {len(completed)} out of "
        f"{len(completed) + len(failed)} papers"
    )
    for t in sorted(
        set(sum([l for _, _, l in completed] + [l for _, _, l in failed], []))
    ):
        c, f = (sum(t in l for _, _, l in completed), sum(t in l for _, _, l in failed))
        logger.info(f"{t}:{c}/{c+f}")


if __name__ == "__main__":
    main()
