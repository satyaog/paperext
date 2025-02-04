import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

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
  $ {PROG} paperoni-2024-07-04.json
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
  $ {PROG} paperoni-2024-07-04.json > data/query_set.txt
    [DEBUG]
    Successfully downloaded and converted 587 out of 867 papers
    arxiv:455/455
    html:3/3
    openreview:52/52
    pdf:77/145
"""


def paperoni_download(paper_data: dict, cache_dir: Path):
    paper = Paper(paper_data)

    if paper.pdfs:
        return paper.get_link_id_pdf(), ["EXISTING"]

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
                    "hatch",
                    "run",
                    "paperoni:paperoni",
                    "download",
                    "--config",
                    _f.name,
                    "--title",
                    paper_data["title"],
                ],
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

    return paper.get_link_id_pdf(), link_types


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
    options = parser.parse_args(argv)

    options.cache_dir.mkdir(parents=True, exist_ok=True)

    completed = []
    failed = []

    for paper in json.loads(options.paperoni.read_text() if options.paperoni else "{}"):
        text_file, link_types = paperoni_download(paper, options.cache_dir)

        if text_file:
            completed.append((paper["paper_id"], text_file, link_types))
        else:
            failed.append((paper["paper_id"], text_file, link_types))

    urls = [
        (
            f"https://arxiv.org/pdf/{arxiv_id}",
            options.cache_dir / f"arxiv/{arxiv_id}.pdf",
            "arxiv",
        )
        for arxiv_id in options.arxiv
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

    print(*sorted(str(text_file) for _, text_file, _ in completed), sep="\n")

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
