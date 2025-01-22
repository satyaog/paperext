import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

from paperext import CFG
from paperext.log import logger

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
        "input",
        metavar="JSON",
        type=Path,
        help="Paperoni json output of papers to download and convert pdfs -> txts",
    )
    parser.add_argument(
        "--cache-dir",
        metavar="DIR",
        type=Path,
        default=CFG.dir.cache,
        help="Directory to store downloaded and converted pdfs -> txts",
    )
    options = parser.parse_args(argv)

    paperoni = json.loads(options.input.read_text())

    options.cache_dir.mkdir(parents=True, exist_ok=True)

    completed = []
    failed = []
    for p in paperoni:
        text = None
        links = [l for l in p["links"] if l["type"].lower().startswith("arxiv")]
        links.extend(
            [
                l
                for l in p["links"]
                if "pdf" in l["type"].lower().split(".")
                and l["type"].lower().startswith("openreview")
            ]
        )
        links.extend([l for l in p["links"] if "pdf" in l["type"].lower().split(".")])
        links.extend([l for l in p["links"] if "pdf" in l["link"].lower()])
        if not links:
            logger.warning(
                "\n".join(
                    [
                        f"Could not find any pdf links for paper {p['title']} in",
                        *[str(l) for l in p["links"]],
                    ]
                )
            )
        else:
            logger.info(f'Downloading and converting {p["title"]}')

        for check_only in (True, False):
            text, link_types = download_and_convert_paper(
                p["paper_id"], links[:], options.cache_dir, check_only=check_only
            )
            if text is not None:
                completed.append((p["paper_id"], text, link_types))
                break
        else:
            failed.append((p["paper_id"], text, link_types))
            logger.error(
                f'Failed to download or convert {p["paper_id"]}:{p["title"]} with links {link_types}'
            )

    print(*sorted(str(text) for _, text, _ in completed), sep="\n")

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
