import json
import pathlib
import subprocess
import urllib.request


def convert_pdf(pdf, text, pdf_link):
    pdf.parent.mkdir(parents=True, exist_ok=True)
    if not pdf.exists():
        print('Downloading from', pdf_link, 'to', str(pdf))
        try:
            urllib.request.urlretrieve(pdf_link, str(pdf))
        except (urllib.error.HTTPError, ValueError) as e:
            print(f"Failed to download {pdf_link}: {e}")
            pdf.unlink(missing_ok=True)
            return None
    if not text.exists():
        # pdftotext comes from https://poppler.freedesktop.org/
        try:
            subprocess.run(
                ["pdftotext", str(pdf), str(text)],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {pdf} to {text}: {e}")
            pdf.unlink(missing_ok=True)
            text.unlink(missing_ok=True)
            return None
    return text


def download_and_convert_paper(links, check_only=False):
    text = None
    link_types = []
    while links:
        l:dict = links.pop(0)
        link_type = l['type'].split(".")[0]
        for _if, pdf, pdf_link in (
            (
                l["type"].lower().startswith("arxiv"),
                cache_dir / f"arxiv/{l['link']}.pdf",
                f"https://arxiv.org/pdf/{l['link']}"
            ),
            (
                "url" in l,
                cache_dir / link_type / f"{l['link']}.pdf",
                l.get("url", None)
            ),
            (
                True,
                cache_dir / link_type / f"{p['paper_id']}.pdf",
                l["link"]
            ),
        ):
            if not _if:
                continue
            text = pdf.with_suffix(".txt")
            link_types.append(link_type)
            if text.exists():
                print(f"Found existing {text}")
                links[:] = []
                break
            if check_only:
                continue
            if convert_pdf(pdf, text, pdf_link) is not None:
                links[:] = []
                break
            print("retrying...")
        else:
            text = None
    if text is not None:
        link_types = link_types[-1:]
    return text, sorted(set(link_types))


if __name__ == "__main__":
    paperoni = json.loads(pathlib.Path("paperoni-2023-2024-PR_2024-07-03.json").read_text())

    cache_dir = pathlib.Path("data/cache/")
    cache_dir.mkdir(parents=True, exist_ok=True)

    completed = []
    failed = []
    for p in paperoni:
        pdf = None
        pdf_link = None
        text = None
        links = [
            l for l in p["links"]
            if l["type"].lower().startswith("arxiv")
        ]
        links.extend(
            [
                l for l in p["links"]
                if "pdf" in l["type"].lower().split(".") and l["type"].lower().startswith("openreview")
            ]
        )
        links.extend(
            [
                l for l in p["links"]
                if "pdf" in l["type"].lower().split(".")
            ]
        )
        links.extend(
            [
                l for l in p["links"]
                if "pdf" in l["link"].lower()
            ]
        )
        if not links:
            print(f"Could not find any pdf links for paper {p['links']} in", *p["links"], sep="\n")
        else:
            print(f'Downloading and converting {p["title"]}')
        text, link_types = download_and_convert_paper(links[:], check_only=True)
        if text is None:
            text, link_types = download_and_convert_paper(links[:], check_only=False)
        if text is None:
            failed.append((p['paper_id'], link_types))
            print(f'Failed to download or convert {p["paper_id"]}:{p["title"]} with links {link_types}')
        else:
            completed.append((p['paper_id'], link_types))

    for t in sorted(set(sum([l for _, l in completed] + [l for _, l in failed], []))):
        print(f"{t}:{sum(t in l for _, l in completed)}\{sum(t in l for _, l in failed)}")