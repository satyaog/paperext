import argparse
import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path

from paperext.config import CFG, Config
from paperext.query import PLATFORMS, PROG, ignore_exceptions

from paperext.structured_output.papaff.state import State
from paperext.utils import Paper


def main(argv: list = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--papers",
        nargs="*",
        type=Path,
        default=[],
        help="Paperoni json report of papers to analyse",
    )
    options = parser.parse_args(argv)

    paperoni = sum([json.loads(_p.read_text()) for _p in options.papers], [])

    with Config.push():
        CFG.platform.select = "llamaparse"
        CFG.platform.struct = "parse_doc"
        CFG.dir.queries = CFG.dir.data / CFG.platform.struct / "queries"

        papers = [Paper(p) for p in paperoni]
        papers = [(p, p.get_link_id_pdf()) for p in papers if p.queries]

    with Config.push():
        CFG.platform.struct = Path(__file__).parent.name

        LOG_FILE = CFG.dir.log / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logging.basicConfig(
            filename=LOG_FILE.with_suffix(f".{PROG}.{CFG.platform.struct}.dbg"),
            level=logging.DEBUG,
            force=True,
        )

        client = PLATFORMS[CFG.platform.select]()

        asyncio.run(
            ignore_exceptions(
                client,
                [
                    (paper, pdf_txt.absolute())
                    for paper, pdf_txt in sorted(
                        papers, key=lambda x: x[0]._paper_id if x[0] else ""
                    )
                ],
                destination=CFG.dir.data
                / CFG.platform.struct
                / "queries"
                / CFG.platform.select,
                state_cls=State,
            )
        )


if __name__ == "__main__":
    main()
