import argparse
import json
import subprocess
import urllib
from datetime import date, datetime
from pathlib import Path

from paperext import CFG
from paperext.log import logger
from paperext.paperoni.utils import parse_curl


def date_type(string: str):
    return datetime.strptime(string, "%Y-%m-%d")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
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
    options = parser.parse_args(argv)
    start = options.start.strftime("%Y-%m-%d") if options.start else ""
    end = options.end.strftime("%Y-%m-%d") if options.end else ""
    output = (
        options.output
        or CFG.dir.data
        / f"paperoni-{start}-{end}-PR_{date.today().strftime('%Y-%m-%d')}.json"
    )

    curl_options = [
        # TODO: remove this as soon as the certificate becomes valid again
        "--insecure",
        "-o",
        str(output),
        *parse_curl(),
    ]

    # building paperoni query params to get something like:
    # ?start=2022-01-01&end=2025-01-01&validation=validated&peer-reviewed=True&sort=-date&format=json
    paperoni_query_params = {
        "start": start,
        "end": end,
        "validation": "validated",
        "peer-reviewed": "True",
        "sort": "-date",
        "format": "json",
    }
    paperoni_query_params = {
        **{key: value for key, value in paperoni_query_params.items() if value}
    }
    encoded_params = urllib.parse.urlencode(paperoni_query_params)

    # Combine base URL and query string
    url = f"{CFG.paperoni.url}/report?{encoded_params}"

    subprocess.run(["curl", url, *curl_options], check=True)

    # Check that the output is a valid JSON
    try:
        json.loads(output.read_text())
    except json.decoder.JSONDecodeError as e:
        logger.error(f"Paperoni report is not a valid JSON: {e}", exc_info=True)


if __name__ == "__main__":
    main()
