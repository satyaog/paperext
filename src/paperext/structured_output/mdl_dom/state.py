import json
from pathlib import Path
from typing import Generator

import pandas as pd

from paperext.config import CFG
from paperext.utils import Paper
from .model import FIRST_MESSAGE, SYSTEM_MESSAGE, ExtractionResponse, PaperExtractions


def build_models_dataframe(models):
    def build_models_tree(models, parent=None) -> list:
        rows = []
        for key in models:
            rows.append({"model": key})
            if models[key]:
                submodels = build_models_tree(models[key])
                for submodel in submodels:
                    submodel[f"parent{len(submodel)}"] = key
                rows.extend(submodels)

        return rows

    return pd.DataFrame(build_models_tree(models))


def truncate_dict(d, depth):
    if depth == 0:
        return {}

    if isinstance(d, dict):
        return {k: truncate_dict(v, depth - 1) for k, v in d.items()}
    else:
        return d


class State:
    def __init__(self, paper: Paper, pdf_txt: Path, domain_name: str):
        self._paper = paper
        self._pdf_txt = pdf_txt
        self._domain_name = domain_name.lower()
        self._categories_refs = json.loads(
            (CFG.dir.data / "categorized_domains.json").read_text().lower()
        )
        # self._categories_refs = (
        #     truncate_dict(
        #         json.loads(
        #             (CFG.dir.data / "categorized_domains.json").read_text().lower()
        #         ),
        #         depth=2,
        #     )
        # )
        self.responses: list[ExtractionResponse] = []

    @property
    def categories_refs(self):
        return self._categories_refs

    def format_messages(self) -> Generator[list[dict[str:str]], None, None]:
        yield [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE.format(self._categories_refs),
            },
            {
                "role": "user",
                "content": FIRST_MESSAGE.format(
                    self._domain_name, self._pdf_txt.read_text()
                ),
            },
        ]

    def push_response(self, response: ExtractionResponse):
        self.responses.append(response)

    def get_extraction_response(
        self,
    ):
        return ExtractionResponse

    def get_paper_extractions(
        self,
    ):
        return PaperExtractions
