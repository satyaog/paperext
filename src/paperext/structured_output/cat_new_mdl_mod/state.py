from pathlib import Path
from typing import Generator

from paperext.sanitize_categorization import (
    _update_sanitized_map,
    default_sanitize_key,
)
from paperext.log import logger
from paperext.structured_output._base import BaseState
from paperext.utils import Paper
from paperext.structured_output.cat_new_mdl_mod.model import (
    FIRST_MESSAGE,
    RETRY_MESSAGE,
    SYSTEM_MESSAGE,
    Analysis,
    Response,
)


class State(BaseState):
    AnalysisCls: Analysis = Analysis
    ResponseCls: Response = Response

    def __init__(
        self,
        paper: Paper,
        pdf_txt: Path,
        model: str,
        other_models: list,
        sanitized_map: dict[str, str],
        **kwargs,
    ):
        super().__init__(
            paper=paper,
            pdf_txt=pdf_txt,
            model=model,
            other_models=other_models,
            sanitized_map=sanitized_map,
            **kwargs,
        )
        self._model = model
        self._other_models = other_models
        self._sanitized_map = sanitized_map.copy()

        self.responses: list[Response]

    def format_messages(self) -> Generator[list[dict[str:str]], None, None]:
        listed_models = set()
        missing = None

        _messages = [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": FIRST_MESSAGE.format(
                    f'"{self._model}"',
                    "\n".join([f'"{d}"' for d in self._other_models]),
                ),
            },
        ]

        while missing is None or (len(missing) / len(self._other_models)) > 0.1:
            self._query_data.append(
                {
                    "model": self._model,
                    "model_pool": self._other_models,
                    "missing": missing,
                }
            )

            _messages[0] = {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            }

            yield _messages

            _analysis = self.responses[-1].analysis
            _domains = list(
                map(
                    lambda x: x.value,
                    sum(
                        (
                            [
                                _analysis.closest_parent_model,
                                _analysis.closest_child_model,
                                _analysis.closest_sibling_model,
                            ],
                            _analysis.semantically_equivalent_models,
                            _analysis.parent_models,
                            _analysis.child_models,
                            _analysis.sibling_models,
                            _analysis.unrelated_models,
                        ),
                        [],
                    ),
                )
            )
            list(_update_sanitized_map(self._sanitized_map, *_domains))

            sanitized_models = set(
                self._sanitized_map[_model] for _model in self._other_models
            )

            listed_models.update(_domains)

            missing = sanitized_models - listed_models
            _messages[1] = {
                "role": "user",
                "content": RETRY_MESSAGE.format(
                    f'"{self._model}"',
                    "\n".join([f'"{d}"' for d in self._other_models]),
                    "\n".join([f'"{d}"' for d in missing]),
                ),
            }
