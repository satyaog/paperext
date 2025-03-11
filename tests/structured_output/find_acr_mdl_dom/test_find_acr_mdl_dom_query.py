import json

import pytest


from paperext.config import CFG, Config
from paperext.structured_output.find_acr_mdl_dom.model import (
    Response,
    empty_response,
)
import paperext.structured_output.find_acr_mdl_dom.query
from paperext.structured_output.find_acr_mdl_dom.query import list_domains, main


def test_list_domains(data_regression, cfg: Config):
    with Config.push(Config(cfg.dir.root / "../config.mdl.ini")):
        CFG.platform.select = "openai"
        CFG.platform.struct = "mdl"

        data_regression.check(
            sorted(
                set(
                    list_domains(
                        json.loads(
                            (
                                cfg.dir.root / "data/paperoni_find_acr_mdl_dom.json"
                            ).read_text()
                        )
                    )
                )
            )
        )


def test_query(monkeypatch, tmp_path, data_regression, cfg: Config):
    async def batch_queries(*args, **kwargs):
        _, filename = args[1][0]
        return list(
            map(
                lambda x: Response.model_validate_json(x.read_text()),
                sorted(
                    (
                        cfg.dir.root
                        / "data"
                        / CFG.platform.struct
                        / "queries"
                        / CFG.platform.select
                    ).glob(f"{filename}_*.json")
                ),
            )
        )

    monkeypatch.setattr(
        paperext.structured_output.find_acr_mdl_dom.query,
        "batch_queries",
        batch_queries,
    )

    categorized_domains_minimal = cfg.dir.data / "mdl/minimal_categorized_domains.json"

    with Config.push(Config(cfg.dir.root / "../config.mdl.ini")):
        CFG.platform.select = "openai"
        CFG.platform.struct = "mdl"

        (tmp_path / categorized_domains_minimal.name).write_text(
            categorized_domains_minimal.read_text(), encoding="utf8"
        )

        main(
            [
                str(cfg.dir.root / "data/paperoni_find_acr_mdl_dom.json"),
                "--categorized-terms",
                str(tmp_path / categorized_domains_minimal.name),
            ]
        )

        data_regression.check(
            json.loads(
                (
                    tmp_path
                    / categorized_domains_minimal.with_stem(
                        f"{categorized_domains_minimal.stem}_acronyms"
                    ).name
                ).read_text(encoding="utf8")
            )
        )


@pytest.mark.parametrize(
    ("acronym", "full_term", "is_in"),
    [
        # Terms not in the concurrent_terms list should be ignored
        ("nad", "not a domain", False),
        # Single char acronyms should be ignored
        ("r", "reinforcement learning", False),
        # Acronyms to itself should be ignored
        ("rl", "rl", False),
        ("rl", "reinforcement learning", True),
    ],
)
def test_query_in_concurrent_terms(
    monkeypatch, tmp_path, acronym, full_term, is_in, cfg: Config
):
    async def batch_queries(*args, **kwargs):
        response = empty_response(Response)
        acr = response.analysis.acronyms[0]
        acr.acronym_abbreviation.value = acronym
        acr.full_form.value = full_term
        return [response.model_copy()]

    monkeypatch.setattr(
        paperext.structured_output.find_acr_mdl_dom.query,
        "batch_queries",
        batch_queries,
    )

    categorized_domains_minimal = cfg.dir.data / "mdl/minimal_categorized_domains.json"

    with Config.push(Config(cfg.dir.root / "../config.mdl.ini")):
        CFG.platform.select = "openai"
        CFG.platform.struct = "mdl"

        (tmp_path / categorized_domains_minimal.name).write_text(
            categorized_domains_minimal.read_text(), encoding="utf8"
        )

        main(
            [
                str(cfg.dir.root / "data/paperoni_find_acr_mdl_dom.json"),
                "--categorized-terms",
                str(tmp_path / categorized_domains_minimal.name),
            ]
        )

        acronyms_map = json.loads(
            (
                tmp_path
                / categorized_domains_minimal.with_stem(
                    f"{categorized_domains_minimal.stem}_acronyms"
                ).name
            ).read_text(encoding="utf8")
        )

        assert (acronym in acronyms_map.keys()) == is_in
        assert (full_term in acronyms_map.values()) == is_in
