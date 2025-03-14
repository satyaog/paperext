import json
from pathlib import Path

import pytest


from paperext.config import CFG, Config
from paperext.structured_output.find_acr_mdl_dom.model import (
    Response,
    empty_response,
)
import paperext.structured_output.find_acr_el.query
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


@pytest.mark.usefixtures("no_query")
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
        paperext.structured_output.find_acr_el.query, "batch_queries", batch_queries
    )

    categorized_domains_minimal = cfg.dir.data / "mdl/minimal_categorized_domains.json"

    with Config.push(Config(cfg.dir.root / "../config.mdl.ini")):
        CFG.platform.select = "openai"
        CFG.platform.struct = "mdl"

        (tmp_path / categorized_domains_minimal.name).write_text(
            categorized_domains_minimal.read_text()
        )

        main(
            [
                str(cfg.dir.root / "data/paperoni_find_acr_mdl_dom.json"),
                "--categorized-domains",
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
                ).read_text()
            )
        )


@pytest.mark.usefixtures("no_query")
def test_query_pipeline(monkeypatch, tmp_path, data_regression, cfg: Config):
    async def batch_queries(*args, **_kwargs):
        _, filename = args[1][0]
        return list(
            map(
                lambda x: Response.model_validate_json(x.read_text()),
                sorted(
                    (
                        cfg.dir.root
                        / "../data"
                        / CFG.platform.struct
                        / "queries"
                        / CFG.platform.select
                    ).glob(f"{filename}_*.json")
                ),
            )
        )

    monkeypatch.setattr(
        paperext.structured_output.find_acr_el.query, "batch_queries", batch_queries
    )

    paperoni = (
        Path(__file__).parent / "paperoni-2022-01-01-2025-01-01-PR_2025-02-05.json"
    )

    with Config.push(Config(cfg.dir.root / "../config.mdl.ini")):
        CFG.platform.select = "openai"
        CFG.platform.struct = "mdl"
        paperoni = CFG.dir.data / "paperoni-2022-01-01-2025-01-01-PR_2025-02-05.json"
        categorized_domains = (
            CFG.dir.data / CFG.platform.struct / "categorized_domains.json"
        )

        (tmp_path / paperoni.name).write_text(paperoni.read_text())
        (tmp_path / categorized_domains.name).write_text(
            categorized_domains.read_text()
        )

        main(
            [
                str(tmp_path / paperoni.name),
                "--categorized-domains",
                str(tmp_path / categorized_domains.name),
            ]
        )

        data_regression.check(
            json.loads((tmp_path / categorized_domains.name).read_text())
        )


@pytest.mark.usefixtures("no_query")
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
        paperext.structured_output.find_acr_el.query, "batch_queries", batch_queries
    )

    categorized_domains_minimal = cfg.dir.data / "mdl/minimal_categorized_domains.json"

    with Config.push(Config(cfg.dir.root / "../config.mdl.ini")):
        CFG.platform.select = "openai"
        CFG.platform.struct = "mdl"

        (tmp_path / categorized_domains_minimal.name).write_text(
            categorized_domains_minimal.read_text()
        )

        main(
            [
                str(cfg.dir.root / "data/paperoni_find_acr_mdl_dom.json"),
                "--categorized-domains",
                str(tmp_path / categorized_domains_minimal.name),
            ]
        )

        acronyms_map = json.loads(
            (
                tmp_path
                / categorized_domains_minimal.with_stem(
                    f"{categorized_domains_minimal.stem}_acronyms"
                ).name
            ).read_text()
        )

        assert (acronym in acronyms_map.keys()) == is_in
        assert (full_term in acronyms_map.values()) == is_in
