import json
from pathlib import Path

import pytest


from paperext.config import CFG, Config
from paperext.structured_output.mdl_cat_new_dom.model import (
    Response,
)
import paperext.structured_output.mdl_cat_new_dom.query
from paperext.structured_output.mdl_cat_new_dom.query import main


def test_query(monkeypatch, tmp_path, data_regression, cfg: Config):
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

    # magicmock = MagicMock()
    # magicmock.return_value = 100
    monkeypatch.setattr(
        paperext.structured_output.mdl_cat_new_dom.query, "batch_queries", batch_queries
    )
    monkeypatch.setattr(
        paperext.structured_output.mdl_cat_new_dom.query,
        "sanitize_categories",
        lambda x, *args, **kwargs: x,
    )
    monkeypatch.setattr(
        paperext.structured_output.mdl_cat_new_dom.query,
        "_make_sanitized_map",
        lambda *args, **kwargs: json.loads(sanitized_map.read_text()),
    )

    paperoni = (
        Path(__file__).parent / "paperoni-2022-01-01-2025-01-01-PR_2025-02-05.json"
    )
    categorized_domains = Path(__file__).parent / "categorized_domains.json"
    acronyms_domains = Path(__file__).parent / "acronyms_domains.json"
    sanitized_map = Path(__file__).parent / "sanitized_map.json"

    with Config.push(Config(cfg.dir.root / "../config.mdl.ini")):
        CFG.platform.select = "openai"
        CFG.platform.struct = "mdl"

        (tmp_path / paperoni.name).write_text(paperoni.read_text(), encoding="utf8")
        (tmp_path / categorized_domains.name).write_text(
            categorized_domains.read_text(), encoding="utf8"
        )
        (tmp_path / acronyms_domains.name).write_text(
            acronyms_domains.read_text(), encoding="utf8"
        )

        main(
            [
                str(tmp_path / paperoni.name),
                "--categorized-domains",
                str(tmp_path / categorized_domains.name),
                "--accronyms",
                str(tmp_path / acronyms_domains.name),
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

    # magicmock = MagicMock()
    # magicmock.return_value = 100
    monkeypatch.setattr(
        paperext.structured_output.mdl_find_acr.query, "batch_queries", batch_queries
    )

    categorized_domains_minimal = (
        cfg.dir.root / "data/mdl/categorized_domains_minimal.json"
    )

    with Config.push(Config(cfg.dir.root / "../config.mdl.ini")):
        CFG.platform.select = "openai"
        CFG.platform.struct = "mdl"

        (tmp_path / categorized_domains_minimal.name).write_text(
            categorized_domains_minimal.read_text(), encoding="utf8"
        )

        main(
            [
                str(cfg.dir.root / "data/paperoni_mdl_find_acr.json"),
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
