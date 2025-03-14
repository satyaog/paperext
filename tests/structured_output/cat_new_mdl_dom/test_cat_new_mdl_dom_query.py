import json
from pathlib import Path

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer


from paperext.config import CFG, Config
from paperext.sanitize_categorization import _flatten_dict
from paperext.structured_output._base import _EMPTY_FLAG
from paperext.structured_output.cat_new_el.query import _update_sorted_propositions
from paperext.structured_output.mdl.stats.build_domains_tree import (
    build_domains_dataframe,
    get_proposition,
)
from paperext.structured_output.mdl.stats.stats import load_analysis
from paperext.structured_output.cat_new_mdl_dom.model import Response, empty_response
import paperext.structured_output.cat_new_el.query
import paperext.structured_output.cat_new_mdl_dom.query
from paperext.structured_output.cat_new_mdl_dom.query import (
    main,
)


def gen_categorized_domains():
    categorized_domains = {"ignore": {"0:ignore": {}, "1:ignore": {}}}

    for i in ["abstract_research_topics", "application_domains"]:
        cat = {}
        categorized_domains[i] = cat
        for j in range(5):
            sub_cat = {}
            cat[f"{i.replace('_', ' ')}:{j}:entry"] = sub_cat
            for k in range(1):
                sub_cat[f"{i.replace('_', ' ')}:{j}:{k}:entry"] = {}
        # Hack to register _EMPTY_FLAG in the sanitized_map
        cat[_EMPTY_FLAG] = {}

    return categorized_domains


def test_update_sorted_propositions(cfg, data_regression):
    paperoni_path = (
        Path(__file__).parent / "paperoni-2022-01-01-2025-01-01-PR_2025-02-05.json"
    )

    domains = json.loads(
        (cfg.dir.root / "data/mdl/minimal_categorized_domains.json").read_text()
    )
    df = build_domains_dataframe(domains)

    with Config.push(Config(cfg.dir.root / "../config.mdl.ini")):
        CFG.platform.select = "openai"
        CFG.platform.struct = "mdl"
        analysis, _ = load_analysis(
            json.loads(paperoni_path.read_text())[:20],
            CFG.dir.queries / CFG.platform.select,
        )

    remainings = sorted(
        (set(analysis["attrs"]["research_fields"].explode()) - set(df["domain"]))
    )

    data_regression.check(remainings, "remainings")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    entries = sorted((set(remainings) | set(df["domain"])))
    embeddings = model.encode(entries)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = model.similarity(embeddings, embeddings)
    similarities = {
        (domain, other): similarities[i, j]
        for i, domain in enumerate(entries)
        for j, other in enumerate(entries)
    }

    all_distances = get_proposition(
        remainings,
        None,
        k=100,
        df=df,
        similarities=similarities,
    )

    while remainings:
        remaining = all_distances[0][1]
        domains["ignore"][remaining] = {}
        remainings.remove(remaining)
        df = build_domains_dataframe(domains)

        all_distances = _update_sorted_propositions(
            all_distances, similarities, [], k=100
        )
        assert (
            get_proposition(
                remainings,
                None,
                k=100,
                df=df,
                similarities=similarities,
            )
            == all_distances
        )


@pytest.mark.usefixtures("no_query")
@pytest.mark.parametrize(
    (
        "new_domain",
        "sem_equ_domains",
        "closest_par_domain",
        "closest_chi_domain",
        "closest_sib_domain",
    ),
    [
        # semantically_equivalent_domains should be selected
        (
            "application domains:2:0:0:sem_equ_domain",
            ["application domains:2:0:entry", "application domains:3:entry"],
            (1, "abstract research topics:1:entry"),
            (2, "application domains:2:entry"),
            (2, "abstract research topics:4:0:entry"),
        ),
        # closest_parent_domain should be selected
        (
            "abstract research topics:1:closest_par_domain",
            [],
            (1, "abstract research topics:1:entry"),
            (2, "application domains:2:entry"),
            (2, "abstract research topics:4:0:entry"),
        ),
        # closest_child_domain should be selected
        (
            "application domains:2:closest_chi_domain",
            [],
            (2, "abstract research topics:1:entry"),
            (1, "application domains:2:entry"),
            (2, "abstract research topics:4:0:entry"),
        ),
        # closest_sibling_domain should be selected
        (
            "abstract research topics:4:0:closest_sib_domain",
            [],
            (2, "abstract research topics:1:entry"),
            (2, "application domains:2:entry"),
            (1, "abstract research topics:4:0:entry"),
        ),
        # skip the new domain
        (
            "skipped",
            [],
            (False, "abstract research topics:1:entry"),
            (False, "application domains:2:entry"),
            (False, "abstract research topics:4:0:entry"),
        ),
    ],
)
def test_query(
    new_domain,
    sem_equ_domains,
    closest_par_domain,
    closest_chi_domain,
    closest_sib_domain,
    cfg: Config,
    monkeypatch,
    data_regression,
    tmp_path,
):
    def find_selection(categorized_domains: dict[str, str], selection: str):
        *keys, _ = selection.split(":")
        parent = categorized_domains
        index = []
        while keys:
            index.append(keys.pop(0))
            for _k in sorted(parent):
                if selection in parent[_k]:
                    return parent[_k][selection], parent[_k]
                elif index == _k.replace("_", " ").split(":")[:1] + _k.split(":")[1:-1]:
                    parent = parent[_k]
                    break

    async def batch_queries(*_, **__):
        response: Response = empty_response(Response)
        _empty_dom = response.analysis.closest_parent_domain.model_copy()

        response.analysis.semantically_equivalent_domains = []
        for d in sem_equ_domains:
            _empty_dom.value = d
            response.analysis.semantically_equivalent_domains.append(
                _empty_dom.model_copy()
            )
        _empty_dom.value = closest_par_domain[1] if closest_par_domain[0] else ""
        response.analysis.closest_parent_domain = _empty_dom.model_copy()
        _empty_dom.value = closest_chi_domain[1] if closest_chi_domain[0] else ""
        response.analysis.closest_child_domain = _empty_dom.model_copy()
        _empty_dom.value = closest_sib_domain[1] if closest_sib_domain[0] else ""
        response.analysis.closest_sibling_domain = _empty_dom.model_copy()
        return [response.model_copy()]

    monkeypatch.setattr(
        paperext.structured_output.cat_new_el.query, "batch_queries", batch_queries
    )

    monkeypatch.setattr(
        paperext.structured_output.cat_new_el.query,
        "get_proposition",
        lambda *_, **__: (
            [
                0,
                new_domain,
                *map(
                    lambda x: x[1],
                    sorted(
                        (closest_par_domain, closest_chi_domain, closest_sib_domain)
                    ),
                ),
                *sem_equ_domains,
                *[_EMPTY_FLAG] * (7 - len(sem_equ_domains)),
            ],
        ),
    )

    domains_filename = "min_cat_dom.json"

    with Config.push(Config(cfg.dir.root / "../config.mdl.ini")):
        CFG.platform.select = "openai"
        CFG.platform.struct = "mdl"

        (tmp_path / domains_filename).write_text(
            json.dumps(gen_categorized_domains(), indent=2, sort_keys=True),
        )

        main(
            [
                str(cfg.dir.root / "data/paperoni_find_acr_mdl_dom.json"),
                "--categorized-domains",
                str(tmp_path / domains_filename),
            ]
        )

        categorized_domains = json.loads((tmp_path / domains_filename).read_text())

        if sem_equ_domains:
            assert (
                new_domain in find_selection(categorized_domains, sem_equ_domains[0])[0]
            ), f"Semantically equivalent [{new_domain}] not under [{sem_equ_domains[0]}]"

        elif closest_par_domain[0] == 1:
            assert (
                new_domain
                in find_selection(categorized_domains, closest_par_domain[1])[0]
            ), f"[{new_domain}] not under parent domain [{closest_par_domain[1]}]"

        elif closest_chi_domain[0] == 1:
            assert (
                closest_chi_domain[1]
                in find_selection(categorized_domains, new_domain)[0]
            ), f"[{closest_chi_domain[1]}] child domain not under [{new_domain}]"

        elif closest_sib_domain[0] == 1:
            assert (
                find_selection(categorized_domains, new_domain)[1]
                is find_selection(categorized_domains, closest_sib_domain[1])[1]
            ), f"[{new_domain}] not a sibling of [{closest_sib_domain[1]}]"

        else:
            assert new_domain not in set(_flatten_dict(categorized_domains))

        data_regression.check(
            json.loads((tmp_path / domains_filename).read_text(encoding="utf8"))
        )


@pytest.mark.usefixtures("no_query")
def test_query_full_pipeline(monkeypatch, tmp_path, data_regression, cfg: Config):
    # TODO: Fix this test
    return

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
        paperext.structured_output.cat_new_el.query, "batch_queries", batch_queries
    )
    monkeypatch.setattr(
        paperext.structured_output.cat_new_mdl_dom.query,
        "sanitize_categories",
        lambda x, *args, **kwargs: x,
    )
    monkeypatch.setattr(
        paperext.structured_output.cat_new_mdl_dom.query,
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
            json.loads((tmp_path / categorized_domains.name).read_text(encoding="utf8"))
        )


# @pytest.mark.parametrize(
#     ("acronym", "full_term", "is_in"),
#     [
#         # Terms not in the concurrent_terms list should be ignored
#         ("nad", "not a domain", False),
#         # Single char acronyms should be ignored
#         ("r", "reinforcement learning", False),
#         # Acronyms to itself should be ignored
#         ("rl", "rl", False),
#         ("rl", "reinforcement learning", True),
#     ],
# )
# def test_query_in_concurrent_terms(
#     monkeypatch, tmp_path, acronym, full_term, is_in, cfg: Config
# ):
#     async def batch_queries(*args, **kwargs):
#         response = empty_response(Response)
#         acr = response.analysis.acronyms[0]
#         acr.acronym_abbreviation.value = acronym
#         acr.full_form.value = full_term
#         return [response.model_copy()]

#     # magicmock = MagicMock()
#     # magicmock.return_value = 100
#     monkeypatch.setattr(
#         paperext.structured_output.mdl_find_acr.query, "batch_queries", batch_queries
#     )

#     categorized_domains_minimal = (
#         cfg.dir.root / "data/mdl/categorized_domains_minimal.json"
#     )

#     with Config.push(Config(cfg.dir.root / "../config.mdl.ini")):
#         CFG.platform.select = "openai"
#         CFG.platform.struct = "mdl"

#         (tmp_path / categorized_domains_minimal.name).write_text(
#             categorized_domains_minimal.read_text(), encoding="utf8"
#         )

#         main(
#             [
#                 str(cfg.dir.root / "data/paperoni_mdl_find_acr.json"),
#                 "--categorized-terms",
#                 str(tmp_path / categorized_domains_minimal.name),
#             ]
#         )

#         acronyms_map = json.loads(
#             (
#                 tmp_path
#                 / categorized_domains_minimal.with_stem(
#                     f"{categorized_domains_minimal.stem}_acronyms"
#                 ).name
#             ).read_text(encoding="utf8")
#         )

#         assert (acronym in acronyms_map.keys()) == is_in
#         assert (full_term in acronyms_map.values()) == is_in
