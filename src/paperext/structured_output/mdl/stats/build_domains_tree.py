# Read sources and tree.
# Find source with closest hamming distance.
# Give 10 proposition to merge them.
# Otherwise move to next best match.
import argparse
import copy
import json
import os
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Generator, List, Tuple

import Levenshtein
import pandas as pd

from paperext.config import CFG

from paperext.structured_output.mdl.stats.stats import load_analysis


def build_domains_dataframe(domains):
    def build_domains_tree(domains, parent=None) -> List:
        rows = []
        for key in domains:
            rows.append({"domain": key})
            if domains[key]:
                subdomains = build_domains_tree(domains[key])
                for subdomain in subdomains:
                    subdomain[f"parent{len(subdomain)}"] = key
                rows.extend(subdomains)

        return rows

    return pd.DataFrame(build_domains_tree(domains))


def is_probably_an_acronym(domain_name):
    return False  # len(domain_name) < 5 and " " not in domain_name


def levenshtein_for_sequence(remaining, domain):
    remaining = remaining.replace("-", " ")
    domain = domain.replace("-", " ")
    dists = defaultdict(list)
    for a in remaining.split(" "):
        for b in domain.split(" "):
            dists[b].append(Levenshtein.distance(a, b))

    dists = [(min(v), k) for k, v in dists.items()]
    sorted(dists)

    return sum(dists)


def get_proposition(remainings, domains, k=10, df=None):
    if df is None:
        df = build_domains_dataframe(domains)
    propositions = []
    for remaining in remainings:
        remaining_domain_is_probably_an_acronym = is_probably_an_acronym(remaining)
        distances = []
        for domain in list(df["domain"]):
            if remaining_domain_is_probably_an_acronym and not is_probably_an_acronym(
                domain
            ):
                distance = Levenshtein.distance(
                    remaining, "".join(m[0] for m in domain.split(" "))
                )
            elif remaining_domain_is_probably_an_acronym:
                continue
            else:
                distance = Levenshtein.distance(remaining.lower(), domain.lower())

            distances.append((distance, domain))

        distances = sorted(distances)
        propositions.append(
            (distances[0][0], remaining) + tuple(d[1] for d in distances[:k])
        )

    # longest_first = [p for p in propositions if len(p[1]) > 4]
    # if longest_first:
    #     propositions = longest_first

    return sorted(propositions)


def get_domain(domain, domains, df=None):
    if df is None:
        df = build_domains_dataframe(domains)

    try:
        domain_row = df[df["domain"] == domain].iloc[0]
    except:
        import pdb

        pdb.set_trace()
    for parent in domain_row[::-1]:
        if pd.isnull(parent):
            continue
        domains = domains[parent]

    return domains


def truncate_dict(d, depth):
    if depth == 0:
        return {}

    if isinstance(d, dict):
        return {k: truncate_dict(v, depth - 1) for k, v in d.items()}
    else:
        return d


def chose_proposition(
    remaining, propositions, papers: pd.DataFrame, domains, df
) -> Generator[Tuple[str, str], None, None]:
    if isinstance(remaining, tuple):
        remaining, *extra_context = remaining
    else:
        extra_context = []

    papers_exploded = papers.explode("research_fields")
    titles = list(
        papers_exploded[
            papers_exploded["research_fields"].isin([remaining, *extra_context])
        ]["title"].unique()
    )
    related_papers = papers[papers["title"].isin(titles)]
    fields = related_papers["research_fields"].explode()

    # for f in fields.unique():
    #     if "".join([part[0] for part in f.split(" ") if part]).upper() == remaining:
    #         print(f"Dropping [{remaining}] in favor of [{f}].")
    #         return "drop", "drop"
    # else:
    #     return "", ""

    pd.options.display.max_rows = 999
    pd.options.display.max_colwidth = 300

    print()
    print("--------------------")
    print()
    print(f"Categorizing '{remaining}'")
    print()
    print("Concurrent research fields proportions:")
    fields.isin([remaining, *extra_context])
    counts = fields[~fields.isin([remaining, *extra_context])].value_counts()
    counts /= related_papers.shape[0]
    print(
        counts[~counts.isin([remaining, *extra_context])]
        .to_frame()
        .reset_index()
        .to_string(index=False, header=False)
    )
    print()
    print("Titles for this domain:")
    print(
        papers[papers["title"].isin(titles)][["title", "research_fields"]].to_string(
            index=False
        )
    )

    valid_choices = list(range(len(propositions)))
    choice = None
    while choice not in valid_choices:
        if choice is not None:
            print()
            print(f"Categorizing '{remaining}'")
        print()
        print(f"Propositions:")
        print("\n".join(f"{i}: {d}" for i, d in enumerate(propositions)))
        print(
            f"a: drop acronym/abreviation [{remaining}] to categorize the complete terms"
        )
        print("d: drop")
        print("s: skip")
        print("l: list titles for this domain")
        print("r: search")
        print("t{number}: print current domains tree with depth 'number'")
        print("wq: save and exit")
        print("q: exit without saving")
        print("pdb: debug")
        print()

        choice = input("Choose a proposition: ")
        if choice == "a":
            yield ("drop", "drop")
            complete_terms = input(f"Enter the complete terms for [{remaining}]: ")
            _propositions = get_proposition([complete_terms], domains, df=df)[0][2:]
            if complete_terms and complete_terms.lower() != _propositions[0].lower():
                for result in chose_proposition(
                    (complete_terms, remaining),
                    _propositions + propositions[10:],
                    papers,
                    domains,
                    df,
                ):
                    yield (*result, complete_terms)
            break
        if choice == "s":
            yield ("", "")
            break
        elif choice == "wq":
            yield ("wq", "wq")
            break
        elif choice == "q":
            yield ("q", "q")
            break
        elif choice.startswith("t"):
            try:
                tmp = copy.deepcopy(domains)
                tmp.pop("ignore")
                pprint(truncate_dict(tmp, int(choice[1:])))
            except:
                print("Invalid depth")
        elif choice == "l":
            # papers_exploded = papers.explode("research_fields")
            # titles = list(
            #     papers_exploded[
            #         papers_exploded["research_fields"].isin([remaining, *extra_context])
            #     ]["title"].unique()
            # )
            pd.options.display.max_rows = 999
            pd.options.display.max_colwidth = 300
            print(papers[papers["title"].isin(titles)][["title", "research_fields"]])
            continue
        elif choice == "r":
            look_for = input("Search: ")
            propositions = get_proposition([look_for], domains, df=df)
            propositions = propositions[0][2:]
            continue
        elif choice == "d":
            yield ("drop", "drop")
            break
        elif choice == "pdb":
            import pdb

            pdb.set_trace()

        try:
            choice = int(choice)
        except:
            continue

        if choice >= len(propositions):
            continue

        chosen_domain = propositions[choice]
        parent = df[df["domain"] == chosen_domain]["parent1"]
        if parent.isnull().any():
            parent = ""
        else:
            print(parent)
            try:
                parent = parent.item()
            except ValueError:
                parent = parent[0]

        yield (chosen_domain, parent)
        return


def chose_action(chosen_domain, parent, domains, df):

    print()
    print(f"Chosen {chosen_domain}, which is in current context:")
    if parent:
        pprint({parent: get_domain(parent, domains, df)})
    else:
        pprint({chosen_domain: get_domain(chosen_domain, domains, df)})

    valid_choices = list("uapns")
    choice = None

    while choice not in valid_choices:
        print()
        print("u: under chosen domain")
        print("a: above chosen domain")
        print("p: above all siblings of chosen domain")
        print("n: next to chosen domain")
        print("s: skip")

        choice = input("Choose an action: ")

    return choice


def handle_propositions(remaining, propositions, domains, papers, df=None):
    if df is None:
        df = build_domains_dataframe(domains)

    domains_groups = [key for key in domains.keys() if key != "ignore"]
    top_domains = sum([list(domains[key].keys()) for key in domains_groups], [])
    propositions = propositions + tuple(sorted(top_domains)) + tuple(domains_groups)

    for chosen_domain, parent, *_remaining in chose_proposition(
        remaining, propositions, papers, domains, df
    ):
        if _remaining:
            remaining = _remaining.pop(0)

        if not chosen_domain:
            yield "skip"
            continue

        if chosen_domain == "wq" and parent == "wq":
            yield "wq"
            continue

        if chosen_domain == "q" and parent == "q":
            yield "q"
            continue

        if chosen_domain == "drop" and parent == "drop":
            if "ignore" not in domains:
                domains["ignore"] = {}

            domains["ignore"][remaining] = {}
            yield "drop"
            continue

        choice = chose_action(chosen_domain, parent, domains, df)

        print(choice)

        # add under
        if choice == "u":
            get_domain(chosen_domain, domains, df)[remaining] = {}
        # add above
        elif choice == "a":
            parent_group = get_domain(parent, domains, df)
            chosen_group = parent_group.pop(chosen_domain)
            parent_group[remaining] = {chosen_domain: chosen_group}
        # add above siblings
        elif choice == "p":
            grand_parent = df[df["domain"] == chosen_domain]["parent2"]
            grand_parent_group = get_domain(grand_parent.item(), domains, df)
            parent_group = grand_parent_group.pop(parent)
            grand_parent_group[parent] = {remaining: parent_group}
        # Add with siblings
        elif choice == "n":
            get_domain(parent, domains, df)[remaining] = {}
        # Skip for now
        elif choice == "s":
            yield "skip"
            continue

        grand_parent = df[df["domain"] == chosen_domain]["parent2"]
        if not grand_parent.isnull().any():
            pprint(get_domain(grand_parent.item(), domains, df))
        elif parent:
            pprint(get_domain(parent, domains, df))
        else:
            pprint(get_domain(chosen_domain, domains, df))


def print_status(skipped, remainings, domains, df=None):
    if df is None:
        df = build_domains_dataframe(domains)

    total = len(skipped) + len(remainings) + df.shape[0]
    print(f"{len(skipped) + len(remainings)} / {total} remainings")


def any_remainings(domains_df, analysis, skipped):
    return list(
        (
            set(analysis["attrs"]["research_fields"].explode())
            - set(domains_df["domain"])
        )
        - set(skipped)
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--papers",
        nargs="*",
        type=Path,
        default=[],
        help="Paperoni json report of papers to analyse",
    )
    parser.add_argument(
        "--categorized-domains",
        type=Path,
        default=CFG.dir.data / "mdl/categorized_domains.json",
        help="Path to categorized domains",
    )

    options = parser.parse_args(argv)

    domains = json.loads(options.categorized_domains.read_text())

    papers = []
    for papers_json_path in options.papers:
        papers.extend(json.loads(Path(papers_json_path).read_text()))

    analysis, _ = load_analysis(papers, CFG.dir.queries / CFG.platform.select)

    print(analysis["attrs"].shape)

    skipped = []
    force_end = False

    df = build_domains_dataframe(domains)
    remainings = any_remainings(df, analysis, skipped)

    while remainings and not force_end:
        print_status(skipped, remainings, domains)
        distances = get_proposition(remainings, domains, df=df)
        remaining = distances[0][1]

        if distances[0][0:2] == (0, distances[0][2]):
            print(f"[{remaining}] already in categorized domains. Removing it.")

        elif (distances[0][0], distances[0][1].lower()) == (
            0,
            distances[0][2].lower(),
        ):
            get_domain(distances[0][2], domains, df)[remaining] = {}
            print(
                f"[{remaining}] inserted under [{distances[0][2]}] in categorized domains."
            )
            pprint(get_domain(distances[0][2], domains, df))

        else:
            for rval in handle_propositions(
                remaining, distances[0][2:], domains, analysis["attrs"], df=df
            ):
                if rval == "skip":
                    skipped.append(remaining)
                elif rval == "wq":
                    # Will end loop
                    force_end = True
                    skipped.append(remaining)
                elif rval == "q":
                    return

        df = build_domains_dataframe(domains)
        remainings = any_remainings(df, analysis, skipped)

        options.categorized_domains.with_suffix(".tmp").write_text(
            json.dumps(domains, indent=2, sort_keys=True)
        )

    if options.categorized_domains.with_suffix(".tmp").exists():
        os.rename(
            options.categorized_domains.with_suffix(".tmp"),
            options.categorized_domains,
        )


if __name__ == "__main__":
    main()
