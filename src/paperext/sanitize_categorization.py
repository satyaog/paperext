import argparse
import copy
from io import StringIO
import json
from pathlib import Path
from pprint import pprint
from typing import Any

from paperext.log import logger
from paperext.utils import split_entry


def _dict_heads(dictionary: dict[Any, dict], level: int = None):
    match max(0, level):
        case None:
            return dictionary
        case 0:
            return {}
        case _:
            return {
                key: _dict_heads(value, level=level - 1)
                for key, value in sorted(dictionary.items())
            }


def _flatten_dict(dictionary: dict[Any, dict]):
    for key, value in dictionary.items():
        yield key
        yield from _flatten_dict(value)


def _merge_dict(first: dict[Any, dict], other: dict[Any, dict]):
    other = other.copy()

    merged = {}
    for key in first.keys():
        if key in other:
            merged[key] = _merge_dict(first[key], other.pop(key))
        else:
            merged[key] = first[key]

    return {**merged, **other}


def _debug_msg_merge_dict(key: str | list, first: dict, other: dict):
    _first = StringIO()
    _other = StringIO()
    _result = StringIO()
    pprint(_dict_heads(first, level=2), _first)
    pprint(_dict_heads(other, level=2), _other)
    pprint(_dict_heads(_merge_dict(first, other), level=2), _result)

    return (
        f"Merging of [{key}]\n{_first.getvalue().strip()} with\n{_other.getvalue().strip()}\n"
        f"Result is\n{_result.getvalue().strip()}"
    )


def _update_sanitized_map(sanitized_key_map: dict[str, str], *keys):
    for key in keys:
        if key in sanitized_key_map:
            continue

        sane_key = default_sanitize_key(key)
        sane_key, *_ = split_entry(sane_key, sep_left="(", sep_right=")")
        same_keys = [k for k in sanitized_key_map if _eq_keys(sane_key, k)]

        if same_keys:
            sane_key = _make_sane_key(sane_key, *same_keys)

        sanitized_key_map.update(**{k: sane_key for k in (key, sane_key, *same_keys)})

    return sanitized_key_map


def _make_sanitized_map(dict_or_keys: dict[str, dict] | set[str]):
    if isinstance(dict_or_keys, dict):
        keys = _flatten_dict(dict_or_keys)
    else:
        keys = dict_or_keys
    keys = sorted(set(keys))

    sanitized_key_map = {}

    while keys:
        if (key := keys.pop(0)) in sanitized_key_map:
            continue

        _update_sanitized_map(sanitized_key_map, key)

    return sanitized_key_map


def split_words(key: str, separators=" "):
    for sep in separators:
        key = key.replace(sep, " ")
    return [word.strip() for word in key.split(" ") if word.strip()]


def default_sanitize_key(key: str, replace="_"):
    return " ".join(split_words(key.lower(), separators=replace))


def _eq_keys(
    key: str,
    other: str,
    sanitize_key: callable = lambda key: default_sanitize_key(key.replace("-", " ")),
):
    key = sanitize_key(key)
    other = sanitize_key(other)
    return key == other


def _make_sane_key(key, *others, sanitize_key: callable = default_sanitize_key):
    key = sanitize_key(key)
    others = [sanitize_key(o) for o in others]

    dash_locs = set(
        sum([[i for i, c in enumerate(k) if c == "-"] for k in (key, *others)], [])
    )

    key = "".join((("-" if i in dash_locs else c) for i, c in enumerate(key)))

    while (_ := key.replace("-" * 2, "-")) != key:
        key = _

    return key


def _sanitize_categories(
    categories: dict[str, dict],
    ignore: dict[str, dict] = None,
    sanitize_key: callable = default_sanitize_key,
):
    if ignore is None:
        ignore = {}

    keys = sorted(categories.keys())

    while keys:
        key = keys.pop(0)
        sane_key = sanitize_key(key)
        _, *_ignore = split_entry(key, sep_left="(", sep_right=")")

        ignore.update(**{k: {} for k in _ignore})

        if _same_keys := [k for k in keys if _eq_keys(key, k, sanitize_key)]:
            logger.debug(f"Duplicates of [{key}] in {_same_keys}")

        _sanitize_categories(categories[key], ignore, sanitize_key)

        for other in keys:
            _sanitize_categories(categories[other], ignore, sanitize_key)

            if _eq_keys(key, other, sanitize_key):
                logger.debug(
                    _debug_msg_merge_dict(
                        (key, other), categories[key], categories[other]
                    )
                )
                categories[key] = _merge_dict(categories[key], categories.pop(other))

        keys = [k for k in keys if k in categories]

        categories[sane_key] = categories.pop(key)

    for key, values in list(categories.items()):
        _sanitize_categories(values, ignore, sanitize_key)

        if key in values:
            # Pull up categories
            _subs = values.pop(key)
            logger.debug(_debug_msg_merge_dict(key, values, _subs))
            categories[key] = _merge_dict(values, _subs)

    return ignore


def sanitize_categories(categories: dict[str, dict]):
    categories = copy.deepcopy(categories)

    ignore = {k: {} for k in _flatten_dict(categories.pop("ignore", {}))}
    sanitized_key_map = _make_sanitized_map(categories)

    ignore.update(
        **_sanitize_categories(
            categories,
            sanitize_key=lambda key: sanitized_key_map[key],
        )
    )

    ignore = {k: v for k, v in ignore.items() if k not in sanitized_key_map}

    if ignore:
        _sanitized_ignore_map = _make_sanitized_map(ignore)
        _sanitize_categories(
            ignore,
            sanitize_key=lambda key: _sanitized_ignore_map[key],
        )

    categories["ignore"] = ignore

    return categories


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "categorization",
        metavar="PATH",
        type=Path,
        help="Path to categorization JSON file",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        type=Path,
        default=None,
        help="Path to sanitized categorization JSON file",
    )
    options = parser.parse_args(argv)
    options.out = options.out or options.categorization

    domains = json.loads(options.categorization.read_text())
    domains = {k.replace(" ", "_"): v for k, v in sanitize_categories(domains).items()}

    (options.out.write_text if str(options.out) != "-" else print)(
        json.dumps(domains, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
