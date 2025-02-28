import argparse
import copy
from io import StringIO
import json
import math
from pathlib import Path
from pprint import pprint
from typing import Any

from paperext.config import CFG
from paperext.log import logger
from paperext.utils import split_entry


def _dict_heads(
    dictionary: dict[Any, dict],
    start_level: int = None,
    level: int = None,
):
    if start_level is None and level is None:
        return dictionary

    if level is None:
        level = math.inf
    if start_level is None:
        start_level = 0

    match max(0, level):
        case 0:
            return {}
        case _:
            heads = {
                key: _dict_heads(value, start_level=start_level - 1, level=level - 1)
                for key, value in dictionary.items()
            }
            if start_level <= 0:
                return heads
            else:
                d = {}
                for value in heads.values():
                    d.update(value)
                return d


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


def _update_sanitized_map(sanitized_map: dict[str, str], *keys, return_bare=False):
    for key in keys:
        if key in sanitized_map:
            continue

        sane_key = default_sanitize_key(key)
        sane_key, *_ = split_entry(sane_key, sep_left="(", sep_right=")")
        bare_key = bare_sanitize_key(sane_key)
        same_keys = [k for k in sanitized_map if _eq_keys(sane_key, k)]
        if same_keys:
            sane_key = _make_sane_key(sane_key, bare_key, *same_keys)

        sanitized_map.update(
            **{
                k: sane_key
                for k in (
                    key,
                    bare_key,
                    sane_key,
                    *same_keys,
                    # update accronyms sane key
                    *(k for k, v in sanitized_map.items() if _eq_keys(sane_key, v)),
                )
            }
        )

    return (
        (bare_sanitize_key(key) if return_bare else sanitized_map[key]) for key in keys
    )


def _make_sanitized_map(dict_or_list: dict[str, dict] | set[str]):
    if isinstance(dict_or_list, dict):
        keys = _flatten_dict(dict_or_list)
    else:
        keys = dict_or_list
    keys = sorted(set(keys))

    sanitized_key_map = {}

    while keys:
        if (key := keys.pop(0)) in sanitized_key_map:
            continue

        _update_sanitized_map(sanitized_key_map, key)

    return sanitized_key_map


def split_words(key: str, separators=" "):
    for sep in set("_" + separators):
        key = key.replace(sep, " ")
    for sep in "-_ ":
        key = sep.join([word.strip() for word in key.split(sep) if word.strip()])
    return key.split(" ")


# def _infer_words(key, *others):
#     key_others = (key, *others)
#     words_for_each = [split_words(k, separators=" -_") for k in key_others]
#     word_index = 0
#     while word_index < max(len(wfe) for wfe in words_for_each):
#         min_len = min([len(wfe[word_index]) for wfe in words_for_each])
#         for key_index, word in enumerate(wfe[word_index] for wfe in words_for_each):
#             if len(word) > min_len:
#                 words = [word[:min_len], word[min_len:]]
#                 words_for_each[key_index].pop(word_index)
#                 while words:
#                     words_for_each[key_index].insert(word_index, words.pop(-1))
#         word_index += 1

#     assert all(wfe == words_for_each[0] for wfe in words_for_each[1:])

#     return words_for_each[0]


def default_sanitize_key(key: str, replace="_"):
    return " ".join(split_words(key.lower(), separators=replace))


def bare_sanitize_key(key: str):
    return default_sanitize_key(key, replace="-_").replace(" ", "")


def _eq_keys(
    key: str,
    other: str,
    sanitize_key: callable = bare_sanitize_key,
):
    key = sanitize_key(key)
    other = sanitize_key(other)
    return key == other


def _make_sane_key(key, *others, sanitize_key: callable = default_sanitize_key):
    all_keys = [sanitize_key(o) for o in (key, *others)]

    all_chars = [list(words) for words in all_keys]

    char_index = 1
    while char_index < max(len(chars) for chars in all_chars):
        col = [chars[char_index] for chars in all_chars]
        sep = "-" if "-" in col else " "
        for key_index, c in enumerate(col):
            if set(col) & set(" -") and c not in " -":
                all_chars[key_index].insert(char_index, " ")
            all_chars[key_index][char_index] = (
                all_chars[key_index][char_index].strip() or sep
            )

        char_index += 1

    assert all(chars == all_chars[0] for chars in all_chars[1:])

    return "".join(all_chars[0])


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


def sanitize_categories(
    categories: dict[str, dict],
    accronyms: dict[str, str] = None,
    sanitized_map: dict[str, str] = None,
):
    categories = copy.deepcopy(categories)

    accronyms = accronyms or {}
    accronyms = accronyms.copy()

    sanitized_map = sanitized_map or {}
    sanitized_map = sanitized_map.copy()

    ignore = {k: {} for k in _flatten_dict(categories.pop("ignore", {}))}
    _update_sanitized_map(
        sanitized_map,
        *set(_flatten_dict(categories)),
        *accronyms.keys(),
        *accronyms.values(),
    )
    accronyms = {sanitized_map[k]: sanitized_map[v] for k, v in accronyms.items()}
    sanitized_map = {
        k: accronyms.get(sanitized_map[k], v) for k, v in sanitized_map.items()
    }

    ignore.update(
        **{k: {} for k in accronyms.keys()},
        **_sanitize_categories(
            categories,
            sanitize_key=lambda key: sanitized_map[key],
        ),
    )

    ignore = {k: v for k, v in ignore.items() if k not in sanitized_map}

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
        "--accronyms",
        type=Path,
        default=CFG.dir.data / "mdl_find_acr/acronyms_or_abbreviations_domains.json",
        help="Path to categorized domains",
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

    if options.accronyms:
        accronyms_map = json.loads(options.accronyms.read_text().lower())
    else:
        accronyms_map = None

    categories = json.loads(options.categorization.read_text())
    categories = {
        k.replace(" ", "_"): v
        for k, v in sanitize_categories(categories, accronyms_map).items()
    }

    (options.out.write_text if str(options.out) != "-" else print)(
        json.dumps(categories, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
