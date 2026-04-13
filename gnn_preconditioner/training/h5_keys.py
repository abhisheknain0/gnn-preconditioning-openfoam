from __future__ import annotations

from typing import Iterable
import re

_TRAJECTORY_RE = re.compile(r"^(.*?)(\d+)$")


def trajectory_numeric_id(key: str) -> int | None:
    match = _TRAJECTORY_RE.match(str(key))
    if match is None:
        return None
    return int(match.group(2))


def trajectory_sort_key(key: str) -> tuple[str, int | float, str]:
    key = str(key)
    numeric = trajectory_numeric_id(key)
    if numeric is None:
        return (key, float("inf"), key)
    prefix = key[: key.rfind(str(numeric))]
    return (prefix, numeric, key)


def sorted_trajectory_keys(keys: Iterable[str]) -> list[str]:
    return sorted((str(k) for k in keys), key=trajectory_sort_key)
