"""Loads prompt text/schemas from prompts.yaml (cached)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml

_PATH = Path(__file__).parent / "prompts.yaml"


@lru_cache(maxsize=1)
def _data() -> dict:
    with open(_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _node(keys: tuple[str, ...]):
    node = _data()
    for k in keys:
        node = node[k]
    return node


def text(*keys: str) -> str:
    return str(_node(keys)).rstrip("\n")


def schema_json(*keys: str) -> str:
    return json.dumps(_node(keys), indent=2)
