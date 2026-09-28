"""Generic, domain-agnostic helpers: token counting, JSON repair, prompt
budget trimming. Stateless utilities shared across modules.
"""

from __future__ import annotations

import functools
import json
import re
from typing import Optional

try:
    import tiktoken
except ImportError:  # tiktoken is optional at runtime
    tiktoken = None


# --- token counting (provider usage is preferred; this is the fallback) ---

@functools.lru_cache(maxsize=4)
def _encoder(name: str = "cl100k_base"):
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        return None


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)  # rough heuristic without tiktoken


# --- lenient JSON extraction/repair for LLM outputs ---

def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    obj = _try_load(s)
    if obj is not None:
        return obj
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        return _try_load(s[start : end + 1])
    return None


def _try_load(s: str) -> Optional[dict]:
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# --- prompt budget trimming (shapes what enters a prompt; DB stays truth) ---

def fit_text(text: str, max_tokens: int) -> str:
    if count_tokens(text) <= max_tokens:
        return text
    # Binary-search a character cut that fits the token budget.
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + " …"


def fit_items(items: list, max_tokens: int) -> list:
    """Drop trailing items until the JSON-encoded list fits the budget."""
    out = list(items)
    while out and count_tokens(json.dumps(out)) > max_tokens:
        out.pop()
    return out


def _size(obj) -> int:
    return count_tokens(json.dumps(obj, default=str))


def trim_ctx(ctx: dict, budget: int) -> dict:
    """Shrink an Attack-context dict toward the budget, low-value fields first."""
    out = dict(ctx)
    if _size(out) <= budget:
        return out
    for key in ("similar_cases", "failed_patterns", "partial_cases", "top_strategies"):
        while isinstance(out.get(key), list) and out[key] and _size(out) > budget:
            out[key] = out[key][:-1]
        if _size(out) <= budget:
            return out
    if isinstance(out.get("conv_summary"), str):
        out["conv_summary"] = fit_text(out["conv_summary"], max(budget // 2, 32))
    if isinstance(out.get("last_target_summary"), str):
        out["last_target_summary"] = fit_text(out["last_target_summary"], max(budget // 4, 16))
    return out
