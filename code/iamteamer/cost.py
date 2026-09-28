"""Cost domain: model pricing + per-run cost tracking.

Prices are fetched live from OpenRouter's public `/models` endpoint and cached
for the process lifetime, so new models are priced automatically. The static
`PRICES` table is only an offline fallback (no network/key, tests). Unknown
models cost 0 + warn. Cost tracking prefers provider usage and falls back to a
tokenizer estimate.

Lookups normalize a short name or a raw OpenRouter slug to the same slug, so
both `gpt-4o` and `openai/gpt-4o` resolve identically.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass

from . import keys
from .models import CostRec, LlmResult, Usage
from .providers.openrouter_provider import _SLUGS
from .utils import count_tokens

log = logging.getLogger(__name__)

# 오프라인 폴백 가격표. OpenRouter 슬러그가 키, 값은 (1M당 입력가, 1M당 출력가).
PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-5.1": (1.25, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
}

_MODELS_URL = f"{keys.OPENROUTER_BASE_URL}/models"
_FETCH_TIMEOUT = 6.0

# False면 라이브 조회를 끄고 오프라인 폴백 테이블만 사용(conftest가 테스트에서 끔).
LIVE_FETCH = True

# 라이브 OpenRouter 가격 캐시(지연 초기화). None은 아직 미조회 상태.
_live: dict[str, tuple[float, float]] | None = None


def _fetch_live() -> dict[str, tuple[float, float]]:
    """Fetch per-1M prices for all OpenRouter models. {} on any failure."""
    try:
        req = urllib.request.Request(_MODELS_URL, headers={"User-Agent": "iamteamer"})
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
            data = json.load(resp)
    except Exception as e:
        log.warning("OpenRouter price fetch failed (%s); using fallback table.", e)
        return {}
    out: dict[str, tuple[float, float]] = {}
    for m in data.get("data", []):
        slug = m.get("id")
        p = m.get("pricing") or {}
        try:
            # OpenRouter는 토큰당 USD로 표기 → 1M 단위로 환산.
            p_in = float(p.get("prompt", 0.0)) * 1_000_000.0
            p_out = float(p.get("completion", 0.0)) * 1_000_000.0
        except (TypeError, ValueError):
            continue
        if slug:
            out[slug] = (p_in, p_out)
    return out


def _live_prices() -> dict[str, tuple[float, float]]:
    global _live
    if _live is None:
        _live = _fetch_live() if LIVE_FETCH else {}
    return _live


def _to_slug(model: str) -> str:
    key = model.lower()
    return _SLUGS.get(key, key)


def price_for(model: str) -> tuple[float, float]:
    slug = _to_slug(model)
    live = _live_prices()
    if slug in live:
        return live[slug]
    if slug in PRICES:
        return PRICES[slug]
    log.warning("No price for model %r; treating cost as 0.", model)
    return (0.0, 0.0)


def calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p_in, p_out = price_for(model)
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000.0


def usage_or_estimate(res: LlmResult, prompt_text: str) -> Usage:
    u = res.usage
    if u.total > 0:
        return u
    return Usage(
        input_tokens=count_tokens(prompt_text),
        output_tokens=count_tokens(res.text),
    )


def cost_of(res: LlmResult, prompt_text: str) -> CostRec:
    u = usage_or_estimate(res, prompt_text)
    return CostRec(
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cost=calc_cost(res.model, u.input_tokens, u.output_tokens),
        latency_sec=res.latency_sec,
    )


@dataclass
class CostTracker:
    """Accumulates spend for one run (tracking only, no budget ceiling)."""

    total_cost: float = 0.0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_time_sec: float = 0.0

    def add(self, rec: CostRec) -> None:
        self.total_cost += rec.cost
        self.total_tokens += rec.total_tokens
        self.input_tokens += rec.input_tokens
        self.output_tokens += rec.output_tokens
        self.total_time_sec += rec.latency_sec
