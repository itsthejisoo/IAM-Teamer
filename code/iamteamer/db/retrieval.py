"""Compact SQL retrieval for the next-turn Attack context.

Reads summary tables (strategies, turns, refusal_patterns) and returns small
dicts. Raw llm_io is intentionally untouched here.
"""

from __future__ import annotations

from ..config import RetrievalLimits
from ..memory.embeddings import cosine, embed, from_blob
from ..memory.thompson import Posterior, rank
from .store import Store


def _score_per_1k(avg_score: float, avg_tokens: float) -> float:
    return (avg_score / avg_tokens * 1000.0) if avg_tokens > 0 else 0.0


def top_strategies(store: Store, k: int) -> list[dict]:
    rows = store.conn.execute(
        "SELECT * FROM strategies WHERE family=? AND trials>0"
        " ORDER BY (CAST(successes AS REAL)/trials) DESC, avg_elite_score DESC LIMIT ?",
        (store.family, k),
    ).fetchall()
    return [_strategy_summary(r) for r in rows]


def _strategy_summary(r) -> dict:
    sr = (r["successes"] / r["trials"]) if r["trials"] else 0.0
    return {
        "name": r["strategy_name"],
        "family": r["strategy_family"],
        "success_rate": round(sr, 3),
        "avg_score": round(r["avg_elite_score"], 2),
        "refusal_rate": round(r["refusal_rate"], 3),
        "avg_token_spent": round(r["avg_token_spent"], 1),
        "avg_time_spent_sec": round(r["avg_time_spent_sec"], 2),
        "score_per_1k_tokens": round(_score_per_1k(r["avg_elite_score"], r["avg_token_spent"]), 3),
        "cost_per_success": round(r["avg_token_spent"] / max(r["successes"], 1), 1),
        "note": (r["reasoning_summary"] or "")[:160],
    }


def similar_cases(store: Store, goal: str, k: int) -> list[dict]:
    vec = embed(goal)
    scored = []
    for sid, blob in store.load_embeddings():
        scored.append((cosine(vec, from_blob(blob)), sid))
    scored.sort(reverse=True)
    out = []
    for sim, sid in scored[:k]:
        r = store.conn.execute(
            "SELECT * FROM strategies WHERE strategy_id=?", (sid,)
        ).fetchone()
        if r:
            d = _strategy_summary(r)
            d["similarity"] = round(sim, 3)
            out.append(d)
    return out


def failed_patterns(store: Store, k: int) -> list[dict]:
    # strategy_name은 turn마다 자유 라벨이라 매번 달라짐 -> embedding dedup된
    # canonical strategy_id로 묶어야 같은 아이디어의 반복 실패가 잡힘
    rows = store.conn.execute(
        "SELECT s.strategy_name, s.strategy_family, COUNT(*) AS n FROM turns t"
        " JOIN strategies s ON s.strategy_id = t.strategy_id"
        " WHERE t.success=0 AND t.strategy_id IS NOT NULL GROUP BY t.strategy_id"
        " ORDER BY n DESC LIMIT ?",
        (k,),
    ).fetchall()
    return [{"name": r["strategy_name"], "family": r["strategy_family"], "fail_count": r["n"]} for r in rows]


def partial_cases(store: Store, k: int) -> list[dict]:
    rows = store.conn.execute(
        "SELECT t.strategy_name, t.refusal_summary, t.toxicity FROM turns t"
        " JOIN refusal_patterns r ON r.turn_id=t.turn_id"
        " WHERE r.partial_compliance=1 ORDER BY t.created_at DESC LIMIT ?",
        (k,),
    ).fetchall()
    return [
        {"name": r["strategy_name"], "toxicity": r["toxicity"], "note": (r["refusal_summary"] or "")[:120]}
        for r in rows
    ]


def posteriors(store: Store) -> list[Posterior]:
    rows = store.conn.execute(
        "SELECT * FROM strategies WHERE family=?", (store.family,)
    ).fetchall()
    out = []
    for r in rows:
        out.append(
            Posterior(
                strategy_id=r["strategy_id"],
                strategy_name=r["strategy_name"],
                family=r["strategy_family"],
                trials=r["trials"],
                successes=r["successes"],
                avg_score=r["avg_elite_score"],
                score_per_1k=_score_per_1k(r["avg_elite_score"], r["avg_token_spent"]),
            )
        )
    return out


def ts_hints(store: Store, k: int, seed: int | None = None) -> list[dict]:
    return rank(posteriors(store), k, seed)


def _factor_view(r) -> dict:
    return {
        "mechanism": r["mechanism"],
        "evidence": (r["evidence"] or "")[:160],
        "uses": r["occurrences"],
    }


def winning_mechanisms(store: Store, k: int) -> dict:
    """Distilled success mechanisms, scoped to avoid cross-strategy bleed.

    `composable` are family-agnostic techniques (safe to stack on any strategy);
    `by_family` keeps strategy-specific mechanisms grouped so the attacker only
    reuses the ones matching the family it commits to.
    """
    composable = [_factor_view(r) for r in store.top_factors("general", None, k)]
    by_family: dict[str, list] = {}
    for r in store.top_factors("strategy_specific", None, k * 4):
        bucket = by_family.setdefault(r["strategy_family"], [])
        if len(bucket) < k:
            bucket.append(_factor_view(r))
    out: dict = {}
    if composable:
        out["composable"] = composable
    if by_family:
        out["by_family"] = by_family
    return out


def transitions(store: Store, from_family: str, k: int) -> list[dict]:
    """Past family transitions (from_family -> X) that improved scores."""
    rows = store.top_transitions(from_family, k)
    return [
        {
            "from": r["from_family"],
            "to": r["to_family"],
            "trials": r["trials"],
            "win_rate": round(r["wins"] / r["trials"], 3) if r["trials"] else 0.0,
            "avg_gain": round(r["avg_gain"], 2),
        }
        for r in rows
    ]


def gather(store: Store, goal: str, limits: RetrievalLimits, seed: int | None = None) -> dict:
    out = {
        "top_strategies": top_strategies(store, limits.strategy_top_k),
        "similar_cases": similar_cases(store, goal, limits.case_top_k),
        "failed_patterns": failed_patterns(store, limits.failure_top_k),
        "partial_cases": partial_cases(store, limits.partial_top_k),
        "ts_hints": ts_hints(store, limits.strategy_top_k, seed),
    }
    mech = winning_mechanisms(store, limits.factor_top_k)
    if mech:
        out["success_mechanisms"] = mech
    return out
