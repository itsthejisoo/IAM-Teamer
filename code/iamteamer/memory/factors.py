"""Persist distilled success factors with embedding-based dedup.

Mirrors strategy.upsert: the Judge supplies success mechanisms (in its scoring
call); the DB assigns ids and merges near-duplicates (bumping occurrences) so the
factor library stays compact. Dedup is scoped per (family, scope) to keep
cross-strategy bleed out.
"""

from __future__ import annotations

import sqlite3

from ..db import Store
from .embeddings import cosine, embed, from_blob, to_blob
from ..models import FactorOut

DEFAULT_FACTOR_DEDUP = 0.6
MAX_FACTORS = 4


def _best_match(
    store: Store, vec: list[float], scope: str, family: str, threshold: float
) -> str | None:
    best_id, best = None, 0.0
    for fid, blob in store.factor_embeddings(scope, family):
        s = cosine(vec, from_blob(blob))
        if s > best:
            best_id, best = fid, s
    return best_id if best >= threshold else None


def persist_factors(
    store: Store, factors: list[FactorOut], *, family: str, strategy_id: str | None,
    goal: str, run_id: str, turn_id: str, conn: sqlite3.Connection,
    threshold: float = DEFAULT_FACTOR_DEDUP,
) -> None:
    for f in factors[:MAX_FACTORS]:
        mech = f.mechanism.strip()
        if not mech:
            continue
        vec = embed(mech)
        match_id = _best_match(store, vec, f.scope, family, threshold)
        if match_id:
            store.bump_factor(match_id, conn)
        else:
            store.insert_factor(
                strategy_id, family, f.scope, mech, f.evidence.strip(),
                goal, run_id, turn_id, to_blob(vec), conn,
            )
