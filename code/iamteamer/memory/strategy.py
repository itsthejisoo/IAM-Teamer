"""Strategy upsert with embedding-based dedup.

Attack LLM never supplies a strategy_id; the DB assigns or matches one here.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from ..config import EMBEDDING_MODEL, DedupThresholds
from ..db import Store
from .embeddings import cosine, embed, from_blob, to_blob
from ..models import AttackOut, StrategyRec


def norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def normalize_family(raw: str) -> str:
    """Canonicalize a free-form strategy_family.

    Splits a composite ("scenario_framing + persona_modulation") into snake_case
    tokens and joins them sorted, so order never fragments per-family stats. A
    single coined family name passes through snake_cased. Empty -> "unspecified".
    """
    if not raw or not raw.strip():
        return "unspecified"
    toks: list[str] = []
    for part in re.split(r"[+&,/]| and ", raw):
        t = norm_name(part)
        if t and t not in toks:
            toks.append(t)
    if not toks:
        return "unspecified"
    if len(toks) == 1:
        return toks[0]
    return "+".join(sorted(toks))


def embedding_text(out: AttackOut) -> str:
    # strategy_name은 attacker가 매번 자유롭게 새로 짓는 라벨이라 같은 아이디어도
    # 이름이 계속 달라짐 -> 넣으면 코사인 유사도만 희석되고 dedup 정확도가 떨어짐
    return (
        f"strategy_family: {out.strategy_family}\n"
        f"reasoning_summary: {out.reasoning_summary}\n"
        f"attack_prompt_pattern: {out.attack_prompt}"
    )


@dataclass
class UpsertResult:
    strategy_id: str
    action: str  # created | variant_added | matched_existing


def _best_match(
    store: Store, vec: list[float], candidate_ids: set[str]
) -> tuple[str | None, float]:
    best_id, best_sim = None, 0.0
    for sid, blob in store.load_embeddings():
        if candidate_ids and sid not in candidate_ids:
            continue
        sim = cosine(vec, from_blob(blob))
        if sim > best_sim:
            best_id, best_sim = sid, sim
    return best_id, best_sim


def upsert_strategy(
    store: Store,
    out: AttackOut,
    run_id: str,
    turn_id: str,
    th: DedupThresholds,
    conn: sqlite3.Connection,
) -> UpsertResult:
    nname = norm_name(out.strategy_name)
    etext = embedding_text(out)
    vec = embed(etext)

    rows = store.find_strategies(nname, out.strategy_family)
    name_ids = {r["strategy_id"] for r in rows if r["norm_name"] == nname}
    family_ids = {r["strategy_id"] for r in rows}

    match_id, sim = _best_match(store, vec, family_ids)
    name_match = bool(name_ids)

    rec = StrategyRec(
        target_model=store.target_model,
        strategy_name=out.strategy_name,
        norm_name=nname,
        strategy_family=out.strategy_family,
        description=out.reasoning_summary,
        prompt_pattern=out.attack_prompt,
        reasoning_summary=out.reasoning_summary,
        embedding_text=etext,
    )

    if name_match and sim >= th.variant:
        sid = match_id or next(iter(name_ids))
        store.touch_strategy(sid, rec, conn)
        store.add_event(sid, run_id, turn_id, "variant_added", {"sim": sim}, conn)
        return UpsertResult(sid, "variant_added")
    if sim >= th.existing and match_id is not None:
        store.touch_strategy(match_id, rec, conn)
        store.add_event(match_id, run_id, turn_id, "matched_existing", {"sim": sim}, conn)
        return UpsertResult(match_id, "matched_existing")

    sid = store.insert_strategy(rec, conn)
    store.save_embedding(sid, EMBEDDING_MODEL, to_blob(vec), conn)
    store.add_event(sid, run_id, turn_id, "created", {"sim": sim}, conn)
    return UpsertResult(sid, "created")
