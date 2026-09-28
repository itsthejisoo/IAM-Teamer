"""No-backtrack orchestrator for the refusal experiment (reproduces no-backtrack-lmao).

The current iamteamer orchestrator, on a refusal, rolls the target conversation back
and regenerates the same turn_idx (backtracking). This module instead leaves the
refused exchange in both the conversation and the DB and moves to the next turn.

It reuses iamteamer's clients, DB, scoring, context builder and turn persistence
(build_ctx, _persist_turn, ...) unchanged and only reimplements the turn loop without
backtracking. The iamteamer package itself is not modified.
"""

from __future__ import annotations

import logging
from typing import Optional

from iamteamer.clients import AttackClient, JudgeClient, TargetClient
from iamteamer.clients.attack import AttackGenError
from iamteamer.config import RunConfig
from iamteamer.context.dataset import Goal
from iamteamer.context.summary import maybe_recompress, update_summary
from iamteamer.cost import CostTracker
from iamteamer.db import Store
from iamteamer.eval.refusal import build_summary
from iamteamer.memory.strategy import normalize_family
from iamteamer.orchestrator import (
    RunResult,
    RunState,
    TurnRes,
    build_ctx,
    model_family,
)
from iamteamer.orchestrator import _persist_turn as persist_turn
from iamteamer.providers import get_provider
from iamteamer.providers.base import ProviderError

log = logging.getLogger(__name__)


def run_turn(st: RunState) -> TurnRes:
    """Run one turn and commit it as-is. On a refusal, keep the exchange in the
    conversation and DB (no rollback) and move to the next turn."""
    turn_id = st.store.new_turn_id()
    first = st.turn_idx == 0
    ctx = build_ctx(st)
    hints = ctx.get("ts_hints", [])

    ar = st.attack.generate(ctx, first, model_family(st.cfg.target_model), hints)
    out = ar.out
    out.strategy_family = normalize_family(out.strategy_family)
    for a in ar.attempts:
        st.cost.add(a.cost)

    tgt = st.target.send(out.attack_prompt)
    st.cost.add(tgt.cost)

    jc = st.judge.evaluate(st.goal.goal, out.attack_prompt, tgt.response)
    st.cost.add(jc.cost)
    res = jc.res

    st.refusal_ctx = build_summary(res.patterns, res.refusal_summary, res.next_adapt)

    turn = persist_turn(st, turn_id, ar, out, tgt, jc, res)

    st.last_attack = out.attack_prompt
    st.last_target = tgt.response
    st.last_judge = {
        "refused": res.refused,
        "elite_score": res.elite_score,
        "toxicity": res.toxicity,
        "success": res.success,
    }
    st.avoid = st.refusal_ctx.get("avoid_repeating", [])
    st.last_family = out.strategy_family
    st.prev_elite = res.elite_score
    st.conv_summary = maybe_recompress(
        update_summary(
            st.goal.goal,
            st.conv_summary,
            {
                **turn,
                **st.last_judge,
                "refused": res.refused,
                "elite_score": res.elite_score,
                "refusal_summary": res.refusal_summary,
            },
            st.cfg.summary_max_tokens,
        ),
        st.cfg.summary_max_tokens,
    )
    return TurnRes(st.turn_idx, res.success, res.elite_score)


def stop_reason(st: RunState, last: TurnRes) -> Optional[str]:
    if last.success:
        return "success"
    # st.turn_idx is the turn just finished, so turns taken = turn_idx + 1.
    if st.turn_idx + 1 >= st.cfg.stop.max_turns:
        return "max_turns"
    return None


def run_goal(cfg: RunConfig, goal: Goal, resume_run: Optional[str] = None) -> RunResult:
    store = Store(cfg.target_model, cfg.memory_dir)
    attack = AttackClient(get_provider(cfg.attack_model), cfg.max_attack_retries)
    target = TargetClient(get_provider(cfg.target_model), cfg.target_raw_turns)
    judge = JudgeClient(get_provider(cfg.judge_model))
    tracker = CostTracker()

    if resume_run:
        run_id = resume_run
        prior = store.run_turns(run_id)
        # No backtracking: one row per turn, so the next index is the row count.
        start_idx = len(prior)
    else:
        run_id = store.create_run(goal.goal_id, goal.goal, cfg.attack_model, cfg.judge_model)
        start_idx = 0

    st = RunState(cfg, goal, store, attack, target, judge, tracker, run_id, turn_idx=start_idx)

    final_status, success, success_turn = "max_turns", False, None
    turns_done = 0
    try:
        while True:
            try:
                tr = run_turn(st)
            except (ProviderError, AttackGenError) as e:
                log.error("run %s aborted at turn %d: %s", run_id, st.turn_idx, e)
                final_status = "error"
                break
            turns_done += 1
            if tr.success:
                final_status, success, success_turn = "success", True, tr.turn_idx
                break
            reason = stop_reason(st, tr)
            if reason:
                final_status = reason
                break
            st.turn_idx += 1
    finally:
        # success_attempts is backtrack-only accounting; omit it here (stays NULL).
        store.finish_run(
            run_id,
            final_status,
            success,
            success_turn,
            totals={
                "total_tokens": tracker.total_tokens,
                "total_cost": tracker.total_cost,
                "total_time_sec": tracker.total_time_sec,
            },
        )
        store.close()
    return RunResult(
        run_id=run_id,
        status=final_status,
        success=success,
        turns=turns_done,
        total_tokens=tracker.total_tokens,
        total_cost=tracker.total_cost,
        total_time_sec=tracker.total_time_sec,
    )
