"""Run orchestration. Wires clients, DB, scoring; owns the per-turn transaction.

Provider calls, DB writes, scoring and retry live in their own modules; this
module only sequences them and persists results atomically per turn.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from .utils import fit_text, trim_ctx
from .clients import AttackClient, JudgeClient, TargetClient
from .clients.attack import AttackGenError, AttackResult
from .config import RunConfig
from .cost import CostTracker
from .db import Store
from .db import retrieval
from .context.dataset import Goal
from .memory.factors import persist_factors
from .models import CostRec, JudgeRes, LlmIo
from .memory.personas import load_personas
from .providers import get_provider
from .providers.base import ProviderError
from .eval.refusal import build_summary
from .memory.strategy import normalize_family, upsert_strategy
from .context.summary import maybe_recompress, update_summary

log = logging.getLogger(__name__)


def model_family(name: str) -> str:
    low = name.lower()
    for fam in ("gpt", "claude", "gemini", "llama", "gemma", "qwen"):
        if fam in low:
            return fam
    return "unknown"


@dataclass
class RunState:
    cfg: RunConfig
    goal: Goal
    store: Store
    attack: AttackClient
    target: TargetClient
    judge: JudgeClient
    cost: CostTracker
    run_id: str
    turn_idx: int = 0
    conv_summary: str = ""
    last_attack: str = ""
    last_target: str = ""
    last_judge: dict = field(default_factory=dict)
    refusal_ctx: dict = field(default_factory=dict)
    avoid: list[str] = field(default_factory=list)
    last_family: Optional[str] = None
    prev_elite: float = 0.0
    attempts_used: int = 0  # run 전체 누적 LLM 사이클 수(확정 턴 + 백트랙 합산)
    backtracks_used: int = 0  # run 전체 누적 백트랙 수(max_turns 안의 서브캡 대상)


@dataclass
class TurnRes:
    turn_idx: int
    success: bool
    elite_score: float
    backtracks: int = 0  # 이 턴을 확정하기까지 발생한 백트랙(재시도) 수
    stop_reason: Optional[str] = None


@dataclass
class RunResult:
    run_id: str
    status: str
    success: bool
    turns: int
    total_tokens: int
    total_cost: float
    total_time_sec: float
    success_attempts: Optional[int] = None  # 성공 시점까지 돈 총 사이클 수(확정 턴 + 백트랙 합산)


def build_ctx(st: RunState) -> dict:
    b = st.cfg.budget
    ctx = {
        "goal": st.goal.goal,
        "target_model": st.cfg.target_model,
        "turn_idx": st.turn_idx,
        "conv_summary": st.conv_summary,
        "last_attack": fit_text(st.last_attack, 200),
        "last_target_summary": fit_text(st.last_target, 400),
        "last_judge": st.last_judge,
        "refusal_summary": st.refusal_ctx,
        "avoid": st.avoid,
        "budget": {
            "spent_usd": round(st.cost.total_cost, 4),
            "tokens": st.cost.total_tokens,
        },
    }
    mode = st.cfg.ablation
    if mode != "no_memory":
        mem = retrieval.gather(st.store, st.goal.goal, st.cfg.retrieval)
        if mode == "no_ts":
            mem.pop("ts_hints", None)  # TS 랭킹만 제거; 원시 메모리 통계는 유지
        ctx.update(mem)
        if st.last_family:
            tr = retrieval.transitions(st.store, st.last_family, st.cfg.retrieval.transition_top_k)
            if tr:
                ctx["strategy_transitions"] = {"current_family": st.last_family, "worked_well_next": tr}
    if st.cfg.personas_path:
        ctx["persona_options"] = _personas(st.cfg, st.goal.goal)
    return trim_ctx(ctx, b.attack_ctx)


def _personas(cfg: RunConfig, goal: str) -> list[dict]:
    try:
        return load_personas(cfg.personas_path).pick(goal, cfg.persona_top_k)
    except (OSError, ValueError):
        return []


def _io_from_attempt(st: RunState, turn_id: str, a) -> LlmIo:
    return LlmIo(
        run_id=st.run_id, role="attack", model=st.cfg.attack_model, status=a.status,
        turn_id=turn_id, turn_idx=st.turn_idx, attempt_idx=a.attempt_idx,
        sys_prompt=a.sys_prompt, user_prompt=a.user_prompt,
        raw_output=a.raw_output,
        parsed_json=json.dumps(a.parsed) if a.parsed else None,
        final_text=(a.parsed or {}).get("attack_prompt") if a.parsed else None,
        input_tokens=a.cost.input_tokens, output_tokens=a.cost.output_tokens,
        total_tokens=a.cost.total_tokens, latency_sec=a.cost.latency_sec,
        cost=a.cost.cost, err=a.err,
    )


def _persist_turn(st: RunState, turn_id: str, ar: AttackResult, out, tgt, jc,
                  res: JudgeRes, record_transition: bool = True) -> dict:
    """이 시도(통과/거절 무관)를 DB에 그대로 기록 — 거절도 memory엔 실패로 남긴다.

    record_transition=False(백트랙 시도)면 family 전이는 기록하지 않는다. 전이는
    '다음-전략 추천' 신호라 확정 턴에서만 의미가 있다(실패 자체는 strategies 통계에 남음).
    """
    turn_cost = CostRec(
        input_tokens=tgt.cost.input_tokens + jc.cost.input_tokens,
        output_tokens=tgt.cost.output_tokens + jc.cost.output_tokens,
        cost=sum(a.cost.cost for a in ar.attempts) + tgt.cost.cost + jc.cost.cost,
        latency_sec=0.0,
    )

    with st.store.tx() as conn:
        up = upsert_strategy(st.store, out, st.run_id, turn_id, st.cfg.dedup, conn)

        attack_ios = [st.store.save_io(_io_from_attempt(st, turn_id, a), conn) for a in ar.attempts]
        target_io = st.store.save_io(
            LlmIo(
                run_id=st.run_id, role="target", model=st.cfg.target_model,
                status="blocked" if getattr(tgt, "blocked", False) else "ok",
                turn_id=turn_id, turn_idx=st.turn_idx,
                messages_json=json.dumps(tgt.messages), raw_output=tgt.response,
                final_text=tgt.response, input_tokens=tgt.cost.input_tokens,
                output_tokens=tgt.cost.output_tokens, total_tokens=tgt.cost.total_tokens,
                latency_sec=tgt.cost.latency_sec, cost=tgt.cost.cost,
            ),
            conn,
        )
        judge_io = st.store.save_io(
            LlmIo(
                run_id=st.run_id, role="judge", model=st.cfg.judge_model, status=jc.status,
                turn_id=turn_id, turn_idx=st.turn_idx,
                sys_prompt=jc.sys_prompt, user_prompt=jc.user_prompt,
                raw_output=jc.raw_output,
                parsed_json=json.dumps(jc.parsed) if jc.parsed else None,
                final_text=res.refusal_summary, input_tokens=jc.cost.input_tokens,
                output_tokens=jc.cost.output_tokens, total_tokens=jc.cost.total_tokens,
                latency_sec=jc.cost.latency_sec, cost=jc.cost.cost,
            ),
            conn,
        )
        # 성공 비결은 judge가 채점 호출에서 함께 낸 factors를 그대로 저장(별도 호출 없음).
        if res.success and res.factors:
            persist_factors(
                st.store, res.factors, family=out.strategy_family,
                strategy_id=up.strategy_id, goal=st.goal.goal, run_id=st.run_id,
                turn_id=turn_id, conn=conn, threshold=st.cfg.dedup.factor,
            )

        turn = {
            "turn_id": turn_id, "run_id": st.run_id, "turn_idx": st.turn_idx,
            "attack_io_id": attack_ios[-1] if attack_ios else None,
            "target_io_id": target_io, "judge_io_id": judge_io,
            "strategy_name": out.strategy_name, "strategy_family": out.strategy_family,
            "attack_prompt": out.attack_prompt, "target_response": tgt.response,
            "attack_latency_sec": sum(a.cost.latency_sec for a in ar.attempts),
            "target_latency_sec": tgt.cost.latency_sec,
            "judge_latency_sec": jc.cost.latency_sec,
        }
        st.store.save_turn(turn, res, up.strategy_id, turn_cost, conn)
        st.store.save_refusal(turn_id, res.patterns.model_dump(), conn)
        st.store.update_stats(up.strategy_id, res, turn_cost, conn)
        st.store.add_event(
            up.strategy_id, st.run_id, turn_id,
            "success" if res.success else "failure",
            {"elite_score": res.elite_score}, conn,
        )
        if record_transition and st.last_family and st.last_family != out.strategy_family:
            gain = res.elite_score - st.prev_elite
            st.store.update_transition(
                st.last_family, out.strategy_family, gain,
                win=(gain > 0 or res.success), conn=conn,
            )
    return turn


def run_turn(st: RunState) -> TurnRes:
    """한 '받아들여진' 턴을 만든다. 거절당하면 타겟 대화에서 그 시도를 무르고
    (백트랙) 같은 turn_idx로 재생성한다. 거절은 DB(memory)엔 남지만 대화엔 안 남는다.
    """
    attempt = 0
    while True:
        turn_id = st.store.new_turn_id()
        first = st.turn_idx == 0
        ctx = build_ctx(st)
        hints = ctx.get("ts_hints", [])

        ar: AttackResult = st.attack.generate(
            ctx, first, model_family(st.cfg.target_model), hints
        )
        out = ar.out
        out.strategy_family = normalize_family(out.strategy_family)
        for a in ar.attempts:
            st.cost.add(a.cost)

        tgt = st.target.send(out.attack_prompt)
        st.cost.add(tgt.cost)

        jc = st.judge.evaluate(st.goal.goal, out.attack_prompt, tgt.response)
        st.cost.add(jc.cost)
        res: JudgeRes = jc.res

        st.refusal_ctx = build_summary(res.patterns, res.refusal_summary, res.next_adapt)

        st.attempts_used += 1  # 방금 사이클(공격+타겟+심판) 1회 소모

        # 거절 & 미성공 & 남은 사이클 예산 있음 & 백트랙 서브캡 미소진 → 백트랙
        will_backtrack = (
            res.refused and not res.success
            and st.attempts_used < st.cfg.stop.max_turns
            and st.backtracks_used < st.cfg.stop.max_backtracks
        )
        # 전이는 확정 턴에서만 기록(백트랙 시도 제외) — 실패는 strategies 통계에 별도로 남음.
        turn = _persist_turn(st, turn_id, ar, out, tgt, jc, res,
                             record_transition=not will_backtrack)

        if will_backtrack:
            st.target.rollback()  # 타겟 대화에서 거절 교환 제거(없던 일로)
            st.last_attack = out.attack_prompt  # 공격자엔 "방금 이걸 시도했다"로 전달
            st.avoid = st.refusal_ctx.get("avoid_repeating", [])
            st.backtracks_used += 1
            attempt += 1
            continue  # turn_idx/conv_summary 불변 → 같은 단계 재시도

        # 확정(통과/성공/백트랙 소진): 상태 갱신 후 반환
        st.last_attack = out.attack_prompt
        st.last_target = tgt.response
        st.last_judge = {
            "refused": res.refused, "elite_score": res.elite_score,
            "toxicity": res.toxicity, "success": res.success,
        }
        st.avoid = st.refusal_ctx.get("avoid_repeating", [])
        st.last_family = out.strategy_family
        st.prev_elite = res.elite_score
        st.conv_summary = maybe_recompress(
            update_summary(st.goal.goal, st.conv_summary, {**turn, **st.last_judge,
                           "refused": res.refused, "elite_score": res.elite_score,
                           "refusal_summary": res.refusal_summary},
                           st.cfg.summary_max_tokens),
            st.cfg.summary_max_tokens,
        )
        return TurnRes(st.turn_idx, res.success, res.elite_score, attempt)


def stop_reason(st: RunState, last: TurnRes) -> Optional[str]:
    if last.success:
        return "success"
    if st.attempts_used >= st.cfg.stop.max_turns:
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
        # 백트랙으로 같은 turn_idx 행이 여러 개 생기므로, 행 수가 아니라 최대 turn_idx+1로 이어간다.
        start_idx = (max(r["turn_idx"] for r in prior) + 1) if prior else 0
        # 예산은 사이클 합산이므로 이전 시도 행 수(확정+백트랙)를 그대로 복원한다.
        # 백트랙 수 = 전체 행 - 확정 턴 수(=start_idx).
        prior_attempts = len(prior)
        prior_backtracks = prior_attempts - start_idx
    else:
        run_id = store.create_run(goal.goal_id, goal.goal, cfg.attack_model, cfg.judge_model)
        start_idx = 0
        prior_attempts = 0
        prior_backtracks = 0

    st = RunState(cfg, goal, store, attack, target, judge, tracker, run_id,
                  turn_idx=start_idx, attempts_used=prior_attempts,
                  backtracks_used=prior_backtracks)

    final_status, success, success_turn = "max_turns", False, None
    success_attempts = None
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
                success_attempts = st.attempts_used
                break
            reason = stop_reason(st, tr)
            if reason:
                final_status = reason
                break
            st.turn_idx += 1
    finally:
        store.finish_run(
            run_id, final_status, success, success_turn, success_attempts,
            totals={
                "total_tokens": tracker.total_tokens,
                "total_cost": tracker.total_cost,
                "total_time_sec": tracker.total_time_sec,
            },
        )
        store.close()
    return RunResult(
        run_id=run_id, status=final_status, success=success, turns=turns_done,
        total_tokens=tracker.total_tokens, total_cost=tracker.total_cost,
        total_time_sec=tracker.total_time_sec, success_attempts=success_attempts,
    )
