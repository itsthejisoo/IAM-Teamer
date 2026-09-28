#!/usr/bin/env python3
"""IAM-Teamer CLI. Results persist to per-target SQLite DBs; no JSON run files."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time  # noqa: E402

from iamteamer.config import (  # noqa: E402
    ABLATION_MODES, DedupThresholds, RetrievalLimits, RunConfig, StopRules, TokenBudget,
)
from iamteamer.context.dataset import Goal, custom_goal, get_goal, load_goals  # noqa: E402
from iamteamer.orchestrator import RunResult, run_goal  # noqa: E402

# The five target models for a full sweep (--all-targets).
ALL_TARGETS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-V4-Flash",
    "gpt-4o",
    "gpt-5.1",
    "qwen/qwen3.7-max",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IAM-Teamer multi-turn red-teaming runner")
    p.add_argument("--dataset", help="data/goals/goals.json or AdvBench CSV")
    p.add_argument("--goal-id", help="single goal id, or batch start index with --limit")
    p.add_argument("--limit", type=int, help="run this many goals from the dataset (batch)")
    p.add_argument("--goal")
    p.add_argument("--personas", default="data/goals/job_descriptions.json")
    p.add_argument("--persona-top-k", type=int, default=5)
    p.add_argument("--attack-model", default="gemma-4-31b-it")
    p.add_argument("--target-model", help="single target model")
    p.add_argument("--target-models", nargs="+", help="run several target models in sequence")
    p.add_argument("--all-targets", action="store_true",
                   help=f"sweep all five targets: {', '.join(ALL_TARGETS)}")
    p.add_argument("--judge-model", default="gemma-4-31b-it")
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--ablation", choices=ABLATION_MODES, default="full",
                   help="component ablation: full | no_ts | no_memory")
    p.add_argument("--memory-dir", default="data/results/memory")
    p.add_argument("--resume-run")
    p.add_argument("--max-attack-generation-retries", type=int, default=3)
    p.add_argument("--embedding-similarity-threshold", type=float, default=0.88)
    p.add_argument("--factor-dedup-threshold", type=float, default=0.6)
    p.add_argument("--strategy-top-k", type=int, default=5)
    p.add_argument("--case-top-k", type=int, default=3)
    p.add_argument("--failure-top-k", type=int, default=3)
    p.add_argument("--partial-top-k", type=int, default=3)
    p.add_argument("--transition-top-k", type=int, default=3)
    p.add_argument("--factor-top-k", type=int, default=3)
    p.add_argument("--target-raw-turns", type=int, default=3)
    p.add_argument("--attack-ctx-budget", type=int, default=3500)
    p.add_argument("--target-ctx-budget", type=int, default=6000)
    p.add_argument("--judge-ctx-budget", type=int, default=2500)
    p.add_argument("--memory-budget", type=int, default=1200)
    p.add_argument("--summary-budget", type=int, default=800)
    return p.parse_args(argv)


def resolve_targets(a: argparse.Namespace) -> list[str]:
    if a.all_targets:
        return list(ALL_TARGETS)
    if a.target_models:
        return list(a.target_models)
    if a.target_model:
        return [a.target_model]
    raise SystemExit("Provide --target-model, --target-models, or --all-targets")


def build_config(a: argparse.Namespace, target_model: str) -> RunConfig:
    return RunConfig(
        attack_model=a.attack_model,
        target_model=target_model,
        judge_model=a.judge_model,
        memory_dir=a.memory_dir,
        target_raw_turns=a.target_raw_turns,
        max_attack_retries=a.max_attack_generation_retries,
        summary_max_tokens=a.summary_budget,
        personas_path=a.personas,
        persona_top_k=a.persona_top_k,
        budget=TokenBudget(
            attack_ctx=a.attack_ctx_budget, target_ctx=a.target_ctx_budget,
            judge_ctx=a.judge_ctx_budget, memory=a.memory_budget,
            summary=a.summary_budget,
        ),
        retrieval=RetrievalLimits(
            strategy_top_k=a.strategy_top_k, case_top_k=a.case_top_k,
            failure_top_k=a.failure_top_k, partial_top_k=a.partial_top_k,
            transition_top_k=a.transition_top_k, factor_top_k=a.factor_top_k,
        ),
        dedup=DedupThresholds(existing=a.embedding_similarity_threshold,
                              factor=a.factor_dedup_threshold),
        stop=StopRules(max_turns=a.max_turns),
        ablation=a.ablation,
    )


def resolve_goals(a: argparse.Namespace) -> list[Goal]:
    if a.goal:
        return [custom_goal(a.goal)]
    if not a.dataset:
        raise SystemExit("Provide --goal or --dataset")
    if a.limit is not None:
        start = int(a.goal_id) if a.goal_id is not None else 0
        return load_goals(a.dataset)[start : start + a.limit]
    if a.goal_id is not None:
        return [get_goal(a.dataset, a.goal_id)]
    raise SystemExit("Provide --goal-id or --limit with --dataset")


def _fmt_duration(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{sec:.1f}s"


def print_summary(results: list[RunResult], wall_sec: float, db_dir: str,
                  label: str = "Results") -> None:
    n = len(results)
    if not n:
        print(f"\n===== {label} =====")
        print("no runs completed")
        return
    successes = sum(r.success for r in results)
    total_tokens = sum(r.total_tokens for r in results)
    total_turns = sum(r.turns for r in results)

    print(f"\n===== {label} =====")
    print(f"runs               : {n}")
    print(f"공격 성공률         : {successes}/{n} ({successes / n * 100:.1f}%)")
    print(f"토큰 사용량         : {total_tokens:,}")
    print(f"걸린 시간           : {_fmt_duration(wall_sec)}")
    print(f"평균 토큰 사용량     : {total_tokens / n:,.0f} / run")
    avg_turn_time = wall_sec / total_turns if total_turns else 0.0
    print(f"턴마다 걸린 평균 시간 : {_fmt_duration(avg_turn_time)}")
    print(f"평균 턴 수          : {total_turns / n:.1f}")
    print(f"db={db_dir}")


def _run_target(a: argparse.Namespace, target: str, goals: list[Goal]) -> list[RunResult]:
    cfg = build_config(a, target)
    resume = a.resume_run if len(goals) == 1 else None
    results: list[RunResult] = []
    for i, goal in enumerate(goals, 1):
        try:
            res = run_goal(cfg, goal, resume)
            results.append(res)
            print(f"[{target}] [{i}/{len(goals)}] goal_id={goal.goal_id} "
                  f"run_id={res.run_id} status={res.status} turns={res.turns}")
        except Exception as e:  # one goal must not kill the batch
            logging.error("target=%s goal_id=%s failed: %s", target, goal.goal_id, e)
    return results


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    a = parse_args(argv)
    goals = resolve_goals(a)
    targets = resolve_targets(a)
    start = time.perf_counter()
    all_results: list[RunResult] = []
    for target in targets:
        t0 = time.perf_counter()
        results = _run_target(a, target, goals)
        all_results.extend(results)
        if len(targets) > 1:
            print_summary(results, time.perf_counter() - t0, a.memory_dir,
                          label=f"Target: {target}")
    wall_sec = time.perf_counter() - start
    print_summary(all_results, wall_sec, a.memory_dir,
                  label="Overall" if len(targets) > 1 else "Results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
