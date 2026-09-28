#!/usr/bin/env python3
"""Ablation 드라이버: memory/Thompson 컴포넌트를 끄고 같은 goal·예산으로 재실행.

조건(논문의 두 contribution에 매핑):
  full       메모리 검색 + TS 랭킹           (기존 data/results/memory_cold 재사용)
  no_ts      메모리 검색만, TS hint 제거       (contribution 1만)
  no_memory  메모리 파생 컨텍스트 전부 제거    (순수 baseline attacker)

goal 집합은 goals.json을 순서대로 [start:start+limit](기본 0~199) 로드한다.
조건별로 별도 memory_dir에 기록하므로 본 결과 DB와 섞이지 않는다. 재실행 가능:
이미 끝난(status in success/max_turns) goal_id는 건너뛴다.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iamteamer.config import RunConfig, StopRules  # noqa: E402
from iamteamer.context.dataset import load_goals  # noqa: E402
from iamteamer.db.store import memory_family, norm_model  # noqa: E402
from iamteamer.orchestrator import run_goal  # noqa: E402

log = logging.getLogger("ablation")

TARGETS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "gpt-5.1": "gpt-5.1",
}
CONDITIONS = ("no_ts", "no_memory")  # full은 기존 cold-start DB 재사용


def db_path(memory_dir: str, target_model: str) -> Path:
    return Path(memory_dir) / f"{norm_model(memory_family(target_model))}.sqlite"


def done_goal_ids(cond_db: Path) -> set[str]:
    if not cond_db.exists():
        return set()
    conn = sqlite3.connect(cond_db)
    try:
        rows = conn.execute(
            "SELECT DISTINCT goal_id FROM runs WHERE status IN ('success','max_turns')"
        ).fetchall()
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()
    return {str(r[0]) for r in rows}


def run_condition(target_model: str, condition: str, goals: list[Goal],
                  out_dir: str, attack_model: str, judge_model: str,
                  max_turns: int, force: bool) -> None:
    memory_dir = str(Path(out_dir) / condition)
    cond_db = db_path(memory_dir, target_model)
    done = set() if force else done_goal_ids(cond_db)
    pending = [g for g in goals if g.goal_id not in done]
    log.info("[%s | %s] %d goals, %d done, %d pending -> %s",
             target_model, condition, len(goals), len(done), len(pending), cond_db)

    cfg = RunConfig(
        attack_model=attack_model,
        target_model=target_model,
        judge_model=judge_model,
        memory_dir=memory_dir,
        ablation=condition,
        stop=StopRules(max_turns=max_turns),
    )
    for i, g in enumerate(pending, 1):
        try:
            res = run_goal(cfg, g)
            log.info("[%s | %s] [%d/%d] goal_id=%s status=%s turns=%d succ_attempts=%s",
                     target_model, condition, i, len(pending), g.goal_id,
                     res.status, res.turns, res.success_attempts)
        except Exception as e:  # 한 goal 실패가 배치를 죽이지 않게
            log.error("[%s | %s] goal_id=%s failed: %s", target_model, condition, g.goal_id, e)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IAM-Teamer component ablation runner (Llama, GPT-5.1)")
    p.add_argument("--targets", nargs="+", choices=list(TARGETS), default=list(TARGETS))
    p.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    p.add_argument("--dataset", default="data/goals/goals.json",
                   help="goal 데이터셋(JSON/CSV); load_goals 순서대로 [start:start+limit]")
    p.add_argument("--start", type=int, default=0, help="시작 인덱스")
    p.add_argument("--limit", type=int, default=200, help="goal 개수(기본 200 = 0~199)")
    p.add_argument("--out-dir", default="data/results/ablation",
                   help="조건별 결과 DB를 쓸 루트(조건명 하위 폴더로 분리)")
    p.add_argument("--attack-model", default="gemma-4-31b-it")
    p.add_argument("--judge-model", default="gemma-4-31b-it")
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--force", action="store_true", help="이미 끝난 goal도 다시 실행")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    a = parse_args(argv)
    goals = load_goals(a.dataset)[a.start : a.start + a.limit]
    log.info("goals: %d (%s[%d:%d])", len(goals), a.dataset, a.start, a.start + a.limit)
    start = time.perf_counter()
    for tkey in a.targets:
        target_model = TARGETS[tkey]
        for condition in a.conditions:
            run_condition(target_model, condition, goals, a.out_dir,
                          a.attack_model, a.judge_model, a.max_turns, a.force)
    log.info("done in %.1fs", time.perf_counter() - start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
