#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iamteamer.context.dataset import load_goals  # noqa: E402
from iamteamer.db.store import memory_family, norm_model  # noqa: E402

TARGETS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "gpt-5.1": "gpt-5.1",
}
KS = (1, 3, 5, 10)


def db_path(memory_dir: str, target_model: str) -> Path:
    return Path(memory_dir) / f"{norm_model(memory_family(target_model))}.sqlite"


def load_outcomes(db: Path, goal_ids: list[str] | None) -> dict[str, tuple[bool, int | None]]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT goal_id, success, success_attempts, started_at FROM runs"
        " WHERE goal_id IS NOT NULL ORDER BY started_at"
    ).fetchall()
    conn.close()
    out: dict[str, tuple[bool, int | None]] = {}
    for r in rows:
        gid = str(r["goal_id"])
        succ = bool(r["success"])
        t = r["success_attempts"] if succ else None
        prev = out.get(gid)
        if prev is None:
            out[gid] = (succ, t)
            continue
        # 성공 우선, 성공끼리는 더 빠른 t 우선
        p_succ, p_t = prev
        if succ and (not p_succ or (p_t is not None and t is not None and t < p_t)):
            out[gid] = (succ, t)
    if goal_ids is not None:
        out = {g: out.get(g, (False, None)) for g in goal_ids}
    return out


def metrics(outcomes: dict[str, tuple[bool, int | None]]) -> tuple[int, dict[int, tuple[float, float]]]:
    vals = list(outcomes.values())
    n = len(vals)
    res: dict[int, tuple[float, float]] = {}
    for k in KS:
        if n == 0:
            res[k] = (0.0, 0.0)
            continue
        succ = sum(1 for s, t in vals if s and t is not None and t <= k)
        asr = succ / n
        # 각 goal은 min(k, t) 기여; k 안에 성공 못 하면 full budget k
        turns = sum((t if (s and t is not None and t <= k) else k) for s, t in vals) / n
        res[k] = (asr, turns)
    return n, res


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ablation ASR@k / turns@k aggregator")
    p.add_argument("--targets", nargs="+", choices=list(TARGETS), default=list(TARGETS))
    p.add_argument("--dataset", default="data/goals/goals.json")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=200, help="goal 개수(기본 200 = 0~199)")
    p.add_argument("--full-db-dir", default="data/results/memory_cold")
    p.add_argument("--ablation-dir", default="data/results/ablation")
    p.add_argument("--latex", action="store_true", help="LaTeX 표 조각도 출력")
    a = p.parse_args(argv)

    conditions = [
        ("full", a.full_db_dir),
        ("no_ts", str(Path(a.ablation_dir) / "no_ts")),
        ("no_memory", str(Path(a.ablation_dir) / "no_memory")),
    ]

    goal_ids = [g.goal_id for g in load_goals(a.dataset)[a.start : a.start + a.limit]]

    rows_for_latex: list[tuple[str, str, int, dict[int, tuple[float, float]], tuple[float, float]]] = []
    for tkey in a.targets:
        target_model = TARGETS[tkey]
        print(f"\n===== {tkey} ({target_model}) — N={len(goal_ids)} goals =====")
        header = f"{'condition':<11} " + " ".join(f"ASR@{k:<2}" for k in KS) + "   " + \
                 " ".join(f"T@{k:<2}" for k in KS) + "   ASR@10 95%CI"
        print(header)
        for cond, mdir in conditions:
            db = db_path(mdir, target_model)
            if not db.exists():
                print(f"{cond:<11} (DB 없음: {db})")
                continue
            outcomes = load_outcomes(db, goal_ids)
            n, m = metrics(outcomes)
            asr10 = m[10][0]
            lo, hi = wilson(asr10, n)
            asr_cells = " ".join(f"{fmt_pct(m[k][0]):>5}" for k in KS)
            turn_cells = " ".join(f"{m[k][1]:>4.2f}" for k in KS)
            print(f"{cond:<11} {asr_cells}   {turn_cells}   [{fmt_pct(lo)}, {fmt_pct(hi)}]")
            rows_for_latex.append((tkey, cond, n, m, (lo, hi)))

    if a.latex and rows_for_latex:
        print("\n% --- LaTeX fragment ---")
        print("\\begin{tabular}{ll" + "c" * (len(KS) * 2) + "}")
        print("\\toprule")
        ks = " & ".join(f"ASR@{k}" for k in KS) + " & " + " & ".join(f"$\\overline{{T}}$@{k}" for k in KS)
        print(f"Target & Condition & {ks} \\\\")
        print("\\midrule")
        for tkey, cond, _n, m, _ci in rows_for_latex:
            asr = " & ".join(fmt_pct(m[k][0]) for k in KS)
            turn = " & ".join(f"{m[k][1]:.2f}" for k in KS)
            print(f"{tkey} & {cond} & {asr} & {turn} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
