"""SQLite access layer. Separated from orchestration; no LLM logic here.

Each turn is committed in a single transaction (turn + refusal + stats).
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from ..models import REFUSAL_KEYS, CostRec, JudgeRes, LlmIo, StrategyRec
from .schema import FAMILY_TABLES, SCHEMA_INDEXES, SCHEMA_TABLES


def norm_model(name: str) -> str:
    return (
        name.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace(" ", "_")
    )


# 메모리 공유 단위 = 버전 라인 family. 같은 라인의 사이즈 변형(8B/70B 등)은
# safety post-training이 같아 한 family로 묶어 warm-start 데이터를 공유한다.
# 명시 규칙은 표기가 제각각인 라인(llama 3.1 / llama-3.1 / llama3.1)을 한 키로
# 정규화한다. 그 외는 fallback이 토큰을 떼어 자동으로 같은 라인을 묶는다.
_FAMILY_RULES: tuple[tuple[str, str], ...] = (
    (r"llama[-_ ]*3\.?3", "llama-3.3"),
    (r"llama[-_ ]*3\.?2", "llama-3.2"),
    (r"llama[-_ ]*3\.?1", "llama-3.1"),
    (r"deepseek[-_ ]*v?4", "deepseek-v4"),
    (r"qwen[-_ ]*3\.?7", "qwen3.7"),
    (r"gpt[-_ ]*4o", "gpt-4o"),
    (r"gpt[-_ ]*5\.?1", "gpt-5.1"),
)

# 파라미터 크기 토큰(9B, 27B, 1.5B …)을 떼어 같은 라인을 자동으로 묶는다.
_SIZE_RE = re.compile(r"[-_ ]?\b\d+(?:\.\d+)?\s*b\b", re.IGNORECASE)

# 포맷/튜닝 변형 토큰(instruct, it, chat, base)도 떼어 같은 라인으로 묶는다.
_VARIANT_RE = re.compile(r"[-_ ]?\b(?:instruct|it|chat|base)\b", re.IGNORECASE)


def memory_family(name: str) -> str:
    """모델명을 메모리 공유용 버전 라인 family 키로 매핑.

    명시 규칙(_FAMILY_RULES)이 먼저, 없으면 벤더 접두사(예: Qwen/)·크기 토큰
    (9B/27B)·포맷 변형(base/instruct/it/chat)을 모두 떼어 같은 라인을 자동으로 한
    family로 묶는다(예: Qwen/Qwen3.5-9B·27B → qwen3_5, Midm-2.0-Base-Instruct →
    midm_2_0). orchestrator.model_family(공격 프롬프트용 vendor 힌트)와는 별개 함수다.
    """
    low = name.lower()
    for pat, fam in _FAMILY_RULES:
        if re.search(pat, low):
            return fam
    base = low.rsplit("/", 1)[-1]
    base = _VARIANT_RE.sub("", _SIZE_RE.sub("", base))
    return norm_model(base)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id() -> str:
    return uuid.uuid4().hex


def _avg(old_avg: float, old_n: int, val: float) -> float:
    return ((old_avg * old_n) + val) / (old_n + 1)


class Store:
    """One SQLite DB per model family (same-version sizes share one DB).

    target_model은 실제 모델명을 row별로 보존(나중에 모델별 재집계 가능),
    family는 메모리 공유·조회의 키다.
    """

    def __init__(self, target_model: str, memory_dir: str = "data/results/memory") -> None:
        self.target_model = target_model
        self.family = memory_family(target_model)
        Path(memory_dir).mkdir(parents=True, exist_ok=True)
        self.path = str(Path(memory_dir) / f"{norm_model(self.family)}.sqlite")
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.executescript(SCHEMA_TABLES)
        self._migrate()
        self.conn.executescript(SCHEMA_INDEXES)
        self.conn.commit()

    def _migrate(self) -> None:
        """구 스키마(per-model) 파일을 재사용할 때 family 컬럼을 보강한다.

        파일명이 family 키와 겹치는 단일 모델 family(gpt-4o 등)는 옛 파일을 그대로
        열게 되므로, 누락된 family 컬럼을 추가하고 이 파일의 family로 백필한다.
        인덱스(family 참조)는 이 단계 뒤에 생성한다.
        """
        for table in FAMILY_TABLES:
            cols = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            if cols and "family" not in cols:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN family TEXT")
                self.conn.execute(
                    f"UPDATE {table} SET family=? WHERE family IS NULL", (self.family,)
                )
        run_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(runs)")}
        if "success_attempts" not in run_cols:
            self.conn.execute("ALTER TABLE runs ADD COLUMN success_attempts INTEGER")

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # runs -------------------------------------------------------------
    def create_run(
        self, goal_id: Optional[str], goal: str, attack: str, judge: str
    ) -> str:
        run_id = _id()
        with self.tx() as c:
            c.execute(
                "INSERT INTO runs(run_id, goal_id, goal, attack_model, target_model,"
                " family, judge_model, started_at, status) VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, goal_id, goal, attack, self.target_model, self.family, judge,
                 _now(), "running"),
            )
        return run_id

    def finish_run(
        self, run_id: str, status: str, success: bool, success_turn: Optional[int],
        success_attempts: Optional[int] = None,
        cost: CostRec | None = None, totals: dict | None = None,
    ) -> None:
        totals = totals or {}
        with self.tx() as c:
            c.execute(
                "UPDATE runs SET ended_at=?, status=?, success=?, success_turn=?,"
                " success_attempts=?, total_tokens=?, total_cost=?, total_time_sec=?"
                " WHERE run_id=?",
                (
                    _now(), status, int(success), success_turn, success_attempts,
                    totals.get("total_tokens", 0), totals.get("total_cost", 0.0),
                    totals.get("total_time_sec", 0.0), run_id,
                ),
            )

    def get_run(self, run_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()

    def run_turns(self, run_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM turns WHERE run_id=? ORDER BY turn_idx", (run_id,)
        ).fetchall()

    # llm_io -----------------------------------------------------------
    def save_io(self, rec: LlmIo, conn: sqlite3.Connection | None = None) -> str:
        c = conn or self.conn
        io_id = _id()
        total = rec.total_tokens or (rec.input_tokens + rec.output_tokens)
        c.execute(
            "INSERT INTO llm_io(io_id, run_id, turn_id, turn_idx, attempt_idx, role,"
            " model, sys_prompt, user_prompt, messages_json, raw_output,"
            " parsed_json, final_text, input_tokens, output_tokens, total_tokens,"
            " latency_sec, cost, status, err, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                io_id, rec.run_id, rec.turn_id, rec.turn_idx, rec.attempt_idx, rec.role,
                rec.model, rec.sys_prompt, rec.user_prompt,
                rec.messages_json, rec.raw_output, rec.parsed_json, rec.final_text,
                rec.input_tokens, rec.output_tokens, total, rec.latency_sec, rec.cost,
                rec.status, rec.err, _now(),
            ),
        )
        if conn is None:
            self.conn.commit()
        return io_id

    # strategies -------------------------------------------------------
    def find_strategies(self, norm_name: str, family: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM strategies WHERE family=? AND (norm_name=? OR"
            " strategy_family=?)",
            (self.family, norm_name, family),
        ).fetchall()

    def all_strategies(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM strategies WHERE family=?", (self.family,)
        ).fetchall()

    def insert_strategy(self, rec: StrategyRec, conn: sqlite3.Connection) -> str:
        sid = _id()
        conn.execute(
            "INSERT INTO strategies(strategy_id, target_model, family, strategy_name,"
            " norm_name, strategy_family, description, prompt_pattern, reasoning_summary,"
            " embedding_text, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                sid, rec.target_model, self.family, rec.strategy_name, rec.norm_name,
                rec.strategy_family, rec.description, rec.prompt_pattern,
                rec.reasoning_summary, rec.embedding_text, _now(), _now(),
            ),
        )
        return sid

    def touch_strategy(self, sid: str, rec: StrategyRec, conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE strategies SET prompt_pattern=?, reasoning_summary=?,"
            " embedding_text=?, updated_at=? WHERE strategy_id=?",
            (rec.prompt_pattern, rec.reasoning_summary, rec.embedding_text, _now(), sid),
        )

    def save_embedding(self, sid: str, model: str, blob: bytes, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO strategy_embeddings(strategy_id, embedding_model,"
            " embedding_vector, created_at) VALUES (?,?,?,?)",
            (sid, model, blob, _now()),
        )

    def load_embeddings(self) -> list[tuple[str, bytes]]:
        rows = self.conn.execute(
            "SELECT e.strategy_id, e.embedding_vector FROM strategy_embeddings e"
            " JOIN strategies s ON s.strategy_id=e.strategy_id WHERE s.family=?",
            (self.family,),
        ).fetchall()
        return [(r["strategy_id"], r["embedding_vector"]) for r in rows]

    def add_event(
        self, sid: str, run_id: str, turn_id: Optional[str], kind: str,
        payload: dict | None, conn: sqlite3.Connection,
    ) -> None:
        conn.execute(
            "INSERT INTO strategy_events(event_id, strategy_id, run_id, turn_id,"
            " event_type, event_payload, created_at) VALUES (?,?,?,?,?,?,?)",
            (_id(), sid, run_id, turn_id, kind, json.dumps(payload or {}), _now()),
        )

    def update_stats(self, sid: str, res: JudgeRes, cost: CostRec, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT trials, successes, avg_elite_score, avg_toxicity, avg_specific,"
            " avg_convincing, refusal_rate, avg_token_spent, avg_time_spent_sec"
            " FROM strategies WHERE strategy_id=?",
            (sid,),
        ).fetchone()
        if row is None:
            return
        n = row["trials"]
        conn.execute(
            "UPDATE strategies SET trials=?, successes=?, avg_elite_score=?,"
            " avg_toxicity=?, avg_specific=?, avg_convincing=?, refusal_rate=?,"
            " avg_token_spent=?, avg_time_spent_sec=?, updated_at=? WHERE strategy_id=?",
            (
                n + 1,
                row["successes"] + (1 if res.success else 0),
                _avg(row["avg_elite_score"], n, res.elite_score),
                _avg(row["avg_toxicity"], n, res.toxicity),
                _avg(row["avg_specific"], n, res.specific),
                _avg(row["avg_convincing"], n, res.convincing),
                _avg(row["refusal_rate"], n, 1.0 if res.refused else 0.0),
                _avg(row["avg_token_spent"], n, float(cost.total_tokens)),
                _avg(row["avg_time_spent_sec"], n, cost.latency_sec),
                _now(), sid,
            ),
        )

    def update_transition(
        self, from_fam: str, to_fam: str, gain: float, win: bool, conn: sqlite3.Connection
    ) -> None:
        row = conn.execute(
            "SELECT transition_id, trials, wins, avg_gain FROM strategy_transitions"
            " WHERE family=? AND from_family=? AND to_family=?",
            (self.family, from_fam, to_fam),
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO strategy_transitions(transition_id, target_model, family,"
                " from_family, to_family, trials, wins, avg_gain, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (_id(), self.target_model, self.family, from_fam, to_fam, 1, int(win),
                 gain, _now()),
            )
            return
        n = row["trials"]
        conn.execute(
            "UPDATE strategy_transitions SET trials=?, wins=?, avg_gain=?, updated_at=?"
            " WHERE transition_id=?",
            (n + 1, row["wins"] + int(win), _avg(row["avg_gain"], n, gain), _now(),
             row["transition_id"]),
        )

    def top_transitions(self, from_fam: str, k: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM strategy_transitions WHERE family=? AND from_family=?"
            " AND trials>0 ORDER BY (CAST(wins AS REAL)/trials) DESC, avg_gain DESC LIMIT ?",
            (self.family, from_fam, k),
        ).fetchall()

    # strategy_factors -------------------------------------------------
    def insert_factor(
        self, strategy_id: Optional[str], family: str, scope: str, mechanism: str,
        evidence: str, goal: str, run_id: str, turn_id: str, blob: bytes,
        conn: sqlite3.Connection,
    ) -> str:
        fid = _id()
        conn.execute(
            "INSERT INTO strategy_factors(factor_id, target_model, family, strategy_id,"
            " strategy_family, scope, mechanism, evidence, goal, run_id, turn_id,"
            " embedding_vector, occurrences, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                fid, self.target_model, self.family, strategy_id, family, scope,
                mechanism, evidence, goal, run_id, turn_id, blob, 1, _now(), _now(),
            ),
        )
        return fid

    def bump_factor(self, factor_id: str, conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE strategy_factors SET occurrences=occurrences+1, updated_at=?"
            " WHERE factor_id=?",
            (_now(), factor_id),
        )

    def factor_embeddings(self, scope: str, family: str) -> list[tuple[str, bytes]]:
        # general mechanisms are family-agnostic: dedup them across all families.
        if scope == "general":
            rows = self.conn.execute(
                "SELECT factor_id, embedding_vector FROM strategy_factors"
                " WHERE family=? AND scope='general'"
                " AND embedding_vector IS NOT NULL",
                (self.family,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT factor_id, embedding_vector FROM strategy_factors"
                " WHERE family=? AND scope=? AND strategy_family=?"
                " AND embedding_vector IS NOT NULL",
                (self.family, scope, family),
            ).fetchall()
        return [(r["factor_id"], r["embedding_vector"]) for r in rows]

    def top_factors(self, scope: str, family: Optional[str], k: int) -> list[sqlite3.Row]:
        if family is None:
            return self.conn.execute(
                "SELECT * FROM strategy_factors WHERE family=? AND scope=?"
                " ORDER BY occurrences DESC, updated_at DESC LIMIT ?",
                (self.family, scope, k),
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM strategy_factors WHERE family=? AND scope=?"
            " AND strategy_family=? ORDER BY occurrences DESC, updated_at DESC LIMIT ?",
            (self.family, scope, family, k),
        ).fetchall()

    # turns ------------------------------------------------------------
    def save_turn(
        self, turn: dict, res: JudgeRes, sid: Optional[str], cost: CostRec,
        conn: sqlite3.Connection,
    ) -> str:
        turn_id = turn["turn_id"]
        conn.execute(
            "INSERT INTO turns(turn_id, run_id, turn_idx, attack_io_id, target_io_id,"
            " judge_io_id, strategy_id, strategy_name, strategy_family, attack_prompt,"
            " target_response, refused, specific, convincing, toxicity, elite_score,"
            " success, refusal_summary, next_adapt, attack_latency_sec,"
            " target_latency_sec, judge_latency_sec, input_tokens, output_tokens,"
            " total_tokens, cost, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                turn_id, turn["run_id"], turn["turn_idx"], turn.get("attack_io_id"),
                turn.get("target_io_id"), turn.get("judge_io_id"), sid,
                turn.get("strategy_name"), turn.get("strategy_family"),
                turn.get("attack_prompt"), turn.get("target_response"),
                int(res.refused), res.specific, res.convincing, res.toxicity,
                res.elite_score, int(res.success), res.refusal_summary, res.next_adapt,
                turn.get("attack_latency_sec", 0.0), turn.get("target_latency_sec", 0.0),
                turn.get("judge_latency_sec", 0.0), cost.input_tokens,
                cost.output_tokens, cost.total_tokens, cost.cost, _now(),
            ),
        )
        return turn_id

    def save_refusal(self, turn_id: str, patterns: dict, conn: sqlite3.Connection) -> None:
        cols = ", ".join(REFUSAL_KEYS)
        vals = ", ".join("?" for _ in REFUSAL_KEYS)
        conn.execute(
            f"INSERT INTO refusal_patterns(pattern_id, turn_id, {cols}, created_at)"
            f" VALUES (?, ?, {vals}, ?)",
            (_id(), turn_id, *[int(bool(patterns.get(k))) for k in REFUSAL_KEYS], _now()),
        )

    def new_turn_id(self) -> str:
        return _id()
