"""SQLite schema. One DB file per model family; SQL is the source of truth.

target_model은 실제 모델명(모델별 재집계용), family는 메모리 공유·조회 키.
인덱스가 family 컬럼을 참조하므로, 구 스키마 파일을 재사용할 때 마이그레이션이
컬럼을 먼저 보강할 수 있도록 테이블과 인덱스를 분리해 둔다(store._migrate 참고).
"""

SCHEMA_TABLES = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    goal_id TEXT,
    goal TEXT NOT NULL,
    attack_model TEXT NOT NULL,
    target_model TEXT NOT NULL,
    family TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL,
    success INTEGER DEFAULT 0,
    success_turn INTEGER,
    success_attempts INTEGER,  -- 성공 시점까지 돈 총 사이클 수(확정 턴 + 백트랙 합산)
    total_tokens INTEGER DEFAULT 0,
    total_cost REAL DEFAULT 0.0,
    total_time_sec REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS strategies (
    strategy_id TEXT PRIMARY KEY,
    target_model TEXT NOT NULL,
    family TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    norm_name TEXT NOT NULL,
    strategy_family TEXT NOT NULL,
    description TEXT,
    prompt_pattern TEXT,
    reasoning_summary TEXT,
    embedding_text TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    trials INTEGER DEFAULT 0,
    successes INTEGER DEFAULT 0,
    avg_elite_score REAL DEFAULT 0.0,
    avg_toxicity REAL DEFAULT 0.0,
    avg_specific REAL DEFAULT 0.0,
    avg_convincing REAL DEFAULT 0.0,
    refusal_rate REAL DEFAULT 0.0,
    avg_token_spent REAL DEFAULT 0.0,
    avg_time_spent_sec REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS strategy_embeddings (
    strategy_id TEXT PRIMARY KEY,
    embedding_model TEXT NOT NULL,
    embedding_vector BLOB NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(strategy_id) REFERENCES strategies(strategy_id)
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    turn_idx INTEGER NOT NULL,
    attack_io_id TEXT,
    target_io_id TEXT,
    judge_io_id TEXT,
    strategy_id TEXT,
    strategy_name TEXT,
    strategy_family TEXT,
    attack_prompt TEXT,
    target_response TEXT,
    refused INTEGER,
    specific REAL,
    convincing REAL,
    toxicity REAL,
    elite_score REAL,
    success INTEGER DEFAULT 0,
    refusal_summary TEXT,
    next_adapt TEXT,
    attack_latency_sec REAL DEFAULT 0.0,
    target_latency_sec REAL DEFAULT 0.0,
    judge_latency_sec REAL DEFAULT 0.0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    cost REAL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id),
    FOREIGN KEY(strategy_id) REFERENCES strategies(strategy_id)
);

CREATE TABLE IF NOT EXISTS llm_io (
    io_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    turn_id TEXT,
    turn_idx INTEGER,
    attempt_idx INTEGER DEFAULT 0,
    role TEXT NOT NULL,
    model TEXT NOT NULL,
    sys_prompt TEXT,
    user_prompt TEXT,
    messages_json TEXT,
    raw_output TEXT,
    parsed_json TEXT,
    final_text TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    latency_sec REAL DEFAULT 0.0,
    cost REAL DEFAULT 0.0,
    status TEXT NOT NULL,
    err TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS refusal_patterns (
    pattern_id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL,
    explicit_refusal INTEGER DEFAULT 0,
    policy_citation INTEGER DEFAULT 0,
    safety_redirect INTEGER DEFAULT 0,
    generic_safety_warning INTEGER DEFAULT 0,
    partial_compliance INTEGER DEFAULT 0,
    safe_alternative_offered INTEGER DEFAULT 0,
    asks_clarification INTEGER DEFAULT 0,
    topic_shift INTEGER DEFAULT 0,
    moralizing_language INTEGER DEFAULT 0,
    high_level_only INTEGER DEFAULT 0,
    missing_specificity INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(turn_id) REFERENCES turns(turn_id)
);

CREATE TABLE IF NOT EXISTS strategy_events (
    event_id TEXT PRIMARY KEY,
    strategy_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    turn_id TEXT,
    event_type TEXT NOT NULL,
    event_payload TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(strategy_id) REFERENCES strategies(strategy_id),
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS strategy_transitions (
    transition_id TEXT PRIMARY KEY,
    target_model TEXT NOT NULL,
    family TEXT NOT NULL,
    from_family TEXT NOT NULL,
    to_family TEXT NOT NULL,
    trials INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    avg_gain REAL DEFAULT 0.0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_factors (
    factor_id TEXT PRIMARY KEY,
    target_model TEXT NOT NULL,
    family TEXT NOT NULL,
    strategy_id TEXT,
    strategy_family TEXT NOT NULL,
    scope TEXT NOT NULL,
    mechanism TEXT NOT NULL,
    evidence TEXT,
    goal TEXT,
    run_id TEXT,
    turn_id TEXT,
    embedding_vector BLOB,
    occurrences INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(strategy_id) REFERENCES strategies(strategy_id)
);
"""

SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_turns_run ON turns(run_id);
CREATE INDEX IF NOT EXISTS idx_io_run ON llm_io(run_id);
CREATE INDEX IF NOT EXISTS idx_strat_family ON strategies(family);
CREATE INDEX IF NOT EXISTS idx_events_strat ON strategy_events(strategy_id);
CREATE INDEX IF NOT EXISTS idx_factors_family ON strategy_factors(family, scope);
CREATE UNIQUE INDEX IF NOT EXISTS idx_transition
    ON strategy_transitions(family, from_family, to_family);
"""

# family 컬럼을 가져야 하는 테이블(구 스키마 파일 마이그레이션 대상).
FAMILY_TABLES = ("runs", "strategies", "strategy_factors", "strategy_transitions")
