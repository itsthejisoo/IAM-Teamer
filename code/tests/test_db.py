from iamteamer.db import Store, memory_family, norm_model
from iamteamer.models import CostRec, JudgeRes, LlmIo, RefusalPatterns


def _judge_res(success=False):
    return JudgeRes(
        refused=not success, specific=4, convincing=4, toxicity=3,
        elite_score=12.0 if success else 0.0, success=success,
        effective=True, patterns=RefusalPatterns(partial_compliance=True),
        refusal_summary="sum", next_adapt="adapt",
    )


def test_norm_model():
    assert norm_model("Claude-Sonnet-4.6") == "claude_sonnet_4_6"
    assert norm_model("Qwen/Qwen3.6-27B") == "qwen_qwen3_6_27b"


def test_memory_family_version_line():
    # 같은 라인의 사이즈 변형은 한 family로 묶인다.
    assert memory_family("meta-llama/Llama-3.1-8B-Instruct") == "llama-3.1"
    assert memory_family("meta-llama/Llama-3.1-70B-Instruct") == "llama-3.1"
    # 다른 버전 라인은 분리된다.
    assert memory_family("meta-llama/Llama-3.3-70B-Instruct") == "llama-3.3"
    # 세대가 다른 GPT는 합쳐지지 않는다.
    assert memory_family("gpt-4o") == "gpt-4o"
    assert memory_family("gpt-5.1") == "gpt-5.1"
    assert memory_family("deepseek-ai/DeepSeek-V4-Flash") == "deepseek-v4"
    assert memory_family("qwen/qwen3.7-max") == "qwen3.7"
    # 미등록 모델은 벤더 접두사를 떼고 자기 이름이 family(공유 안 됨).
    assert memory_family("some/Unknown-Model") == "unknown_model"
    # 규칙 없이도 벤더 접두사+크기 토큰을 떼어 사이즈 변형이 자동으로 묶인다.
    assert memory_family("Qwen/Qwen3.5-9B") == memory_family("Qwen/Qwen3.5-27B")
    assert memory_family("Qwen/Qwen3.5-9B") == "qwen3_5"
    # 버전이 다르면(같은 크기여도) 분리된다.
    assert memory_family("Qwen/Qwen3.5-9B") != memory_family("Qwen/Qwen2.5-9B")
    # 포맷 변형 토큰(base/instruct/it)도 떼어 더 묶는다.
    assert memory_family("K-intelligence/Midm-2.0-Base-Instruct") == "midm_2_0"
    assert memory_family("NCSOFT/Llama-VARCO-8B-Instruct") == "llama_varco"


def test_family_shares_db_and_strategies(tmp_path):
    # 같은 family의 두 모델이 같은 DB·전략 풀을 공유한다.
    from iamteamer.models import StrategyRec

    s8 = Store("meta-llama/Llama-3.1-8B-Instruct", str(tmp_path))
    s70 = Store("meta-llama/Llama-3.1-70B-Instruct", str(tmp_path))
    assert s8.path == s70.path

    with s8.tx() as conn:
        s8.insert_strategy(
            StrategyRec("meta-llama/Llama-3.1-8B-Instruct", "S", "s", "scenario_framing"),
            conn,
        )
    # 70B Store는 8B가 만든 전략을 그대로 본다(warm-start).
    assert len(s70.all_strategies()) == 1
    s8.close()
    s70.close()


def test_migrates_legacy_per_model_db(tmp_path):
    # family 키가 옛 파일명과 겹치는 단일 모델(gpt-4o)은 구 스키마 파일을 재사용한다.
    import sqlite3

    legacy = tmp_path / "gpt_4o.sqlite"
    conn = sqlite3.connect(str(legacy))
    conn.executescript(
        "CREATE TABLE strategies (strategy_id TEXT PRIMARY KEY,"
        " target_model TEXT NOT NULL, strategy_name TEXT, norm_name TEXT,"
        " strategy_family TEXT, trials INTEGER DEFAULT 0, successes INTEGER DEFAULT 0,"
        " avg_elite_score REAL DEFAULT 0.0);"
        "INSERT INTO strategies(strategy_id, target_model, strategy_name, norm_name,"
        " strategy_family, trials) VALUES ('x','gpt-4o','S','s','scenario_framing',2);"
    )
    conn.commit()
    conn.close()

    s = Store("gpt-4o", str(tmp_path))  # 마이그레이션 발동
    cols = {r["name"] for r in s.conn.execute("PRAGMA table_info(strategies)")}
    assert "family" in cols
    rows = s.all_strategies()  # family 백필되어 조회됨
    assert len(rows) == 1 and rows[0]["family"] == "gpt-4o"
    s.create_run("0", "g", "a", "j")  # family 컬럼 INSERT 정상
    s.close()


def test_raw_io_saved(tmp_path):
    s = Store("GPT-4o", str(tmp_path))
    run_id = s.create_run("0", "goal", "gpt-4o", "gpt-4o")
    io = LlmIo(run_id=run_id, role="attack", model="gpt-4o", status="ok",
               raw_output="raw", input_tokens=10, output_tokens=5)
    io_id = s.save_io(io)
    row = s.conn.execute("SELECT * FROM llm_io WHERE io_id=?", (io_id,)).fetchone()
    assert row["raw_output"] == "raw"
    assert row["total_tokens"] == 15
    s.close()


def test_turn_and_stats_same_transaction(tmp_path):
    s = Store("GPT-4o", str(tmp_path))
    run_id = s.create_run("0", "goal", "gpt-4o", "gpt-4o")
    turn_id = s.new_turn_id()
    res = _judge_res(success=True)
    with s.tx() as conn:
        from iamteamer.models import StrategyRec
        sid = s.insert_strategy(
            StrategyRec("GPT-4o", "S", "s", "scenario_framing"), conn
        )
        turn = {"turn_id": turn_id, "run_id": run_id, "turn_idx": 0,
                "strategy_name": "S", "strategy_family": "scenario_framing",
                "attack_prompt": "ap", "target_response": "tr"}
        s.save_turn(turn, res, sid, CostRec(10, 5, 0.0, 0.1), conn)
        s.save_refusal(turn_id, res.patterns.model_dump(), conn)
        s.update_stats(sid, res, CostRec(10, 5, 0.0, 0.1), conn)
    row = s.conn.execute("SELECT * FROM strategies WHERE strategy_id=?", (sid,)).fetchone()
    assert row["trials"] == 1
    assert row["successes"] == 1
    rp = s.conn.execute("SELECT * FROM refusal_patterns WHERE turn_id=?", (turn_id,)).fetchone()
    assert rp["partial_compliance"] == 1
    s.close()


def test_no_json_run_files(tmp_path):
    s = Store("GPT-4o", str(tmp_path))
    s.create_run("0", "goal", "gpt-4o", "gpt-4o")
    s.close()
    assert not list(tmp_path.glob("*.json"))
    assert list(tmp_path.glob("*.sqlite"))
