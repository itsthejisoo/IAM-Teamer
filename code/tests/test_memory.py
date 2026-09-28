import random

from iamteamer.config import DedupThresholds, RunConfig
from iamteamer.cost import CostTracker
from iamteamer.db import Store, retrieval
from iamteamer.context.dataset import custom_goal
from iamteamer.memory.strategy import norm_name, upsert_strategy
from iamteamer.memory.thompson import Posterior, rank, sample
from iamteamer.models import AttackOut
from iamteamer.orchestrator import RunState, build_ctx


def test_posterior_prior():
    p = Posterior("s1", "n", "f", trials=0, successes=0)
    assert p.alpha == 1.0 and p.beta == 1.0


def test_posterior_update_counts():
    p = Posterior("s1", "n", "f", trials=10, successes=4)
    assert p.alpha == 5.0
    assert p.beta == 7.0


def test_rank_prefers_higher_success():
    strong = Posterior("a", "strong", "f", trials=20, successes=18)
    weak = Posterior("b", "weak", "f", trials=20, successes=1)
    wins = 0
    for seed in range(50):
        order = rank([strong, weak], top_k=1, seed=seed)
        if order[0]["strategy_id"] == "a":
            wins += 1
    assert wins > 40


def test_sample_in_unit_interval():
    p = Posterior("s", "n", "f", trials=3, successes=2)
    v = sample(p, random.Random(0))
    assert 0.0 <= v <= 1.0


def _out(name, family="scenario_framing", prompt="some attack prompt about evaluation framing"):
    return AttackOut(strategy_name=name, strategy_family=family, attack_prompt=prompt)


def test_norm_name():
    assert norm_name("Scenario Framing!") == "scenario_framing"


def test_create_then_match_same(tmp_path):
    s = Store("GPT-4o", str(tmp_path))
    run_id = s.create_run("0", "g", "gpt-4o", "gpt-4o")
    th = DedupThresholds()
    out = _out("Scenario Framing")
    with s.tx() as conn:
        r1 = upsert_strategy(s, out, run_id, s.new_turn_id(), th, conn)
    assert r1.action == "created"
    with s.tx() as conn:
        r2 = upsert_strategy(s, _out("Scenario Framing"), run_id, s.new_turn_id(), th, conn)
    assert r2.strategy_id == r1.strategy_id
    assert r2.action in ("variant_added", "matched_existing")
    s.close()


def test_distinct_strategy_creates_new(tmp_path):
    s = Store("GPT-4o", str(tmp_path))
    run_id = s.create_run("0", "g", "gpt-4o", "gpt-4o")
    th = DedupThresholds()
    with s.tx() as conn:
        r1 = upsert_strategy(s, _out("Alpha", prompt="completely different wording one"),
                             run_id, s.new_turn_id(), th, conn)
    with s.tx() as conn:
        r2 = upsert_strategy(
            s, _out("Beta", family="persona_modulation",
                    prompt="adopt the persona of a forensic specialist entirely"),
            run_id, s.new_turn_id(), th, conn,
        )
    assert r1.strategy_id != r2.strategy_id
    assert len(s.all_strategies()) == 2
    s.close()


def _store(tmp_path):
    s = Store("T", str(tmp_path))
    s.create_run("0", "g", "a", "j")
    return s


def test_transition_upsert_and_stats(tmp_path):
    s = _store(tmp_path)
    with s.tx() as c:
        s.update_transition("scenario_framing", "persona_modulation", gain=5.0, win=True, conn=c)
        s.update_transition("scenario_framing", "persona_modulation", gain=1.0, win=True, conn=c)
        s.update_transition("scenario_framing", "state_injection", gain=-2.0, win=False, conn=c)
    rows = s.top_transitions("scenario_framing", 5)
    # persona_modulation has the better win_rate, ranked first
    assert rows[0]["to_family"] == "persona_modulation"
    assert rows[0]["trials"] == 2
    assert rows[0]["wins"] == 2
    assert abs(rows[0]["avg_gain"] - 3.0) < 1e-9
    s.close()


def test_transitions_retrieval_shape(tmp_path):
    s = _store(tmp_path)
    with s.tx() as c:
        s.update_transition("a", "b", gain=4.0, win=True, conn=c)
    out = retrieval.transitions(s, "a", 3)
    assert out == [{"from": "a", "to": "b", "trials": 1, "win_rate": 1.0, "avg_gain": 4.0}]
    assert retrieval.transitions(s, "nonexistent", 3) == []
    s.close()


def test_transitions_advisory_in_ctx(tmp_path):
    s = _store(tmp_path)
    with s.tx() as c:
        s.update_transition("scenario_framing", "persona_modulation", gain=6.0, win=True, conn=c)
    cfg = RunConfig("a", "T", "j", memory_dir=str(tmp_path), personas_path=None)
    st = RunState(cfg, custom_goal("g"), s, attack=None, target=None, judge=None,
                  cost=CostTracker(), run_id="r", last_family="scenario_framing")
    ctx = build_ctx(st)
    assert ctx["strategy_transitions"]["current_family"] == "scenario_framing"
    assert ctx["strategy_transitions"]["worked_well_next"][0]["to"] == "persona_modulation"
    s.close()


def test_no_transitions_without_last_family(tmp_path):
    s = _store(tmp_path)
    cfg = RunConfig("a", "T", "j", memory_dir=str(tmp_path), personas_path=None)
    st = RunState(cfg, custom_goal("g"), s, attack=None, target=None, judge=None,
                  cost=CostTracker(), run_id="r")
    assert "strategy_transitions" not in build_ctx(st)
    s.close()
