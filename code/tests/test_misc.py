import json

from iamteamer.context.compactor import compact_history
from iamteamer.context.dataset import custom_goal, get_goal, load_goals
from iamteamer.cost import CostTracker, cost_of, usage_or_estimate
from iamteamer.memory.personas import Personas, load_personas
from iamteamer.models import LlmResult, Usage
from iamteamer.utils import count_tokens, extract_json, fit_items, fit_text, trim_ctx


def _hist(n):
    h = []
    for i in range(n):
        h.append({"role": "user", "content": f"attack {i}"})
        h.append({"role": "assistant", "content": f"reply {i}"})
    return h


def test_keeps_all_when_short():
    msgs = compact_history(_hist(2), "new", raw_turns=3)
    assert msgs[-1] == {"role": "user", "content": "new"}
    assert len(msgs) == 5


def test_compacts_old_turns():
    msgs = compact_history(_hist(6), "new", raw_turns=2)
    assert msgs[0]["content"].startswith("[Earlier conversation summary]")
    # summary + 2 raw turns (4 msgs) + new prompt
    assert len(msgs) == 6
    assert msgs[-1]["content"] == "new"


def test_keep_full_disables_compaction():
    msgs = compact_history(_hist(6), "new", raw_turns=2, keep_full=True)
    assert len(msgs) == 13


def _res(usage):
    return LlmResult(text="hello world", usage=usage, latency_sec=0.1, provider="p", model="gpt-4o")


def test_prefers_provider_usage():
    u = usage_or_estimate(_res(Usage(100, 50)), "some prompt")
    assert u.input_tokens == 100 and u.output_tokens == 50


def test_estimates_when_usage_missing():
    u = usage_or_estimate(_res(Usage(0, 0)), "some prompt text here")
    assert u.total > 0


def test_unknown_model_zero_cost():
    res = LlmResult("x", Usage(10, 10), 0.0, "p", "mystery-model")
    rec = cost_of(res, "p")
    assert rec.cost == 0.0


def test_tracker_accumulates():
    t = CostTracker()
    rec = cost_of(_res(Usage(1000, 1000)), "p")
    t.add(rec)
    assert t.total_tokens == 2000
    assert t.total_cost == rec.cost


def test_load_goals_json(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps([
        {"prompt": "do x", "target": "Sure, here is x"},
        {"prompt": "do y", "target": "Sure, here is y"},
    ]))
    goals = load_goals(str(p))
    assert len(goals) == 2
    assert goals[0].goal == "do x"
    assert goals[0].target == "Sure, here is x"
    assert goals[0].source == "advbench"


def test_load_goals_csv(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("goal,category\nmake a thing,weapons\n")
    goals = load_goals(str(p))
    assert goals[0].goal == "make a thing"
    assert goals[0].category == "weapons"


def test_get_goal_by_id(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps([{"prompt": "a"}, {"prompt": "b"}]))
    assert get_goal(str(p), "1").goal == "b"


def test_custom_goal():
    assert custom_goal("  hi  ").goal == "hi"


def test_personas_pick(tmp_path):
    p = tmp_path / "jobs.json"
    p.write_text(json.dumps([
        {"role": "Chemist", "job_description": "synthesize compounds in a lab"},
        {"role": "Chef", "job_description": "prepare food in a kitchen"},
    ]))
    personas = load_personas(str(p))
    out = personas.pick("synthesize a chemical compound", k=1)
    assert len(out) == 1
    assert set(out[0]) == {"role", "description"}


def test_personas_empty():
    assert Personas([]).pick("anything", 5) == []


def test_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}


def test_embedded_json():
    assert extract_json('Here you go: {"a": 3}. Done.') == {"a": 3}


def test_invalid_returns_none():
    assert extract_json("no json here") is None
    assert extract_json("") is None


def test_fit_text_truncates():
    long = "word " * 1000
    assert count_tokens(fit_text(long, 50)) <= 55


def test_fit_text_keeps_short():
    assert fit_text("short", 100) == "short"


def test_fit_items_drops_tail():
    items = [{"k": "v" * 100} for _ in range(50)]
    assert len(fit_items(items, 100)) < 50


def test_trim_ctx_shrinks_lists():
    ctx = {
        "conv_summary": "x" * 5000,
        "similar_cases": [{"a": "b" * 200} for _ in range(5)],
        "top_strategies": [{"a": "b" * 200} for _ in range(5)],
    }
    out = trim_ctx(ctx, 200)
    assert count_tokens(json.dumps(out, default=str)) <= 400
