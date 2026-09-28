"""Judge LLM client. Scoring is verified in code, not trusted from the judge."""

from __future__ import annotations

from dataclasses import dataclass

from ..cost import cost_of
from ..utils import extract_json
from ..models import CostRec, JudgeOut, JudgeRes, RefusalPatterns
from ..prompts import judge as jp
from ..providers.base import Provider
from ..eval.refusal import heuristic_patterns
from ..eval.scoring import calc_elite, is_effective, is_success


@dataclass
class JudgeCall:
    res: JudgeRes
    sys_prompt: str
    user_prompt: str
    raw_output: str
    parsed: dict | None
    cost: CostRec
    status: str


def _verify(out: JudgeOut) -> JudgeRes:
    # partial_hedging은 내용이 있으면 judge가 refused=true를 잘못 설정한 것
    refused = out.refused
    if (
        out.refusal_pattern_summary == "partial_hedging"
        and refused
        and (out.specific > 0 or out.convincing > 0 or out.toxicity > 0)
    ):
        refused = False
    score = calc_elite(refused, out.specific, out.convincing, out.toxicity)
    return JudgeRes(
        refused=refused,
        specific=out.specific,
        convincing=out.convincing,
        toxicity=out.toxicity,
        elite_score=score,
        success=is_success(refused, score),
        effective=is_effective(out.toxicity),
        patterns=out.detected_refusal_patterns,
        refusal_summary=out.refusal_pattern_summary,
        next_adapt=out.recommended_adaptation,
        factors=out.success_factors if not refused else [],
    )


class JudgeClient:
    def __init__(self, provider: Provider) -> None:
        self.provider = provider

    def evaluate(self, goal: str, attack_prompt: str, target_response: str) -> JudgeCall:
        sys = jp.JUDGE_SYS
        user = jp.judge_user_prompt(goal, attack_prompt, target_response)
        res = self.provider.chat([
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ])
        parsed = extract_json(res.text)
        cost = cost_of(res, sys + user)
        if parsed is None:
            jr = _fallback_res(target_response)
            return JudgeCall(jr, sys, user, res.text, None, cost, "parse_failed")
        try:
            out = JudgeOut(**parsed)
            jr = _verify(out)
            return JudgeCall(jr, sys, user, res.text, parsed, cost, "ok")
        except Exception:
            jr = _fallback_res(target_response)
            return JudgeCall(jr, sys, user, res.text, parsed, cost, "parse_failed")


def _fallback_res(target_response: str) -> JudgeRes:
    """Conservative result when the judge output cannot be parsed."""
    patterns: RefusalPatterns = heuristic_patterns(target_response)
    refused = patterns.explicit_refusal
    return JudgeRes(
        refused=refused, specific=0.0, convincing=0.0, toxicity=0.0,
        elite_score=0.0, success=False, effective=False, patterns=patterns,
        refusal_summary="judge parse failed; heuristic fallback",
        next_adapt="retry judge or rephrase",
    )
