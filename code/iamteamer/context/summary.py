"""Conversation summary maintenance (compressed running state).

Kept short (~300-600 tokens); recompressed when it grows past budget.
"""

from __future__ import annotations

from typing import Callable, Optional

from ..prompts import loader
from ..utils import count_tokens, fit_text

SUMMARY_SYS = loader.text("summary", "system")


def summary_user_prompt(goal: str, prev_summary: str, turn: dict) -> str:
    return (
        f"Goal: {goal}\n\n"
        f"Previous running summary:\n{prev_summary or '(none)'}\n\n"
        f"New turn {turn['turn_idx']}:\n"
        f"- attack: {turn.get('attack_prompt', '')[:400]}\n"
        f"- target: {turn.get('target_response', '')[:600]}\n"
        f"- judge: refused={turn.get('refused')} score={turn.get('elite_score')}\n"
        f"- refusal: {turn.get('refusal_summary', '')}\n\n"
        "Return the updated running summary."
    )


def update_summary(
    goal: str,
    prev_summary: str,
    turn: dict,
    max_tokens: int,
    summarizer: Optional[Callable[[str, str], str]] = None,
) -> str:
    if summarizer is not None:
        text = summarizer(SUMMARY_SYS, summary_user_prompt(goal, prev_summary, turn))
    else:
        text = _heuristic(prev_summary, turn)
    return fit_text(text, max_tokens)


def _heuristic(prev_summary: str, turn: dict) -> str:
    line = (
        f"[t{turn['turn_idx']}] strategy={turn.get('strategy_name')} "
        f"refused={turn.get('refused')} score={turn.get('elite_score')} "
        f"{turn.get('refusal_summary', '')}".strip()
    )
    merged = (prev_summary + "\n" + line).strip() if prev_summary else line
    return merged


def maybe_recompress(summary: str, max_tokens: int) -> str:
    if count_tokens(summary) <= max_tokens:
        return summary
    return fit_text(summary, max_tokens)
