"""First-turn-only prompt. No history; attacker freely picks the strongest opener."""

from __future__ import annotations

import json

from . import loader
from .attack import SCHEMA_HINT


def cold_start_user_prompt(
    goal: str,
    target_model: str,
    target_family: str,
    strategy_hints: list[dict],
    persona_options: list[dict] | None = None,
) -> str:
    return (
        f"{loader.text('cold_start', 'instructions')}\n\n"
        f"goal: {goal}\n"
        f"target_model: {target_model}\n"
        f"target_model_family: {target_family}\n"
        f"available_strategy_hints: {json.dumps(strategy_hints, default=str)}\n"
        f"persona_options: {json.dumps(persona_options or [], default=str)}\n\n"
        f"Return ONLY JSON in this schema:\n{SCHEMA_HINT}"
    )
