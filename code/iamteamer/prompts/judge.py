"""Judge LLM prompts. Rubric/schema loaded from prompts.yaml. Minimal input."""

from __future__ import annotations

from . import loader

JUDGE_SYS = loader.text("judge", "system")
SCHEMA_HINT = loader.schema_json("judge", "schema")


def judge_user_prompt(goal: str, attack_prompt: str, target_response: str) -> str:
    return (
        f"goal:\n{goal}\n\n"
        f"attack_prompt:\n{attack_prompt}\n\n"
        f"target_response:\n{target_response}\n\n"
        f"Return ONLY JSON in this schema:\n{SCHEMA_HINT}"
    )
