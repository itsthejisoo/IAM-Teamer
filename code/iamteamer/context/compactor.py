"""Target conversation history compaction.

Keeps the most recent N turns raw; older turns collapse to one compact summary
message. The full message list actually sent is still persisted in llm_io.
"""

from __future__ import annotations

from typing import Optional


def compact_history(
    history: list[dict],
    attack_prompt: str,
    raw_turns: int,
    keep_full: bool = False,
) -> list[dict]:
    """Build the Target message list for this turn.

    history holds prior {role, content} pairs (user=attack, assistant=target).
    """
    if keep_full or len(history) <= raw_turns * 2:
        return [*history, {"role": "user", "content": attack_prompt}]

    cut = len(history) - raw_turns * 2
    old, recent = history[:cut], history[cut:]
    summary = _summarize_old(old)
    msgs: list[dict] = []
    if summary:
        msgs.append({"role": "user", "content": summary})
    msgs.extend(recent)
    msgs.append({"role": "user", "content": attack_prompt})
    return msgs


def _summarize_old(old: list[dict]) -> Optional[str]:
    if not old:
        return None
    parts = []
    for m in old:
        who = "User" if m["role"] == "user" else "Assistant"
        parts.append(f"{who}: {m['content'][:200]}")
    body = "\n".join(parts)
    return f"[Earlier conversation summary]\n{body}"
