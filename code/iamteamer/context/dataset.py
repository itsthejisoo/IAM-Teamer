"""Goal dataset loader. Supports AdvBench CSV and JSON ({prompt, target})."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Goal:
    goal_id: str
    goal: str
    category: Optional[str]
    source: str
    target: Optional[str] = None


def load_goals(path: str) -> list[Goal]:
    if Path(path).suffix.lower() == ".json":
        rows = json.load(open(path, encoding="utf-8"))
    else:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    goals: list[Goal] = []
    for i, row in enumerate(rows):
        text = (row.get("goal") or row.get("prompt") or row.get("behavior") or "").strip()
        if not text:
            continue
        goals.append(Goal(str(i), text, row.get("category"), "advbench", row.get("target")))
    return goals


def get_goal(path: str, goal_id: str) -> Goal:
    for g in load_goals(path):
        if g.goal_id == goal_id:
            return g
    raise KeyError(f"goal_id {goal_id!r} not found in {path}")


def custom_goal(text: str) -> Goal:
    return Goal("custom", text.strip(), None, "custom")
