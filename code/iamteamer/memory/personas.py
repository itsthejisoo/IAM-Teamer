"""Persona role library for persona_modulation strategies.

Loads job_descriptions.json ({role, job_description}) and selects goal-relevant
roles via embedding similarity, surfaced to the Attack LLM as hints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from .embeddings import cosine, embed


@dataclass
class Role:
    role: str
    description: str
    vector: list[float]


class Personas:
    def __init__(self, roles: list[Role]) -> None:
        self.roles = roles

    def pick(self, goal: str, k: int) -> list[dict]:
        if not self.roles:
            return []
        gv = embed(goal)
        scored = sorted(
            self.roles, key=lambda r: cosine(gv, r.vector), reverse=True
        )
        return [
            {"role": r.role, "description": r.description[:160]} for r in scored[:k]
        ]


@lru_cache(maxsize=4)
def load_personas(path: str) -> Personas:
    data = json.load(open(path, encoding="utf-8"))
    roles = []
    for row in data:
        name = row.get("role") or ""
        desc = row.get("job_description") or row.get("description") or ""
        if not name.strip():
            continue
        roles.append(Role(name.strip(), desc.strip(), embed(f"{name} {desc}")))
    return Personas(roles)
