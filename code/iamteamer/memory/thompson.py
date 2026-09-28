"""Thompson sampling over per-strategy Beta posteriors.

Reward uses the success criterion (elite_score >= 10 and not refused), never
toxicity. Output is an advisory hint; it does not force strategy selection.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Posterior:
    strategy_id: str
    strategy_name: str
    family: str
    trials: int
    successes: int
    avg_score: float = 0.0
    score_per_1k: float = 0.0

    @property
    def alpha(self) -> float:
        return 1.0 + self.successes

    @property
    def beta(self) -> float:
        return 1.0 + (self.trials - self.successes)


def sample(post: Posterior, rng: random.Random) -> float:
    return rng.betavariate(post.alpha, post.beta)


def rank(posteriors: list[Posterior], top_k: int, seed: int | None = None) -> list[dict]:
    """Sample each posterior; break ties by cost efficiency (score_per_1k)."""
    rng = random.Random(seed)
    scored = [(sample(p, rng), p.score_per_1k, p) for p in posteriors]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [
        {
            "strategy_id": p.strategy_id,
            "name": p.strategy_name,
            "family": p.family,
            "ts_value": round(s, 4),
            "trials": p.trials,
            "successes": p.successes,
            "avg_score": round(p.avg_score, 2),
            "score_per_1k": round(p.score_per_1k, 3),
        }
        for s, _, p in scored[:top_k]
    ]
