"""Sentence-embedding helper for strategy dedup. Lazy-loaded, cached.

Falls back to a deterministic hashing vector when the model is unavailable,
so the pipeline and tests run without the heavy dependency.
"""

from __future__ import annotations

import hashlib
import struct
from functools import lru_cache

from ..config import EMBEDDING_MODEL

_DIM = 384

# Set False to force the deterministic hash embedding (tests disable the heavy
# model in conftest to stay offline and fast).
USE_MODEL = True


@lru_cache(maxsize=1)
def _model():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception:
        return None


def embed(text: str) -> list[float]:
    if USE_MODEL:
        m = _model()
        if m is not None:
            return m.encode(text, normalize_embeddings=True).tolist()
    return _hash_embed(text)


def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * _DIM
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % _DIM] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def from_blob(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
