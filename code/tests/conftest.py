"""Keep the suite fully offline and fast: no live price fetch, no heavy
embedding model (use the deterministic hash embedding instead)."""

import iamteamer.cost as cost
import iamteamer.memory.embeddings as embeddings

cost.LIVE_FETCH = False
embeddings.USE_MODEL = False
