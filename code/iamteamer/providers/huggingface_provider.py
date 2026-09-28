"""HuggingFace Inference Providers (OpenAI-compatible router).

Open-source models (gemma, llama, deepseek) route here instead of OpenRouter.
The model id carries a routing-policy suffix (default ":fastest") so HF picks
the fastest available serverless provider.
"""

from __future__ import annotations

from .. import keys
from ..models import Usage
from .base import Provider, ProviderError

# Short name -> HuggingFace repo id (case-sensitive on the Hub).
_REPOS = {
    "gemma-4-31b": "google/gemma-4-31B-it",
    "gemma-4-31b-it": "google/gemma-4-31B-it",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-v4-flash": "deepseek-ai/DeepSeek-V4-Flash",
}

DEFAULT_POLICY = "fastest"  # HF router policy: fastest | cheapest | preferred


class HuggingFaceProvider(Provider):
    name = "huggingface"

    def __init__(self, model: str, **opts) -> None:
        super().__init__(model, **opts)
        # HF 서버리스는 콜드스타트/일시 장애가 잦아 기본적으로 무한 대기로 재시도.
        self.retry_forever = bool(opts.get("retry_forever", True))
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ProviderError("openai package not installed") from e
        key = keys.hf_key()
        if not key:
            raise ProviderError("HF token missing (set HF_TOKEN or HF_KEY)")
        self._client = OpenAI(api_key=key, base_url=keys.HF_BASE_URL)
        repo = _REPOS.get(model.lower(), model)
        policy = opts.get("hf_policy", DEFAULT_POLICY)
        # Don't double-append a policy the caller already specified.
        self._slug = repo if ":" in repo else f"{repo}:{policy}"

    def _chat(self, messages: list[dict], **kw) -> tuple[str, Usage, dict]:
        try:
            resp = self._client.chat.completions.create(
                model=self._slug,
                messages=messages,
                temperature=kw.get("temperature", 0.9),
                max_tokens=kw.get("max_tokens", 4096),
            )
        except Exception as e:
            raise ProviderError(f"huggingface call failed: {e}") from e
        text = resp.choices[0].message.content or ""
        u = resp.usage
        usage = Usage(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
        return text, usage, {"id": resp.id, "slug": self._slug}
