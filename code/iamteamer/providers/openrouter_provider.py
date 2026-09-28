"""OpenRouter provider (OpenAI-compatible). Routes closed models (gpt, qwen)
through the OpenRouter gateway. Open-source models go via HuggingFace instead
(see registry.get_provider).
"""

from __future__ import annotations

from .. import keys
from ..models import Usage
from .base import Provider, ProviderError

# Internal model name -> OpenRouter slug. Adjust to match available models.
_SLUGS = {
    "gpt-4o": "openai/gpt-4o",
    "gpt-5.1": "openai/gpt-5.1",
    "qwen3.7-max": "qwen/qwen3.7-max",
}


class OpenRouterProvider(Provider):
    name = "openrouter"

    def __init__(self, model: str, **opts) -> None:
        super().__init__(model, **opts)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ProviderError("openai package not installed") from e
        self._client = OpenAI(api_key=keys.openrouter_key(), base_url=keys.OPENROUTER_BASE_URL)
        self._slug = _SLUGS.get(model.lower(), model)

    def _chat(self, messages: list[dict], **kw) -> tuple[str, Usage, dict]:
        try:
            resp = self._client.chat.completions.create(
                model=self._slug,
                messages=messages,
                temperature=kw.get("temperature", 0.9),
                max_tokens=kw.get("max_tokens", 1024),
            )
        except Exception as e:
            raise ProviderError(f"openrouter call failed: {e}") from e
        text = resp.choices[0].message.content or ""
        u = resp.usage
        usage = Usage(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
        return text, usage, {"id": resp.id, "slug": self._slug}
