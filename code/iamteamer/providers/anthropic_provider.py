"""Anthropic provider (direct api.anthropic.com via its OpenAI-compatible
endpoint). Routes claude models straight to Anthropic using ANTHROPIC_API_KEY,
bypassing the OpenRouter gateway. Reuses the openai SDK so the message format
and base.py retry/error handling match the other providers.
"""

from __future__ import annotations

from .. import keys
from ..models import Usage
from .base import Provider, ProviderError

# 내부 모델명/슬러그 -> Anthropic 모델 id. OpenRouter식 슬러그도 받아 둘 다 허용.
_SLUGS = {
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-sonnet-4.6": "claude-sonnet-4-6",
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4-6",
}


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str, **opts) -> None:
        super().__init__(model, **opts)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ProviderError("openai package not installed") from e
        key = keys.anthropic_key()
        if not key:
            raise ProviderError("Anthropic key missing (set ANTHROPIC_API_KEY)")
        self._client = OpenAI(api_key=key, base_url=keys.ANTHROPIC_BASE_URL)
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
            raise ProviderError(f"anthropic call failed: {e}") from e
        text = resp.choices[0].message.content or ""
        u = resp.usage
        usage = Usage(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
        return text, usage, {"id": resp.id, "slug": self._slug}
