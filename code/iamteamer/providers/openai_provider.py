"""OpenAI provider (direct api.openai.com). Routes gpt models straight to
OpenAI using OPENAI_API_KEY, bypassing the OpenRouter gateway.
"""

from __future__ import annotations

from .. import keys
from ..models import Usage
from .base import Provider, ProviderError

# Internal model name -> OpenAI model id. Accepts the bare name or the
# "openai/" OpenRouter-style slug so callers can pass either form.
_SLUGS = {
    "gpt-4o": "gpt-4o",
    "gpt-5.1": "gpt-5.1",
    "openai/gpt-4o": "gpt-4o",
    "openai/gpt-5.1": "gpt-5.1",
}


def _is_reasoning_era(slug: str) -> bool:
    """gpt-5+ and the o-series require max_completion_tokens and reject any
    non-default temperature."""
    s = slug.lower()
    return s.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, model: str, **opts) -> None:
        super().__init__(model, **opts)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ProviderError("openai package not installed") from e
        key = keys.openai_key()
        if not key:
            raise ProviderError("OpenAI key missing (set OPENAI_API_KEY)")
        self._client = OpenAI(api_key=key, base_url=keys.OPENAI_BASE_URL)
        self._slug = _SLUGS.get(model.lower(), model)

    def _chat(self, messages: list[dict], **kw) -> tuple[str, Usage, dict]:
        params: dict = {"model": self._slug, "messages": messages}
        if _is_reasoning_era(self._slug):
            params["max_completion_tokens"] = kw.get("max_tokens", 1024)
        else:
            params["max_tokens"] = kw.get("max_tokens", 1024)
            params["temperature"] = kw.get("temperature", 0.9)
        try:
            resp = self._client.chat.completions.create(**params)
        except Exception as e:
            raise ProviderError(f"openai call failed: {e}") from e
        text = resp.choices[0].message.content or ""
        u = resp.usage
        usage = Usage(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
        return text, usage, {"id": resp.id, "slug": self._slug}
