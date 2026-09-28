"""Model -> provider resolution. No model-specific logic lives in the runner.

Open-source families (gemma, llama, deepseek, qwen) route through HuggingFace
Inference Providers; gpt routes directly to OpenAI; claude routes directly to
Anthropic. OpenRouter remains only as a catch-all fallback. Both short names
and raw slugs/repo ids are accepted.
"""

from __future__ import annotations

from .base import Provider
from .anthropic_provider import AnthropicProvider
from .huggingface_provider import HuggingFaceProvider
from .openai_provider import OpenAIProvider
from .openrouter_provider import OpenRouterProvider

# Substrings that select each gateway.
_HF_FAMILIES = ("gemma", "llama", "deepseek", "qwen")
_OPENAI_FAMILIES = ("gpt",)
_ANTHROPIC_FAMILIES = ("claude",)


def get_provider(model: str, **opts) -> Provider:
    m = model.lower()
    if any(fam in m for fam in _HF_FAMILIES):
        return HuggingFaceProvider(model, **opts)
    if any(fam in m for fam in _OPENAI_FAMILIES):
        return OpenAIProvider(model, **opts)
    if any(fam in m for fam in _ANTHROPIC_FAMILIES):
        return AnthropicProvider(model, **opts)
    return OpenRouterProvider(model, **opts)
