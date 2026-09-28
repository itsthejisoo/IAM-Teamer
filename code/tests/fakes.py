"""In-memory fake providers for deterministic tests (no API keys, no network)."""

from __future__ import annotations

from iamteamer.models import Usage
from iamteamer.providers.base import Provider


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, model: str, replies: list[str], usage: Usage | None = None) -> None:
        super().__init__(model)
        self._replies = list(replies)
        self._usage = usage or Usage(50, 30)
        self.calls: list[list[dict]] = []

    def _chat(self, messages: list[dict], **kw) -> tuple[str, Usage, dict]:
        self.calls.append(messages)
        if len(self._replies) > 1:
            reply = self._replies.pop(0)
        else:
            reply = self._replies[0] if self._replies else "{}"
        return reply, self._usage, {}
