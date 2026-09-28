"""Target LLM client. Maintains multi-turn state; compacts old history.

Exposes the exact message list sent, so it can be persisted to llm_io.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..context.compactor import compact_history
from ..cost import cost_of
from ..models import CostRec
from ..providers.base import ContentBlocked, Provider

# 입력단 차단 시 합성하는 거절 응답. judge가 거절(미성공)로 판정하도록 명시적 거절 문구.
BLOCKED_RESPONSE = (
    "I'm sorry, but I can't help with that request. "
    "[provider safety filter blocked this prompt before the model saw it]"
)


@dataclass
class TargetResult:
    response: str
    messages: list[dict]
    cost: CostRec
    blocked: bool = False


class TargetClient:
    def __init__(self, provider: Provider, raw_turns: int = 3, keep_full: bool = False) -> None:
        self.provider = provider
        self.raw_turns = raw_turns
        self.keep_full = keep_full
        self.history: list[dict] = []

    def send(self, attack_prompt: str) -> TargetResult:
        messages = compact_history(self.history, attack_prompt, self.raw_turns, self.keep_full)
        try:
            res = self.provider.chat(messages)
        except ContentBlocked:
            # 모델 미도달 → 거절 턴으로 처리. 무비용, blocked 표식.
            self.history.append({"role": "user", "content": attack_prompt})
            self.history.append({"role": "assistant", "content": BLOCKED_RESPONSE})
            return TargetResult(BLOCKED_RESPONSE, messages, CostRec(), blocked=True)
        prompt_text = "\n".join(m["content"] for m in messages)
        self.history.append({"role": "user", "content": attack_prompt})
        self.history.append({"role": "assistant", "content": res.text})
        return TargetResult(res.text, messages, cost_of(res, prompt_text))

    def rollback(self) -> None:
        """직전 (user, assistant) 쌍을 히스토리에서 제거 — 거절 턴 백트래킹용.

        타겟 입장에선 그 교환이 '없던 일'이 되어, 자기 거절에 일관되어
        계속 거절하는 것을 막는다. (실패 자체는 호출측에서 DB에 남긴다.)
        """
        if len(self.history) >= 2:
            self.history.pop()  # assistant(거절 응답)
            self.history.pop()  # user(거절당한 질문)
