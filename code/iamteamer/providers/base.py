"""Common provider interface. Provider-specific code stays behind this."""

from __future__ import annotations

import logging
import random
import sys
import time
from abc import ABC, abstractmethod

from ..models import LlmResult, Usage

log = logging.getLogger(__name__)

# Substrings marking a transient error worth retrying with backoff.
_RETRYABLE = ("429", "500", "502", "503", "504", "too many requests",
              "rate limit", "timeout", "timed out", "overloaded", "temporarily",
              "service unavailable", "unavailable", "loading", "connection",
              "bad gateway", "gateway")

# 입력단(safety classifier)에서 프롬프트 자체가 차단된 경우의 지문. 모델 미도달.
_CONTENT_BLOCK = ("invalid_prompt", "content_policy", "content_filter",
                  "content management policy", "responsibleaipolicyviolation",
                  "your prompt was flagged")


class ProviderError(RuntimeError):
    pass


class ContentBlocked(ProviderError):
    """프롬프트가 제공자 안전 필터에 입력 단계에서 차단됨(모델이 보지 못함).

    연결/서버 에러와 달리 결정적이라 재시도해도 동일하게 막힌다. 호출측은 이를
    '타겟이 막아낸 거절 턴'으로 처리해 run을 죽이지 않고 max_turns까지 진행한다.
    """


def _is_retryable(err: Exception) -> bool:
    msg = str(err).lower()
    return any(m in msg for m in _RETRYABLE)


def _is_content_block(err: Exception) -> bool:
    msg = str(err).lower()
    return any(m in msg for m in _CONTENT_BLOCK)


class Provider(ABC):
    """One model behind a uniform chat interface."""

    name: str = "base"

    def __init__(self, model: str, **opts) -> None:
        self.model = model
        self.opts = opts
        self.max_retries = int(opts.get("max_retries", 5))
        # True면 재시도 가능한(서버) 에러에서 성공할 때까지 무한 대기한다.
        self.retry_forever = bool(opts.get("retry_forever", False))
        self.base_delay = float(opts.get("retry_base_delay", 2.0))
        self.max_delay = float(opts.get("retry_max_delay", 30.0))

    @abstractmethod
    def _chat(self, messages: list[dict], **kw) -> tuple[str, Usage, dict]:
        """Return (text, usage, raw_meta). Raise ProviderError on failure."""

    def chat(self, messages: list[dict], **kw) -> LlmResult:
        attempt = 0
        while True:
            t0 = time.perf_counter()
            try:
                text, usage, raw = self._chat(messages, **kw)
            except ContentBlocked:
                raise
            except ProviderError as e:
                # 안전 필터 입력 차단은 결정적 → 재시도/대기 없이 전용 예외로 승격.
                if _is_content_block(e):
                    raise ContentBlocked(str(e)) from e
                retryable = _is_retryable(e)
                # 무한 모드가 아니면 횟수 소진/비재시도 에러에서 포기.
                if not retryable or (not self.retry_forever
                                     and attempt >= self.max_retries):
                    raise
                # 무한 모드에선 지수 백오프가 max_delay에서 멈추도록 지수를 제한.
                delay = min(self.base_delay * 2 ** min(attempt, 6), self.max_delay)
                delay += random.uniform(0, delay * 0.25)
                if self.retry_forever:
                    # 다음 단계로 넘어가지 않고 서버가 살아날 때까지 대기.
                    print(
                        f"[WARNING] Waiting for {self.name} API to recover "
                        f"(attempt {attempt + 1}): {e} — retrying in {delay:.1f}s",
                        file=sys.stderr, flush=True,
                    )
                else:
                    log.warning(
                        "%s transient error (attempt %d/%d), retrying in %.1fs: %s",
                        self.name, attempt + 1, self.max_retries, delay, e,
                    )
                time.sleep(delay)
                attempt += 1
                continue
            return LlmResult(
                text=text,
                usage=usage,
                latency_sec=time.perf_counter() - t0,
                provider=self.name,
                model=self.model,
                raw=raw,
            )
