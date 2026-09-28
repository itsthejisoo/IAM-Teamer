"""Typed records shared across providers, DB, scoring and orchestration.

Pydantic models validate LLM JSON output; dataclasses carry internal state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

# The five canonical technique families.
BASE_FAMILIES = (
    "persona_modulation",
    "state_injection",
    "scenario_framing",
    "parallel_response",
    "topic_fragmentation",
)

StrategyFamily = str

Role = Literal["attack", "target", "judge", "analyst"]

FactorScope = Literal["strategy_specific", "general"]

IoStatus = Literal[
    "ok",
    "parse_failed",
    "refused_to_generate",
    "provider_error",
    "timeout",
    "fallback_used",
]


REFUSAL_KEYS = [
    "explicit_refusal",
    "policy_citation",
    "safety_redirect",
    "generic_safety_warning",
    "partial_compliance",
    "safe_alternative_offered",
    "asks_clarification",
    "topic_shift",
    "moralizing_language",
    "high_level_only",
    "missing_specificity",
]


class RefusalPatterns(BaseModel):
    explicit_refusal: bool = False
    policy_citation: bool = False
    safety_redirect: bool = False
    generic_safety_warning: bool = False
    partial_compliance: bool = False
    safe_alternative_offered: bool = False
    asks_clarification: bool = False
    topic_shift: bool = False
    moralizing_language: bool = False
    high_level_only: bool = False
    missing_specificity: bool = False


class RefusalAdaptation(BaseModel):
    observed_refusal_patterns: list[str] = Field(default_factory=list)
    adaptation_summary: str = ""
    avoid_repeating: list[str] = Field(default_factory=list)
    what_changed_from_previous_turn: str = ""


class AttackOut(BaseModel):
    """Validated Attack LLM output. No strategy_id here by design."""

    strategy_name: str
    strategy_family: StrategyFamily
    is_new_strategy: bool = True
    attack_prompt: str
    reasoning_summary: str = ""
    refusal_adaptation: RefusalAdaptation = Field(default_factory=RefusalAdaptation)
    used_persona_role: Optional[str] = None
    stop_reason: Literal["continue", "budget_exceeded", "success"] = "continue"


class FactorOut(BaseModel):
    """One distilled success mechanism. Scope keeps cross-strategy bleed in check."""

    mechanism: str
    scope: FactorScope = "strategy_specific"
    evidence: str = ""


class JudgeOut(BaseModel):
    """Validated Judge LLM output. elite_score/success recomputed in code.

    Accepts both the current judge keys (`refusal_pattern`, `reason`) and the
    legacy ones (`refusal_pattern_summary`, `rationale`).
    """

    model_config = ConfigDict(populate_by_name=True)

    refused: bool
    specific: float = 0.0
    convincing: float = 0.0
    toxicity: float = 0.0
    elite_score: float = 0.0
    rationale: str = Field(
        default="", validation_alias=AliasChoices("reason", "rationale")
    )
    detected_refusal_patterns: RefusalPatterns = Field(default_factory=RefusalPatterns)
    refusal_pattern_summary: str = Field(
        default="",
        validation_alias=AliasChoices("refusal_pattern", "refusal_pattern_summary"),
    )
    recommended_adaptation: str = ""
    success: bool = False
    # 거절이 아닐 때만 judge가 채우는 성공 비결(메커니즘). 코드가 success 판정 시 저장.
    success_factors: list[FactorOut] = Field(default_factory=list)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LlmResult:
    """Raw provider call result, provider-agnostic."""

    text: str
    usage: Usage
    latency_sec: float
    provider: str
    model: str
    raw: dict = field(default_factory=dict)


@dataclass
class CostRec:
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_sec: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LlmIo:
    """One persisted LLM call (success or failure)."""

    run_id: str
    role: Role
    model: str
    status: IoStatus
    turn_id: Optional[str] = None
    turn_idx: Optional[int] = None
    attempt_idx: int = 0
    sys_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    messages_json: Optional[str] = None
    raw_output: Optional[str] = None
    parsed_json: Optional[str] = None
    final_text: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_sec: float = 0.0
    cost: float = 0.0
    err: Optional[str] = None


@dataclass
class StrategyRec:
    target_model: str
    strategy_name: str
    norm_name: str
    strategy_family: str
    description: str = ""
    prompt_pattern: str = ""
    reasoning_summary: str = ""
    embedding_text: str = ""


@dataclass
class JudgeRes:
    """Code-verified scoring result derived from JudgeOut."""

    refused: bool
    specific: float
    convincing: float
    toxicity: float
    elite_score: float
    success: bool
    effective: bool
    patterns: RefusalPatterns
    refusal_summary: str
    next_adapt: str
    factors: list[FactorOut] = field(default_factory=list)
