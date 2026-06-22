from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from solvent.env.pricing import TokenUsage

MODEL_ALIAS_PREFIX = "SOLVENT_MODEL_ALIAS_"


@dataclass(frozen=True)
class ModelRequest:
    model: str
    system_prompt: str
    observation: dict[str, Any]
    tools: dict[str, dict[str, Any]]
    history: list[dict[str, Any]]
    temperature: float = 0.0
    cache_hint: bool = True
    # Ask the provider to emit its reasoning/thinking trace. Off by default so
    # direct payload construction stays unchanged; the harness turns it on.
    reasoning: bool = False


@dataclass(frozen=True)
class ModelResponse:
    tool_call: dict[str, Any]
    usage: TokenUsage = TokenUsage()
    content: str = ""
    # The model's reasoning / chain-of-thought, when the provider returns it.
    reasoning: str = ""


class ModelClient(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse:
        ...


def model_alias_env_var(model: str) -> str:
    safe = "".join(char.upper() if char.isalnum() else "_" for char in model)
    return f"{MODEL_ALIAS_PREFIX}{safe}"


def resolve_model_name(model: str) -> str:
    return os.environ.get(model_alias_env_var(model), model)
