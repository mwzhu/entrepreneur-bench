from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


VALID_CONTEXT_POLICIES = {"none", "sliding_window", "scratchpad"}


@dataclass(frozen=True)
class ContextManager:
    policy: str = "sliding_window"
    window_tokens: int = 24000

    def __post_init__(self) -> None:
        if self.policy not in VALID_CONTEXT_POLICIES:
            raise ValueError(f"unknown context policy: {self.policy}")
        if self.window_tokens < 1:
            raise ValueError("window_tokens must be at least 1")

    def build(self, history: list[dict[str, Any]], reserved_tokens: int = 0) -> list[dict[str, Any]]:
        if self.policy == "none":
            return list(history)
        if self.policy == "scratchpad":
            return self._scratchpad(history, token_limit=self.window_tokens - reserved_tokens)
        return self._sliding_window(history, token_limit=self.window_tokens - reserved_tokens)

    def trim_memory(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.policy == "none":
            return list(history)
        # Keep a little more than the prompt window so scratchpad/window construction
        # has recent raw events without allowing memory to grow forever.
        return self._sliding_window(history, token_limit=self.window_tokens * 2)

    def _sliding_window(self, history: list[dict[str, Any]], token_limit: int | None = None) -> list[dict[str, Any]]:
        limit = self.window_tokens if token_limit is None else token_limit
        if limit <= 0:
            return []
        kept: list[dict[str, Any]] = []
        total = 0
        for item in reversed(history):
            item_tokens = estimate_tokens(item)
            if item_tokens > limit:
                continue
            if total + item_tokens > limit:
                break
            kept.append(item)
            total += item_tokens
            if total >= limit:
                break
        kept.reverse()
        return kept

    def _scratchpad(self, history: list[dict[str, Any]], token_limit: int | None = None) -> list[dict[str, Any]]:
        if not history:
            return []
        limit = self.window_tokens if token_limit is None else token_limit
        if limit <= 0:
            return []
        summary = {
            "turns_seen": len(history),
            "recent_results": [
                {
                    "tool_call": item.get("tool_call"),
                    "ok": item.get("result", {}).get("ok"),
                    "result": item.get("result", {}).get("result"),
                    "error": item.get("result", {}).get("error"),
                }
                for item in history[-8:]
            ],
        }
        while summary["recent_results"] and estimate_tokens([{"scratchpad": summary}]) > limit:
            summary["recent_results"].pop(0)
        return [{"scratchpad": summary}]


def estimate_tokens(value: Any) -> int:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    # Conservative enough for tests and provider-neutral budgeting without pulling
    # tokenizer dependencies into the core package.
    return max(1, (len(text) + 3) // 4)
