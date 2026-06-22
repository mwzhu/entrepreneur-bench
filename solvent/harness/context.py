from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


VALID_CONTEXT_POLICIES = {"none", "sliding_window", "anchored", "scratchpad"}


@dataclass
class ContextManager:
    policy: str = "anchored"
    window_tokens: int = 30000
    _anchor: int = field(default=0, init=False)
    _tok_cache: dict[int, int] = field(default_factory=dict, init=False)

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
        if self.policy == "anchored":
            return self._anchored(history, token_limit=self.window_tokens - reserved_tokens)
        return self._sliding_window(history, token_limit=self.window_tokens - reserved_tokens)

    def trim_memory(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.policy in {"none", "anchored"}:
            # Anchored windowing indexes into the full history with a monotonic
            # anchor, so the stored history must NOT be evicted from the front
            # (that would shift indices and break the stable cacheable prefix).
            return list(history)
        # Keep a little more than the prompt window so scratchpad/window construction
        # has recent raw events without allowing memory to grow forever.
        return self._sliding_window(history, token_limit=self.window_tokens * 2)

    def _anchored(self, history: list[dict[str, Any]], token_limit: int | None = None) -> list[dict[str, Any]]:
        """Bounded window that keeps a *stable* prefix so prompt caching survives.

        A plain sliding window evicts the oldest item every turn, so the prompt
        prefix changes each turn and the provider's prefix cache is invalidated
        constantly (measured: hit rate collapses ~78% -> ~0% once the window
        fills). Instead we hold a monotonic anchor and only advance it in *chunks*
        when the tail overflows the limit, dropping down to ~half the limit so the
        new prefix stays stable for many turns. Between trims the prefix is
        append-only and fully cacheable; only the trim turn (~every 60-120 turns)
        pays a cache miss.
        """
        limit = self.window_tokens if token_limit is None else token_limit
        if limit <= 0:
            return []
        n = len(history)
        if n == 0:
            return []
        if self._anchor >= n:
            self._anchor = max(0, n - 1)

        def tail_tokens(start: int) -> int:
            return sum(self._item_tokens(index, history[index]) for index in range(start, n))

        if tail_tokens(self._anchor) <= limit:
            return list(history[self._anchor :])
        # Overflow: jump the anchor forward to free headroom (drop to ~half) so the
        # prefix stays stable for many subsequent turns. Always keep >=1 item.
        target = max(1, limit // 2)
        while self._anchor < n - 1 and tail_tokens(self._anchor) > target:
            self._anchor += 1
        return list(history[self._anchor :])

    def _item_tokens(self, index: int, item: dict[str, Any]) -> int:
        cached = self._tok_cache.get(index)
        if cached is None:
            cached = estimate_tokens(item)
            self._tok_cache[index] = cached
        return cached

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
    # ~2.7 chars/token, calibrated against live deepseek-v3.2 requests (median
    # actual/estimate was 1.485x under the old chars/4 heuristic). Undercounting
    # under-fills the window and makes the cost estimate run hot, so we match the
    # measured ratio. Provider-neutral; avoids a tokenizer dependency.
    return max(1, (len(text) * 10 + 26) // 27)
