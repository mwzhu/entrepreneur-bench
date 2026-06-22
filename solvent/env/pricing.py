from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


PRICING_TABLE_VERSION = "pricing_v0_5_2026_06_21"


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True)
class BrainPrice:
    input_per_million: Decimal
    output_per_million: Decimal
    cache_read_per_million: Decimal
    cache_write_per_million: Decimal
    source_url: str
    verified_date: str
    version: str


DEFAULT_BRAIN_PRICES: dict[str, BrainPrice] = {
    "fake": BrainPrice(
        Decimal("0"),
        Decimal("0"),
        Decimal("0"),
        Decimal("0"),
        "local://solvent/fake-client",
        "2026-06-20",
        PRICING_TABLE_VERSION,
    ),
    "recorded": BrainPrice(
        Decimal("0"),
        Decimal("0"),
        Decimal("0"),
        Decimal("0"),
        "local://solvent/recorded-client",
        "2026-06-20",
        PRICING_TABLE_VERSION,
    ),
    "stub": BrainPrice(
        Decimal("0"),
        Decimal("0"),
        Decimal("0"),
        Decimal("0"),
        "local://solvent/stub-harness",
        "2026-06-21",
        PRICING_TABLE_VERSION,
    ),
    "gpt-5.5": BrainPrice(
        Decimal("5.00"),
        Decimal("30.00"),
        Decimal("0.50"),
        Decimal("5.00"),
        "https://openai.com/api/pricing/",
        "2026-06-21",
        PRICING_TABLE_VERSION,
    ),
    "gpt-5.4-mini": BrainPrice(
        Decimal("0.75"),
        Decimal("4.50"),
        Decimal("0.075"),
        Decimal("0.75"),
        "https://openai.com/api/pricing/",
        "2026-06-21",
        PRICING_TABLE_VERSION,
    ),
    "gemini-2.5-flash": BrainPrice(
        Decimal("0.30"),
        Decimal("2.50"),
        Decimal("0.03"),
        Decimal("0.30"),
        "https://ai.google.dev/gemini-api/docs/pricing",
        "2026-06-21",
        PRICING_TABLE_VERSION,
    ),
    "gemini-2.5-flash-lite": BrainPrice(
        Decimal("0.10"),
        Decimal("0.40"),
        Decimal("0.01"),
        Decimal("0.10"),
        "https://ai.google.dev/gemini-api/docs/pricing",
        "2026-06-21",
        PRICING_TABLE_VERSION,
    ),
    "kimi-k2.6": BrainPrice(
        Decimal("0.95"),
        Decimal("4.00"),
        Decimal("0.16"),
        Decimal("0.95"),
        "https://platform.kimi.ai/docs/pricing/chat-k26",
        "2026-06-21",
        PRICING_TABLE_VERSION,
    ),
    "kimi-k2-6": BrainPrice(
        Decimal("0.95"),
        Decimal("4.00"),
        Decimal("0.16"),
        Decimal("0.95"),
        "https://platform.kimi.ai/docs/pricing/chat-k26",
        "2026-06-21",
        PRICING_TABLE_VERSION,
    ),
    "glm-5.2": BrainPrice(
        Decimal("1.40"),
        Decimal("4.40"),
        Decimal("0.26"),
        Decimal("1.40"),
        "https://docs.z.ai/guides/overview/pricing",
        "2026-06-21",
        PRICING_TABLE_VERSION,
    ),
    "glm-5": BrainPrice(
        Decimal("1.00"),
        Decimal("3.20"),
        Decimal("0.20"),
        Decimal("1.00"),
        "https://docs.z.ai/guides/overview/pricing",
        "2026-06-21",
        PRICING_TABLE_VERSION,
    ),
    "glm-4.5": BrainPrice(
        Decimal("0.60"),
        Decimal("2.20"),
        Decimal("0.11"),
        Decimal("0.60"),
        "https://docs.z.ai/guides/overview/pricing",
        "2026-06-21",
        PRICING_TABLE_VERSION,
    ),
    "minimax-m3": BrainPrice(
        Decimal("0.30"),
        Decimal("1.20"),
        Decimal("0.06"),
        Decimal("0.30"),
        "https://platform.minimax.io/docs/guides/pricing-paygo",
        "2026-06-21",
        PRICING_TABLE_VERSION,
    ),
    "minimax-m2.7": BrainPrice(
        Decimal("0.30"),
        Decimal("1.20"),
        Decimal("0.06"),
        Decimal("0.375"),
        "https://platform.minimax.io/docs/guides/pricing-paygo",
        "2026-06-21",
        PRICING_TABLE_VERSION,
    ),
    "claude-opus-4-8": BrainPrice(
        Decimal("5.00"),
        Decimal("25.00"),
        Decimal("0.50"),
        Decimal("5.00"),
        "https://docs.anthropic.com/en/docs/about-claude/pricing",
        "2026-06-20",
        PRICING_TABLE_VERSION,
    ),
    "claude-sonnet-4-6": BrainPrice(
        Decimal("3.00"),
        Decimal("15.00"),
        Decimal("0.30"),
        Decimal("3.00"),
        "https://docs.anthropic.com/en/docs/about-claude/pricing",
        "2026-06-20",
        PRICING_TABLE_VERSION,
    ),
    # --- v0.5 web-verified additions (verified 2026-06-20). cache_write_per_million
    # follows the registry convention of mirroring input where a provider does not
    # publish a distinct write rate; it is unused by the canonical estimate. ---
    # Anthropic (platform.claude.com/docs/en/about-claude/pricing)
    "claude-opus-4-7": BrainPrice(
        Decimal("5.00"), Decimal("25.00"), Decimal("0.50"), Decimal("5.00"),
        "https://platform.claude.com/docs/en/about-claude/pricing", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "claude-opus-4-6": BrainPrice(
        Decimal("5.00"), Decimal("25.00"), Decimal("0.50"), Decimal("5.00"),
        "https://platform.claude.com/docs/en/about-claude/pricing", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "claude-opus-4-5": BrainPrice(
        Decimal("5.00"), Decimal("25.00"), Decimal("0.50"), Decimal("5.00"),
        "https://platform.claude.com/docs/en/about-claude/pricing", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "claude-sonnet-4-5": BrainPrice(
        Decimal("3.00"), Decimal("15.00"), Decimal("0.30"), Decimal("3.00"),
        "https://platform.claude.com/docs/en/about-claude/pricing", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "claude-haiku-4-5": BrainPrice(
        Decimal("1.00"), Decimal("5.00"), Decimal("0.10"), Decimal("1.00"),
        "https://platform.claude.com/docs/en/about-claude/pricing", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "claude-fable-5": BrainPrice(
        Decimal("10.00"), Decimal("50.00"), Decimal("1.00"), Decimal("10.00"),
        "https://platform.claude.com/docs/en/about-claude/pricing", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    # OpenAI (developers.openai.com/api/docs/pricing)
    "gpt-5.4": BrainPrice(
        Decimal("2.50"), Decimal("15.00"), Decimal("0.25"), Decimal("2.50"),
        "https://developers.openai.com/api/docs/pricing", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "gpt-5.3-codex": BrainPrice(
        Decimal("1.75"), Decimal("14.00"), Decimal("0.175"), Decimal("1.75"),
        "https://developers.openai.com/api/docs/models/gpt-5.3-codex", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "gpt-5.2": BrainPrice(
        Decimal("1.75"), Decimal("14.00"), Decimal("0.175"), Decimal("1.75"),
        "https://developers.openai.com/api/docs/models/gpt-5.2", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "gpt-5.1": BrainPrice(
        Decimal("1.25"), Decimal("10.00"), Decimal("0.125"), Decimal("1.25"),
        "https://developers.openai.com/api/docs/models/gpt-5.1", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "gpt-5-mini": BrainPrice(
        Decimal("0.25"), Decimal("2.00"), Decimal("0.025"), Decimal("0.25"),
        "https://developers.openai.com/api/docs/models/gpt-5-mini", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    # GPT-OSS-120b is open-weight; rate is a representative Together AI hosted price (no provider prompt caching).
    "gpt-oss-120b": BrainPrice(
        Decimal("0.15"), Decimal("0.60"), Decimal("0.15"), Decimal("0.15"),
        "https://www.together.ai/models/gpt-oss-120b", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    # Google Gemini (ai.google.dev/gemini-api/docs/pricing; Pro = base <=200k tier)
    "gemini-3.1-pro": BrainPrice(
        Decimal("2.00"), Decimal("12.00"), Decimal("0.20"), Decimal("2.00"),
        "https://ai.google.dev/gemini-api/docs/pricing", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "gemini-3.5-flash": BrainPrice(
        Decimal("1.50"), Decimal("9.00"), Decimal("0.15"), Decimal("1.50"),
        "https://ai.google.dev/gemini-api/docs/pricing", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "gemini-3-flash": BrainPrice(
        Decimal("0.50"), Decimal("3.00"), Decimal("0.05"), Decimal("0.50"),
        "https://ai.google.dev/gemini-api/docs/pricing", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "gemini-2.5-pro": BrainPrice(
        Decimal("1.25"), Decimal("10.00"), Decimal("0.125"), Decimal("1.25"),
        "https://ai.google.dev/gemini-api/docs/pricing", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    # xAI (docs.x.ai). grok-4.1-fast is OpenRouter-only (no official page); flagged unverified.
    "grok-4.20": BrainPrice(
        Decimal("1.25"), Decimal("2.50"), Decimal("0.20"), Decimal("1.25"),
        "https://docs.x.ai/developers/models/grok-4.20-0309-reasoning", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "grok-4.3": BrainPrice(
        Decimal("1.25"), Decimal("2.50"), Decimal("0.20"), Decimal("1.25"),
        "https://docs.x.ai/developers/models/grok-4.3", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "grok-4.1-fast": BrainPrice(
        Decimal("0.20"), Decimal("0.50"), Decimal("0.05"), Decimal("0.20"),
        "https://openrouter.ai/x-ai/grok-4.1-fast", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    # DeepSeek (api-docs.deepseek.com). V4-Pro current; V3.2 last-known list (superseded by V4).
    "deepseek-v4-pro": BrainPrice(
        Decimal("0.435"), Decimal("0.87"), Decimal("0.003625"), Decimal("0.435"),
        "https://api-docs.deepseek.com/quick_start/pricing", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "deepseek-v3.2": BrainPrice(
        Decimal("0.28"), Decimal("0.42"), Decimal("0.028"), Decimal("0.28"),
        "https://api-docs.deepseek.com/quick_start/pricing", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    # MiniMax (platform.minimax.io/docs/guides/pricing-paygo; base tier)
    "minimax-m2.5": BrainPrice(
        Decimal("0.30"), Decimal("1.20"), Decimal("0.03"), Decimal("0.30"),
        "https://platform.minimax.io/docs/guides/pricing-paygo", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "minimax-m2": BrainPrice(
        Decimal("0.30"), Decimal("1.20"), Decimal("0.03"), Decimal("0.30"),
        "https://platform.minimax.io/docs/guides/pricing-paygo", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    # Moonshot/Kimi (platform.kimi.ai)
    "kimi-k2.7-code": BrainPrice(
        Decimal("0.95"), Decimal("4.00"), Decimal("0.19"), Decimal("0.95"),
        "https://platform.kimi.ai/docs/pricing/chat-k27-code", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "kimi-k2.5": BrainPrice(
        Decimal("0.60"), Decimal("3.00"), Decimal("0.10"), Decimal("0.60"),
        "https://platform.kimi.ai/docs/pricing/chat-k25", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    # Zhipu/GLM (docs.z.ai/guides/overview/pricing)
    "glm-5.1": BrainPrice(
        Decimal("1.40"), Decimal("4.40"), Decimal("0.26"), Decimal("1.40"),
        "https://docs.z.ai/guides/overview/pricing", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "glm-4.7": BrainPrice(
        Decimal("0.60"), Decimal("2.20"), Decimal("0.11"), Decimal("0.60"),
        "https://docs.z.ai/guides/overview/pricing", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    # Alibaba Qwen (alibabacloud.com/help/en/model-studio/model-pricing; Intl, base tier).
    # Alibaba does not publish a per-token cache-read rate, so cache_read mirrors input
    # (no caching discount assumed -> conservative, will not understate cost).
    "qwen3-max": BrainPrice(
        Decimal("1.20"), Decimal("6.00"), Decimal("1.20"), Decimal("1.20"),
        "https://www.alibabacloud.com/help/en/model-studio/model-pricing", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "qwen3.6-max": BrainPrice(
        Decimal("1.30"), Decimal("7.80"), Decimal("1.30"), Decimal("1.30"),
        "https://openrouter.ai/qwen/qwen3.6-max-preview", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "qwen3.6-plus": BrainPrice(
        Decimal("0.50"), Decimal("3.00"), Decimal("0.50"), Decimal("0.50"),
        "https://www.alibabacloud.com/help/en/model-studio/model-pricing", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "qwen3.5-plus": BrainPrice(
        Decimal("0.40"), Decimal("2.40"), Decimal("0.40"), Decimal("0.40"),
        "https://www.alibabacloud.com/help/en/model-studio/model-pricing", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "qwen3.5-35b-a3b": BrainPrice(
        Decimal("0.14"), Decimal("1.00"), Decimal("0.14"), Decimal("0.14"),
        "https://openrouter.ai/qwen/qwen3.5-35b-a3b", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "qwen3.5-27b": BrainPrice(
        Decimal("0.195"), Decimal("1.56"), Decimal("0.195"), Decimal("0.195"),
        "https://openrouter.ai/qwen/qwen3.5-27b", "2026-06-20", PRICING_TABLE_VERSION,
    ),
    "qwen3-235b-a22b-thinking": BrainPrice(
        Decimal("0.23"), Decimal("2.30"), Decimal("0.23"), Decimal("0.23"),
        "https://www.alibabacloud.com/help/en/model-studio/model-pricing", "2026-06-20", PRICING_TABLE_VERSION,
    ),
}


def price_for_model(model: str) -> BrainPrice:
    try:
        return DEFAULT_BRAIN_PRICES[model]
    except KeyError as exc:
        raise KeyError(f"unknown brain pricing for model: {model}") from exc


def brain_cost(model: str, usage: TokenUsage) -> Decimal:
    price = price_for_model(model)
    input_cost = Decimal(usage.input_tokens) * price.input_per_million / Decimal("1000000")
    output_cost = Decimal(usage.output_tokens) * price.output_per_million / Decimal("1000000")
    cache_read_cost = Decimal(usage.cache_read_tokens) * price.cache_read_per_million / Decimal("1000000")
    cache_write_cost = Decimal(usage.cache_write_tokens) * price.cache_write_per_million / Decimal("1000000")
    return (input_cost + output_cost + cache_read_cost + cache_write_cost).quantize(Decimal("0.000001"))
