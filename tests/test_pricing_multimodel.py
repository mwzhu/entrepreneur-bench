from decimal import Decimal

from solvent.doctor import doctor
from solvent.env.pricing import DEFAULT_BRAIN_PRICES, TokenUsage, brain_cost, price_for_model


def test_opus_4_8_pricing_is_corrected_and_versioned() -> None:
    price = price_for_model("claude-opus-4-8")

    assert price.input_per_million == Decimal("5.00")
    assert price.output_per_million == Decimal("25.00")
    assert price.source_url
    assert price.verified_date
    assert price.version


def test_unknown_model_pricing_fails_loudly() -> None:
    try:
        brain_cost("unknown-model", TokenUsage(input_tokens=1))
    except KeyError as exc:
        assert "unknown brain pricing" in str(exc)
    else:
        raise AssertionError("unknown model pricing should fail loudly")


def test_local_harness_pricing_is_zero_cost_and_versioned() -> None:
    for model in ["fake", "recorded", "stub"]:
        price = price_for_model(model)

        assert price.input_per_million == Decimal("0")
        assert price.output_per_million == Decimal("0")
        assert price.source_url.startswith("local://")
        assert price.version
        assert brain_cost(model, TokenUsage(input_tokens=10, output_tokens=10)) == Decimal("0.000000")


def test_brain_cost_includes_cache_token_rates() -> None:
    usage = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    price = DEFAULT_BRAIN_PRICES["claude-opus-4-8"]
    expected = (
        price.input_per_million
        + price.output_per_million
        + price.cache_read_per_million
        + price.cache_write_per_million
    ).quantize(Decimal("0.000001"))

    assert brain_cost("claude-opus-4-8", usage) == expected


def test_non_claude_registry_rows_have_provenance() -> None:
    for model in ["gpt-5.4-mini", "gemini-2.5-flash-lite", "kimi-k2.6", "glm-5", "minimax-m3"]:
        price = price_for_model(model)

        assert price.input_per_million > 0
        assert price.output_per_million > 0
        assert price.cache_read_per_million >= 0
        assert price.source_url.startswith("https://")
        assert price.verified_date == "2026-06-21"
        assert price.version


def test_doctor_surfaces_missing_pricing_before_live_run() -> None:
    report = doctor("unknown-model:base")
    pricing = [check for check in report["checks"] if check["name"] == "brain_pricing"][0]

    assert not pricing["ok"]
    assert "unknown brain pricing" in pricing["detail"]
