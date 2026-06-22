from copy import deepcopy
from decimal import Decimal

from solvent.delivery.menu import DIFFICULTIES, DeliveryMenu, delivery_draw_key


def test_delivery_menu_schema_and_profiles_are_valid() -> None:
    menu = DeliveryMenu.load_default()

    assert menu.schema_version == "solvent_delivery_menu_v0_4"
    assert menu.version == "menu_v0_4"
    assert len(menu.checksum) == 64
    assert menu.calibration["basis"]
    assert [model.name for model in menu.public_models()] == ["tool-mini", "tool-mid", "tool-pro"]
    assert all(model.price > Decimal("0") for model in menu.public_models())


def test_delivery_profile_is_monotone_by_difficulty() -> None:
    menu = DeliveryMenu.load_default()

    for task_type in ["data_clean", "extract"]:
        for model in menu.public_models():
            pass_rates = [menu.pass_prob(task_type, model.name, difficulty) for difficulty in DIFFICULTIES]
            durations = [menu.duration(task_type, model.name, difficulty) for difficulty in DIFFICULTIES]
            assert pass_rates == sorted(pass_rates, reverse=True)
            assert durations == sorted(durations)


def test_delivery_resolve_is_deterministic_and_charges_model_price() -> None:
    menu = DeliveryMenu.load_default()

    first = menu.resolve("data_clean", "tool-mini", "easy", "seed:job:model")
    second = menu.resolve("data_clean", "tool-mini", "easy", "seed:job:model")

    assert first == second
    assert first.price_charged == Decimal("0.02")
    assert first.pass_prob == menu.pass_prob("data_clean", "tool-mini", "easy")
    # The draw is exposed and consistent with the pass/fail decision.
    assert 0.0 <= first.draw < 1.0
    assert first.passed == (first.draw < first.pass_prob)


def test_pass_prob_by_model_and_shared_draw_key() -> None:
    menu = DeliveryMenu.load_default()

    probs = menu.pass_prob_by_model("extract", "easy")
    assert set(probs) == {"tool-mini", "tool-mid", "tool-pro"}
    assert probs["tool-mini"] == menu.pass_prob("extract", "tool-mini", "easy")

    # The shared key reproduces the environment's exact draw for reconstruction.
    key = delivery_draw_key(141, "ex-141-1", "tool-mini", 0, "menu_v0_4")
    assert key == "141:ex-141-1:tool-mini:0:menu_v0_4"
    resolution = menu.resolve("extract", "tool-mini", "easy", key)
    assert menu.resolve("extract", "tool-mini", "easy", key).draw == resolution.draw


def test_delivery_menu_requires_calibration_provenance() -> None:
    data = deepcopy(DeliveryMenu.load_default().data)
    del data["calibration"]

    try:
        DeliveryMenu(data, checksum="test")
    except ValueError as exc:
        assert "calibration provenance" in str(exc)
    else:
        raise AssertionError("missing calibration provenance should fail loader validation")


def test_delivery_menu_rejects_dominated_tool_frontier() -> None:
    data = deepcopy(DeliveryMenu.load_default().data)
    for task_profile in data["profile"].values():
        task_profile["tool-mid"] = deepcopy(task_profile["tool-mini"])
    for tool in data["tools"]:
        if tool["name"] == "tool-mid":
            tool["price"] = "0.10"

    try:
        DeliveryMenu(data, checksum="test")
    except ValueError as exc:
        assert "dominated tools" in str(exc)
        assert "tool-mid" in str(exc)
    else:
        raise AssertionError("dominated tool should fail loader validation")
