import pytest

from mequa.blockspec.block_validator import validate_block_spec
from mequa.blockspec.rule_translator import parse_description, translate_offline
from mequa.blockspec.semantic_block_translator import translate_to_block_spec
from mequa.blockspec.xcos_importer import simulate_block_spec


def _chain(res):
    return [(b["type"], b["parameters"]) for b in res["block_spec"]["blocks"]]


def test_first_order_prompt_from_the_editor_default():
    res = translate_offline("Simulate a first-order transfer function with gain 2 and time constant 1, driven by a step input.")
    assert res["success"] and res["source"] == "rules"
    tf = _chain(res)[1][1]
    assert tf["numerator"] == [2.0] and tf["denominator"] == [1.0, 1.0]
    sim = simulate_block_spec(res["block_spec"], t_end=12)
    assert sim["output"][-1] == pytest.approx(2.0, abs=1e-2)  # DC gain 2


def test_second_order_with_step_amplitude():
    res = translate_offline("second-order system with natural frequency 3 and damping ratio 0.4, step of amplitude 2")
    tf = _chain(res)[1][1]
    assert tf["denominator"] == [1.0, pytest.approx(2.4), 9.0] and tf["numerator"] == [9.0]
    assert _chain(res)[0][1]["final_value"] == 2.0


def test_separate_gain_block_is_not_folded_into_the_transfer_function():
    res = translate_offline("An integrator with gain 5 followed by a gain block of 3")
    kinds = [k for k, _ in _chain(res)]
    assert kinds == ["step_input", "transfer_function", "gain"]
    assert _chain(res)[1][1]["numerator"] == [5.0] and _chain(res)[2][1]["gain_value"] == 3.0
    assert validate_block_spec(res["block_spec"])["valid"]


def test_unreadable_description_is_refused_with_a_reason():
    res = translate_offline("make something nice")
    assert not res["success"] and "first-order" in res["error"]
    assert not translate_offline("second-order system")["success"]


def test_ollama_down_gives_friendly_error_and_optional_fallback():
    down = lambda **kw: {"error": "URLError: <urlopen error [WinError 10061] refused>"}
    off = translate_to_block_spec("first-order, gain 2, time constant 1", generate_fn=down)
    assert not off["success"] and "not reachable" in off["error"] and "source" not in off
    on = translate_to_block_spec("first-order, gain 2, time constant 1", generate_fn=down, offline_fallback=True)
    assert on["success"] and on["source"] == "rules" and "not reachable" in on["llm_error"]
