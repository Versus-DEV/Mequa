import math

from mequa.blockspec.block_validator import validate_block_spec
from mequa.blockspec.normalize import normalize_block_spec
from mequa.blockspec.semantic_block_translator import translate_to_block_spec
from mequa.blockspec.xcos_importer import simulate_block_spec


def _chain(tf):
    return {"blocks": [{"id": "block_1", "type": "step_input", "parameters": {"step_time": 0.0, "initial_value": 0.0, "final_value": 1.0}},
                       {"id": "block_2", "type": "transfer_function", "parameters": tf}],
            "connections": [{"from": "block_1", "to": "block_2"}]}


def test_second_order_form_expands_to_the_textbook_coefficients():
    spec = normalize_block_spec(_chain({"form": "second_order", "wn": 2.0, "zeta": 0.3}))
    p = spec["blocks"][1]["parameters"]
    assert p["numerator"] == [4.0] and p["denominator"] == [1.0, 1.2, 4.0]
    sim = simulate_block_spec(spec, t_end=10.0)
    y = sim["output"]
    assert abs((max(y) / y[-1] - 1) - math.exp(-math.pi * 0.3 / math.sqrt(1 - 0.09))) < 0.03


def test_first_order_and_explicit_coefficients_are_untouched_or_expanded():
    p = normalize_block_spec(_chain({"form": "first_order", "K": 2.0, "tau": 1.5}))["blocks"][1]["parameters"]
    assert p["numerator"] == [2.0] and p["denominator"] == [1.5, 1.0]
    explicit = {"numerator": [9.0], "denominator": [1.0, 1.0]}
    assert normalize_block_spec(_chain(explicit))["blocks"][1]["parameters"] == explicit


def test_bad_form_is_left_for_the_validator_to_reject():
    spec = normalize_block_spec(_chain({"form": "second_order", "wn": -1.0, "zeta": 0.3}))
    assert not validate_block_spec(spec)["valid"]


def test_translator_repairs_one_invalid_answer():
    answers = iter(['{"blocks": []}',
                    '{"blocks":[{"id":"block_1","type":"step_input","parameters":{"final_value":1.0}},'
                    '{"id":"block_2","type":"transfer_function","parameters":{"form":"first_order","K":1.0,"tau":2.0}}],'
                    '"connections":[{"from":"block_1","to":"block_2"}]}'])
    out = translate_to_block_spec("first order, tau 2", model="x", generate_fn=lambda **kw: {"response": next(answers)})
    assert out["success"] and out["attempts"] == 2
    assert validate_block_spec(out["block_spec"])["valid"]


def test_translator_does_not_loop_forever_on_garbage():
    out = translate_to_block_spec("anything", model="x", generate_fn=lambda **kw: {"response": "no json here"})
    assert not out["success"] and out["attempts"] == 2
