import math

from mequa.blockspec.block_validator import validate_block_spec
from mequa.blockspec.xcos_importer import simulate_block_spec


def spec(den=(1.0, 1.0), gain=None):
    blocks = [
        {"id": "u", "type": "step_input", "parameters": {"step_time": 0.0, "initial_value": 0.0, "final_value": 1.0}},
        {"id": "g", "type": "transfer_function", "parameters": {"numerator": [2.0], "denominator": list(den)}},
    ]
    conns = [{"from": "u", "to": "g"}]
    if gain is not None:
        blocks.append({"id": "k", "type": "gain", "parameters": {"gain_value": gain}})
        conns.append({"from": "g", "to": "k"})
    return {"blocks": blocks, "connections": conns}


def test_first_order_step_matches_the_analytical_answer():
    out = simulate_block_spec(spec(), t_end=5.0, n=2001)  # 2/(s+1): y(t) = 2(1 - e^-t)
    assert out["success"]
    assert abs(out["output"][-1] - 2 * (1 - math.exp(-5.0))) < 1e-3


def test_gain_block_and_connections_are_honoured():
    out = simulate_block_spec(spec(gain=3.0), t_end=5.0, n=2001)
    assert out["chain"] == ["u", "g", "k"]
    assert abs(out["output"][-1] - 3 * 2 * (1 - math.exp(-5.0))) < 5e-3


def test_unsupported_and_malformed_specs_are_rejected_with_reasons():
    s = spec()
    s["blocks"].append({"id": "s", "type": "sum"})
    assert not validate_block_spec(s)["valid"]
    bad = spec(den=(0.0, 1.0))
    assert any("non-zero" in e for e in validate_block_spec(bad)["errors"])
    improper = {"blocks": spec()["blocks"], "connections": spec()["connections"]}
    improper["blocks"][1]["parameters"] = {"numerator": [1, 2, 3], "denominator": [1, 1]}
    assert any("proper" in e for e in validate_block_spec(improper)["errors"])
    loop = spec()
    loop["connections"].append({"from": "g", "to": "u"})
    assert not validate_block_spec(loop)["valid"]
    assert not simulate_block_spec({"blocks": [], "connections": []})["success"]
