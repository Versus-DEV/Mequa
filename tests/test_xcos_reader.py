import pytest

from mequa.blockspec.block_validator import validate_block_spec
from mequa.blockspec.xcos_importer import simulate_block_spec
from mequa.blockspec.xcos_reader import XcosError, parse_polynomial, read_xcos


def test_polynomial_parser_handles_scilab_syntax():
    assert parse_polynomial("1+0.5*s") == [0.5, 1.0]
    assert parse_polynomial("s^2+2*s+1") == [1.0, 2.0, 1.0]
    assert parse_polynomial("(s+1)*(s+2)") == [1.0, 3.0, 2.0]
    assert parse_polynomial("3") == [3.0]


@pytest.mark.parametrize("bad", ["__import__('os').system('x')", "K*s+1", "s**99", "open('f')", ""])
def test_polynomial_parser_rejects_unsafe_or_unresolved_input(bad):
    with pytest.raises(XcosError):
        parse_polynomial(bad)


def test_example_diagram_imports_validates_and_simulates_to_the_right_answer():
    out = read_xcos("data/examples/step_gain_lag.xcos")
    assert out["ignored_blocks"] == ["CSCOPE"]
    assert validate_block_spec(out["block_spec"])["valid"]
    y = simulate_block_spec(out["block_spec"], t_end=10.0)["output"]
    assert abs(y[-1] - 2 * 3 * 1) < 0.01


def test_unsupported_block_is_rejected_not_dropped(tmp_path):
    text = open("data/examples/step_gain_lag.xcos", encoding="utf-8").read().replace("GAINBLK_f", "SUMMATION")
    f = tmp_path / "x.xcos"
    f.write_text(text, encoding="utf-8")
    with pytest.raises(XcosError, match="not supported"):
        read_xcos(str(f))


def test_entity_declarations_and_zcos_are_refused(tmp_path):
    f = tmp_path / "e.xcos"
    f.write_text('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]><XcosDiagram/>')
    with pytest.raises(XcosError):
        read_xcos(str(f))
    z = tmp_path / "d.zcos"
    z.write_bytes(b"PK")
    with pytest.raises(XcosError):
        read_xcos(str(z))
