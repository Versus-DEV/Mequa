import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from mequa import simulink, slx_demo


def make_slx(path, blocks, lines):
    system = ET.Element("System")
    for sid, kind, name, params in blocks:
        b = ET.SubElement(system, "Block", BlockType=kind, Name=name, SID=str(sid))
        for key, value in params.items():
            ET.SubElement(b, "P", Name=key).text = value
    for src, dst in lines:
        line = ET.SubElement(system, "Line")
        ET.SubElement(line, "P", Name="Src").text = src
        ET.SubElement(line, "P", Name="Dst").text = dst
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("simulink/systems/system_root.xml", ET.tostring(system, encoding="unicode"))
    return path


def test_windkessel_matches_scipy(tmp_path):
    path = slx_demo.make_demo_slx(tmp_path / "w.slx")
    model = simulink.load_slx(path)
    res = simulink.simulate(model, t_end=5.0, dt=0.002, variables={"R": 1.0, "C": 0.3})
    q = lambda t: 4 * np.sin(2 * np.pi * 1.2 * t) + 5
    ref = solve_ivp(lambda t, p: (q(t) - p / 1.0) / 0.3, (0, 5.0), [5.0], t_eval=res.time, rtol=1e-10, atol=1e-12)
    assert np.max(np.abs(res.signals["Pressure"] - ref.y[0])) < 1e-6


def test_missing_variable_is_reported(tmp_path):
    model = simulink.load_slx(slx_demo.make_demo_slx(tmp_path / "w.slx"))
    with pytest.raises(simulink.ModelError, match="variable"):
        simulink.simulate(model, t_end=1.0)


def test_expressions_cannot_execute_code():
    with pytest.raises(simulink.ModelError):
        simulink.evaluate("__import__('os').system('echo hi')", {})
    with pytest.raises(simulink.ModelError):
        simulink.evaluate("(1).__class__", {})
    assert simulink.evaluate("2*pi", {}) == pytest.approx(2 * np.pi)
    assert simulink.evaluate("1/R", {"R": 4.0}) == 0.25


def test_unsupported_block_is_rejected(tmp_path):
    path = make_slx(
        tmp_path / "t.slx",
        [(1, "Constant", "c", {"Value": "1"}), (2, "TransferFcn", "tf", {}), (3, "Outport", "out", {})],
        [("1#out:1", "2#in:1"), ("2#out:1", "3#in:1")],
    )
    with pytest.raises(simulink.ModelError, match="unsupported blocks"):
        simulink.simulate(simulink.load_slx(path), t_end=1.0)


def test_algebraic_loop_is_detected(tmp_path):
    path = make_slx(
        tmp_path / "loop.slx",
        [(1, "Gain", "g", {"Gain": "2"}), (2, "Outport", "out", {})],
        [("1#out:1", "1#in:1"), ("1#out:1", "2#in:1")],
    )
    with pytest.raises(simulink.ModelError, match="algebraic loop"):
        simulink.simulate(simulink.load_slx(path), t_end=0.1)


def test_unconnected_input_is_reported(tmp_path):
    path = make_slx(tmp_path / "u.slx", [(1, "Gain", "g", {"Gain": "2"}), (2, "Outport", "out", {})], [("1#out:1", "2#in:1")])
    with pytest.raises(simulink.ModelError, match="not connected"):
        simulink.simulate(simulink.load_slx(path), t_end=0.1)


def test_not_a_zip_is_reported(tmp_path):
    bad = tmp_path / "bad.slx"
    bad.write_text("not a zip")
    with pytest.raises(simulink.ModelError):
        simulink.load_slx(bad)
