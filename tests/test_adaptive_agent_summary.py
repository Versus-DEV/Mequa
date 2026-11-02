from mequa.agents.adaptive_investigator import ensure_threshold_stated


def _trace(tl):
    return [{"type": "tool_call", "tool": "sweep_load", "result": {"rows": [], "threshold_load": tl, "strain_limit": 0.05}}]


def test_threshold_appended_when_missing():
    out = ensure_threshold_stated("Strain looks fine.", _trace(0.12))
    assert "0.12" in out and out.startswith("Strain looks fine.")


def test_threshold_not_duplicated():
    t = "Threshold is 0.12."
    assert ensure_threshold_stated(t, _trace(0.12)) == t


def test_no_sweep_unchanged():
    assert ensure_threshold_stated("x", []) == "x"
    assert ensure_threshold_stated("x", _trace(None)) == "x"
