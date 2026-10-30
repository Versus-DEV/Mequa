"""Number-grounding check: every number in a generated summary must come from the solver output.

Llama writes the summary, but it is not trusted with the numbers. We pull every number out of the
text and look for it among the numbers in the measured facts. A number is accepted when it

  * equals a fact value (to the precision it was written with, or within 0.5 percent relative),
  * is a fact value expressed as a percentage (0.12 may be written as 12 or 12 percent),
  * is a small ordinal that names a zone that exists ("zone 2") or counts the zones,
  * is a difference or ratio between two zones' values stated in the facts only if
    allow_derived is set (off by default, because derived numbers are exactly where a small
    model drifts).

Anything else is reported as ungrounded. The agent regenerates once or twice with the offending
numbers listed, and falls back to a deterministic template if the model keeps inventing numbers.
"""

import re

NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")


def _flatten(obj, out):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten(v, out)


def fact_numbers(facts):
    out = []
    _flatten(facts, out)
    return out


def extract_numbers(text):
    """Return (token, value, decimals) for each number written in the text."""
    found = []
    for m in NUMBER.finditer(text):
        token = m.group(0).rstrip(".,")
        clean = token.replace(",", "")
        try:
            value = float(clean)
        except ValueError:
            continue
        decimals = len(clean.split(".")[1].split("e")[0]) if "." in clean else 0
        found.append((token, value, decimals))
    return found


def _close(written, fact, decimals):
    if decimals == 0 and float(fact).is_integer():
        return written == fact
    if round(fact, decimals) == round(written, decimals):
        return True
    return abs(written - fact) <= 0.005 * max(abs(fact), 1e-12)


DEFINITION = re.compile(r"\b(90|95)(?:st|nd|rd|th)\s+percentile", re.IGNORECASE)


def ungrounded_numbers(text, facts, zone_ids=()):
    """Numbers in text that cannot be traced to facts. Empty list means fully grounded.

    The phrases "90th percentile" and "95th percentile" are definitions of the reported
    quantities (s_crit and p95 strain), not results, so those two numbers are allowed there only.
    """
    text = DEFINITION.sub("percentile", text)
    allowed = fact_numbers(facts)
    zone_ids = {float(z) for z in zone_ids}
    bad = []
    for token, value, decimals in extract_numbers(text):
        ok = any(_close(value, f, decimals) for f in allowed)
        if not ok:
            ok = any(_close(value / 100.0, f, decimals + 2) for f in allowed if 0 <= f <= 1.0)
        if not ok and value in zone_ids:
            ok = True
        if not ok:
            bad.append(token)
    return bad


def template_summary(facts):
    """Deterministic fallback summary built only from the facts. No model involved."""
    zones = facts.get("per_zone", [])
    parts = [
        f"The analysis covered {facts['cells']} cells in {facts['zones']} density zones, with "
        f"{facts['mechanical_links']} mechanical links and a pull of {facts['load_fraction']} of the tissue width.",
        f"SPHI is {facts['sphi']}, mean strain is {facts['mean_strain']} and maximum strain is {facts['max_strain']} "
        f"against a critical strain (90th percentile) of {facts['s_crit']}.",
    ]
    if zones:
        top = max(zones, key=lambda z: z["mean_strain"])
        low = min(zones, key=lambda z: z["mean_strain"])
        parts.append(
            f"Zone {top['zone']} has the highest mean strain ({top['mean_strain']}) and zone {low['zone']} the lowest "
            f"({low['mean_strain']})."
        )
    return " ".join(parts)


def check_number_grounding(summary, solver_outputs, zone_ids=()):
    """Compatibility wrapper used by the dashboard: same keys as before, tolerant matching."""
    bad = ungrounded_numbers(summary, solver_outputs, zone_ids)
    total = len(extract_numbers(summary))
    return {
        "pass": not bad,
        "grounded": total - len(bad),
        "total": total,
        "ungrounded": bad,
        "message": "All numbers grounded in solver output." if not bad else f"{len(bad)} number(s) not found in solver output: {bad}",
    }
