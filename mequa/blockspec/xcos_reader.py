"""Read a Scilab Xcos diagram (.xcos XML) into a Mequa block specification.

Free alternative to Simulink: Xcos ships with Scilab. Supported interface functions, which map onto
the block types the simulator already validates:
  STEP_FUNCTION / STEP -> step_input     GAINBLK_f / GAINBLK -> gain
  CLR -> transfer_function (num/den polynomials in s)     INTEGRAL_f / INTEGRAL_m -> transfer_function 1/s
Display and clock blocks (CLOCK_c, CSCOPE, CMSCOPE, AFFICH_m, TIME_f, ...) are ignored and reported.
Anything else (SUMMATION, PRODUCT, feedback, unknown blocks) is rejected with a reason rather than
silently dropped, because a half-imported diagram would simulate something the user did not draw.

Parameters are read from each block's `exprs`; numbers and polynomials in s are parsed with a small
safe evaluator (no eval of file content). Context variables are not resolved and are reported.

Verified against hand-built files in the documented Xcos XML layout (tests/test_xcos_reader.py). It
has NOT been verified against files exported by Scilab itself; if an export fails to import, send the
error and the file.

  python -m mequa.blockspec.xcos_reader diagram.xcos --simulate
"""

from __future__ import annotations

import ast
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

MAX_BYTES = 5_000_000
IGNORED = {"CLOCK_c", "SampleCLK", "CSCOPE", "CMSCOPE", "CSCOPXY", "AFFICH_m", "AFFICH_f", "TIME_f", "OUT_f", "IN_f", "TOWS_c", "CLKSPLIT_f", "SPLIT_f", "CANIMXY"}


class XcosError(ValueError):
    pass


# ---------------------------------------------------------------- safe polynomial parser
def parse_polynomial(text: str) -> List[float]:
    """'2*s^2+3*s+1' -> [2, 3, 1] (descending powers). Only numbers, s, + - * ^ and parentheses."""
    src = text.strip().replace("^", "**")
    if not src:
        raise XcosError("empty expression")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as exc:
        raise XcosError(f"cannot parse {text!r}: {exc.msg}") from exc

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return np.poly1d([float(node.value)])
        if isinstance(node, ast.Name):
            if node.id != "s":
                raise XcosError(f"unresolved variable {node.id!r} in {text!r} (context variables are not supported)")
            return np.poly1d([1.0, 0.0])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a = ev(node.left)
            if isinstance(node.op, ast.Pow):
                if not (isinstance(node.right, ast.Constant) and isinstance(node.right.value, int) and 0 <= node.right.value <= 12):
                    raise XcosError(f"only small non-negative integer powers are supported in {text!r}")
                return a ** node.right.value
            b = ev(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
        raise XcosError(f"unsupported syntax in {text!r}")

    poly = ev(tree)
    coeffs = [float(c) for c in np.atleast_1d(poly.coeffs)]
    return coeffs


def _number(text: str, what: str) -> float:
    try:
        coeffs = parse_polynomial(text)
    except XcosError as exc:
        raise XcosError(f"{what}: {exc}") from exc
    if len(coeffs) != 1:
        raise XcosError(f"{what}: expected a number, got {text!r}")
    return coeffs[0]


# ---------------------------------------------------------------- xml helpers
def _exprs(block: ET.Element) -> List[str]:
    for child in block:
        if child.get("as") == "exprs":
            items = sorted(child.findall("data"), key=lambda d: int(d.get("line", "0")))
            return [d.get("value", "") for d in items]
    return []


def _find_inner(block: ET.Element, name: str) -> Optional[ET.Element]:
    for el in block.iter():
        if el is not block and el.get("interfaceFunctionName") == name:
            return el
    return None


def _convert(block: ET.Element, bid: str) -> Optional[Dict[str, Any]]:
    fn = block.get("interfaceFunctionName", "")
    if fn in IGNORED:
        return None
    if fn == "STEP_FUNCTION":
        inner = _find_inner(block, "STEP")
        ex = _exprs(inner) if inner is not None else _exprs(block)
        fn = "STEP"
    else:
        ex = _exprs(block)
    if fn == "STEP":
        if len(ex) < 3:
            raise XcosError(f"{bid}: STEP block needs step time, initial and final value")
        return {"id": bid, "type": "step_input", "parameters": {
            "step_time": _number(ex[0], f"{bid} step time"), "initial_value": _number(ex[1], f"{bid} initial value"),
            "final_value": _number(ex[2], f"{bid} final value")}}
    if fn in ("GAINBLK_f", "GAINBLK", "GAIN_f"):
        if not ex:
            raise XcosError(f"{bid}: gain block has no value")
        return {"id": bid, "type": "gain", "parameters": {"gain_value": _number(ex[0], f"{bid} gain")}}
    if fn == "CLR":
        if len(ex) < 2:
            raise XcosError(f"{bid}: CLR needs numerator and denominator")
        return {"id": bid, "type": "transfer_function", "parameters": {
            "numerator": parse_polynomial(ex[0]), "denominator": parse_polynomial(ex[1])}}
    if fn in ("INTEGRAL_f", "INTEGRAL_m"):
        return {"id": bid, "type": "transfer_function", "parameters": {"numerator": [1.0], "denominator": [1.0, 0.0]}}
    raise XcosError(f"{bid}: block type {fn or '(none)'} is not supported by the importer")


def read_xcos(path: str) -> Dict[str, Any]:
    p = Path(path)
    if p.suffix.lower() == ".zcos":
        raise XcosError(".zcos is a zipped format; in Xcos use File > Save as and choose .xcos")
    raw = p.read_bytes()
    if len(raw) > MAX_BYTES:
        raise XcosError("file is larger than 5 MB")
    if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise XcosError("DOCTYPE/ENTITY declarations are not accepted")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise XcosError(f"not valid XML: {exc}") from exc

    top = None
    for el in root.iter("root"):
        top = el
        break
    if top is None:
        raise XcosError("no <root> element: this does not look like an Xcos diagram")

    blocks = [el for el in top if el.tag.endswith("Block") and el.get("interfaceFunctionName")]
    port_owner: Dict[str, str] = {}
    for el in top:
        if el.tag.endswith("Port") and el.get("parent"):
            port_owner[el.get("id")] = el.get("parent")
    block_ids = {b.get("id"): f"block_{i + 1}" for i, b in enumerate(blocks)}

    spec_blocks, kept, ignored = [], {}, []
    for b in blocks:
        bid = block_ids[b.get("id")]
        converted = _convert(b, bid)
        if converted is None:
            ignored.append(b.get("interfaceFunctionName"))
        else:
            spec_blocks.append(converted)
            kept[b.get("id")] = bid

    conns = []
    for el in top:
        if el.tag.endswith("Link"):
            src, dst = port_owner.get(el.get("source")), port_owner.get(el.get("target"))
            if src in kept and dst in kept:
                conns.append({"from": kept[src], "to": kept[dst]})
            elif (src in kept) != (dst in kept) and (src in block_ids or dst in block_ids):
                continue  # link to an ignored display or clock block
    return {"block_spec": {"blocks": spec_blocks, "connections": conns}, "ignored_blocks": sorted(set(ignored))}


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--t-end", type=float, default=10.0)
    args = ap.parse_args(argv)
    try:
        out = read_xcos(args.file)
    except XcosError as exc:
        print(f"IMPORT FAILED: {exc}")
        return 1
    from mequa.blockspec.block_validator import validate_block_spec

    check = validate_block_spec(out["block_spec"])
    print(json.dumps({**out, "validation": check}, indent=2))
    if args.simulate and check["valid"]:
        from mequa.blockspec.xcos_importer import simulate_block_spec

        sim = simulate_block_spec(out["block_spec"], t_end=args.t_end)
        print("simulation:", "ok, final output %.4f" % sim["output"][-1] if sim["success"] else sim["error"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
