import ast
import math
import operator
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SUPPORTED = {"Constant", "Sin", "Step", "Gain", "Sum", "Product", "Saturate", "Integrator", "Outport", "Scope"}
PORT = re.compile(r"^(\d+)#(out|in):(\d+)$")
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
FUNCS = {"sin": math.sin, "cos": math.cos, "exp": math.exp, "sqrt": math.sqrt, "abs": abs}
CONSTS = {"pi": math.pi, "e": math.e, "inf": math.inf}


class ModelError(Exception):
    pass


@dataclass
class Block:
    sid: str
    name: str
    kind: str
    params: dict


@dataclass
class Model:
    name: str
    blocks: dict
    wires: dict
    variables: dict = field(default_factory=dict)


@dataclass
class SimResult:
    time: np.ndarray
    signals: dict


def _eval_node(node, variables):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, variables)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in variables:
            return float(variables[node.id])
        if node.id in CONSTS:
            return CONSTS[node.id]
        raise ModelError(f"the model uses the variable '{node.id}'; provide it with --set {node.id}=value")
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval_node(node.operand, variables))
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ModelError("exponent too large")
        return OPS[type(node.op)](left, right)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in FUNCS
        and len(node.args) == 1
        and not node.keywords
    ):
        return FUNCS[node.func.id](_eval_node(node.args[0], variables))
    raise ModelError("unsupported expression in a block parameter")


def evaluate(expr, variables, default=None):
    text = (expr or "").strip()
    if text == "":
        return default
    text = text.replace("^", "**")
    try:
        return float(_eval_node(ast.parse(text, mode="eval"), variables))
    except ModelError:
        raise
    except ZeroDivisionError as exc:
        raise ModelError(f"division by zero in parameter '{expr}'") from exc
    except (SyntaxError, ValueError, OverflowError) as exc:
        raise ModelError(f"cannot evaluate parameter '{expr}'") from exc


def _params(element):
    out = {}
    for p in element.findall("P"):
        out[p.get("Name")] = (p.text or "").strip()
    for inst in element.findall("InstanceData"):
        for p in inst.findall("P"):
            out[p.get("Name")] = (p.text or "").strip()
    return out


def load_slx(path):
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ModelError("not a valid .slx file (a .slx is a zip archive)") from exc
    with archive as z:
        names = sorted(n for n in z.namelist() if n.startswith("simulink/") and n.endswith(".xml"))
        if not names:
            raise ModelError("no simulink/*.xml found; save the model as .slx, older .mdl files are not supported")
        model_name = Path(str(path)).stem
        system = None
        for n in names:
            root = ET.fromstring(z.read(n))
            if n.endswith("blockdiagram.xml"):
                for p in root.iter("P"):
                    if p.get("Name") == "Name" and p.text:
                        model_name = p.text.strip()
                        break
            candidates = [root] if root.tag == "System" else list(root.iter("System"))
            for cand in candidates:
                if cand.findall("Block") and (system is None or n.endswith("system_root.xml")):
                    system = cand
        if system is None:
            raise ModelError("no block diagram found in the file")
    blocks = {}
    for b in system.findall("Block"):
        sid = b.get("SID")
        blocks[sid] = Block(sid, b.get("Name", sid), b.get("BlockType", "?"), _params(b))
    wires = {}
    for line in system.findall("Line"):
        src = None
        dsts = []
        for p in line.iter("P"):
            if p.get("Name") == "Src":
                src = (p.text or "").strip()
            elif p.get("Name") == "Dst":
                dsts.append((p.text or "").strip())
        sm = PORT.match(src or "")
        if not sm or sm.group(2) != "out":
            continue
        for d in dsts:
            dm = PORT.match(d)
            if dm and dm.group(2) == "in":
                wires[(dm.group(1), int(dm.group(3)))] = (sm.group(1), int(sm.group(3)))
    return Model(model_name, blocks, wires)


def block_table(model):
    rows = []
    for b in model.blocks.values():
        rows.append({
            "sid": b.sid,
            "name": b.name,
            "type": b.kind,
            "supported": b.kind in SUPPORTED,
        })
    return rows


def check_supported(model):
    bad = [f"{b.name} ({b.kind})" for b in model.blocks.values() if b.kind not in SUPPORTED]
    if bad:
        raise ModelError(
            "unsupported blocks: " + ", ".join(bad) + ". Supported types: " + ", ".join(sorted(SUPPORTED))
        )


def _block_params(b, variables):
    def g(key, default):
        return evaluate(b.params.get(key), variables, default)

    k = b.kind
    if k == "Constant":
        return {"value": g("Value", 1.0)}
    if k == "Sin":
        return {"amp": g("Amplitude", 1.0), "bias": g("Bias", 0.0), "freq": g("Frequency", 1.0), "phase": g("Phase", 0.0)}
    if k == "Step":
        return {"time": g("Time", 1.0), "before": g("Before", 0.0), "after": g("After", 1.0)}
    if k == "Gain":
        return {"gain": g("Gain", 1.0)}
    if k == "Saturate":
        return {"upper": g("UpperLimit", 0.5), "lower": g("LowerLimit", -0.5)}
    if k == "Integrator":
        return {"ic": g("InitialCondition", 0.0)}
    if k == "Sum":
        text = b.params.get("Inputs", "++")
        signs = [1.0 if c == "+" else -1.0 for c in text if c in "+-"]
        if not signs:
            signs = [1.0] * (int(text) if text.strip().isdigit() else 2)
        return {"signs": signs}
    if k == "Product":
        text = b.params.get("Inputs", "**")
        ops = [c for c in text if c in "*/"]
        if not ops:
            ops = ["*"] * (int(text) if text.strip().isdigit() else 2)
        return {"ops": ops}
    return {}


def simulate(model, t_end=10.0, dt=0.002, variables=None):
    variables = dict(variables or {})
    check_supported(model)
    blocks = model.blocks
    params = {sid: _block_params(b, variables) for sid, b in blocks.items()}
    integrators = [sid for sid, b in blocks.items() if b.kind == "Integrator"]
    index = {sid: i for i, sid in enumerate(integrators)}
    x0 = np.array([params[s]["ic"] for s in integrators], dtype=float)
    sinks = [sid for sid, b in blocks.items() if b.kind in ("Outport", "Scope")]
    if not sinks:
        raise ModelError("the model has no Outport or Scope block to record")

    def inp(sid, port, t, x, cache, stack):
        src = model.wires.get((sid, port))
        if src is None:
            raise ModelError(f"input {port} of block '{blocks[sid].name}' is not connected")
        return out(src[0], t, x, cache, stack)

    def out(sid, t, x, cache, stack):
        if sid in cache:
            return cache[sid]
        b = blocks[sid]
        p = params[sid]
        k = b.kind
        if k == "Integrator":
            v = x[index[sid]]
        else:
            if sid in stack:
                raise ModelError(f"algebraic loop through block '{b.name}'")
            stack.add(sid)
            if k == "Constant":
                v = p["value"]
            elif k == "Sin":
                v = p["amp"] * math.sin(p["freq"] * t + p["phase"]) + p["bias"]
            elif k == "Step":
                v = p["before"] if t < p["time"] else p["after"]
            elif k == "Gain":
                v = p["gain"] * inp(sid, 1, t, x, cache, stack)
            elif k == "Saturate":
                v = min(max(inp(sid, 1, t, x, cache, stack), p["lower"]), p["upper"])
            elif k == "Sum":
                v = sum(s * inp(sid, i + 1, t, x, cache, stack) for i, s in enumerate(p["signs"]))
            elif k == "Product":
                v = 1.0
                for i, op in enumerate(p["ops"]):
                    val = inp(sid, i + 1, t, x, cache, stack)
                    if op == "*":
                        v *= val
                    else:
                        if val == 0:
                            raise ModelError(f"division by zero in block '{b.name}'")
                        v /= val
            else:
                v = inp(sid, 1, t, x, cache, stack)
            stack.discard(sid)
        cache[sid] = v
        return v

    def deriv(t, x):
        cache = {}
        return np.array([inp(sid, 1, t, x, cache, set()) for sid in integrators], dtype=float)

    steps = int(round(t_end / dt))
    times = np.linspace(0.0, steps * dt, steps + 1)
    names = {sid: blocks[sid].name for sid in sinks}
    signals = {names[sid]: np.zeros(steps + 1) for sid in sinks}
    x = x0.copy()
    for n in range(steps + 1):
        t = times[n]
        cache = {}
        for sid in sinks:
            signals[names[sid]][n] = inp(sid, 1, t, x, cache, set())
        if n == steps or len(x) == 0:
            continue
        k1 = deriv(t, x)
        k2 = deriv(t + dt / 2, x + dt / 2 * k1)
        k3 = deriv(t + dt / 2, x + dt / 2 * k2)
        k4 = deriv(t + dt, x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return SimResult(times, signals)


def parse_assignments(text):
    out = {}
    for part in re.split(r"[,\s]+", (text or "").strip()):
        if not part:
            continue
        if "=" not in part:
            raise ModelError(f"expected name=value, got '{part}'")
        key, value = part.split("=", 1)
        try:
            out[key.strip()] = float(value)
        except ValueError as exc:
            raise ModelError(f"value for '{key}' must be a number") from exc
    return out
