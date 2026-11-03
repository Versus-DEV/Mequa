"""Channel-independent Mequa bot.

One incoming text goes in, replies go out through an `out` object with three methods:
text(str), image(png_bytes, caption) and menu(body). WhatsApp and the terminal console both
implement it, so a Discord adapter would only need those three methods too.

Commands run the verified solver directly. Free text goes to the Llama investigator, which checks
the request and the reply with the firewall and verifies every number against solver output.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from mequa import chat as chat_mod
from mequa.workspace import Dataset, make_dataset

log = logging.getLogger("mequa.bot")

MAX_QUESTION_CHARS = 500
MAX_TEXT_CHARS = 3800

HELP = (
    "*Mequa* on WhatsApp\n"
    "Local Llama and the verified solver. Every number comes from the solver.\n\n"
    "*analyze [load%]*  solve at a load (default 5)\n"
    "*sweep [limit%]*  find the load where p95 strain reaches the limit (default 5)\n"
    "*map [load%]*  strain map image\n"
    "*status*  models and dataset\n"
    "Or ask a research question in plain words.\n\n"
    "Research datasets only. Requests for clinical guidance about an individual are blocked."
)


def load_dataset(spec: Optional[str] = None) -> Dataset:
    spec = (spec or os.getenv("MEQUA_BOT_DATASET") or "synthetic").strip()
    if spec.lower() == "synthetic":
        from mequa import numpy_backend as nb

        return make_dataset(nb.synthetic_cells(2000, seed=1, clusters=3), label="synthetic, 2000 cells, seed 1", source="bot")
    path = Path(spec)
    if not path.is_file():
        raise FileNotFoundError(f"MEQUA_BOT_DATASET points to a missing file: {spec}")
    if path.suffix.lower() == ".npy":
        pts = np.load(path)
    elif path.suffix.lower() == ".csv":
        try:
            pts = np.loadtxt(path, delimiter=",", ndmin=2)
        except ValueError:
            pts = np.loadtxt(path, delimiter=",", ndmin=2, skiprows=1)
    else:
        raise ValueError("MEQUA_BOT_DATASET must be 'synthetic', a .npy file or a .csv file of cell positions")
    return make_dataset(pts, label=path.name, source="bot")


def to_whatsapp(text: str) -> str:
    out = str(text).replace("**", "*")
    if len(out) > MAX_TEXT_CHARS:
        out = out[:MAX_TEXT_CHARS].rstrip() + "\n(shortened)"
    return out


def parse_percent(arg: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(arg.strip().rstrip("%")) / 100.0
    except ValueError:
        return default
    return min(hi, max(lo, value))


class Batcher:
    """Coalesces progress messages so a fast run does not send a message per step."""

    def __init__(self, send: Callable[[str], None], min_interval: float = 4.0, clock: Callable[[], float] = time.monotonic):
        self.send = send
        self.min_interval = min_interval
        self.clock = clock
        self.pending: List[str] = []
        self.last: Optional[float] = None

    def push(self, text: str) -> None:
        self.pending.append(text)
        now = self.clock()
        if self.last is None or now - self.last >= self.min_interval:
            self._emit(now)

    def flush(self) -> None:
        if self.pending:
            self._emit(self.clock())

    def _emit(self, now: float) -> None:
        text = "\n".join(self.pending)
        self.pending = []
        self.last = now
        self.send(text)


def _before(name: str, args: Dict[str, Any]) -> str:
    if name == "run_solver":
        try:
            return f"Running the solver at {float(args.get('load_fraction', 0.05)) * 100:.1f}% load..."
        except (TypeError, ValueError):
            return "Running the solver..."
    if name == "sweep_load":
        return "Sweeping the load..."
    return f"Running {name}..."


def _after(name: str, result: Dict[str, Any]) -> str:
    if "error" in result:
        return "A solver step failed; the agent will adapt."
    if name == "run_solver" and "p95_strain" in result:
        return f"Solver done: p95 strain {result['p95_strain']}"
    if name == "sweep_load" and "solves" in result:
        thr = result.get("threshold_load")
        return f"Sweep done ({result['solves']} solves), threshold load: {'not reached' if thr is None else thr}"
    return f"{name} done."


def progress_class(emit: Callable[[str], None], base: Optional[type] = None) -> type:
    if base is None:
        from mequa.agents.adaptive_investigator import AdaptiveInvestigator as base

    class ProgressInvestigator(base):
        def execute(self, name, args):
            emit(_before(name, args))
            result = super().execute(name, args)
            emit(_after(name, result))
            return result

    return ProgressInvestigator


class BotCore:
    def __init__(self, dataset: Dataset, model: Optional[str] = None, strain_limit: Optional[float] = None,
                 answer_fn: Optional[Callable[..., Dict[str, Any]]] = None, investigator_base: Optional[type] = None,
                 min_interval: float = 4.0, clock: Callable[[], float] = time.monotonic):
        self.dataset = dataset
        self.model = model or os.getenv("PLANNER_MODEL", "llama3.2:3b")
        self.strain_limit = float(strain_limit or os.getenv("MEQUA_STRAIN_LIMIT", 0.05))
        self.answer_fn = answer_fn or chat_mod.answer
        self.investigator_base = investigator_base
        self.min_interval = min_interval
        self.clock = clock
        self.lock = threading.Lock()

    def handle(self, text: str, out) -> None:
        text = (text or "").strip()
        if not text:
            return
        cmd, _, arg = text.partition(" ")
        key = cmd.lower().lstrip("/")
        if key in ("help", "hi", "hello", "menu", "start"):
            out.menu(HELP)
            return
        if key == "status":
            out.text(self._status())
            return
        if not self.lock.acquire(blocking=False):
            out.text("Still working on your previous request. Please wait for it to finish.")
            return
        try:
            if key == "analyze":
                self._analyze(arg, out)
            elif key == "sweep":
                self._sweep(arg, out)
            elif key == "map":
                self._map(arg, out)
            else:
                self._ask(text[:MAX_QUESTION_CHARS], out)
        except Exception as exc:
            log.exception("request failed")
            out.text(f"Something went wrong ({type(exc).__name__}). Check the Mequa console for details.")
        finally:
            self.lock.release()

    def _session(self):
        from mequa.session import Session

        ds = self.dataset
        return Session(ds.points, stiffness=ds.stiffness, knn=8)

    def _status(self) -> str:
        main = chat_mod.ollama_status()
        return chat_mod.status_text(self.dataset, main, self.model)

    def _analyze(self, arg: str, out) -> None:
        load = parse_percent(arg, 0.05, 0.005, 0.2)
        out.text(f"Solving at {load * 100:.1f}% load...")
        r = self._session().run(load)
        zones = "\n".join(
            f"Zone {z['zone']}: {z['cells']} cells, mean strain {z['mean_strain']}, max {z['max_strain']}"
            for z in r.get("per_zone", [])
        )
        connected = "yes" if r.get("load_path") else "no (treat strain values with care)"
        out.text(
            f"*Analysis at {r['load_fraction'] * 100:.1f}% load*\n"
            f"Cells: {r['cells']} ({r['dim']}D), links: {r['mechanical_links']}\n"
            f"p95 strain: {r['p95_strain']}, max strain: {r['max_strain']}\n"
            f"SPHI: {r['sphi']}, critical strain: {r['s_crit']}\n"
            f"Load path connected: {connected}\n{zones}"
        )

    def _sweep(self, arg: str, out) -> None:
        from mequa import parallel

        limit = parse_percent(arg, self.strain_limit, 0.005, 0.5)
        loads = np.linspace(0.01, 0.12, 8).tolist()
        out.text(f"Sweeping 8 loads from 1.0% to 12.0% (p95 strain limit {limit}).")
        sess = self._session()
        res = parallel.sweep_loads(sess.points, loads, k=sess.k, knn=sess.knn, stiffness=sess.stiffness, comm=None)
        rows = [{k: round(v, 5) for k, v in r.items()} for r in res["rows"]]
        thr = parallel.find_threshold(rows, limit)
        table = "\n".join(f"{r['load_fraction'] * 100:.1f}% load: p95 strain {r['p95_strain']}" for r in rows)
        verdict = (
            f"p95 strain reaches the limit of {limit} at about {thr * 100:.2f}% load."
            if thr is not None else f"p95 strain does not reach the limit of {limit} within 1% to 12% load."
        )
        out.text(f"*Sweep result*\n{table}\n\n{verdict}\nThe limit is a modelling choice, not a measured tissue property.")

    def _map(self, arg: str, out) -> None:
        from mequa.bot import render

        if self.dataset.dim != 2:
            out.text("The strain map is 2D only for now, and the loaded dataset is 3D.")
            return
        load = parse_percent(arg, 0.05, 0.005, 0.2)
        out.text(f"Drawing the strain map at {load * 100:.1f}% load...")
        summary, res = self._session().run(load, full=True)
        png = render.strain_map(self.dataset.points, res["node_strain"], title=f"Strain at {load * 100:.1f}% load")
        out.image(png, f"Strain map, {self.dataset.n} cells, p95 strain {summary['p95_strain']}")

    def _ask(self, question: str, out) -> None:
        batcher = Batcher(out.text, self.min_interval, self.clock)
        batcher.push(f"Thinking with local {self.model}...")
        cls = progress_class(batcher.push, self.investigator_base)
        res = self.answer_fn(question, self.dataset, model=self.model, strain_limit=self.strain_limit, investigator_cls=cls)
        batcher.flush()
        out.text(to_whatsapp(res.get("text", "")))
