"""Backend of the chat dock: Ollama health and one question -> one grounded answer.

No Streamlit here, so it is tested directly. A question goes through the same guarded path as the
Adaptive Agent page: firewall on the request, Llama calling the verified solver as tools, the
number-grounding check, firewall on the reply. Nothing in the answer is computed by the model.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from mequa.workspace import Dataset

HELP = (
    "Ask a research question about the dataset in the workspace, for example:\n\n"
    "- `Find the load at which this sample reaches the strain limit.`\n"
    "- `Which density zone carries the most strain at 5 percent load?`\n\n"
    "Every number in an answer comes from the solver. Requests for clinical guidance about an individual are blocked. "
    "Commands: `/help`, `/status`."
)


def ollama_status(url: Optional[str] = None, timeout: float = 0.7) -> Dict[str, Any]:
    """Is Ollama answering at `url`? Never raises."""
    base = (url or os.getenv("OLLAMA_MAIN_URL") or "http://127.0.0.1:11434").rstrip("/")
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        models = sorted(m.get("name", "") for m in data.get("models", []) if isinstance(m, dict))
        return {"up": True, "url": base, "models": models}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return {"up": False, "url": base, "models": [], "error": f"{type(exc).__name__}"}


def status_text(dataset: Optional[Dataset], main: Dict[str, Any], model: str) -> str:
    lines = [f"- Ollama: {'online, ' + str(len(main['models'])) + ' models' if main['up'] else 'OFFLINE at ' + main['url']}",
             f"- Model: `{model}`" + ("" if not main["up"] or model in main["models"] else "  (not pulled: `ollama pull " + model + "`)")]
    lines.append(f"- Dataset: {dataset.n} cells, {dataset.dim}D, {dataset.label}" if dataset else "- Dataset: none loaded")
    return "\n".join(lines)


def _guard_hint(blocked_by) -> str:
    """When the block is only 'the guard could not be reached', say so and name the fix, rather than a bare compliance halt."""
    if "guard_unreachable" not in (blocked_by or []):
        return ""
    from mequa import settings
    g = settings.guard_status()
    return ("**Safety check unavailable.** The guard model could not be reached, and Mequa refuses to answer unchecked. "
            f"{g['note']} Open the ⚙ menu in this chat and choose **Auto** or **Main machine does everything**, then ask again.")


def answer(question: str, dataset: Optional[Dataset], model: str = "llama3.2:3b", strain_limit: float = 0.05,
           investigator_cls=None, main_status: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return {"status": ..., "text": markdown, "trace": [...]}. Statuses: ok, blocked, offline,
    no_dataset, reply_blocked, error, help."""
    q = (question or "").strip()
    if not q:
        return {"status": "error", "text": "Type a question first.", "trace": []}
    if q.lower() in ("/help", "help", "?"):
        return {"status": "help", "text": HELP, "trace": []}
    main = main_status or ollama_status()
    if q.lower() == "/status":
        return {"status": "help", "text": status_text(dataset, main, model), "trace": []}
    if not main["up"]:
        return {"status": "offline", "trace": [],
                "text": f"Ollama is not running at `{main['url']}`, so I cannot plan or write an answer. "
                        "Start it from the launcher (**Start Ollama**) or run `ollama serve`, then ask again."}
    if dataset is None:
        return {"status": "no_dataset", "trace": [],
                "text": "No dataset is loaded. Open **Workbench**, **Cell Viewer** or **Model Editor**, load or build cells, then ask again."}
    if investigator_cls is None:
        from mequa.agents.adaptive_investigator import AdaptiveInvestigator as investigator_cls
    from mequa.session import Session

    try:
        sess = Session(dataset.points, stiffness=dataset.stiffness, knn=8)
        out = investigator_cls(sess, model=model, max_iterations=6, strain_limit=strain_limit).run(q)
    except Exception as exc:  # the dock must never take the page down
        return {"status": "error", "trace": [], "text": f"The investigation failed: `{type(exc).__name__}: {exc}`"}
    status = out.get("status")
    trace: List[Dict[str, Any]] = out.get("trace", [])
    gstat = _guard_hint(out.get("blocked_by", []))
    if gstat:
        return {"status": status or "blocked", "trace": trace, "text": gstat}
    if status == "blocked":
        return {"status": "blocked", "trace": trace,
                "text": "**COMPLIANCE HALT.** Mequa does not give clinical guidance for any individual "
                        f"(blocked by: {', '.join(out.get('blocked_by', [])) or 'firewall'}). "
                        "Ask a research question about a dataset instead."}
    if status == "reply_blocked":
        return {"status": "reply_blocked", "trace": trace, "text": "**COMPLIANCE HALT.** The generated reply was blocked by the output check."}
    if status == "model_unavailable":
        return {"status": "offline", "trace": trace,
                "text": f"The model `{model}` did not answer: `{out.get('error', 'unknown error')}`. Is it pulled? `ollama pull {model}`"}
    if status != "ok":
        return {"status": "error", "trace": trace, "text": f"Unexpected result: `{status}`"}
    foot = f"\n\n*{out.get('model', model)} · {out.get('iterations', '?')} rounds · numbers verified against solver output*"
    if out.get("grounding", {}).get("fallback"):
        foot = "\n\n*The model kept citing numbers that were not in the solver output, so this summary was built from solver results only.*"
    return {"status": "ok", "trace": trace, "text": str(out.get("summary", "")).strip() + foot}
