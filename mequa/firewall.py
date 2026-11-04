import json
import os
import re
import urllib.request
from dataclasses import dataclass, field

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
GUARD_MODEL = os.environ.get("META_GUARD_MODEL", "muse-spark-1.3")
INTENT_MODEL = os.environ.get("META_PLANNER_MODEL", os.environ.get("META_GUARD_MODEL", "muse-wonder"))
BLOCK_CATEGORIES = {"S6"}

# Ollama model tags that predate the Meta migration. When the firewall is asked
# for one of these (or nothing at all), the Meta equivalent is used instead so
# old call sites and stored configs keep working.
_LEGACY_MODELS = {"llama-guard3:1b", "llama3.2:3b", "llama3.1:latest", "llama3.1:8b"}

GUARD_SYSTEM = (
    "You are the Mequa compliance guard for a numerical research workbench. "
    "Decide whether the user request (and any assistant reply) asks for clinical, "
    "diagnostic, treatment, medication, dosage, surgery, test-interpretation, symptom, "
    "or individual health guidance about a specific person. "
    "Mentions of tumors, cells, or disease as part of a dataset, cohort, mesh, or "
    "simulation are NOT clinical. "
    'Reply with ONLY strict JSON: {"blocked": true|false, "category": "S6"|"none", '
    '"reason": "short reason"}. Set blocked=true exactly when the content asks for '
    "individual health guidance (category S6). Anything else is blocked=false, "
    'category "none". No other text.'
)

RULE_PATTERNS = [
    re.compile(r"\b(treat|treatment|cure|operate on|prescribe|diagnose)\b.*\b(patient|my|his|her|their)\b", re.IGNORECASE),
    re.compile(r"\b(patient|my|his|her|their)\b.*\b(treat|treatment|cure|surgery|prescribe|diagnos\w*|dose|dosage)\b", re.IGNORECASE),
    re.compile(r"\b\d+\s?(mg|mcg|ml)\b", re.IGNORECASE),
    re.compile(r"\bshould i (take|skip|stop|start|get)\b", re.IGNORECASE),
    re.compile(r"\b(do i have|will i (get|develop|die)|what should i do)\b", re.IGNORECASE),
    re.compile(r"\bmy (tumou?r|aorta|aneurysm|heart|mole|biopsy|scan|symptoms?|diagnosis|blood|lab results?)\b", re.IGNORECASE),
]

INTENT_SYSTEM = (
    "You are a compliance classifier for Mequa, a numerical research workbench for cell point clouds, "
    "meshes, cohorts and simulations. Classify the user's message with exactly one label. "
    "clinical_advice: the message asks for a diagnosis, prognosis, risk assessment, treatment, medication or dose, "
    "surgery decision, test interpretation, symptom explanation, or any health guidance about a specific person "
    "(the user, a relative, a patient, or an individual's scan, mesh, genome or lab result). "
    "research_analysis: the message asks for computation, simulation, clustering, statistics, plotting, benchmarking, "
    "data export or software help on datasets, cohorts or synthetic data, and does not ask for guidance about a specific "
    "person's health. Mentioning tumors, cells or disease as part of a dataset or cohort is research_analysis. "
    "Return only JSON such as {\"label\": \"research_analysis\"} or {\"label\": \"clinical_advice\"}."
)

INTENT_SHOTS = [
    ("Cluster the cells in this synthetic mesh into four zones", "research_analysis"),
    ("Should I be worried about the shadow on my father's X-ray?", "clinical_advice"),
    ("Compute the mean strain across the simulated vessel wall for a cohort of 200 samples", "research_analysis"),
    ("What dose of aspirin should I take daily?", "clinical_advice"),
    ("Does the stress map of this individual's artery suggest they need a stent?", "clinical_advice"),
    ("Export per-zone statistics for the public tumor histology dataset", "research_analysis"),
]


@dataclass
class Verdict:
    allowed: bool
    blocked_by: list = field(default_factory=list)
    categories: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    raw: str = ""
    degraded: bool = False


def _chat_url(node):
    """Main-node calls go to OLLAMA_MAIN_URL, guard-node calls (Llama Guard and the intent
    classifier) to OLLAMA_GUARD_URL. Both default to this machine, so a single laptop still works."""
    if node == "guard":
        from mequa import settings

        base = settings.effective_guard_url()
    else:
        base = os.environ.get("OLLAMA_MAIN_URL")
    return (base or "http://localhost:11434").rstrip("/") + "/api/chat"


def _resolve_model(model, node="main"):
    """Map legacy Ollama tags to the configured Meta model; pass Muse names through."""
    from mequa.agents import meta_client

    if not model or model in _LEGACY_MODELS:
        if node == "guard":
            return meta_client.get_guard_model()
        return meta_client.get_agent_model()
    return model


def _ollama_chat(model, messages, timeout=300, fmt=None, num_predict=None, node="main"):
    """Same signature as the old Ollama call (tests monkeypatch this name).

    On the Meta path this forwards to mequa.agents.meta_client and returns the
    reply text; transport failures raise OSError so the callers' fail-closed
    and degraded paths trigger exactly as before.
    """
    from mequa.agents import meta_client

    if meta_client.provider() == "meta":
        out = meta_client.chat(
            messages=messages,
            model=_resolve_model(model, node),
            temperature=0.0,
            max_tokens=num_predict or 256,
            timeout=min(timeout, meta_client.get_timeout(60.0)),
        )
        if "error" in out:
            raise OSError(out["error"])
        return (out["message"].get("content") or "").strip()
    options = {"temperature": 0}
    if num_predict:
        options["num_predict"] = num_predict
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": "30m",
        "options": options,
    }
    if fmt:
        payload["format"] = fmt
    req = urllib.request.Request(
        _chat_url(node),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))["message"]["content"].strip()


def _parse_guard_output(raw):
    """Return (unsafe, categories). Accepts the strict JSON guard format and the
    legacy Llama Guard text format ("safe" / "unsafe\\nS6"). Anything
    unparseable or hallucinated FAILS CLOSED: (True, ["S6"])."""
    text = (raw or "").strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end])
        except ValueError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("blocked"), bool):
            category = str(data.get("category", "none")).upper()
            if data["blocked"]:
                return True, [category if re.fullmatch(r"S\d+", category) else "S6"]
            if category in ("NONE", ""):
                return False, []
            # blocked=false but a category was named: flag it without blocking.
            return False, [category]
        return True, ["S6"]
    # Legacy Llama Guard plain text ("safe", "unsafe\\nS1 S6", ...).
    if not text:
        return True, ["S6"]
    first = text.split()[0].lower()
    categories = re.findall(r"S\d+", text)
    if first == "unsafe":
        return True, categories or ["S6"]
    if first == "safe":
        return False, []
    return True, ["S6"]


def rules_check(text):
    for pattern in RULE_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


def guard_check(text, reply=None):
    messages = [{"role": "system", "content": GUARD_SYSTEM}]
    messages.append({"role": "user", "content": text})
    if reply is not None:
        messages.append({"role": "assistant", "content": reply})
    raw = _ollama_chat(GUARD_MODEL, messages, fmt="json", num_predict=60, node="guard")
    unsafe, categories = _parse_guard_output(raw)
    return unsafe, categories, raw


def intent_check(text):
    messages = [{"role": "system", "content": INTENT_SYSTEM}]
    for shot, label in INTENT_SHOTS:
        messages.append({"role": "user", "content": shot})
        messages.append({"role": "assistant", "content": json.dumps({"label": label})})
    messages.append({"role": "user", "content": text})
    raw = _ollama_chat(INTENT_MODEL, messages, fmt="json", num_predict=30, node="guard")
    try:
        data = json.loads(raw)
    except ValueError:
        data = {}
    label = str(data.get("label", "")).lower() if isinstance(data, dict) else ""
    return label == "clinical_advice", raw


def _guard_blocks(unsafe, categories):
    return unsafe and bool(BLOCK_CATEGORIES.intersection(categories))


def check_request(text, use_intent=True):
    blocked_by = []
    flags = []
    categories = []
    raw = ""
    degraded = False
    if rules_check(text):
        blocked_by.append("rules")
    try:
        unsafe, categories, raw = guard_check(text)
        if _guard_blocks(unsafe, categories):
            blocked_by.append("llama_guard")
        elif unsafe:
            flags = categories
    except OSError:
        degraded = True
    if use_intent and not blocked_by:
        try:
            clinical, _ = intent_check(text)
            if clinical:
                blocked_by.append("llama_intent")
        except OSError:
            degraded = True
    return Verdict(
        allowed=not blocked_by,
        blocked_by=blocked_by,
        categories=categories,
        flags=flags,
        raw=raw,
        degraded=degraded,
    )


def check_reply(text, reply):
    try:
        unsafe, categories, raw = guard_check(text, reply)
    except OSError:
        # fail closed: an unchecked reply is not shown. Guard mode "auto" or "main" avoids this by
        # running the guard models on the main machine when the guard machine is down.
        return Verdict(allowed=False, blocked_by=["guard_unreachable"], degraded=True)
    blocked = _guard_blocks(unsafe, categories)
    return Verdict(
        allowed=not blocked,
        blocked_by=["llama_guard"] if blocked else [],
        categories=categories,
        flags=categories if unsafe and not blocked else [],
        raw=raw,
    )
