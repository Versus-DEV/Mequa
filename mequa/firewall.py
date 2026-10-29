import json
import re
import urllib.request
from dataclasses import dataclass, field

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
GUARD_MODEL = "llama-guard3:1b"
INTENT_MODEL = "llama3.2:3b"
BLOCK_CATEGORIES = {"S6"}

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


def _ollama_chat(model, messages, timeout=300, fmt=None, num_predict=None):
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
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))["message"]["content"].strip()


def rules_check(text):
    for pattern in RULE_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


def guard_check(text, reply=None):
    messages = [{"role": "user", "content": text}]
    if reply is not None:
        messages.append({"role": "assistant", "content": reply})
    raw = _ollama_chat(GUARD_MODEL, messages)
    first = raw.split()[0].lower() if raw else ""
    categories = re.findall(r"S\d+", raw)
    return first == "unsafe", categories, raw


def intent_check(text):
    messages = [{"role": "system", "content": INTENT_SYSTEM}]
    for shot, label in INTENT_SHOTS:
        messages.append({"role": "user", "content": shot})
        messages.append({"role": "assistant", "content": json.dumps({"label": label})})
    messages.append({"role": "user", "content": text})
    raw = _ollama_chat(INTENT_MODEL, messages, fmt="json", num_predict=30)
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
        return Verdict(allowed=False, blocked_by=["guard_unreachable"], degraded=True)
    blocked = _guard_blocks(unsafe, categories)
    return Verdict(
        allowed=not blocked,
        blocked_by=["llama_guard"] if blocked else [],
        categories=categories,
        flags=categories if unsafe and not blocked else [],
        raw=raw,
    )
