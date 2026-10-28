import json
import re
import urllib.request
from dataclasses import dataclass, field

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
GUARD_MODEL = "llama-guard3:1b"

RULE_PATTERNS = [
    re.compile(r"\b(treat|treatment|cure|operate on|prescribe|diagnose)\b.*\b(patient|my|his|her|their)\b", re.IGNORECASE),
    re.compile(r"\b(patient|my|his|her|their)\b.*\b(treat|treatment|cure|surgery|prescribe|diagnos\w*|dose|dosage)\b", re.IGNORECASE),
    re.compile(r"\b\d+\s?(mg|mcg|ml)\b", re.IGNORECASE),
    re.compile(r"\bshould i (take|skip|stop|start|get)\b", re.IGNORECASE),
]


@dataclass
class Verdict:
    allowed: bool
    blocked_by: list = field(default_factory=list)
    categories: list = field(default_factory=list)
    raw: str = ""
    degraded: bool = False


def _ollama_chat(model, messages, timeout=180):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_CHAT_URL, data=body, headers={"Content-Type": "application/json"}
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


def check_request(text):
    blocked_by = []
    rule_hit = rules_check(text)
    if rule_hit:
        blocked_by.append("rules")
    degraded = False
    categories = []
    raw = ""
    try:
        guard_hit, categories, raw = guard_check(text)
        if guard_hit:
            blocked_by.append("llama_guard")
    except OSError:
        degraded = True
    return Verdict(
        allowed=not blocked_by,
        blocked_by=blocked_by,
        categories=categories,
        raw=raw,
        degraded=degraded,
    )


def check_reply(text, reply):
    try:
        guard_hit, categories, raw = guard_check(text, reply)
    except OSError:
        return Verdict(allowed=False, blocked_by=["guard_unreachable"], degraded=True)
    return Verdict(
        allowed=not guard_hit,
        blocked_by=["llama_guard"] if guard_hit else [],
        categories=categories,
        raw=raw,
    )