"""WhatsApp Cloud API pieces: signature check, payload parsing, dedupe and a small sending client.

Standard library only. Tokens are never logged and never appear in error messages.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import urllib.error
import urllib.request
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

GRAPH = "https://graph.facebook.com"
MAX_WA_TEXT = 4000

MENU_BUTTONS = (("analyze", "Analyze"), ("sweep", "Sweep"), ("map", "Strain map"))


class WhatsAppError(RuntimeError):
    pass


def normalize_number(value: str) -> str:
    return "".join(ch for ch in str(value) if ch.isdigit())


def parse_allowed(raw: str) -> set:
    return {normalize_number(p) for p in str(raw or "").split(",") if normalize_number(p)}


def verify_signature(secret: str, body: bytes, header: Optional[str]) -> bool:
    if not secret or not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header[len("sha256="):])


@dataclass(frozen=True)
class Incoming:
    id: str
    sender: str
    kind: str
    text: str = ""
    media_id: str = ""


def parse_incoming(payload: Dict[str, Any]) -> List[Incoming]:
    found: List[Incoming] = []
    for entry in payload.get("entry", []) or []:
        for change in entry.get("changes", []) or []:
            value = change.get("value", {}) or {}
            for m in value.get("messages", []) or []:
                mid, sender, kind = str(m.get("id", "")), normalize_number(m.get("from", "")), str(m.get("type", ""))
                if not mid or not sender:
                    continue
                if kind == "text":
                    found.append(Incoming(mid, sender, "text", text=str((m.get("text") or {}).get("body", ""))))
                elif kind == "interactive":
                    inter = m.get("interactive") or {}
                    reply = inter.get("button_reply") or inter.get("list_reply") or {}
                    found.append(Incoming(mid, sender, "text", text=str(reply.get("id", ""))))
                elif kind == "image":
                    found.append(Incoming(mid, sender, "image", media_id=str((m.get("image") or {}).get("id", ""))))
                else:
                    found.append(Incoming(mid, sender, kind or "unknown"))
    return found


class Deduper:
    def __init__(self, capacity: int = 2000):
        self.capacity = capacity
        self.items: "OrderedDict[str, None]" = OrderedDict()

    def seen(self, key: str) -> bool:
        if key in self.items:
            self.items.move_to_end(key)
            return True
        self.items[key] = None
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)
        return False


HttpFn = Callable[[str, str, Dict[str, str], Optional[bytes]], Tuple[int, bytes]]


def urllib_http(method: str, url: str, headers: Dict[str, str], body: Optional[bytes]) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def multipart(fields: Dict[str, str], file_field: str, filename: str, content_type: str, data: bytes) -> Tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts: List[bytes] = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n".encode() + data + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class WhatsAppClient:
    def __init__(self, token: str, phone_id: str, version: str = "v23.0", http: Optional[HttpFn] = None):
        self.token = token
        self.phone_id = phone_id
        self.version = version
        self.http = http or urllib_http

    def _url(self, path: str) -> str:
        return f"{GRAPH}/{self.version}/{self.phone_id}/{path}"

    def _call(self, path: str, body: bytes, content_type: str) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": content_type}
        status, raw = self.http("POST", self._url(path), headers, body)
        if status >= 300:
            raise WhatsAppError(f"WhatsApp API returned {status}: {raw[:200].decode('utf-8', 'replace')}")
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except ValueError:
            return {}

    def _message(self, to: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = {"messaging_product": "whatsapp", "to": to, **payload}
        return self._call("messages", json.dumps(body).encode("utf-8"), "application/json")

    def send_text(self, to: str, text: str) -> None:
        self._message(to, {"type": "text", "text": {"body": str(text)[:MAX_WA_TEXT], "preview_url": False}})

    def send_buttons(self, to: str, body: str, buttons: Sequence[Tuple[str, str]] = MENU_BUTTONS) -> None:
        self._message(to, {
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": str(body)[:1000]},
                "action": {"buttons": [{"type": "reply", "reply": {"id": i, "title": t[:20]}} for i, t in buttons[:3]]},
            },
        })

    def send_image(self, to: str, png: bytes, caption: str = "") -> None:
        body, ctype = multipart({"messaging_product": "whatsapp", "type": "image/png"}, "file", "mequa.png", "image/png", png)
        media = self._call("media", body, ctype)
        media_id = media.get("id")
        if not media_id:
            raise WhatsAppError("media upload returned no id")
        self._message(to, {"type": "image", "image": {"id": media_id, "caption": str(caption)[:1000]}})


class WhatsAppOut:
    def __init__(self, client: WhatsAppClient, to: str):
        self.client = client
        self.to = to

    def text(self, s: str) -> None:
        self.client.send_text(self.to, s)

    def menu(self, s: str) -> None:
        try:
            self.client.send_buttons(self.to, s[:1000])
            if len(s) > 1000:
                self.client.send_text(self.to, s)
        except WhatsAppError:
            self.client.send_text(self.to, s)

    def image(self, png: bytes, caption: str) -> None:
        self.client.send_image(self.to, png, caption)
