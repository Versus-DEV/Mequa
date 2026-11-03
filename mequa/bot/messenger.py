"""Messenger Platform pieces: payload parsing and a small sending client for a Facebook Page.

Standard library only, reusing the signature check, dedupe and multipart helpers from the WhatsApp module.
The page token is never logged and never appears in error messages.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mequa.bot.whatsapp import GRAPH, HttpFn, Incoming, multipart, urllib_http

log = logging.getLogger("mequa.bot")

MAX_MESSENGER_TEXT = 1900
QUICK_REPLIES = (("analyze", "Analyze"), ("sweep", "Sweep"), ("map", "Strain map"))


class MessengerError(RuntimeError):
    pass


def plain(text: str) -> str:
    return re.sub(r"[*`]", "", str(text))


def split_text(text: str, limit: int = MAX_MESSENGER_TEXT) -> List[str]:
    text = str(text)
    out: List[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        out.append(text[:cut].rstrip())
        text = text[cut:].lstrip("\n")
    if text.strip():
        out.append(text)
    return [c for c in out if c.strip()]


def parse_messenger(payload: Dict[str, Any]) -> List[Incoming]:
    found: List[Incoming] = []
    if payload.get("object") != "page":
        return found
    for entry in payload.get("entry", []) or []:
        for ev in entry.get("messaging", []) or []:
            sender = str((ev.get("sender") or {}).get("id", ""))
            if not sender:
                continue
            stamp = str(ev.get("timestamp", ""))
            msg = ev.get("message")
            if msg:
                if msg.get("is_echo"):
                    continue
                mid = str(msg.get("mid") or f"{sender}-{stamp}")
                quick = (msg.get("quick_reply") or {}).get("payload")
                if quick:
                    found.append(Incoming(mid, sender, "text", text=str(quick)))
                elif msg.get("text") is not None:
                    found.append(Incoming(mid, sender, "text", text=str(msg["text"])))
                elif msg.get("attachments"):
                    kinds = {str(a.get("type", "")) for a in msg["attachments"]}
                    found.append(Incoming(mid, sender, "image" if "image" in kinds else "attachment"))
                continue
            postback = ev.get("postback")
            if postback:
                mid = str(postback.get("mid") or f"pb-{sender}-{stamp}")
                found.append(Incoming(mid, sender, "text", text=str(postback.get("payload", ""))))
    return found


class MessengerClient:
    def __init__(self, page_token: str, version: str = "v23.0", http: Optional[HttpFn] = None):
        self.token = page_token
        self.version = version
        self.http = http or urllib_http

    def _post(self, body: bytes, content_type: str) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": content_type}
        status, raw = self.http("POST", f"{GRAPH}/{self.version}/me/messages", headers, body)
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except ValueError:
            data = {}
        if status >= 300:
            err = data.get("error", {}) if isinstance(data, dict) else {}
            raise MessengerError(
                f"Messenger API returned {status}: code {err.get('code')} subcode {err.get('error_subcode')} "
                f"{str(err.get('message', ''))[:200]}"
            )
        log.info("Messenger API call ok")
        return data if isinstance(data, dict) else {}

    def _json(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._post(json.dumps(body).encode("utf-8"), "application/json")

    def send_text(self, to: str, text: str) -> None:
        for chunk in split_text(plain(text)):
            self._json({"recipient": {"id": to}, "messaging_type": "RESPONSE", "message": {"text": chunk}})

    def send_quick_replies(self, to: str, text: str, replies: Sequence[Tuple[str, str]] = QUICK_REPLIES) -> None:
        self._json({
            "recipient": {"id": to},
            "messaging_type": "RESPONSE",
            "message": {
                "text": plain(text)[:MAX_MESSENGER_TEXT],
                "quick_replies": [{"content_type": "text", "title": t[:20], "payload": p} for p, t in replies[:13]],
            },
        })

    def typing(self, to: str) -> None:
        self._json({"recipient": {"id": to}, "sender_action": "typing_on"})

    def send_image(self, to: str, png: bytes, caption: str = "") -> None:
        fields = {
            "recipient": json.dumps({"id": to}),
            "messaging_type": "RESPONSE",
            "message": json.dumps({"attachment": {"type": "image", "payload": {"is_reusable": False}}}),
        }
        body, ctype = multipart(fields, "filedata", "mequa.png", "image/png", png)
        self._post(body, ctype)
        if caption:
            self.send_text(to, caption)


class MessengerOut:
    def __init__(self, client: MessengerClient, to: str):
        self.client = client
        self.to = to

    def text(self, s: str) -> None:
        self.client.send_text(self.to, s)

    def menu(self, s: str) -> None:
        s = s.replace("on WhatsApp", "on Messenger")
        try:
            self.client.send_quick_replies(self.to, s)
        except MessengerError:
            self.client.send_text(self.to, s)

    def image(self, png: bytes, caption: str) -> None:
        self.client.send_image(self.to, png, caption)
