"""Webhook server for the Mequa WhatsApp bot: python -m mequa.bot.whatsapp_server

Listens on 127.0.0.1 only; expose it with a tunnel (ngrok or cloudflared). Every POST must carry a valid
X-Hub-Signature-256, only allow-listed numbers are served, and duplicate deliveries are dropped.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

from mequa.bot.whatsapp import Deduper, Incoming, WhatsAppClient, WhatsAppError, WhatsAppOut, parse_allowed, parse_incoming, verify_signature

log = logging.getLogger("mequa.bot")

MAX_BODY = 1_000_000
REQUIRED = ("WHATSAPP_TOKEN", "WHATSAPP_PHONE_ID", "WHATSAPP_VERIFY_TOKEN", "WHATSAPP_APP_SECRET", "WHATSAPP_ALLOWED")


@dataclass
class Config:
    token: str
    phone_id: str
    verify_token: str
    app_secret: str
    allowed: set
    version: str = "v23.0"
    port: int = 8080

    @classmethod
    def from_env(cls) -> "Config":
        missing = [k for k in REQUIRED if not os.getenv(k, "").strip()]
        if missing:
            raise SystemExit("Missing settings in .env or the environment: " + ", ".join(missing))
        allowed = parse_allowed(os.environ["WHATSAPP_ALLOWED"])
        if not allowed:
            raise SystemExit("WHATSAPP_ALLOWED has no usable phone number (digits with country code, comma separated)")
        return cls(
            token=os.environ["WHATSAPP_TOKEN"].strip(),
            phone_id=os.environ["WHATSAPP_PHONE_ID"].strip(),
            verify_token=os.environ["WHATSAPP_VERIFY_TOKEN"].strip(),
            app_secret=os.environ["WHATSAPP_APP_SECRET"].strip(),
            allowed=allowed,
            version=os.getenv("WHATSAPP_API_VERSION", "v23.0").strip() or "v23.0",
            port=int(os.getenv("MEQUA_BOT_PORT", "8080")),
        )


class Dispatcher:
    def __init__(self, cfg: Config, core, client: WhatsAppClient, workers: int = 2):
        self.cfg = cfg
        self.core = core
        self.client = client
        self.dedupe = Deduper()
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def submit(self, msg: Incoming):
        if self.dedupe.seen(msg.id):
            return None
        return self.pool.submit(self.process, msg)

    def process(self, msg: Incoming) -> None:
        if msg.sender not in self.cfg.allowed:
            log.info("ignored a message from a number that is not allow-listed (...%s)", msg.sender[-4:])
            return
        out = WhatsAppOut(self.client, msg.sender)
        try:
            if msg.kind == "text":
                log.info("request from ...%s", msg.sender[-4:])
                self.core.handle(msg.text, out)
            elif msg.kind == "image":
                out.text("Photo analysis is not enabled yet. Send a text command; type help for the menu.")
            else:
                out.text("I can read text messages and the menu buttons. Type help for the menu.")
        except WhatsAppError as exc:
            log.error("WhatsApp send failed: %s", exc)
        except Exception:
            log.exception("unexpected error while handling a message")


def make_handler(cfg: Config, dispatcher: Dispatcher):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _reply(self, code: int, body: str = "") -> None:
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/health":
                return self._reply(200, "ok")
            if url.path != "/webhook":
                return self._reply(404, "not found")
            q = parse_qs(url.query)
            mode = (q.get("hub.mode") or [""])[0]
            token = (q.get("hub.verify_token") or [""])[0]
            challenge = (q.get("hub.challenge") or [""])[0]
            if mode == "subscribe" and hmac.compare_digest(token, cfg.verify_token):
                return self._reply(200, challenge)
            return self._reply(403, "forbidden")

        def do_POST(self):
            if urlparse(self.path).path != "/webhook":
                return self._reply(404, "not found")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._reply(400, "bad length")
            if length <= 0 or length > MAX_BODY:
                return self._reply(413, "bad size")
            body = self.rfile.read(length)
            if not verify_signature(cfg.app_secret, body, self.headers.get("X-Hub-Signature-256")):
                return self._reply(403, "bad signature")
            try:
                payload = json.loads(body.decode("utf-8"))
            except ValueError:
                return self._reply(400, "bad json")
            for msg in parse_incoming(payload):
                dispatcher.submit(msg)
            return self._reply(200, "ok")

    return Handler


def main() -> None:
    from mequa.bot.core import BotCore, load_dataset
    from mequa.utils.env import load_env_file

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_env_file()
    cfg = Config.from_env()
    core = BotCore(load_dataset())
    client = WhatsAppClient(cfg.token, cfg.phone_id, cfg.version)
    server = ThreadingHTTPServer(("127.0.0.1", cfg.port), make_handler(cfg, Dispatcher(cfg, core, client)))
    log.info("Dataset: %s | model: %s | allow-listed numbers: %d", core.dataset.label, core.model, len(cfg.allowed))
    log.info("Listening on http://127.0.0.1:%d/webhook . Start your tunnel, for example: ngrok http %d", cfg.port, cfg.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("stopped")
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
