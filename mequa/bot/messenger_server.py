"""Webhook server for the Mequa Messenger bot: python -m mequa.bot.messenger_server

Same safety model as the WhatsApp server: 127.0.0.1 only, valid X-Hub-Signature-256 on every POST, an allow-list
(of Messenger sender IDs), and duplicate deliveries dropped. With MESSENGER_ALLOWED empty nobody is served, and the
log prints each sender ID so you can copy yours into .env.
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
from urllib.parse import parse_qs, urlparse

from mequa.bot.messenger import MessengerClient, MessengerError, MessengerOut, parse_messenger
from mequa.bot.whatsapp import Deduper, Incoming, parse_allowed, verify_signature

log = logging.getLogger("mequa.bot")

MAX_BODY = 1_000_000


def _pick(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


@dataclass
class Config:
    page_token: str
    verify_token: str
    app_secret: str
    allowed: set
    version: str = "v23.0"
    port: int = 8080

    @classmethod
    def from_env(cls) -> "Config":
        token = _pick("MESSENGER_PAGE_TOKEN")
        verify = _pick("MESSENGER_VERIFY_TOKEN", "WHATSAPP_VERIFY_TOKEN")
        secret = _pick("MESSENGER_APP_SECRET", "WHATSAPP_APP_SECRET")
        missing = []
        if not token:
            missing.append("MESSENGER_PAGE_TOKEN")
        if not verify:
            missing.append("MESSENGER_VERIFY_TOKEN (or WHATSAPP_VERIFY_TOKEN)")
        if not secret:
            missing.append("MESSENGER_APP_SECRET (or WHATSAPP_APP_SECRET)")
        if missing:
            raise SystemExit("Missing settings in .env or the environment: " + ", ".join(missing))
        return cls(
            page_token=token,
            verify_token=verify,
            app_secret=secret,
            allowed=parse_allowed(os.getenv("MESSENGER_ALLOWED", "")),
            version=_pick("MESSENGER_API_VERSION", "WHATSAPP_API_VERSION") or "v23.0",
            port=int(os.getenv("MEQUA_BOT_PORT", "8080")),
        )


class Dispatcher:
    def __init__(self, cfg: Config, core, client: MessengerClient, workers: int = 2):
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
            log.warning("ignored a message from sender ID %s. To allow it, add this ID to MESSENGER_ALLOWED in .env and restart.", msg.sender)
            return
        out = MessengerOut(self.client, msg.sender)
        try:
            if msg.kind == "text":
                log.info("request from ...%s", msg.sender[-4:])
                try:
                    self.client.typing(msg.sender)
                except MessengerError:
                    pass
                self.core.handle(msg.text, out)
                log.info("finished request from ...%s", msg.sender[-4:])
            elif msg.kind == "image":
                out.text("Photo analysis is not enabled yet. Send a text command; type help for the menu.")
            else:
                out.text("I can read text messages and the menu buttons. Type help for the menu.")
        except MessengerError as exc:
            log.error("Messenger send failed: %s", exc)
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
            log.warning("webhook verification rejected: wrong mode or verify token")
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
                log.warning("webhook POST rejected: bad or missing signature, check the app secret")
                return self._reply(403, "bad signature")
            try:
                payload = json.loads(body.decode("utf-8"))
            except ValueError:
                return self._reply(400, "bad json")
            messages = parse_messenger(payload)
            log.info("webhook POST accepted: %d message(s)", len(messages))
            for msg in messages:
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
    client = MessengerClient(cfg.page_token, cfg.version)
    server = ThreadingHTTPServer(("127.0.0.1", cfg.port), make_handler(cfg, Dispatcher(cfg, core, client)))
    log.info("Dataset: %s | model: %s | allow-listed senders: %d", core.dataset.label, core.model, len(cfg.allowed))
    if not cfg.allowed:
        log.warning("MESSENGER_ALLOWED is empty, so nobody is served. Message the Page, copy your sender ID from the log, put it in .env, restart.")
    log.info("Listening on http://127.0.0.1:%d/webhook . Start your tunnel, for example: ngrok http %d", cfg.port, cfg.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("stopped")
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
