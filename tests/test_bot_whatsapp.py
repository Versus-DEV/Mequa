import hashlib
import hmac
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from mequa.bot import whatsapp_server as srv
from mequa.bot.whatsapp import (Deduper, Incoming, WhatsAppClient, WhatsAppError, WhatsAppOut, normalize_number,
                                parse_allowed, parse_incoming, verify_signature)

SECRET = "app-secret"


def sign(body: bytes, secret=SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def payload(*messages):
    return {"entry": [{"changes": [{"value": {"messages": list(messages)}}]}]}


TEXT = {"id": "wamid.1", "from": "919876543210", "type": "text", "text": {"body": "analyze 5"}}
BUTTON = {"id": "wamid.2", "from": "919876543210", "type": "interactive",
          "interactive": {"button_reply": {"id": "sweep", "title": "Sweep"}}}
IMAGE = {"id": "wamid.3", "from": "919876543210", "type": "image", "image": {"id": "media9"}}


def test_signature_valid_and_invalid():
    body = b'{"a":1}'
    assert verify_signature(SECRET, body, sign(body))
    assert not verify_signature(SECRET, body, sign(body, "other"))
    assert not verify_signature(SECRET, body, None)
    assert not verify_signature("", body, sign(body))
    assert not verify_signature(SECRET, body, "sha1=abc")


def test_parse_incoming_kinds():
    msgs = parse_incoming(payload(TEXT, BUTTON, IMAGE, {"id": "w4", "from": "1", "type": "audio"}))
    assert [m.kind for m in msgs] == ["text", "text", "image", "audio"]
    assert msgs[0].text == "analyze 5" and msgs[1].text == "sweep" and msgs[2].media_id == "media9"
    assert parse_incoming({"entry": [{"changes": [{"value": {"statuses": [{}]}}]}]}) == []
    assert parse_incoming({}) == []


def test_allowed_parsing_and_normalising():
    assert normalize_number("+91 98765-43210") == "919876543210"
    assert parse_allowed("+91 98765 43210, 1 555 0100, ,") == {"919876543210", "15550100"}
    assert parse_allowed("") == set()


def test_deduper_drops_repeats_and_bounds_memory():
    d = Deduper(capacity=3)
    assert not d.seen("a") and d.seen("a")
    for k in "bcd":
        d.seen(k)
    assert not d.seen("a")


class FakeHttp:
    def __init__(self, status=200, body=b'{"id": "m1"}'):
        self.calls, self.status, self.body = [], status, body

    def __call__(self, method, url, headers, data):
        self.calls.append((method, url, headers, data))
        return self.status, self.body


def test_client_text_payload_and_auth_header():
    http = FakeHttp()
    c = WhatsAppClient("TOKEN123", "555", "v23.0", http)
    c.send_text("919876543210", "hello")
    method, url, headers, data = http.calls[0]
    assert url == "https://graph.facebook.com/v23.0/555/messages"
    assert headers["Authorization"] == "Bearer TOKEN123"
    body = json.loads(data)
    assert body["messaging_product"] == "whatsapp" and body["to"] == "919876543210"
    assert body["text"]["body"] == "hello"


def test_client_buttons_limits():
    http = FakeHttp()
    WhatsAppClient("t", "1", http=http).send_buttons("9", "pick", [("a", "A" * 40), ("b", "B"), ("c", "C"), ("d", "D")])
    buttons = json.loads(http.calls[0][3])["interactive"]["action"]["buttons"]
    assert len(buttons) == 3 and len(buttons[0]["reply"]["title"]) == 20


def test_client_image_uploads_then_sends():
    http = FakeHttp()
    WhatsAppClient("t", "1", http=http).send_image("9", b"\x89PNGdata", "cap")
    assert http.calls[0][1].endswith("/1/media") and b"\x89PNGdata" in http.calls[0][3]
    assert "multipart/form-data" in http.calls[0][2]["Content-Type"]
    sent = json.loads(http.calls[1][3])
    assert sent["image"]["id"] == "m1" and sent["image"]["caption"] == "cap"


def test_client_error_hides_token():
    c = WhatsAppClient("SECRET-TOKEN", "1", http=FakeHttp(status=401, body=b'{"error":"bad"}'))
    with pytest.raises(WhatsAppError) as exc:
        c.send_text("9", "x")
    assert "SECRET-TOKEN" not in str(exc.value) and "401" in str(exc.value)


class RecordingCore:
    def __init__(self):
        self.seen, self.event = [], threading.Event()

    def handle(self, text, out):
        self.seen.append(text)
        out.text("reply to " + text)
        self.event.set()


@pytest.fixture
def running():
    cfg = srv.Config("tok", "555", "verify-me", SECRET, {"919876543210"})
    http = FakeHttp()
    core = RecordingCore()
    disp = srv.Dispatcher(cfg, core, WhatsAppClient("tok", "555", http=http))
    server = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(cfg, disp))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", core, http
    server.shutdown()
    server.server_close()


def post(base, body: bytes, signature):
    headers = {"Content-Type": "application/json"}
    if signature:
        headers["X-Hub-Signature-256"] = signature
    req = urllib.request.Request(base + "/webhook", data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_webhook_verification_handshake(running):
    base, _, _ = running
    with urllib.request.urlopen(base + "/webhook?hub.mode=subscribe&hub.verify_token=verify-me&hub.challenge=777", timeout=5) as r:
        assert r.read() == b"777"
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(base + "/webhook?hub.mode=subscribe&hub.verify_token=wrong&hub.challenge=1", timeout=5)
    assert exc.value.code == 403


def test_signed_message_is_processed_and_answered(running):
    base, core, http = running
    body = json.dumps(payload(TEXT)).encode()
    assert post(base, body, sign(body)) == 200
    assert core.event.wait(5)
    assert core.seen == ["analyze 5"]
    assert json.loads(http.calls[0][3])["text"]["body"] == "reply to analyze 5"


def test_bad_or_missing_signature_rejected(running):
    base, core, _ = running
    body = json.dumps(payload(TEXT)).encode()
    assert post(base, body, sign(body, "wrong")) == 403
    assert post(base, body, None) == 403
    assert core.seen == []


def test_duplicate_delivery_processed_once(running):
    base, core, _ = running
    body = json.dumps(payload(TEXT)).encode()
    assert post(base, body, sign(body)) == 200
    assert post(base, body, sign(body)) == 200
    assert core.event.wait(5)
    assert core.seen == ["analyze 5"]


def test_number_not_on_allowlist_is_ignored(running):
    base, core, http = running
    stranger = dict(TEXT, id="wamid.99", **{"from": "15550100"})
    body = json.dumps(payload(stranger)).encode()
    assert post(base, body, sign(body)) == 200
    assert not core.event.wait(0.5)
    assert core.seen == [] and http.calls == []


def test_image_gets_not_enabled_notice(running):
    base, core, http = running
    body = json.dumps(payload(IMAGE)).encode()
    assert post(base, body, sign(body)) == 200
    for _ in range(50):
        if http.calls:
            break
        threading.Event().wait(0.1)
    assert "not enabled yet" in json.loads(http.calls[0][3])["text"]["body"]


def test_config_requires_all_settings_and_allowlist(monkeypatch):
    for k in srv.REQUIRED:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit) as exc:
        srv.Config.from_env()
    assert "WHATSAPP_TOKEN" in str(exc.value)
    for k in srv.REQUIRED:
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("WHATSAPP_ALLOWED", " , ")
    with pytest.raises(SystemExit):
        srv.Config.from_env()
    monkeypatch.setenv("WHATSAPP_ALLOWED", "+91 98765 43210")
    assert srv.Config.from_env().allowed == {"919876543210"}
