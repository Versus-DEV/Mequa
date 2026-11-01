import hashlib
import hmac
import json
import logging
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from mequa.bot import messenger_server as srv
from mequa.bot.messenger import MessengerClient, MessengerError, MessengerOut, parse_messenger, plain, split_text

SECRET = "app-secret"
PSID = "7654321098765432"


def sign(body: bytes, secret=SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def event(**kw):
    base = {"sender": {"id": PSID}, "recipient": {"id": "page1"}, "timestamp": 1793550000000}
    base.update(kw)
    return {"object": "page", "entry": [{"id": "page1", "time": 1, "messaging": [base]}]}


TEXT = event(message={"mid": "m_1", "text": "analyze 5"})


def test_parse_text_quick_reply_postback_attachment():
    assert parse_messenger(TEXT)[0].text == "analyze 5"
    quick = parse_messenger(event(message={"mid": "m_2", "text": "Sweep", "quick_reply": {"payload": "sweep"}}))
    assert quick[0].text == "sweep"
    post = parse_messenger(event(postback={"title": "Map", "payload": "map", "mid": "m_3"}))
    assert post[0].text == "map" and post[0].id == "m_3"
    img = parse_messenger(event(message={"mid": "m_4", "attachments": [{"type": "image", "payload": {"url": "u"}}]}))
    assert img[0].kind == "image"
    other = parse_messenger(event(message={"mid": "m_5", "attachments": [{"type": "audio"}]}))
    assert other[0].kind == "attachment"


def test_parse_ignores_echoes_receipts_and_other_objects():
    assert parse_messenger(event(message={"mid": "m_6", "text": "x", "is_echo": True})) == []
    assert parse_messenger(event(delivery={"mids": ["m_1"]})) == []
    assert parse_messenger(event(read={"watermark": 1})) == []
    assert parse_messenger({"object": "whatsapp_business_account", "entry": []}) == []
    assert parse_messenger({}) == []


def test_plain_and_split_text():
    assert plain("*Bold* and `code`") == "Bold and code"
    parts = split_text("line\n" * 1000, limit=500)
    assert all(len(p) <= 500 for p in parts) and len(parts) > 1
    assert split_text("   ") == []


class FakeHttp:
    def __init__(self, status=200, body=b'{"message_id": "m1"}'):
        self.calls, self.status, self.body = [], status, body

    def __call__(self, method, url, headers, data):
        self.calls.append((method, url, headers, data))
        return self.status, self.body


def test_client_text_payload_and_auth_header():
    http = FakeHttp()
    MessengerClient("PAGETOKEN", "v23.0", http).send_text(PSID, "*hi* there")
    method, url, headers, data = http.calls[0]
    assert url == "https://graph.facebook.com/v23.0/me/messages"
    assert headers["Authorization"] == "Bearer PAGETOKEN"
    body = json.loads(data)
    assert body["recipient"] == {"id": PSID} and body["messaging_type"] == "RESPONSE"
    assert body["message"]["text"] == "hi there"


def test_client_quick_replies_titles_trimmed():
    http = FakeHttp()
    MessengerClient("t", http=http).send_quick_replies(PSID, "pick", [("a", "A" * 40), ("b", "B")])
    replies = json.loads(http.calls[0][3])["message"]["quick_replies"]
    assert len(replies[0]["title"]) == 20 and replies[1]["payload"] == "b"


def test_client_image_is_multipart_with_recipient_and_caption_follows():
    http = FakeHttp()
    MessengerClient("t", http=http).send_image(PSID, b"\x89PNGdata", "caption here")
    assert "multipart/form-data" in http.calls[0][2]["Content-Type"]
    assert b"\x89PNGdata" in http.calls[0][3] and b'name="recipient"' in http.calls[0][3]
    assert b'name="filedata"' in http.calls[0][3]
    assert json.loads(http.calls[1][3])["message"]["text"] == "caption here"


def test_client_error_shows_code_but_not_token():
    err = b'{"error": {"message": "(#10) outside allowed window", "code": 10, "error_subcode": 2018278}}'
    c = MessengerClient("SECRET-PAGE-TOKEN", http=FakeHttp(status=400, body=err))
    with pytest.raises(MessengerError) as exc:
        c.send_text(PSID, "x")
    text = str(exc.value)
    assert "SECRET-PAGE-TOKEN" not in text and "code 10" in text and "2018278" in text


def test_menu_falls_back_to_plain_text_on_error():
    class Flaky(MessengerClient):
        def __init__(self):
            self.sent = []

        def send_quick_replies(self, to, text, replies=()):
            raise MessengerError("nope")

        def send_text(self, to, text):
            self.sent.append(text)

    client = Flaky()
    MessengerOut(client, PSID).menu("menu text")
    assert client.sent == ["menu text"]


class RecordingCore:
    def __init__(self):
        self.seen, self.event = [], threading.Event()

    def handle(self, text, out):
        self.seen.append(text)
        out.text("reply to " + text)
        self.event.set()


@pytest.fixture
def running():
    cfg = srv.Config("tok", "verify-me", SECRET, {PSID})
    http = FakeHttp()
    core = RecordingCore()
    disp = srv.Dispatcher(cfg, core, MessengerClient("tok", http=http))
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


def test_handshake(running):
    base, _, _ = running
    with urllib.request.urlopen(base + "/webhook?hub.mode=subscribe&hub.verify_token=verify-me&hub.challenge=555", timeout=5) as r:
        assert r.read() == b"555"
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(base + "/webhook?hub.mode=subscribe&hub.verify_token=bad&hub.challenge=1", timeout=5)
    assert exc.value.code == 403


def test_signed_message_processed_and_answered_with_typing(running):
    base, core, http = running
    body = json.dumps(TEXT).encode()
    assert post(base, body, sign(body)) == 200
    assert core.event.wait(5)
    assert core.seen == ["analyze 5"]
    bodies = [json.loads(c[3]) for c in http.calls]
    assert any(b.get("sender_action") == "typing_on" for b in bodies)
    assert any(b.get("message", {}).get("text") == "reply to analyze 5" for b in bodies)


def test_bad_signature_rejected_and_duplicates_dropped(running):
    base, core, _ = running
    body = json.dumps(TEXT).encode()
    assert post(base, body, sign(body, "wrong")) == 403
    assert post(base, body, None) == 403
    assert core.seen == []
    assert post(base, body, sign(body)) == 200
    assert post(base, body, sign(body)) == 200
    assert core.event.wait(5)
    assert core.seen == ["analyze 5"]


def test_unknown_sender_ignored_and_id_logged(running, caplog):
    base, core, http = running
    stranger = event(message={"mid": "m_99", "text": "hi"})
    stranger["entry"][0]["messaging"][0]["sender"] = {"id": "1111222233334444"}
    body = json.dumps(stranger).encode()
    with caplog.at_level(logging.WARNING, logger="mequa.bot"):
        assert post(base, body, sign(body)) == 200
        assert not core.event.wait(0.5)
    assert core.seen == [] and http.calls == []
    assert "1111222233334444" in caplog.text and "MESSENGER_ALLOWED" in caplog.text


def test_echo_and_status_events_do_nothing(running):
    base, core, http = running
    body = json.dumps(event(message={"mid": "m_7", "text": "bot reply", "is_echo": True})).encode()
    assert post(base, body, sign(body)) == 200
    assert not core.event.wait(0.5)
    assert http.calls == []


def test_config_fallbacks_and_empty_allowlist(monkeypatch):
    for k in ("MESSENGER_PAGE_TOKEN", "MESSENGER_VERIFY_TOKEN", "MESSENGER_APP_SECRET", "MESSENGER_ALLOWED",
              "WHATSAPP_VERIFY_TOKEN", "WHATSAPP_APP_SECRET", "MESSENGER_API_VERSION", "WHATSAPP_API_VERSION"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit) as exc:
        srv.Config.from_env()
    assert "MESSENGER_PAGE_TOKEN" in str(exc.value)
    monkeypatch.setenv("MESSENGER_PAGE_TOKEN", "t")
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "v")
    monkeypatch.setenv("WHATSAPP_APP_SECRET", "s")
    cfg = srv.Config.from_env()
    assert cfg.verify_token == "v" and cfg.app_secret == "s" and cfg.allowed == set() and cfg.version == "v23.0"
    monkeypatch.setenv("MESSENGER_ALLOWED", "7654321098765432, 1")
    assert srv.Config.from_env().allowed == {"7654321098765432", "1"}


def test_menu_names_the_right_channel():
    http = FakeHttp()
    MessengerOut(MessengerClient("t", http=http), PSID).menu("*Mequa* on WhatsApp\nhelp text")
    assert "on Messenger" in json.loads(http.calls[0][3])["message"]["text"]
