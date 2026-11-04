import pytest

from mequa import firewall, settings


@pytest.fixture(autouse=True)
def _iso(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PATH", tmp_path / "s.json")
    monkeypatch.delenv("MEQUA_GUARD_MODE", raising=False)
    # These tests cover legacy two-machine Ollama routing, so pin the provider.
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MAIN_URL", "http://main:11434")
    monkeypatch.setenv("OLLAMA_GUARD_URL", "http://guard:11434")
    settings._probe_cache.clear()


def _probe(monkeypatch, up):
    monkeypatch.setattr(settings, "reachable", lambda url, *a, **k: up)


def test_auto_uses_guard_when_up_and_main_when_down(monkeypatch):
    _probe(monkeypatch, True)
    s = settings.guard_status()
    assert s["effective"] == "http://guard:11434" and not s["fallback"] and s["ok"]
    _probe(monkeypatch, False)
    s = settings.guard_status()
    assert s["effective"] == "http://main:11434" and s["fallback"] and s["ok"] and "main machine" in s["note"]


def test_dedicated_fails_closed_with_a_reason(monkeypatch):
    settings.save(guard_mode="dedicated")
    _probe(monkeypatch, False)
    s = settings.guard_status()
    assert not s["ok"] and s["effective"] == "http://guard:11434" and "refused" in s["note"]


def test_main_mode_ignores_the_guard_machine(monkeypatch):
    settings.save(guard_mode="main")
    _probe(monkeypatch, False)
    assert settings.effective_guard_url() == "http://main:11434"
    assert firewall._chat_url("guard") == "http://main:11434/api/chat"
    assert firewall._chat_url("main") == "http://main:11434/api/chat"


def test_env_overrides_file_and_bad_values_are_rejected(monkeypatch):
    settings.save(guard_mode="dedicated")
    monkeypatch.setenv("MEQUA_GUARD_MODE", "main")
    assert settings.guard_mode() == "main"
    with pytest.raises(ValueError):
        settings.save(guard_mode="off")
    settings.PATH.write_text("{not json")
    assert settings.load()["guard_mode"] == "auto"


def test_settings_guard_url_beats_env_and_no_guard_means_main(monkeypatch):
    settings.save(guard_url="http://other:1/")
    assert settings.configured_guard_url() == "http://other:1"
    monkeypatch.delenv("OLLAMA_GUARD_URL")
    settings.save(guard_url="")
    _probe(monkeypatch, False)
    assert settings.guard_status()["on_main"]


def test_reply_check_with_guard_down_in_auto_goes_to_main(monkeypatch):
    """The reported bug: with the guard laptop off, every reply was blocked. Now the request goes to the main machine."""
    seen = []
    monkeypatch.setattr(settings, "reachable", lambda url, *a, **k: False)

    def fake_urlopen(req, timeout=0):
        seen.append(req.full_url)
        import io, json
        return io.BytesIO(json.dumps({"message": {"content": "safe"}}).encode())

    monkeypatch.setattr(firewall.urllib.request, "urlopen", fake_urlopen)
    v = firewall.check_reply("hello", "hi there")
    assert v.allowed and seen == ["http://main:11434/api/chat"]


def test_theme_setting_roundtrip_and_validation(tmp_path, monkeypatch):
    from mequa import settings

    monkeypatch.setattr(settings, "PATH", tmp_path / "s.json")
    assert settings.load()["theme"] == "light"
    settings.save(theme="green")
    assert settings.load()["theme"] == "green"
    with pytest.raises(ValueError):
        settings.save(theme="purple")


def test_both_palettes_define_the_same_variables_and_css_builds():
    from mequa.ui import theme

    assert set(theme.PALETTES["light"]) == set(theme.PALETTES["green"])
    for mode in theme.PALETTES:
        css = theme._css(True, mode)
        assert "--bg:" in css and "#fff;" not in css.lower().replace("#ffffff", "")  # no stray hard-coded white
    assert "invert(1)" in theme._css(True, "green") and "invert(1)" not in theme._css(True, "light")
