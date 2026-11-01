import numpy as np

from mequa.bot import core as botcore
from mequa.bot.core import Batcher, BotCore, parse_percent, to_whatsapp
from mequa.workspace import make_dataset


class Out:
    def __init__(self):
        self.texts, self.images, self.menus = [], [], []

    def text(self, s):
        self.texts.append(s)

    def menu(self, s):
        self.menus.append(s)

    def image(self, png, caption):
        self.images.append((png, caption))


def small_dataset(n=300):
    from mequa import numpy_backend as nb

    return make_dataset(nb.synthetic_cells(n, seed=3, clusters=2), label="test cells", source="test")


def test_parse_percent_clamps_and_defaults():
    assert parse_percent("5", 0.05, 0.005, 0.2) == 0.05
    assert parse_percent("500", 0.05, 0.005, 0.2) == 0.2
    assert parse_percent("abc", 0.05, 0.005, 0.2) == 0.05
    assert parse_percent("7%", 0.05, 0.005, 0.2) == 0.07


def test_to_whatsapp_bold_and_truncation():
    assert to_whatsapp("**x**") == "*x*"
    assert len(to_whatsapp("a" * 10000)) < 3900


def test_batcher_coalesces():
    sent, now = [], [0.0]
    b = Batcher(sent.append, min_interval=4.0, clock=lambda: now[0])
    b.push("a")
    b.push("b")
    now[0] = 1.0
    b.push("c")
    assert sent == ["a"]
    now[0] = 5.0
    b.push("d")
    assert sent == ["a", "b\nc\nd"]
    b.flush()
    assert len(sent) == 2


def test_help_status_and_menu():
    core = BotCore(small_dataset())
    out = Out()
    core.handle("hi", out)
    assert out.menus and "analyze" in out.menus[0]


def test_analyze_runs_real_solver():
    core = BotCore(small_dataset())
    out = Out()
    core.handle("analyze 5", out)
    assert any("p95 strain" in t for t in out.texts)


def test_sweep_reports_threshold_or_not_reached():
    core = BotCore(small_dataset())
    out = Out()
    core.handle("sweep 5", out)
    last = out.texts[-1]
    assert "Sweep result" in last and ("reaches the limit" in last or "does not reach" in last)


def test_map_returns_png():
    core = BotCore(small_dataset())
    out = Out()
    core.handle("map", out)
    assert out.images and out.images[0][0].startswith(b"\x89PNG")


def test_map_refuses_3d():
    pts = np.random.default_rng(1).random((50, 3))
    core = BotCore(make_dataset(pts, label="3d"))
    out = Out()
    core.handle("map", out)
    assert "2D only" in out.texts[-1]


def test_busy_lock_blocks_second_request():
    core = BotCore(small_dataset())
    out = Out()
    core.lock.acquire()
    core.handle("analyze", out)
    core.lock.release()
    assert "Still working" in out.texts[0]


def test_free_text_streams_progress_and_answer():
    class Base:
        def __init__(self, session, **kw):
            pass

        def execute(self, name, args):
            return {"p95_strain": 0.031}

    def fake_answer(question, dataset, model, strain_limit, investigator_cls=None, **kw):
        inv = investigator_cls(None, model=model, max_iterations=1, strain_limit=strain_limit)
        inv.execute("run_solver", {"load_fraction": 0.05})
        return {"status": "ok", "text": "**Answer** p95 is 0.031"}

    core = BotCore(small_dataset(), answer_fn=fake_answer, investigator_base=Base, min_interval=0.0)
    out = Out()
    core.handle("Which zone carries the most strain?", out)
    joined = "\n".join(out.texts)
    assert "Thinking with local" in joined
    assert "Running the solver at 5.0% load" in joined
    assert "Solver done: p95 strain 0.031" in joined
    assert out.texts[-1] == "*Answer* p95 is 0.031"


def test_errors_do_not_leak_details():
    def boom(*a, **k):
        raise RuntimeError("secret path D:\\x")

    core = BotCore(small_dataset(), answer_fn=boom)
    out = Out()
    core.handle("a research question", out)
    assert "RuntimeError" in out.texts[-1] and "secret path" not in out.texts[-1]


def test_load_dataset_synthetic_and_missing(tmp_path):
    ds = botcore.load_dataset("synthetic")
    assert ds.n == 2000 and ds.dim == 2
    try:
        botcore.load_dataset(str(tmp_path / "nope.npy"))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
    path = tmp_path / "cells.csv"
    path.write_text("x,y\n0,0\n1,1\n2,0\n")
    assert botcore.load_dataset(str(path)).n == 3
