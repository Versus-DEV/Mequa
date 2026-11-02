"""Drive the real Streamlit pages with AppTest: every page renders, and the failures seen in the
dashboard screenshots (Workbench KeyError 'labels', Simulink URLError) cannot come back."""

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
PAGES = ["views/0_workbench.py", "views/1_sam_mechanics.py", "views/2_two_machine_status.py", "views/3_adaptive_agent.py",
         "views/3b_research_pivot.py", "views/4_simulink_editor.py", "views/5_mesh_viewer.py", "views/6_solver_benchmark.py",
         "views/7_model_editor.py", "views/8_viewer.py", "views/9_image_lab.py"]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("MEQUA_FORCE_NUMPY", "1")
    monkeypatch.setenv("OLLAMA_MAIN_URL", "http://127.0.0.1:9")     # nothing listens here: Ollama is "offline"
    monkeypatch.delenv("OLLAMA_GUARD_URL", raising=False)
    monkeypatch.chdir(ROOT)


def _app():
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _btn(container, label):
    return next(b for b in container.button if b.label == label)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    at = _app()
    at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]


def test_workbench_survives_foreign_results_in_session_state():
    """v5 crashed with KeyError 'labels' because the SAM page and the Workbench shared st.session_state['result']."""
    at = _app()
    at.session_state["result"] = {"per_zone": [], "cells": 5}                 # what the SAM page used to leave behind
    at.session_state["wb_result"] = {"per_zone": [], "cells": 5}               # and even a malformed Workbench result
    at.switch_page("views/0_workbench.py").run()
    assert not at.exception, [e.value for e in at.exception]
    _btn(at.sidebar, "Run Analysis").click().run()
    assert not at.exception, [e.value for e in at.exception]
    res = at.session_state["wb_result"]
    assert "labels" in res and len(res["labels"]) == 2000


def test_simulink_editor_offline_uses_rules_and_says_so():
    at = _app()
    at.switch_page("views/4_simulink_editor.py").run()
    _btn(at.sidebar, "Generate Block Spec").click().run()
    assert not at.exception, [e.value for e in at.exception]
    tr = at.session_state["translation"]
    assert tr["success"] and tr["source"] == "rules"
    assert any("Ollama is not reachable" in w.value for w in at.warning)


def test_simulink_editor_offline_unparseable_text_gives_friendly_error():
    at = _app()
    at.switch_page("views/4_simulink_editor.py").run()
    at.sidebar.text_area[0].set_value("make it nice").run()
    _btn(at.sidebar, "Generate Block Spec").click().run()
    assert not at.exception
    assert any("Ollama is not reachable" in e.value for e in at.error)


def test_model_editor_end_to_end():
    at = _app()
    at.switch_page("views/7_model_editor.py").run()
    assert not at.exception
    at.sidebar.selectbox[0].set_value("Synthetic 3D blob").run()
    at.sidebar.slider[0].set_value(400)
    _btn(at.sidebar, "Load model").click().run()
    assert not at.exception, [e.value for e in at.exception]
    m = at.session_state["ed_model"]
    assert m.n == 400 and m.dim == 3
    x0 = float(m.points[:, 0].mean())
    _btn(at.sidebar, "All").click().run()
    assert at.session_state["ed_sel"].sum() == 400
    # move all cells +1 in x through the Transform tab, then undo
    move_x = next(n for n in at.number_input if n.label == "Move x")
    move_x.set_value(1.0).run()
    _btn(at.main, "Move").click().run()
    assert not at.exception, [e.value for e in at.exception]
    moved = at.session_state["ed_model"].points[:, 0].mean()
    assert moved == pytest.approx(x0 + 1.0, abs=1e-6)
    assert len(at.session_state["ed_model"].log) == 1
    _btn(at.main, "Undo").click().run()
    assert not at.exception
    # solve and compare
    _btn(at.main, "Solve and compare").click().run()
    assert not at.exception, [e.value for e in at.exception]
    solve = at.session_state["ed_solve"]
    assert solve["before"]["cells"] == solve["after"]["cells"] == 400
    # the edited model is the workspace dataset for other pages
    from mequa import workspace
    ds = workspace.current(at.session_state)
    assert ds is not None and ds.n == 400 and ds.source == "Model Editor"


def test_model_editor_delete_and_guards():
    at = _app()
    at.switch_page("views/7_model_editor.py").run()
    at.sidebar.selectbox[0].set_value("Synthetic 3D blob").run()
    _btn(at.sidebar, "Load model").click().run()
    n0 = at.session_state["ed_model"].n
    _btn(at.sidebar, "All").click().run()
    _btn(at.main, "Delete selection").click().run()          # deleting everything must be refused, not crash
    assert not at.exception
    assert at.session_state["ed_model"].n == n0
    assert any("at least 8" in e.value for e in at.error)


def test_chat_dock_commands_and_offline_message():
    at = _app()
    at.chat_input[0].set_value("/help").run()
    assert not at.exception
    at.chat_input[0].set_value("Find the load at which this sample reaches the strain limit.").run()
    assert not at.exception, [e.value for e in at.exception]
    texts = " ".join(m.value for m in at.markdown)
    assert "Ollama is not running" in texts


def test_cell_viewer_reads_the_edited_workspace_dataset():
    from mequa import numpy_backend as nb, workspace
    at = _app()
    workspace.publish(at.session_state, nb.synthetic_cells(300, seed=1, clusters=1, dim=3), None, label="x", source="test")
    at.switch_page("views/5_mesh_viewer.py").run()
    at.sidebar.selectbox[0].set_value("Workspace dataset (editor or workbench)")
    _btn(at.sidebar, "Load and Analyze").click().run()
    assert not at.exception, [e.value for e in at.exception]


def test_image_lab_synthetic_stack_finds_cells_and_publishes_to_workspace():
    from mequa import workspace

    at = _app()
    at.switch_page("views/9_image_lab.py").run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Synthetic test stack").run()
    assert any("made-up" in w.value for w in at.warning)          # synthetic data is labelled as such
    _btn(at.main, "Find cells").click().run()
    assert not at.exception, [e.value for e in at.exception]
    ds = workspace.current(at.session_state)
    assert ds is not None and ds.dim == 3 and ds.n >= 8 and ds.source == "Image Lab"


def test_dock_follows_the_dataset_a_page_publishes():
    at = _app()
    at.switch_page("views/9_image_lab.py").run()
    at.sidebar.radio[0].set_value("Synthetic test stack").run()
    _btn(at.main, "Find cells").click().run()
    assert any("cells" in (m.value if hasattr(m, "value") else "") and "no dataset" not in m.value for m in at.markdown if "mq-chat-sub" in m.value)
