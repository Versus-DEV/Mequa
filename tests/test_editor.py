import numpy as np
import pytest

from mequa import numpy_backend as nb
from mequa import workspace
from mequa.editor import CellModel, apply_selection
from mequa.editor import io as eio
from mequa.session import Session


def _model(n=60, dim=3, seed=3):
    return CellModel.from_arrays(nb.synthetic_cells(n, seed=seed, clusters=1, dim=dim), name="t")


def test_selection_modes():
    cur = np.array([True, True, False, False])
    assert apply_selection(cur, [2], "replace").tolist() == [False, False, True, False]
    assert apply_selection(cur, [2], "add").tolist() == [True, True, True, False]
    assert apply_selection(cur, [1], "remove").tolist() == [True, False, False, False]
    assert apply_selection(cur, [1, 2], "intersect").tolist() == [False, True, False, False]
    assert apply_selection(cur, [99, -1], "replace").sum() == 0  # out-of-range ids are ignored
    with pytest.raises(ValueError):
        apply_selection(cur, [0], "xor")


def test_move_rotate_scale_mirror_geometry():
    m = _model()
    sel = m.select_box([-1e9] * 3, [1e9] * 3)
    before = m.points.copy()
    m.move(sel, (1, 2, 3))
    assert np.allclose(m.points, before + [1, 2, 3])
    c = m.points.mean(axis=0)
    m.rotate(sel, "z", 90, about="selection")
    assert np.allclose(m.points.mean(axis=0), c)                       # rotation about the centroid keeps the centroid
    assert np.allclose(np.linalg.norm(m.points - c, axis=1), np.linalg.norm(before + [1, 2, 3] - c, axis=1))  # rigid
    m.scale(sel, 2.0, about="selection")
    assert np.allclose(np.linalg.norm(m.points - c, axis=1), 2 * np.linalg.norm(before + [1, 2, 3] - c, axis=1))
    x0 = m.points[:, 0].copy()
    m.mirror(sel, "x", about="selection")
    assert np.allclose(m.points[:, 0], 2 * c[0] - x0)


def test_undo_redo_restore_exact_state():
    m = _model()
    base = m.points.copy()
    sel = m.select_nearest(m.points[0], 5)
    m.move(sel, (1, 0, 0))
    m.set_stiffness(sel, 3.0)
    m.delete(m.select_nearest(m.points[10], 3))
    assert m.n == len(base) - 3
    m.undo(); m.undo(); m.undo()
    assert np.array_equal(m.points, base) and np.all(m.stiffness == 1.0) and not m.can_undo
    m.redo(); m.redo()
    assert m.stiffness.max() == 3.0 and m.n == len(base)
    m.redo()
    assert m.n == len(base) - 3
    assert m.redo() is None
    m.move(m.select_nearest(m.points[0], 2), (0, 0, 1))
    assert not m.can_redo                                               # a new edit clears the redo stack


def test_delete_duplicate_add_and_guards():
    m = _model(30)
    with pytest.raises(ValueError, match="nothing is selected"):
        m.move(np.zeros(m.n, bool), (1, 0, 0))
    with pytest.raises(ValueError, match="at least 8"):
        m.delete(np.ones(m.n, bool))
    new = m.duplicate(m.select_nearest(m.points[0], 4), (0.1, 0, 0))
    assert m.n == 34 and new.sum() == 4 and new[-4:].all()
    i = m.add_cell((1, 2, 3), stiffness=2.5, group=7)
    assert m.points[i].tolist() == [1, 2, 3] and m.stiffness[i] == 2.5 and m.group[i] == 7
    with pytest.raises(ValueError):
        m.add_cell((0, 0, 0), stiffness=0)
    with pytest.raises(ValueError):
        m.set_stiffness(m.select_nearest(m.points[0], 2), 1e6)


def test_2d_models_stay_2d():
    m = _model(40, dim=2)
    assert m.dim == 2 and m.analysis_points().shape == (40, 2)
    m.rotate(m.mask(range(10)), "z", 30)
    with pytest.raises(ValueError):
        m.rotate(m.mask(range(10)), "x", 30)
    m.add_cell((1, 1, 5))
    assert m.points[-1, 2] == 0.0 and m.analysis_points().shape[1] == 2


def test_cleanup_ops():
    m = _model(80)
    m.add_cell(m.points.max(axis=0) + 40)           # a far-away stray cell
    assert m.find_outliers(k=6, z=3.0)[-1]
    m.duplicate(m.mask([0]), (0.001, 0, 0))
    removed = m.merge_close(0.05)
    assert removed == 1
    m.downsample(40)
    assert m.n == 40
    m.normalize_extent(10.0)
    lo, hi = m.bounds()
    assert hi[0] - lo[0] == pytest.approx(10.0)


def test_edits_change_the_solver_result_and_stiffness_is_used():
    m = _model(200, seed=5)
    base = Session(m.analysis_points(), knn=8).run(0.05)
    m.set_stiffness(m.select_box(m.bounds()[0], m.points.mean(axis=0)), 4.0)
    stiff = Session(m.analysis_points(), stiffness=m.analysis_stiffness(), knn=8).run(0.05)
    assert stiff["max_strain"] != base["max_strain"] and stiff["cells"] == base["cells"]


def test_io_roundtrips_and_errors(tmp_path):
    m = _model(25)
    m.set_stiffness(m.mask([0, 1]), 2.0)
    m.set_group(m.mask([2]), 4)
    for label in eio.EXPORTERS:
        name, data, _ = eio.export(m, label)
        assert name.startswith("t.") and len(data) > 100
    for kind in ("csv", "json"):
        data = eio.to_csv(m) if kind == "csv" else eio.to_json(m)
        back = eio.from_bytes(data, f"m.{kind}")
        assert np.allclose(back.points, m.points, atol=1e-5) and np.allclose(back.stiffness, m.stiffness) and back.group.tolist() == m.group.tolist()
    two_d = _model(20, dim=2)
    assert eio.to_csv(two_d).decode().splitlines()[0] == "x,y,stiffness,group"
    assert eio.from_bytes(eio.to_csv(two_d), "a.csv").dim == 2
    with pytest.raises(ValueError, match="columns named x and y"):
        eio.from_bytes(b"a,b\n1,2\n", "bad.csv")
    with pytest.raises(ValueError, match="cannot read"):
        eio.from_bytes(b"", "m.stl")
    with pytest.raises(ValueError, match="NaN"):
        CellModel.from_arrays(np.array([[0, 0, np.nan]] * 10))


def test_workspace_slot_is_typed_and_validated():
    state = {}
    assert workspace.current(state) is None
    ds = workspace.publish(state, nb.synthetic_cells(30, dim=3), label="blob", source="test")
    assert workspace.current(state) is ds and ds.n == 30 and ds.dim == 3
    state["workspace_dataset"] = {"points": 1}      # something else under the key is ignored, not trusted
    assert workspace.current(state) is None
    with pytest.raises(ValueError):
        workspace.make_dataset(np.zeros((5, 4)))
    with pytest.raises(ValueError):
        workspace.make_dataset(np.zeros((5, 3)), stiffness=np.ones(4))
