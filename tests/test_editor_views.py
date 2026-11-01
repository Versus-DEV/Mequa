import numpy as np

from mequa import numpy_backend as nb
from mequa.editor import CellModel
from mequa.editor import views


def _m():
    m = CellModel.from_arrays(nb.synthetic_cells(50, seed=2, clusters=1, dim=3))
    m.set_group(m.mask(range(5)), 2)
    return m


def test_all_views_and_modes_build():
    m = _m()
    sel = m.mask([1, 2, 3])
    strain = np.linspace(0, 0.1, m.n)
    for mode in views.COLOR_MODES:
        for v in views.VIEWS:
            fig = views.figure_2d(m, sel, v, mode, strain)
            assert len(fig.data) == 2 and len(fig.data[0].x) == m.n and len(fig.data[1].x) == 3
        assert len(views.figure_3d(m, sel, mode, strain).data) == 2
    assert len(views.figure_2d(m, m.mask([]), "top").data) == 1     # no overlay when nothing is selected


def test_top_view_plots_x_against_y():
    m = _m()
    fig = views.figure_2d(m, m.mask([]), "front")
    assert np.allclose(fig.data[0].x, m.points[:, 0]) and np.allclose(fig.data[0].y, m.points[:, 2])


def test_event_parsing_ignores_overlay_trace_and_handles_shapes():
    ev = {"selection": {"points": [{"curve_number": 0, "customdata": 4}, {"curve_number": 0, "customdata": [7]},
                                   {"curve_number": 1, "point_index": 0}, {"curve_number": 0, "point_index": 9}], "point_indices": [4, 7, 0, 9]}}
    assert views.picked_from_event(ev) == [4, 7, 9]
    assert views.picked_from_event({"selection": {"points": [], "point_indices": [3, 1]}}) == [1, 3]
    assert views.picked_from_event({"selection": {}}) == []
    assert views.picked_from_event(None) == []
