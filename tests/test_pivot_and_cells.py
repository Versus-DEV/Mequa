import numpy as np

from mequa.firewall import Verdict, rules_check
from mequa.guard.research_pivot import BLOCK_MESSAGE, SUGGESTIONS, pivot_to_research
from mequa.vision.cells import cells_from_masks, filter_masks, normalize_stiffness
from mequa.vision.demo_masks import make_demo_masks


def test_blocked_request_gets_message_and_fixed_suggestions_never_a_rewrite():
    q = "Is this tumor malignant and what dose should I give the patient?"
    out = pivot_to_research(q, check_fn=lambda t: Verdict(allowed=False, blocked_by=["llama_intent"]))
    assert out["decision"] == "blocked" and out["message"] == BLOCK_MESSAGE
    assert out["suggestions"] == SUGGESTIONS
    assert "tumor" not in " ".join(out["suggestions"]).lower() and "patient" not in " ".join(out["suggestions"]).lower()
    assert "research_prompt" not in out


def test_allowed_request_passes_through_unchanged():
    out = pivot_to_research("cluster the cells", check_fn=lambda t: Verdict(allowed=True))
    assert out["decision"] == "allowed" and "suggestions" not in out


def test_the_fixed_suggestions_themselves_pass_the_rules_layer():
    assert not any(rules_check(s) for s in SUGGESTIONS)


def disc(shape, cx, cy, r):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def test_filter_masks_drops_background_duplicates_clusters_and_specks():
    s = (200, 200)
    cells = [disc(s, 40, 40, 12), disc(s, 100, 50, 12), disc(s, 60, 120, 14)]
    extra = [disc(s, 100, 100, 90), disc(s, 100, 50, 12), disc(s, 40, 40, 12) | disc(s, 70, 40, 12), disc(s, 5, 5, 3)]
    kept, bg, ov = filter_masks(cells + extra, min_area=60)
    assert len(kept) == 3 and bg == 1 and ov == 2


def test_cells_have_one_stiffness_per_cell_float64_in_range():
    cs = cells_from_masks(make_demo_masks(), min_area=50)
    assert cs.points.shape == (cs.n, 2) and cs.stiffness.shape == (cs.n,)
    assert cs.stiffness.dtype == np.float64 and cs.stiffness.min() >= 0.5 and cs.stiffness.max() <= 2.0


def test_irregular_cell_is_stiffer_than_round_cell():
    s = (200, 200)
    round_cell = disc(s, 50, 50, 20)
    elong = (((np.mgrid[0:200, 0:200][1] - 140) / 10.0) ** 2 + ((np.mgrid[0:200, 0:200][0] - 100) / 40.0) ** 2) <= 1.0
    cs = cells_from_masks([round_cell, elong, disc(s, 150, 40, 20)], min_area=60)
    assert cs.n == 3
    by_ar = sorted(zip([f["aspect_ratio"] for f in cs.features], cs.stiffness))
    assert by_ar[-1][1] >= by_ar[0][1]


def test_normalize_is_median_relative():
    out = normalize_stiffness([1.0, 2.0, 4.0])
    assert out[1] == 1.0 and out[0] == 0.5 and out[2] == 2.0
