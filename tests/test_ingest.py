from mequa import ingest


def test_classical_segmentation_and_csv_roundtrip(tmp_path):
    img_path = tmp_path / "cells.png"
    drawn = ingest.make_demo_image(str(img_path), n=200, seed=5)
    pts, areas = ingest.segment(ingest.load_image(str(img_path)), "classical")
    assert 0.6 * drawn <= len(pts) <= drawn
    csv_path = tmp_path / "cells.csv"
    ingest.save_points_csv(pts, str(csv_path))
    again = ingest.load_points_csv(str(csv_path))
    assert again.shape == pts.shape


def test_checkpoint_detection(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / "sam_vit_h_4b8939.pth").write_bytes(b"x")
    (models / "sam_vit_l_0b3195.pth").write_bytes(b"x")
    monkeypatch.setattr(ingest, "SAM_PATH", models / "sam_vit_b_01ec64.pth")
    path, kind = ingest.find_checkpoint()
    assert kind == "vit_l" and path.name.startswith("sam_vit_l")
    path, kind = ingest.find_checkpoint(str(models / "sam_vit_h_4b8939.pth"))
    assert kind == "vit_h"
