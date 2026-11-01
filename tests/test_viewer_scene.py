import numpy as np
import pytest

from mequa import numpy_backend as nb
from mequa import viewer
from mequa.viewer import pdb

PDB = """HEADER    TEST
ATOM      1  N   ALA A   1      11.104  13.207   9.060  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.207   9.060  1.00 20.00           C
ATOM      3  C   ALA A   1      13.060  14.640   9.060  1.00 20.00           C
ATOM      4  O   ALA A   1      12.300  15.600   9.100  1.00 20.00           O
ATOM      5  N   GLY A   2      14.380  14.760   9.000  1.00 20.00           N
ATOM      6  CA  GLY A   2      15.020  16.060   9.000  1.00 20.00           C
ATOM      7  CA  GLY A   3      18.020  16.060   9.000  1.00 20.00           C
HETATM    8 ZN    ZN A 100      20.000  20.000  20.000  1.00 20.00          ZN
END
"""


def test_scene_from_solver_output_has_fields_edges_and_loaded_shape():
    pts = nb.synthetic_cells(300, seed=2, clusters=1, dim=3)
    res = nb.analyze(pts, k=3, knn=6, load=0.05)
    sc = viewer.scene_from_solve(pts, res, stiffness=np.ones(300), group=np.zeros(300), knn=6, title="t")
    assert sc["n"] == 300 and len(sc["pos"]) == 900 and len(sc["loaded"]) == 900
    assert set(sc["fields"]) >= {"strain", "zone", "failure_prob", "stiffness", "group"}
    assert sc["field_kinds"]["zone"] == "categorical" and sc["field_kinds"]["strain"] == "continuous"
    assert len(sc["edges"]) % 2 == 0 and len(sc["edge_strain"]) == len(sc["edges"]) // 2
    assert viewer.scene_id(sc) == viewer.scene_id(sc)


def test_scene_id_changes_when_cells_move_but_not_otherwise():
    pts = nb.synthetic_cells(60, seed=1, dim=3)
    a = viewer.scene_id(viewer.build_scene(pts))
    assert a == viewer.scene_id(viewer.build_scene(pts.copy()))
    moved = pts.copy(); moved[0, 0] += 0.5
    assert a != viewer.scene_id(viewer.build_scene(moved))


def test_2d_points_are_lifted_and_bad_input_is_rejected():
    sc = viewer.build_scene(nb.synthetic_cells(20, dim=2))
    assert sc["pos"][2] == 0.0
    with pytest.raises(ValueError):
        viewer.build_scene(np.zeros((5, 4)))
    with pytest.raises(ValueError, match="one value per cell"):
        viewer.build_scene(np.zeros((5, 3)), fields={"a": [1, 2]})
    with pytest.raises(ValueError, match="downsample"):
        viewer.build_scene(np.zeros((viewer.MAX_SCENE_CELLS + 1, 3)))


def test_pdb_parsing_bonds_and_backbone():
    mol = pdb.load(PDB.encode(), "x.pdb")
    assert len(mol["xyz"]) == 8 and mol["elements"][:4] == ["N", "C", "C", "O"] and mol["elements"][-1] == "ZN"
    b = {tuple(sorted(p)) for p in mol["bonds"].tolist()}
    assert (0, 1) in b and (1, 2) in b and (2, 3) in b and (0, 7) not in b       # N-CA, CA-C, C=O bonded; zinc far away is not
    bb = pdb.backbone_edges(mol)
    assert bb.tolist() == [[1, 5], [5, 6]] or bb.tolist() == [[1, 5]]           # CA(1)-CA(2) is 3.8 A apart; CA(2)-CA(3) is 3.0 A
    sc = viewer.scene_from_molecule(mol)
    assert sc["colors"][0] == "#3050F8" and sc["field_kinds"]["chain"] == "categorical" and len(sc["radii"]) == 8


def test_xyz_and_errors():
    mol = pdb.load(b"3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n", "w.xyz")
    assert len(mol["bonds"]) == 2
    with pytest.raises(ValueError, match="no ATOM"):
        pdb.load(b"nothing here", "a.pdb")
    with pytest.raises(ValueError, match="XYZ"):
        pdb.load(b"x\ny\n", "a.xyz")
