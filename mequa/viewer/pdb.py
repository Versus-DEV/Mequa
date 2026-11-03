"""Read molecular structures (PDB, XYZ) for the 3D viewer: atoms, element colours, bonds.

This is for looking at a structure the way a molecular viewer does, with the same orbit, measure,
box-select and clipping tools used on cell models. Bonds are inferred from distances using covalent
radii (no CONECT parsing needed). Nothing here claims biological or mechanical meaning.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

CPK = {"H": "#FFFFFF", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050", "P": "#FF8000", "S": "#FFFF30", "CL": "#1FF01F",
       "BR": "#A62929", "I": "#940094", "FE": "#E06633", "ZN": "#7D80B0", "MG": "#8AFF00", "CA": "#3DFF00", "NA": "#AB5CF2", "K": "#8F40D4",
       "CU": "#C88033", "MN": "#9C7AC7", "SE": "#FFA100"}
COVALENT = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "P": 1.07, "S": 1.05, "CL": 1.02, "BR": 1.20, "I": 1.39, "FE": 1.32,
            "ZN": 1.22, "MG": 1.41, "CA": 1.76, "NA": 1.66, "K": 2.03, "CU": 1.32, "MN": 1.39, "SE": 1.20}
VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "P": 1.80, "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98}
MAX_ATOMS = 200_000


def _element(name: str, elem_col: str) -> str:
    e = (elem_col or "").strip().upper()
    if e:
        return e
    letters = "".join(ch for ch in name.strip() if ch.isalpha()).upper()
    return letters[:2] if letters[:2] in CPK else letters[:1]


def parse_pdb(text: str, model: int = 1, hetatm: bool = True) -> Dict:
    """Atoms of one model from PDB text. Returns arrays plus residue and chain labels."""
    xyz, elems, labels, chains, resid, ca = [], [], [], [], [], []
    cur = 1
    for line in text.splitlines():
        rec = line[:6].strip()
        if rec == "MODEL":
            try:
                cur = int(line[10:14])
            except ValueError:
                cur += 1
        elif rec == "ENDMDL" and cur >= model:
            break
        if cur != model or not (rec == "ATOM" or (hetatm and rec == "HETATM")):
            continue
        if len(line) < 54:
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        alt = line[16:17]
        if alt not in (" ", "A", "1"):
            continue
        name = line[12:16].strip()
        xyz.append((x, y, z))
        elems.append(_element(name, line[76:78] if len(line) >= 78 else ""))
        res = line[17:20].strip()
        chains.append(line[21:22].strip() or "A")
        resid.append(int(line[22:26]) if line[22:26].strip().lstrip("-").isdigit() else 0)
        labels.append(f"{name} {res}{resid[-1]} {chains[-1]}")
        ca.append(name == "CA" and rec == "ATOM")
        if len(xyz) > MAX_ATOMS:
            raise ValueError(f"more than {MAX_ATOMS} atoms; trim the file")
    if not xyz:
        raise ValueError("no ATOM or HETATM records found (is this a PDB file?)")
    return {"xyz": np.asarray(xyz, float), "elements": elems, "labels": labels, "chain": chains, "resid": np.asarray(resid), "is_ca": np.asarray(ca)}


def parse_xyz(text: str) -> Dict:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    try:
        n = int(lines[0].split()[0])
        body = lines[2:2 + n]
        if len(body) != n:
            raise ValueError
        xyz, elems = [], []
        for ln in body:
            p = ln.split()
            elems.append(p[0].upper())
            xyz.append([float(p[1]), float(p[2]), float(p[3])])
    except (ValueError, IndexError):
        raise ValueError("not a valid XYZ file (line 1 is the atom count, line 2 a comment, then element x y z)")
    n = len(xyz)
    return {"xyz": np.asarray(xyz, float), "elements": elems, "labels": [f"{e} {i + 1}" for i, e in enumerate(elems)],
            "chain": ["A"] * n, "resid": np.arange(n), "is_ca": np.zeros(n, bool)}


def infer_bonds(xyz: np.ndarray, elements: List[str], tol: float = 0.45) -> np.ndarray:
    """Pairs (i, j) closer than the sum of covalent radii plus tol angstrom; hydrogens only bond to one heavy atom."""
    if len(xyz) < 2:
        return np.empty((0, 2), dtype=np.int64)
    rad = np.array([COVALENT.get(e, 1.0) for e in elements])
    pairs = cKDTree(xyz).query_pairs(float(rad.max() * 2 + tol), output_type="ndarray")
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    d = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
    ok = (d < rad[pairs[:, 0]] + rad[pairs[:, 1]] + tol) & (d > 0.4)
    return pairs[ok]


def backbone_edges(mol: Dict, max_gap: float = 4.4) -> np.ndarray:
    """Consecutive C-alpha atoms of the same chain, the usual trace of a protein."""
    idx = np.flatnonzero(mol["is_ca"])
    out = []
    for a, b in zip(idx, idx[1:]):
        if mol["chain"][a] == mol["chain"][b] and np.linalg.norm(mol["xyz"][a] - mol["xyz"][b]) <= max_gap:
            out.append((a, b))
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


def load(data: bytes, filename: str) -> Dict:
    text = data.decode("utf-8", errors="replace")
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    mol = parse_xyz(text) if ext == "xyz" else parse_pdb(text)
    mol["bonds"] = infer_bonds(mol["xyz"], mol["elements"])
    mol["name"] = filename.rsplit(".", 1)[0]
    return mol


def element_colors(mol: Dict) -> List[str]:
    return [CPK.get(e, "#B0B0B0") for e in mol["elements"]]


def radii(mol: Dict) -> np.ndarray:
    return np.array([VDW.get(e, 1.8) for e in mol["elements"]]) * 0.9
