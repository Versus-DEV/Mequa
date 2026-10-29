import numpy as np

from mequa import numpy_backend as nb


def main():
    pts = nb.synthetic_cells(2000)
    results = {"numpy": nb.analyze(pts, 3, 6, 0.05)}
    try:
        from mequa import rust_backend as rb

        results["rust"] = rb.analyze(pts, 3, 6, 0.05)
    except ImportError as exc:
        print(f"rust unavailable: {exc}")
    try:
        from mequa import julia_bridge as jb

        results["julia"] = jb.analyze(pts, 3, 6, 0.05)
    except Exception as exc:
        print(f"julia unavailable: {type(exc).__name__}: {exc}")
    ref = results["numpy"]
    print(f"{'backend':8} {'SPHI':>20} {'rel diff vs numpy':>20} {'max node strain diff':>22} {'edges':>7}")
    for name, r in results.items():
        diff = abs(r["sphi"] - ref["sphi"]) / abs(ref["sphi"])
        sdiff = float(np.max(np.abs(np.asarray(r["node_strain"]) - ref["node_strain"])))
        print(f"{name:8} {r['sphi']:20.12f} {diff:20.2e} {sdiff:22.2e} {r['n_edges']:7d}")


if __name__ == "__main__":
    main()
