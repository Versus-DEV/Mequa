import numpy as np

import mequa_rs


def main():
    pts = np.random.default_rng(0).random((1000, 2))
    same = mequa_rs.buffer_address(pts) == pts.ctypes.data
    print(f"NumPy array: Rust sees the same memory address as NumPy: {same}")
    try:
        import torch

        t = torch.rand(1000, 2, dtype=torch.float64)
        view = t.numpy()
        same_t = mequa_rs.buffer_address(view) == t.data_ptr()
        print(f"PyTorch CPU tensor: Rust sees the same memory address as the tensor: {same_t}")
    except ImportError:
        print("PyTorch not installed; skipped tensor check")
    try:
        mequa_rs.buffer_address(pts.astype(np.float32))
        print("float32 input: accepted (unexpected)")
    except TypeError:
        print("float32 input: rejected instead of silently copied")
    fort = np.asfortranarray(pts)
    try:
        mequa_rs.knn_edges(fort, 6)
        print("Fortran-ordered input: accepted (unexpected)")
    except ValueError:
        print("Fortran-ordered input: rejected instead of silently copied")
    print(f"Rust worker threads: {mequa_rs.num_threads()}")
    print("Scope: CPU memory only. GPU (CUDA) handles are not supported on this machine.")


if __name__ == "__main__":
    main()
