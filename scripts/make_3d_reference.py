import numpy as np

from mequa import numpy_backend as nb

pts = nb.synthetic_cells(300, seed=7, clusters=1, dim=3)
ei, ej = nb.build_knn_edges(pts, 6)
pos, ns, _ = nb.relax_strain(pts, ei, ej, load=0.05, iters=200)
d = "data/examples/"
np.savetxt(d + "ref3d_points.txt", pts, fmt="%.17g")
np.savetxt(d + "ref3d_edges.txt", np.column_stack([ei, ej]), fmt="%d")
np.savetxt(d + "ref3d_pos.txt", pos, fmt="%.17g")
np.savetxt(d + "ref3d_node_strain.txt", ns, fmt="%.17g")
print("wrote reference files for", len(pts), "cells,", len(ei), "edges")
