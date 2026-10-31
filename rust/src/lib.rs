use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 5000;

fn check_points(points: &PyReadonlyArray2<f64>) -> PyResult<(usize, usize)> {
    let shape = points.shape();
    if shape[1] != 2 && shape[1] != 3 {
        return Err(PyValueError::new_err("points must have shape (n, 2) or (n, 3)"));
    }
    if shape[0] < 2 {
        return Err(PyValueError::new_err("need at least 2 points"));
    }
    if !points.is_c_contiguous() {
        return Err(PyValueError::new_err("points must be C-contiguous float64; use numpy.ascontiguousarray"));
    }
    Ok((shape[0], shape[1]))
}

fn grid_knn(pts: &[f64], n: usize, k: usize) -> Vec<(i64, i64)> {
    let (mut minx, mut maxx, mut miny, mut maxy) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..n {
        let x = pts[2 * i];
        let y = pts[2 * i + 1];
        minx = minx.min(x);
        maxx = maxx.max(x);
        miny = miny.min(y);
        maxy = maxy.max(y);
    }
    let w = (maxx - minx).max(1e-12);
    let h = (maxy - miny).max(1e-12);
    let mut cell = ((w * h * 3.0 / n as f64).sqrt()).max(1e-12);
    let (mut gx, mut gy);
    loop {
        gx = ((w / cell).floor() as usize + 1).max(1);
        gy = ((h / cell).floor() as usize + 1).max(1);
        if gx * gy <= 4 * n + 16 {
            break;
        }
        cell *= 1.5;
    }
    let cell_of = |i: usize| -> (usize, usize) {
        let cx = (((pts[2 * i] - minx) / cell) as usize).min(gx - 1);
        let cy = (((pts[2 * i + 1] - miny) / cell) as usize).min(gy - 1);
        (cx, cy)
    };
    let mut start = vec![0usize; gx * gy + 1];
    for i in 0..n {
        let (cx, cy) = cell_of(i);
        start[cy * gx + cx + 1] += 1;
    }
    for c in 0..gx * gy {
        start[c + 1] += start[c];
    }
    let mut fill = start.clone();
    let mut order = vec![0usize; n];
    for i in 0..n {
        let (cx, cy) = cell_of(i);
        let c = cy * gx + cx;
        order[fill[c]] = i;
        fill[c] += 1;
    }
    let max_r = gx.max(gy);
    let mut res = vec![0usize; n * k];
    let work = |i: usize, out: &mut [usize]| {
        let (cx, cy) = cell_of(i);
        let px = pts[2 * i];
        let py = pts[2 * i + 1];
        let mut best: Vec<(f64, usize)> = Vec::with_capacity(k + 1);
        for r in 0..=max_r {
            let r_i = r as i64;
            let x0 = cx as i64 - r_i;
            let x1 = cx as i64 + r_i;
            let y0 = cy as i64 - r_i;
            let y1 = cy as i64 + r_i;
            for yy in y0..=y1 {
                if yy < 0 || yy >= gy as i64 {
                    continue;
                }
                let edge_row = yy == y0 || yy == y1;
                let mut xx = x0;
                while xx <= x1 {
                    if xx >= 0 && xx < gx as i64 {
                        let c = yy as usize * gx + xx as usize;
                        for &j in &order[start[c]..start[c + 1]] {
                            if j == i {
                                continue;
                            }
                            let dx = pts[2 * j] - px;
                            let dy = pts[2 * j + 1] - py;
                            let d = dx * dx + dy * dy;
                            if best.len() < k || d < best[best.len() - 1].0 {
                                let pos = best.partition_point(|b| b.0 <= d);
                                best.insert(pos, (d, j));
                                if best.len() > k {
                                    best.pop();
                                }
                            }
                        }
                    }
                    if edge_row || r == 0 {
                        xx += 1;
                    } else {
                        xx += (x1 - x0).max(1);
                    }
                }
            }
            if best.len() == k {
                let bound = r as f64 * cell;
                if best[k - 1].0 <= bound * bound {
                    break;
                }
            }
        }
        for (t, b) in best.iter().enumerate() {
            out[t] = b.1;
        }
    };
    if n >= PAR_THRESHOLD {
        res.par_chunks_mut(k).enumerate().for_each(|(i, out)| work(i, out));
    } else {
        for (i, out) in res.chunks_mut(k).enumerate() {
            work(i, out);
        }
    }
    let mut pairs: Vec<(i64, i64)> = Vec::with_capacity(n * k);
    for i in 0..n {
        for t in 0..k {
            let j = res[i * k + t];
            if j != i {
                let (a, b) = if i < j { (i, j) } else { (j, i) };
                pairs.push((a as i64, b as i64));
            }
        }
    }
    pairs.par_sort_unstable();
    pairs.dedup();
    pairs
}

struct Csr {
    start: Vec<usize>,
    nbr: Vec<usize>,
    edge: Vec<usize>,
}

fn build_csr(n: usize, ei: &[i64], ej: &[i64]) -> Csr {
    let m = ei.len();
    let mut start = vec![0usize; n + 1];
    for e in 0..m {
        start[ei[e] as usize + 1] += 1;
        start[ej[e] as usize + 1] += 1;
    }
    for i in 0..n {
        start[i + 1] += start[i];
    }
    let mut fill = start.clone();
    let mut nbr = vec![0usize; 2 * m];
    let mut edge = vec![0usize; 2 * m];
    for e in 0..m {
        let a = ei[e] as usize;
        let b = ej[e] as usize;
        nbr[fill[a]] = b;
        edge[fill[a]] = e;
        fill[a] += 1;
        nbr[fill[b]] = a;
        edge[fill[b]] = e;
        fill[b] += 1;
    }
    Csr { start, nbr, edge }
}

#[allow(clippy::too_many_arguments)]
fn relax(
    pts: &[f64],
    n: usize,
    d: usize,
    ei: &[i64],
    ej: &[i64],
    load: f64,
    band: f64,
    iters: usize,
    step: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let m = ei.len();
    let dist = |p: &[f64], a: usize, b: usize| -> f64 {
        let mut s = 0.0;
        for c in 0..d {
            let t = p[d * a + c] - p[d * b + c];
            s += t * t;
        }
        s.sqrt()
    };
    let rest: Vec<f64> = (0..m).map(|e| dist(pts, ei[e] as usize, ej[e] as usize).max(1e-9)).collect();
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..n {
        lo = lo.min(pts[d * i]);
        hi = hi.max(pts[d * i]);
    }
    let width = hi - lo;
    let mut state = vec![0u8; n];
    for i in 0..n {
        let x = pts[d * i];
        if x <= lo + band * width {
            state[i] = 1;
        } else if x >= hi - band * width {
            state[i] = 2;
        }
    }
    let mut pos = pts.to_vec();
    for i in 0..n {
        if state[i] == 2 {
            pos[d * i] += load * width;
        }
    }
    let csr = build_csr(n, ei, ej);
    let mut force = vec![0.0f64; d * n];
    let parallel = n >= PAR_THRESHOLD;
    for _ in 0..iters {
        {
            let pos_ref = &pos;
            let csr_ref = &csr;
            let rest_ref = &rest;
            let node_force = |i: usize, f: &mut [f64]| {
                let mut acc = [0.0f64; 3];
                let mut dv = [0.0f64; 3];
                for t in csr_ref.start[i]..csr_ref.start[i + 1] {
                    let j = csr_ref.nbr[t];
                    let mut len2 = 0.0;
                    for c in 0..d {
                        dv[c] = pos_ref[d * j + c] - pos_ref[d * i + c];
                        len2 += dv[c] * dv[c];
                    }
                    let len = len2.sqrt();
                    if len > 0.0 {
                        let s = (len - rest_ref[csr_ref.edge[t]]) / len;
                        for c in 0..d {
                            acc[c] += s * dv[c];
                        }
                    }
                }
                f[..d].copy_from_slice(&acc[..d]);
            };
            if parallel {
                force.par_chunks_mut(d).enumerate().for_each(|(i, f)| node_force(i, f));
            } else {
                for (i, f) in force.chunks_mut(d).enumerate() {
                    node_force(i, f);
                }
            }
        }
        for i in 0..n {
            if state[i] == 0 {
                for c in 0..d {
                    pos[d * i + c] += step * force[d * i + c];
                }
            }
        }
    }
    let mut edge_strain = vec![0.0f64; m];
    let mut node_sum = vec![0.0f64; n];
    let mut node_cnt = vec![0usize; n];
    for e in 0..m {
        let a = ei[e] as usize;
        let b = ej[e] as usize;
        let s = (dist(&pos, a, b) - rest[e]).abs() / rest[e];
        edge_strain[e] = s;
        node_sum[a] += s;
        node_sum[b] += s;
        node_cnt[a] += 1;
        node_cnt[b] += 1;
    }
    let node_strain: Vec<f64> = (0..n)
        .map(|i| if node_cnt[i] > 0 { node_sum[i] / node_cnt[i] as f64 } else { 0.0 })
        .collect();
    (pos, node_strain, edge_strain)
}

fn sphi_value(node_strain: &[f64], ei: &[i64], ej: &[i64], iters: usize, tol: f64) -> f64 {
    let n = node_strain.len();
    let csr = build_csr(n, ei, ej);
    let w: Vec<f64> = (0..ei.len())
        .map(|e| (node_strain[ei[e] as usize] + node_strain[ej[e] as usize]) / 2.0)
        .collect();
    let matvec = |v: &[f64]| -> Vec<f64> {
        let row = |i: usize| -> f64 {
            let mut s = 0.0;
            for t in csr.start[i]..csr.start[i + 1] {
                s += w[csr.edge[t]] * v[csr.nbr[t]];
            }
            s
        };
        if n >= PAR_THRESHOLD {
            (0..n).into_par_iter().map(row).collect()
        } else {
            (0..n).map(row).collect()
        }
    };
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut lam = 0.0;
    for _ in 0..iters {
        let wv = matvec(&v);
        let nrm = wv.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm == 0.0 {
            return 0.0;
        }
        for i in 0..n {
            v[i] = wv[i] / nrm;
        }
        let w2 = matvec(&v);
        let newlam: f64 = v.iter().zip(w2.iter()).map(|(a, b)| a * b).sum();
        if (newlam - lam).abs() < tol {
            return newlam;
        }
        lam = newlam;
    }
    lam
}

#[pyfunction]
fn buffer_address(points: PyReadonlyArray2<f64>) -> usize {
    points.as_array().as_ptr() as usize
}

#[pyfunction]
fn num_threads() -> usize {
    rayon::current_num_threads()
}

#[pyfunction]
fn knn_edges<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    k: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let (n, d) = check_points(&points)?;
    if d != 2 {
        return Err(PyValueError::new_err("knn_edges supports 2D only; for 3D build edges with scipy cKDTree and call relax_strain"));
    }
    if k == 0 || k >= n {
        return Err(PyValueError::new_err("k must satisfy 1 <= k < n"));
    }
    let data = points
        .as_slice()
        .map_err(|_| PyValueError::new_err("points must be C-contiguous float64; use numpy.ascontiguousarray"))?;
    let pairs = py.detach(|| grid_knn(data, n, k));
    let ei: Vec<i64> = pairs.iter().map(|p| p.0).collect();
    let ej: Vec<i64> = pairs.iter().map(|p| p.1).collect();
    Ok((ei.into_pyarray(py), ej.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (points, ei, ej, load=0.05, band=0.1, iters=1500, step=0.05))]
#[allow(clippy::too_many_arguments)]
fn relax_strain<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    ei: PyReadonlyArray1<'py, i64>,
    ej: PyReadonlyArray1<'py, i64>,
    load: f64,
    band: f64,
    iters: usize,
    step: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (n, d) = check_points(&points)?;
    let data = points
        .as_slice()
        .map_err(|_| PyValueError::new_err("points must be C-contiguous float64"))?;
    let a = ei.as_slice().map_err(|_| PyValueError::new_err("ei must be contiguous int64"))?;
    let b = ej.as_slice().map_err(|_| PyValueError::new_err("ej must be contiguous int64"))?;
    if a.len() != b.len() {
        return Err(PyValueError::new_err("ei and ej must have the same length"));
    }
    if a.iter().chain(b.iter()).any(|&v| v < 0 || v as usize >= n) {
        return Err(PyValueError::new_err("edge index out of range"));
    }
    let (pos, node_strain, edge_strain) = py.detach(|| relax(data, n, d, a, b, load, band, iters, step));
    let pos_arr = numpy::ndarray::Array2::from_shape_vec((n, d), pos)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((pos_arr.into_pyarray(py), node_strain.into_pyarray(py), edge_strain.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (node_strain, ei, ej, iters=300, tol=1e-10))]
fn sphi(
    py: Python<'_>,
    node_strain: PyReadonlyArray1<'_, f64>,
    ei: PyReadonlyArray1<'_, i64>,
    ej: PyReadonlyArray1<'_, i64>,
    iters: usize,
    tol: f64,
) -> PyResult<f64> {
    let s = node_strain.as_slice().map_err(|_| PyValueError::new_err("node_strain must be contiguous"))?;
    let a = ei.as_slice().map_err(|_| PyValueError::new_err("ei must be contiguous"))?;
    let b = ej.as_slice().map_err(|_| PyValueError::new_err("ej must be contiguous"))?;
    if a.iter().chain(b.iter()).any(|&v| v < 0 || v as usize >= s.len()) {
        return Err(PyValueError::new_err("edge index out of range"));
    }
    Ok(py.detach(|| sphi_value(s, a, b, iters, tol)))
}

#[pymodule]
fn mequa_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(buffer_address, m)?)?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(knn_edges, m)?)?;
    m.add_function(wrap_pyfunction!(relax_strain, m)?)?;
    m.add_function(wrap_pyfunction!(sphi, m)?)?;
    Ok(())
}
