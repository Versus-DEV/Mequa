module MequaCore

using LinearAlgebra
using Random
using SparseArrays
using Base.Threads

export synthetic_cells, kmeans_zones, build_knn_edges, build_knn_edges_bruteforce, relax_strain, relax_strain_edgewise, sphi,
       markov_estimate, stationary_distribution, sigmoid, logistic_failure,
       fit_logistic, run_pipeline

function synthetic_cells(n::Int; seed::Int=1, clusters::Int=3)
    rng = MersenneTwister(seed)
    centers = rand(rng, clusters, 2) .* 10
    pts = Matrix{Float64}(undef, n, 2)
    for i in 1:n
        c = rand(rng, 1:clusters)
        pts[i, 1] = centers[c, 1] + 0.8 * randn(rng)
        pts[i, 2] = centers[c, 2] + 0.8 * randn(rng)
    end
    return pts
end

function kmeans_zones(points::AbstractMatrix{Float64}, k::Int; iters::Int=100, seed::Int=1)
    n, d = size(points)
    rng = MersenneTwister(seed)
    centroids = Matrix{Float64}(points[randperm(rng, n)[1:k], :])
    labels = zeros(Int, n)
    for _ in 1:iters
        changed = Threads.Atomic{Int}(0)
        @threads for i in 1:n
            best = 1
            bestd = Inf
            for c in 1:k
                s = 0.0
                for j in 1:d
                    diff = points[i, j] - centroids[c, j]
                    s += diff * diff
                end
                if s < bestd
                    bestd = s
                    best = c
                end
            end
            if labels[i] != best
                labels[i] = best
                Threads.atomic_add!(changed, 1)
            end
        end
        changed[] == 0 && break
        sums = zeros(k, d)
        counts = zeros(Int, k)
        for i in 1:n
            c = labels[i]
            counts[c] += 1
            for j in 1:d
                sums[c, j] += points[i, j]
            end
        end
        for c in 1:k
            if counts[c] > 0
                for j in 1:d
                    centroids[c, j] = sums[c, j] / counts[c]
                end
            end
        end
    end
    return labels, centroids
end

function build_knn_edges_bruteforce(points::AbstractMatrix{Float64}, k::Int)
    n, d = size(points)
    nbrs = Vector{Vector{Int}}(undef, n)
    @threads for i in 1:n
        dists = Vector{Float64}(undef, n)
        for m in 1:n
            s = 0.0
            for j in 1:d
                diff = points[i, j] - points[m, j]
                s += diff * diff
            end
            dists[m] = s
        end
        dists[i] = Inf
        nbrs[i] = partialsortperm(dists, 1:k)
    end
    edges = Set{Tuple{Int,Int}}()
    for i in 1:n
        for m in nbrs[i]
            push!(edges, i < m ? (i, m) : (m, i))
        end
    end
    sorted = sort!(collect(edges))
    ei = [e[1] for e in sorted]
    ej = [e[2] for e in sorted]
    return ei, ej
end

function build_knn_edges(points::AbstractMatrix{Float64}, k::Int)
    n, d = size(points)
    d == 2 || return build_knn_edges_bruteforce(points, k)
    minx, maxx = extrema(view(points, :, 1))
    miny, maxy = extrema(view(points, :, 2))
    w = max(maxx - minx, 1e-12)
    h = max(maxy - miny, 1e-12)
    cell = max(sqrt(w * h * 3.0 / n), 1e-12)
    gx = 1
    gy = 1
    while true
        gx = max(floor(Int, w / cell) + 1, 1)
        gy = max(floor(Int, h / cell) + 1, 1)
        gx * gy <= 4 * n + 16 && break
        cell *= 1.5
    end
    cx = Vector{Int}(undef, n)
    cy = Vector{Int}(undef, n)
    for i in 1:n
        cx[i] = min(floor(Int, (points[i, 1] - minx) / cell), gx - 1)
        cy[i] = min(floor(Int, (points[i, 2] - miny) / cell), gy - 1)
    end
    ncell = gx * gy
    start = zeros(Int, ncell + 1)
    for i in 1:n
        start[cy[i] * gx + cx[i] + 2] += 1
    end
    start[1] = 1
    for c in 1:ncell
        start[c + 1] += start[c]
    end
    fillpos = copy(start)
    order = Vector{Int}(undef, n)
    for i in 1:n
        c = cy[i] * gx + cx[i] + 1
        order[fillpos[c]] = i
        fillpos[c] += 1
    end
    maxr = max(gx, gy)
    nbrs = Vector{Vector{Int}}(undef, n)
    @threads for i in 1:n
        bd = Vector{Float64}(undef, k)
        bi = Vector{Int}(undef, k)
        cnt = 0
        px = points[i, 1]
        py = points[i, 2]
        for r in 0:maxr
            for yy in (cy[i] - r):(cy[i] + r)
                (yy < 0 || yy >= gy) && continue
                full = (yy == cy[i] - r) || (yy == cy[i] + r)
                step = full ? 1 : max(2 * r, 1)
                xx = cx[i] - r
                while xx <= cx[i] + r
                    if xx >= 0 && xx < gx
                        c = yy * gx + xx + 1
                        for t in start[c]:(start[c + 1] - 1)
                            j = order[t]
                            j == i && continue
                            dx = points[j, 1] - px
                            dy = points[j, 2] - py
                            dd = dx * dx + dy * dy
                            if cnt < k || dd < bd[k]
                                pos = cnt < k ? cnt + 1 : k
                                while pos > 1 && bd[pos - 1] > dd
                                    bd[pos] = bd[pos - 1]
                                    bi[pos] = bi[pos - 1]
                                    pos -= 1
                                end
                                bd[pos] = dd
                                bi[pos] = j
                                cnt < k && (cnt += 1)
                            end
                        end
                    end
                    xx += step
                end
            end
            if cnt == k
                bound = r * cell
                bd[k] <= bound * bound && break
            end
        end
        nbrs[i] = copy(bi)
    end
    pairs = Tuple{Int,Int}[]
    sizehint!(pairs, n * k)
    for i in 1:n
        for m in nbrs[i]
            push!(pairs, i < m ? (i, m) : (m, i))
        end
    end
    sort!(pairs)
    unique!(pairs)
    ei = [e[1] for e in pairs]
    ej = [e[2] for e in pairs]
    return ei, ej
end

function relax_strain_edgewise(points::AbstractMatrix{Float64}, ei::Vector{Int}, ej::Vector{Int};
                      load::Float64=0.05, band::Float64=0.1, iters::Int=1500, step::Float64=0.05)
    n, d = size(points)
    m = length(ei)
    rest = Vector{Float64}(undef, m)
    for e in 1:m
        s = 0.0
        for j in 1:d
            diff = points[ei[e], j] - points[ej[e], j]
            s += diff * diff
        end
        rest[e] = max(sqrt(s), 1e-9)
    end
    lo, hi = extrema(view(points, :, 1))
    width = hi - lo
    fixed = falses(n)
    pulled = falses(n)
    for i in 1:n
        if points[i, 1] <= lo + band * width
            fixed[i] = true
        elseif points[i, 1] >= hi - band * width
            pulled[i] = true
        end
    end
    pos = Matrix{Float64}(points)
    for i in 1:n
        if pulled[i]
            pos[i, 1] += load * width
        end
    end
    force = zeros(n, d)
    for _ in 1:iters
        fill!(force, 0.0)
        for e in 1:m
            a = ei[e]
            b = ej[e]
            s = 0.0
            for j in 1:d
                diff = pos[b, j] - pos[a, j]
                s += diff * diff
            end
            len = sqrt(s)
            if len > 0
                f = (len - rest[e]) / len
                for j in 1:d
                    fj = f * (pos[b, j] - pos[a, j])
                    force[a, j] += fj
                    force[b, j] -= fj
                end
            end
        end
        for i in 1:n
            if !fixed[i] && !pulled[i]
                for j in 1:d
                    pos[i, j] += step * force[i, j]
                end
            end
        end
    end
    node_sum = zeros(n)
    node_cnt = zeros(Int, n)
    edge_strain = Vector{Float64}(undef, m)
    for e in 1:m
        a = ei[e]
        b = ej[e]
        s = 0.0
        for j in 1:d
            diff = pos[b, j] - pos[a, j]
            s += diff * diff
        end
        edge_strain[e] = abs(sqrt(s) - rest[e]) / rest[e]
        node_sum[a] += edge_strain[e]
        node_sum[b] += edge_strain[e]
        node_cnt[a] += 1
        node_cnt[b] += 1
    end
    node_strain = [node_cnt[i] > 0 ? node_sum[i] / node_cnt[i] : 0.0 for i in 1:n]
    return pos, node_strain, edge_strain
end

@inline function node_force!(fx, fy, i, px, py, astart, anbr, aedge, rest)
    sx = 0.0
    sy = 0.0
    xi = px[i]
    yi = py[i]
    for t in astart[i]:(astart[i + 1] - 1)
        j = anbr[t]
        dx = px[j] - xi
        dy = py[j] - yi
        len = sqrt(dx * dx + dy * dy)
        if len > 0
            f = (len - rest[aedge[t]]) / len
            sx += f * dx
            sy += f * dy
        end
    end
    fx[i] = sx
    fy[i] = sy
    return nothing
end

function relax_strain(points::AbstractMatrix{Float64}, ei::Vector{Int}, ej::Vector{Int};
                      load::Float64=0.05, band::Float64=0.1, iters::Int=1500, step::Float64=0.05)
    n, d = size(points)
    d == 2 || return relax_strain_edgewise(points, ei, ej; load=load, band=band, iters=iters, step=step)
    m = length(ei)
    rest = Vector{Float64}(undef, m)
    for e in 1:m
        dx = points[ei[e], 1] - points[ej[e], 1]
        dy = points[ei[e], 2] - points[ej[e], 2]
        rest[e] = max(sqrt(dx * dx + dy * dy), 1e-9)
    end
    lo, hi = extrema(view(points, :, 1))
    width = hi - lo
    state = zeros(Int8, n)
    for i in 1:n
        if points[i, 1] <= lo + band * width
            state[i] = 1
        elseif points[i, 1] >= hi - band * width
            state[i] = 2
        end
    end
    px = Vector{Float64}(points[:, 1])
    py = Vector{Float64}(points[:, 2])
    for i in 1:n
        if state[i] == 2
            px[i] += load * width
        end
    end
    astart = zeros(Int, n + 1)
    for e in 1:m
        astart[ei[e] + 1] += 1
        astart[ej[e] + 1] += 1
    end
    astart[1] = 1
    for i in 1:n
        astart[i + 1] += astart[i]
    end
    fillpos = copy(astart)
    anbr = Vector{Int}(undef, 2 * m)
    aedge = Vector{Int}(undef, 2 * m)
    for e in 1:m
        a = ei[e]
        b = ej[e]
        anbr[fillpos[a]] = b
        aedge[fillpos[a]] = e
        fillpos[a] += 1
        anbr[fillpos[b]] = a
        aedge[fillpos[b]] = e
        fillpos[b] += 1
    end
    fx = zeros(n)
    fy = zeros(n)
    parallel = n >= 5000 && Threads.nthreads() > 1
    for _ in 1:iters
        if parallel
            @threads for i in 1:n
                node_force!(fx, fy, i, px, py, astart, anbr, aedge, rest)
            end
        else
            for i in 1:n
                node_force!(fx, fy, i, px, py, astart, anbr, aedge, rest)
            end
        end
        for i in 1:n
            if state[i] == 0
                px[i] += step * fx[i]
                py[i] += step * fy[i]
            end
        end
    end
    node_sum = zeros(n)
    node_cnt = zeros(Int, n)
    edge_strain = Vector{Float64}(undef, m)
    for e in 1:m
        a = ei[e]
        b = ej[e]
        dx = px[b] - px[a]
        dy = py[b] - py[a]
        edge_strain[e] = abs(sqrt(dx * dx + dy * dy) - rest[e]) / rest[e]
        node_sum[a] += edge_strain[e]
        node_sum[b] += edge_strain[e]
        node_cnt[a] += 1
        node_cnt[b] += 1
    end
    node_strain = [node_cnt[i] > 0 ? node_sum[i] / node_cnt[i] : 0.0 for i in 1:n]
    return hcat(px, py), node_strain, edge_strain
end

function sphi(node_strain::Vector{Float64}, ei::Vector{Int}, ej::Vector{Int};
              iters::Int=300, tol::Float64=1e-10)
    n = length(node_strain)
    w = [(node_strain[ei[e]] + node_strain[ej[e]]) / 2 for e in eachindex(ei)]
    W = sparse(vcat(ei, ej), vcat(ej, ei), vcat(w, w), n, n)
    v = fill(1.0 / sqrt(n), n)
    lam = 0.0
    for _ in 1:iters
        wv = W * v
        nrm = norm(wv)
        nrm == 0 && return 0.0
        v = wv ./ nrm
        newlam = dot(v, W * v)
        abs(newlam - lam) < tol && return newlam
        lam = newlam
    end
    return lam
end

function markov_estimate(seq::AbstractVector{Int}, nstates::Int)
    counts = zeros(nstates, nstates)
    for t in 1:(length(seq) - 1)
        counts[seq[t], seq[t + 1]] += 1
    end
    P = zeros(nstates, nstates)
    for i in 1:nstates
        s = sum(counts[i, :])
        if s > 0
            P[i, :] = counts[i, :] ./ s
        else
            P[i, i] = 1.0
        end
    end
    return P
end

function stationary_distribution(P::Matrix{Float64}; iters::Int=2000)
    n = size(P, 1)
    dist = fill(1.0 / n, n)
    for _ in 1:iters
        dist = P' * dist
    end
    return Vector{Float64}(dist)
end

sigmoid(x) = 1.0 / (1.0 + exp(-x))

function logistic_failure(strain::AbstractVector{Float64}, s_crit::Float64, width::Float64)
    return [sigmoid((s - s_crit) / width) for s in strain]
end

function fit_logistic(x::AbstractVector{Float64}, y::AbstractVector{Float64};
                      lr::Float64=0.5, iters::Int=3000)
    w = 0.0
    b = 0.0
    n = length(x)
    for _ in 1:iters
        gw = 0.0
        gb = 0.0
        for i in 1:n
            p = sigmoid(w * x[i] + b)
            gw += (p - y[i]) * x[i]
            gb += p - y[i]
        end
        w -= lr * gw / n
        b -= lr * gb / n
    end
    return w, b
end

function run_pipeline(points::AbstractMatrix{Float64}, k::Int, knn::Int, load::Float64)
    pts = Matrix{Float64}(points)
    labels, centroids = kmeans_zones(pts, k)
    ei, ej = build_knn_edges(pts, knn)
    pos, node_strain, edge_strain = relax_strain(pts, ei, ej; load=load)
    index = sphi(node_strain, ei, ej)
    s_crit = quantile_value(node_strain, 0.9)
    failure = logistic_failure(node_strain, s_crit, max(s_crit * 0.25, 1e-9))
    return (labels=labels, centroids=centroids, positions=pos, node_strain=node_strain,
            sphi=index, failure_prob=failure, s_crit=s_crit, n_edges=length(ei))
end

function quantile_value(v::AbstractVector{Float64}, q::Float64)
    s = sort(v)
    idx = clamp(ceil(Int, q * length(s)), 1, length(s))
    return s[idx]
end

end
