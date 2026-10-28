module MequaCore

using LinearAlgebra
using Random
using SparseArrays
using Base.Threads

export synthetic_cells, kmeans_zones, build_knn_edges, relax_strain, sphi,
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

function build_knn_edges(points::AbstractMatrix{Float64}, k::Int)
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

function relax_strain(points::AbstractMatrix{Float64}, ei::Vector{Int}, ej::Vector{Int};
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
