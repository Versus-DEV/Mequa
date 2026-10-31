module MequaCore3D

using LinearAlgebra

export relax_strain_nd

function relax_strain_nd(points::AbstractMatrix{Float64}, ei::Vector{Int}, ej::Vector{Int};
                         load::Float64=0.05, band::Float64=0.1, iters::Int=1500, step::Float64=0.05)
    n, d = size(points)
    m = length(ei)
    rest = [max(norm(points[ei[e], :] - points[ej[e], :]), 1e-9) for e in 1:m]
    x = points[:, 1]
    lo, hi = minimum(x), maximum(x)
    width = hi - lo
    fixed = x .<= lo + band * width
    pulled = (.!fixed) .& (x .>= hi - band * width)
    free = .!(fixed .| pulled)
    pos = copy(points)
    pos[pulled, 1] .+= load * width
    force = zeros(n, d)
    for _ in 1:iters
        fill!(force, 0.0)
        for e in 1:m
            i, j = ei[e], ej[e]
            len2 = 0.0
            for c in 1:d
                len2 += (pos[j, c] - pos[i, c])^2
            end
            len = sqrt(len2)
            if len > 0
                s = (len - rest[e]) / len
                for c in 1:d
                    fv = (pos[j, c] - pos[i, c]) * s
                    force[i, c] += fv
                    force[j, c] -= fv
                end
            end
        end
        for i in 1:n
            if free[i]
                for c in 1:d
                    pos[i, c] += step * force[i, c]
                end
            end
        end
    end
    edge_strain = zeros(m)
    node_sum = zeros(n)
    node_cnt = zeros(Int, n)
    for e in 1:m
        i, j = ei[e], ej[e]
        len = norm(pos[j, :] - pos[i, :])
        s = abs(len - rest[e]) / rest[e]
        edge_strain[e] = s
        node_sum[i] += s
        node_sum[j] += s
        node_cnt[i] += 1
        node_cnt[j] += 1
    end
    node_strain = [node_cnt[i] > 0 ? node_sum[i] / node_cnt[i] : 0.0 for i in 1:n]
    return pos, node_strain, edge_strain
end

end
