module MequaBench

using Main.MequaCore

export stage_times

function stage_times(points::Matrix{Float64}, k::Int, knn::Int, load::Float64)
    t0 = time_ns()
    labels, cents = kmeans_zones(points, k)
    t1 = time_ns()
    ei, ej = build_knn_edges(points, knn)
    t2 = time_ns()
    pos, node_strain, edge_strain = relax_strain(points, ei, ej; load=load)
    t3 = time_ns()
    value = sphi(node_strain, ei, ej)
    t4 = time_ns()
    return (kmeans=(t1 - t0) / 1e9, knn=(t2 - t1) / 1e9, relax=(t3 - t2) / 1e9,
            sphi_seconds=(t4 - t3) / 1e9, sphi_value=value)
end

end
