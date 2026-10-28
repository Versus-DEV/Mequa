include(joinpath(@__DIR__, "MequaCore.jl"))
using .MequaCore

pts = synthetic_cells(2000)
labels, cents = kmeans_zones(pts, 3)
println("zone sizes: ", [count(==(c), labels) for c in 1:3])

ei, ej = build_knn_edges(pts, 6)
println("edges: ", length(ei))

pos, node_strain, edge_strain = relax_strain(pts, ei, ej)
println("mean node strain: ", sum(node_strain) / length(node_strain))
println("SPHI: ", sphi(node_strain, ei, ej))

seq = rand(1:3, 5000)
P = markov_estimate(seq, 3)
println("stationary: ", stationary_distribution(P))

x = randn(500)
y = Float64.(x .+ 0.3 .* randn(500) .> 0)
w, b = fit_logistic(x, y)
println("logistic fit w, b: ", w, ", ", b)

res = run_pipeline(pts, 3, 6, 0.05)
println("pipeline SPHI: ", res.sphi)
@time run_pipeline(pts, 3, 6, 0.05)