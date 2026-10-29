include(joinpath(@__DIR__, "MequaCore.jl"))
using .MequaCore
using Random

function check(points, k, label)
    a = build_knn_edges(points, k)
    b = build_knn_edges_bruteforce(points, k)
    ok = a[1] == b[1] && a[2] == b[2]
    println(label, ": grid == bruteforce: ", ok, " (", length(a[1]), " edges)")
    ok || error("mismatch for " * label)
end

check(synthetic_cells(2000), 6, "synthetic 2000")
check(synthetic_cells(777; seed=4, clusters=5), 4, "synthetic 777")
rng = MersenneTwister(11)
check(rand(rng, 1500, 2), 8, "uniform 1500")
check(hcat(rand(rng, 600), fill(0.5, 600)), 5, "collinear 600")
check(rand(rng, 300, 3), 5, "3-dimensional fallback")
pts = synthetic_cells(10000)
build_knn_edges(pts, 6)
println("grid 10000: ", @elapsed build_knn_edges(pts, 6))
println("brute 10000: ", @elapsed build_knn_edges_bruteforce(pts, 6))

function check_relax(points, k, label)
    ei, ej = build_knn_edges(points, k)
    p1, s1, e1 = relax_strain(points, ei, ej; load=0.05)
    p2, s2, e2 = relax_strain_edgewise(points, ei, ej; load=0.05)
    dstrain = maximum(abs.(s1 .- s2))
    dpos = maximum(abs.(p1 .- p2))
    ds = abs(sphi(s1, ei, ej) - sphi(s2, ei, ej)) / sphi(s2, ei, ej)
    println(label, ": max strain diff ", dstrain, ", max position diff ", dpos, ", SPHI rel diff ", ds)
    (dstrain < 1e-9 && dpos < 1e-9 && ds < 1e-9) || error("relaxation mismatch for " * label)
end

check_relax(synthetic_cells(2000), 6, "relax serial path 2000")
check_relax(synthetic_cells(6000), 6, "relax threaded path 6000")
println("threads: ", Threads.nthreads())
big = synthetic_cells(20000)
ei, ej = build_knn_edges(big, 6)
relax_strain(big, ei, ej)
println("relax node-wise 20000: ", @elapsed relax_strain(big, ei, ej))
println("relax edgewise 20000: ", @elapsed relax_strain_edgewise(big, ei, ej))
