using DelimitedFiles
include(joinpath(@__DIR__, "MequaCore3D.jl"))
using .MequaCore3D

dir = joinpath(@__DIR__, "..", "data", "examples")
points = readdlm(joinpath(dir, "ref3d_points.txt"))
edges = Int.(readdlm(joinpath(dir, "ref3d_edges.txt"))) .+ 1
expected_pos = readdlm(joinpath(dir, "ref3d_pos.txt"))
expected_strain = vec(readdlm(joinpath(dir, "ref3d_node_strain.txt")))

pos, node_strain, _ = relax_strain_nd(Matrix{Float64}(points), vec(edges[:, 1]), vec(edges[:, 2]); load=0.05, iters=200)
dpos = maximum(abs.(pos .- expected_pos))
dstr = maximum(abs.(node_strain .- expected_strain))
println("max |position diff| vs NumPy = ", dpos)
println("max |strain diff|   vs NumPy = ", dstr)
if dpos < 1e-9 && dstr < 1e-9
    println("PASS: Julia 3D kernel matches the NumPy reference")
else
    println("FAIL")
    exit(1)
end
