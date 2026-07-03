using Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "src", "MirnanNew.jl"))
using .MirnanNew

words = ["العلم", "نور", "السماء", "hello", "world"]
for w in words
    pv = MirnanNew.Physics.WordPhysics.compute_extended_phase_vector(w)
    n = sum(abs2, pv)
    println("'$w' → dim=$(length(pv)), norm²=$n, first3=$(pv[1:3])")
end
