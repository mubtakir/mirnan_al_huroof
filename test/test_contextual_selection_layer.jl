using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test
using LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "MirnanNew.jl"))
using .MirnanNew

@testset "compact phase and contextual selection layer" begin
    wp = MirnanNew.Physics.WordPhysics

    v = wp.compute_compact_phase_vector("علم")
    @test length(v) == wp.COMPACT_PHASE_DIM
    @test wp.COMPACT_PHASE_DIM == wp.ENHANCED_DIM + 5
    @test isapprox(norm(v), 1.0; atol=1e-6)

    a = wp.artificial_letter_vector('ع')
    b = wp.artificial_letter_vector('ل')
    @test length(a) == 5
    @test a != b

    sim = wp.compact_phase_similarity(
        wp.compute_compact_phase_vector("علم"),
        wp.compute_compact_phase_vector("فهم"),
    )
    @test -1.0 <= sim <= 1.0

    @test MirnanNew.Physics.Generator._context_selection_enabled()
end
