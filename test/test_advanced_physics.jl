using Test
using LinearAlgebra
using SparseArrays

include(joinpath(@__DIR__, "..", "src", "MirnanNew.jl"))
using .MirnanNew

@testset "advanced physics phase 15" begin
    @test isdefined(MirnanNew, :CliffordRotor)
    @test isdefined(MirnanNew, :construct_rotor)
    @test isdefined(MirnanNew, :apply_rotor)
    @test isdefined(MirnanNew.Physics.Generator, :GeneratorState)

    @testset "Clifford rotor projection" begin
        u = [1.0, 0.0, 0.0]
        v = [0.0, 1.0, 0.0]
        rotor = construct_rotor(u, v)
        rotated = apply_rotor(rotor, u)

        @test norm(rotated - v) < 1e-8
        @test norm(apply_rotor(rotor, [0.0, 0.0, 1.0]) - [0.0, 0.0, 1.0]) < 1e-8
        @test_throws DimensionMismatch construct_rotor([1.0, 0.0], [1.0, 0.0, 0.0])
        @test_throws DimensionMismatch apply_rotor(rotor, [1.0, 0.0])
    end

    @testset "deterministic opposite rotor" begin
        u = [1.0, 0.0, 0.0]
        v = [-1.0, 0.0, 0.0]
        r1 = construct_rotor(u, v)
        r2 = construct_rotor(u, v)

        @test r1.theta == r2.theta
        @test r1.e2 == r2.e2
        @test norm(apply_rotor(r1, u) - v) < 1e-8
    end

    @testset "runtime synaptic decay copy" begin
        K = spzeros(Float64, 3, 3)
        K[1, 2] = 10.0
        K[2, 3] = 5.0

        decayed = MirnanNew.Physics.Generator._apply_runtime_synaptic_decay(K)

        @test decayed[1, 2] == 9.8
        @test decayed[2, 3] == 4.9
        @test K[1, 2] == 10.0
        @test K[2, 3] == 5.0
        @test_throws ArgumentError MirnanNew.Physics.Generator._apply_runtime_synaptic_decay(K; factor=1.2)
    end
end
