using LinearAlgebra
include("src/MirnanNew.jl")
using .MirnanNew
using .MirnanNew.Physics

println("=" ^ 60)
println("PHASE 4 TEST: Generation")
println("=" ^ 60)

# Test 1: Generator
println("\n[1] Generator...")
try
    using .Physics.Generator
    vocab = Dict("مرحبا" => 1, "بالعالم" => 2, "كيف" => 3, "حالك" => 4, " اليوم" => 5)
    gen = MirnanGenerator(vocab)
    println("  OK MirnanGenerator created: vocab_size=$(length(gen.vocab))")
catch e
    println("  FAIL: $e")
end

# Test 2: Orchestrator
println("\n[2] Orchestrator...")
try
    using .Physics.Orchestrator
    println("  OK Orchestrator module loaded")
catch e
    println("  FAIL: $e")
end

println("\n" * "=" ^ 60)
println("PHASE 4 COMPLETE")
println("=" ^ 60)
