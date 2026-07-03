cd(@__DIR__)

test_files = [
    "test_phase1.jl",
    "test_phase2.jl",
    "test_clifford_enhanced.jl",
    "test_syntactic_gravity.jl",
    "test_semantic_gravity.jl",
    "test_clifford_words.jl",
    "test_new_model.jl",
]

passed = 0
failed = 0
for f in test_files
    println("\n" * "="^60)
    println("Running: $f")
    println("="^60)
    try
        include(f)
        println("✓ PASSED: $f")
        global passed += 1
    catch e
        println("✗ FAILED: $f")
        println("  Error: ", e)
        global failed += 1
    end
end

println("\n" * "="^60)
println("RESULTS: $passed passed, $failed failed out of $(passed+failed)")
println("="^60)
