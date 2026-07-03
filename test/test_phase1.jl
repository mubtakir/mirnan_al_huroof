using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "MirnanNew.jl"))
using .MirnanNew

println("=== Phase 1 Test ===")
println("✓ MirnanNew loaded")

# Test enhanced phase vector
println("\n--- Testing Enhanced Phase Vector ---")
pv_a = MirnanNew.Physics.WordPhysics.compute_enhanced_vector('ع')
pv_b = MirnanNew.Physics.WordPhysics.compute_enhanced_vector('ل')
pv_m = MirnanNew.Physics.WordPhysics.compute_enhanced_vector('م')
println("Length of enhanced vector: ", length(pv_a))
println("Vector 'ع': ", pv_a[1:5], "...")
println("Vector 'ل': ", pv_b[1:5], "...")

# Test phase similarity
sim_ab = MirnanNew.Physics.WordPhysics.phase_similarity_enhanced(pv_a, pv_b)
sim_am = MirnanNew.Physics.WordPhysics.phase_similarity_enhanced(pv_a, pv_m)
println("\nSimilarity ع-ل: ", sim_ab)
println("Similarity ع-م: ", sim_am)

# Test gravitational force
println("\n--- Testing Semantic Gravity ---")
f = MirnanNew.Physics.SemanticGravity.gravitational_force_enhanced("ع", "ل")
println("Gravity ع-ل: ", f)

# Test syntactic gravity
println("\n--- Testing Syntactic Gravity ---")
f_syn = MirnanNew.Physics.SyntacticGravity.gravitational_force_syntactic(1, pv_a, 2, pv_b, 0.5)
println("Syntactic gravity (pos 1 vs 2): ", f_syn)

# Test resonant chain
println("\n--- Testing Resonant Chain ---")
chain = MirnanNew.Physics.ResonantChain.ResonantChainRLC()
m1 = MirnanNew.Physics.WordPhysics.letter_mass('ع')
m2 = MirnanNew.Physics.WordPhysics.letter_mass('ل')
freq = MirnanNew.Physics.ResonantChain.pair_freq(chain, m1, m2, pv_a, pv_b)
println("Resonant frequency ع-ل: ", freq)

# Test entropy gate
println("\n--- Testing Entropy Gate ---")
gate = MirnanNew.Physics.EntropyGateModule.EntropyGate()
S = MirnanNew.Physics.EntropyGateModule.compute_S(gate, [pv_a, pv_b], pv_m)
println("Entropy: ", S)

# Test RAM core
println("\n--- Testing RAM Core ---")
ram = MirnanNew.Physics.RAMCore.AttractorMemory()
idx = MirnanNew.Physics.RAMCore.observe!(ram, ["ع", "ل", "م"])
println("Attractor stored at index: ", idx)
println("RAM size: ", MirnanNew.Physics.RAMCore.size(ram))

println("\n=== All Phase 1 tests passed! ===")
