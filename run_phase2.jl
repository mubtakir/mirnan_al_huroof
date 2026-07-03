include("src/MirnanNew.jl")
using .MirnanNew
using .MirnanNew.Physics

println("=" ^ 60)
println("PHASE 2 TEST: Advanced Engines")
println("=" ^ 60)

# Test 1: Phase Reinforcement
println("\n[1] Phase Reinforcement...")
try
    using .Physics.PhaseReinforcement
    pr = PhaseReinforcer(lr=0.15, decay_rate=0.005, max_traces=200)
    reinforce!(pr, "test_word", randn(27); reward=1.0)
    strength = get_strength(pr, "test_word")
    println("  OK PhaseReinforcer: strength = $strength")
catch e
    println("  FAIL PhaseReinforcer: $e")
end

# Test 2: Density Matrix
println("\n[2] Density Matrix...")
try
    using .Physics.DensityMatrix
    dm = PhaseDensityMatrix(dim=64, decay_rate=0.8)
    pvs = [randn(64) for _ in 1:5]
    build!(dm, pvs)
    trace_val = get_trace(dm)
    purity = get_purity(dm)
    entropy = get_spectral_entropy(dm)
    res = resonance(dm, randn(64))
    println("  OK PhaseDensityMatrix: trace=$trace_val, purity=$purity, entropy=$entropy, resonance=$res")
catch e
    println("  FAIL PhaseDensityMatrix: $e")
end

# Test 3: Causal Flow
println("\n[3] Causal Flow...")
try
    using .Physics.CausalFlow
    cf = CausalFlowField(dim=27, flow_strength=1.0)
    current_pv = randn(27)
    context_pvs = [randn(27) for _ in 1:3]
    context_ids = [1, 2, 3]
    causal_matrix = rand(5, 5)
    result = compute_flow(cf, current_pv, context_pvs, context_ids; causal_matrix=causal_matrix)
    println("  OK CausalFlowField: magnitude=$(result["flow_magnitude"]), score=$(result["logical_score"])")
catch e
    println("  FAIL CausalFlowField: $e")
end

# Test 4: DCCF
println("\n[4] DCCF...")
try
    using .Physics.DCCF
    dccf = DynamicCCF(decay_rate=0.5, mass_threshold=0.3)
    context_pvs = [randn(27) for _ in 1:4]
    coupling, scores = build_coupling(dccf, context_pvs)
    boost = get_context_boost(dccf, randn(27), context_pvs)
    println("  OK DynamicCCF: coupling_size=$(size(coupling)), boost=$boost")
catch e
    println("  FAIL DynamicCCF: $e")
end

# Test 5: PPM
println("\n[5] PPM...")
try
    using .Physics.PPM
    pf = PromptField(dim=27, decay_rate=0.1, strength=1.0)
    prompt_pvs = [randn(27) for _ in 1:3]
    absorb!(pf, prompt_pvs)
    candidate = randn(27)
    modulated = modulate(pf, candidate)
    score_val = score(pf, candidate)
    step!(pf)
    println("  OK PromptField: active=$(pf.active), score=$score_val")
catch e
    println("  FAIL PromptField: $e")
end

# Test 6: AMFS
println("\n[6] AMFS...")
try
    using .Physics.AMFS
    context_pvs = [randn(27) for _ in 1:3]
    result = adapt_word("test"; context_pvs=context_pvs, w_pv=randn(27))
    println("  OK AMFS: mass=$(result["mass"]), freq=$(result["freq"]), centrality=$(result["centrality"])")
catch e
    println("  FAIL AMFS: $e")
end

# Test 7: Holographic KB
println("\n[7] Holographic KB...")
try
    using .Physics.HolographicKB
    kb = HolographicKnowledgeBase()
    subj_pv = randn(27)
    obj_pv = randn(27)
    store_fact!(kb, subj_pv, obj_pv, "IS_A"; subj_word="cat", obj_word="animal")
    results = query(kb, subj_pv; top_k=5)
    reconstructed = reconstruct_vector(kb, subj_pv)
    println("  OK HolographicKB: facts=$(kb.fact_count), results=$(length(results)), reconstructed=$(reconstructed !== nothing)")
catch e
    println("  FAIL HolographicKB: $e")
end

# Test 8: Potential Cascade
println("\n[8] Potential Cascade...")
try
    using .Physics.PotentialCascade
    layer = PotentialCascadeLayer(lambda_cascade=3.0, gamma=2.0, delta=0.3)
    candidate_pv = randn(27)
    context_pvs = [randn(27) for _ in 1:3]
    context_masses = [1.0, 0.8, 0.6]
    score_val = compute_score(layer, candidate_pv, context_pvs, context_masses, true)
    println("  OK PotentialCascadeLayer: score=$score_val")
catch e
    println("  FAIL PotentialCascadeLayer: $e")
end

println("\n" * "=" ^ 60)
println("PHASE 2 COMPLETE")
println("=" ^ 60)
