push!(LOAD_PATH, joinpath(dirname(@__DIR__), "src"))
include(joinpath(@__DIR__, "src", "MirnanNew.jl")); using .MirnanNew, JSON, SparseArrays, LinearAlgebra, Statistics, Printf

println("Loading generator...")
model_dir = joinpath(dirname(@__DIR__), "model")
vf = joinpath(model_dir, "vocab.json")
raw_vocab = JSON.parsefile(vf)
vocab = Dict{String,Int}(k => Int(v) for (k,v) in raw_vocab)
V = length(vocab)

function _load_sparse_dat(path::String, vocab_size::Int)
    isfile(path) || return spzeros(vocab_size, vocab_size)
    open(path, "r") do io
        header = readline(io)
        header == "SPARSE_CSC" || return spzeros(vocab_size, vocab_size)
        m = read(io, Int32)
        n = read(io, Int32)
        nnz = read(io, Int32)
        colptr = read!(io, Vector{Int32}(undef, n+1))
        rows   = read!(io, Vector{Int32}(undef, nnz))
        vals   = read!(io, Vector{Float64}(undef, nnz))
        return SparseMatrixCSC(Int(m), Int(n), Vector{Int}(colptr), Vector{Int}(rows), vals)
    end
end
K_sem = _load_sparse_dat(joinpath(model_dir, "K_sem.dat"), V)
gen = MirnanGenerator(vocab, K_sem)

prompt_tokens = ["غابة", "غابات", "وادي"]
us = Set(prompt_tokens)
prev = "وادي"
cids = [get(gen.vocab, w, nothing) for w in prompt_tokens]
pvs = [MirnanNew.Physics.Generator._pv(gen, w) for w in prompt_tokens]
cands = MirnanNew.Physics.Generator._resonance_candidates(gen, cids, pvs, us; prev_word=prev)

println("Total candidates: $(length(cands))")
cands_to_test = cands[1:min(100, end)]

# Timings dictionary
times = Dict{String, Float64}(
    "basics" => 0.0,
    "syntax" => 0.0,
    "gravity" => 0.0,
    "heterodyne" => 0.0,
    "oscillator" => 0.0,
    "pragmatic" => 0.0,
    "resonant_chain" => 0.0,
    "dccf" => 0.0,
    "ppm" => 0.0,
    "dialogue" => 0.0,
    "kb" => 0.0,
    "amfs" => 0.0,
    "weight_resonance" => 0.0,
    "spectral" => 0.0,
    "density" => 0.0,
    "sentiment" => 0.0,
    "hierarchical" => 0.0,
    "entity" => 0.0,
    "trace" => 0.0,
    "beamform" => 0.0,
    "anchor" => 0.0,
    "tension" => 0.0,
    "other" => 0.0
)

# Run once for compilation
MirnanNew.Physics.Generator._score(gen, cands_to_test[1], us, pvs, pvs; gen_pos=3, total_pos=10, prev_word=prev, context_ids=cids, context_words=prompt_tokens)

for word in cands_to_test
    wid = get(gen.vocab, word, nothing)
    (wid===nothing || word in us || length(word)<2) && continue
    
    t = time_ns()
    w_pv = MirnanNew.Physics.Generator._pv(gen, word)
    target = isempty(pvs) ? copy(w_pv) : Float64.(pvs[end])
    align = MirnanNew.Physics.Generator._phase_align(gen, w_pv, target)
    prompt_align = isempty(pvs) ? 0.0 : sum(MirnanNew.Physics.Generator._phase_align(gen,w_pv,pp) for pp in pvs)/length(pvs)
    diversity = 1.0
    repulsion_score = 0.0
    times["basics"] += (time_ns() - t) / 1e6

    t = time_ns()
    syntax_score = 0.0
    if prev !== nothing && gen.syntax_field !== nothing
        try; syntax_score = gen.syntax_field.transition_align(prev, word); catch; end
    end
    times["syntax"] += (time_ns() - t) / 1e6

    t = time_ns()
    gravity_score = 0.0
    m_word = MirnanNew.Physics.Generator._mass(gen, word)
    for (i, cw) in enumerate(prompt_tokens[max(1, end-5):end])
        c_pv = MirnanNew.Physics.Generator._pv(gen, cw); m_c = MirnanNew.Physics.Generator._mass(gen, cw)
        r = norm(w_pv .- c_pv) + 1e-6
        gravity_score += MirnanNew.Physics.Constants.GRAVITY_G * m_word * m_c / (r^2 + 0.01) * exp(-0.3*i)
    end
    times["gravity"] += (time_ns() - t) / 1e6

    t = time_ns()
    heterodyne_score = 0.0
    if !isempty(prompt_tokens)
        heterodyne_score = MirnanNew.Physics.Heterodyne.score_candidate(gen.heterodyne, word, prompt_tokens)
    end
    times["heterodyne"] += (time_ns() - t) / 1e6

    t = time_ns()
    # Oscillator
    osc_score = 0.0
    key = join(prompt_tokens[max(1, end-5):end], "\0")
    evolved_phase = get!(gen.pv_cache, "OSC\0" * key) do
        plan = MirnanNew.Physics.Generator._get_fft_plan()
        ctx_omega = [MirnanNew.Physics.WordPhysics.compute_word_frequency_with_irab(w) for w in prompt_tokens[max(1, end-5):end]]
        ctx_pv = [MirnanNew.Physics.Generator._pv(gen,w)[1:MirnanNew.Physics.Constants.PHASE_DIM] for w in prompt_tokens[max(1, end-5):end]]
        ctx_masses = [MirnanNew.Physics.Generator._mass(gen,w) for w in prompt_tokens[max(1, end-5):end]]
        n_ctx = length(ctx_omega)
        phases = zeros(n_ctx, MirnanNew.Physics.Constants.PHASE_DIM)
        for k in 1:n_ctx; phases[k,:] .= angle.(plan * ComplexF64.(ctx_pv[k])); end
        coupling = ones(n_ctx,n_ctx)*0.3
        evolved, _ = MirnanNew.Physics.Oscillator.simulate(gen.osc_engine, ctx_omega, phases, ctx_masses, ctx_pv, coupling; dt=0.01, steps=20, temperature=0.02)
        vec(mean(evolved; dims=1))
    end
    if !isempty(evolved_phase)
        w_phase = get!(gen.pv_cache, "FFT\0" * word) do
            plan = MirnanNew.Physics.Generator._get_fft_plan()
            angle.(plan * ComplexF64.(w_pv[1:MirnanNew.Physics.Constants.PHASE_DIM]))
        end
        osc_score = MirnanNew.Physics.WordPhysics.phase_similarity(evolved_phase, w_phase)
    end
    times["oscillator"] += (time_ns() - t) / 1e6

    t = time_ns()
    intent_frame, alpha_intent, is_int = MirnanNew.Physics.PragmaticField.detect_intent_frame(gen.pragmatic_engine, prompt_tokens)
    pragmatic_align = MirnanNew.Physics.PragmaticField.compute_pragmatic_score(gen.pragmatic_engine, w_pv, intent_frame, alpha_intent)
    times["pragmatic"] += (time_ns() - t) / 1e6

    t = time_ns()
    resonant_chain_score = 0.0
    m_prev = MirnanNew.Physics.Generator._mass(gen, prev)
    m_cand = MirnanNew.Physics.Generator._mass(gen, word)
    pv_prev = MirnanNew.Physics.Generator._pv(gen, prev)
    prev_freqs = Float64[]
    for k in 2:length(prompt_tokens)
        ma = MirnanNew.Physics.Generator._mass(gen, prompt_tokens[k-1]); mb = MirnanNew.Physics.Generator._mass(gen, prompt_tokens[k])
        pa = MirnanNew.Physics.Generator._pv(gen, prompt_tokens[k-1]); pb = MirnanNew.Physics.Generator._pv(gen, prompt_tokens[k])
        push!(prev_freqs, MirnanNew.Physics.ResonantChain.pair_freq(gen.resonant_chain, ma, mb, pa, pb))
    end
    resonant_chain_score = MirnanNew.Physics.ResonantChain.score_candidate(gen.resonant_chain, m_prev, m_cand, pv_prev, w_pv; prev_freqs=prev_freqs)
    times["resonant_chain"] += (time_ns() - t) / 1e6

    t = time_ns()
    dccf_score = MirnanNew.Physics.DCCF.get_context_boost(gen.dccf, word, prompt_tokens)
    times["dccf"] += (time_ns() - t) / 1e6

    t = time_ns()
    ppm_score = MirnanNew.Physics.PPM.score(gen.prompt_field, w_pv)
    times["ppm"] += (time_ns() - t) / 1e6

    t = time_ns()
    dialogue_gravity = 0.0
    if gen.dialogue_mode
        dialogue_gravity = MirnanNew.Physics.DialogueMemoryModule.get_dialogue_gravity(gen.dialogue_memory, w_pv)
    end
    times["dialogue"] += (time_ns() - t) / 1e6

    t = time_ns()
    kb_score = 0.0
    if gen.kb.built
        kr = MirnanNew.Physics.HolographicKB.query(gen.kb, w_pv; top_k=3)
        if !isempty(kr); kb_score = kr[1][1]; end
    end
    times["kb"] += (time_ns() - t) / 1e6

    t = time_ns()
    # AMFS
    precomputed_ctx = if !isempty(prompt_tokens)
        key = join(prompt_tokens, "\0")
        ctx_pvs = [MirnanNew.Physics.Generator._pv(gen, w) for w in prompt_tokens]
        ctx_norms = get!(gen.pv_cache, "AMFS_NORMS\0" * key) do
            [norm(cpv) for cpv in ctx_pvs]
        end
        ctx_mean = get!(gen.pv_cache, "AMFS_MEAN\0" * key) do
            sum(ctx_pvs) ./ length(ctx_pvs)
        end
        ctx_mean_norm = get!(gen.pv_cache, "AMFS_MEAN_NORM\0" * key) do
            [norm(ctx_mean)]
        end[1]
        (pvs=ctx_pvs, norms=ctx_norms, mean=ctx_mean, mean_norm=ctx_mean_norm)
    else
        nothing
    end
    base_freq = get!(gen.pv_cache, "FREQ\0" * word) do
        [MirnanNew.Physics.WordPhysics.compute_word_frequency(word)]
    end[1]
    amfs_result = MirnanNew.Physics.AMFS.adapt_word(word; context_words=prompt_tokens, context_pvs=pvs, w_pv=w_pv, base_mass=m_cand, base_freq=base_freq, precomputed_ctx=precomputed_ctx)
    amfs_boost = max(0.0, amfs_result["centrality"] * amfs_result["mass"]/3.0)
    times["amfs"] += (time_ns() - t) / 1e6

    t = time_ns()
    weight_resonance_score = 0.0
    wr_prev = MirnanNew.Physics.WeightResonance.get_weight(gen.weight_resonance, prev)
    wr_word = MirnanNew.Physics.WeightResonance.get_weight(gen.weight_resonance, word)
    if wr_prev !== nothing && wr_word !== nothing
        weight_resonance_score = MirnanNew.Physics.WeightResonance.transition_score(gen.weight_resonance, wr_prev, wr_word)
    end
    times["weight_resonance"] += (time_ns() - t) / 1e6

    t = time_ns()
    spectral_score = 0.0
    w_spec = get!(gen.pv_cache, "SPEC\0" * word) do
        s = MirnanNew.Physics.WordSpectrum.compute_word_spectrum(word)
        s === nothing ? Float64[] : s["spectral_vector"]
    end
    if !isempty(w_spec)
        specs = Float64[]
        for cw in prompt_tokens[max(1, end-5):end]
            cw == word && continue
            cw_spec = get!(gen.pv_cache, "SPEC\0" * cw) do
                s = MirnanNew.Physics.WordSpectrum.compute_word_spectrum(cw)
                s === nothing ? Float64[] : s["spectral_vector"]
            end
            if !isempty(cw_spec)
                dim = min(length(w_spec), length(cw_spec))
                if dim > 0
                    cs = dot(w_spec[1:dim], cw_spec[1:dim]) / (norm(w_spec[1:dim])*norm(cw_spec[1:dim]) + 1e-10)
                    push!(specs, clamp((cs+1.0)/2.0, 0.0, 1.0))
                end
            end
        end
        if !isempty(specs); spectral_score = mean(specs); end
    end
    times["spectral"] += (time_ns() - t) / 1e6

    t = time_ns()
    density_resonance_score = MirnanNew.Physics.DensityMatrix.resonance(gen.density_matrix, w_pv)
    times["density"] += (time_ns() - t) / 1e6

    t = time_ns()
    sentiment_score = MirnanNew.Physics.SentimentPolarity.sentiment_fidelity(word, prompt_tokens; cache=gen.pv_cache)
    times["sentiment"] += (time_ns() - t) / 1e6

    t = time_ns()
    hier_ctx = MirnanNew.Physics.HierarchicalMemoryModule.get_hierarchical_context(gen.hierarchical_memory)
    hmod = norm(hier_ctx) > 1e-10 ? max(0.0, dot(w_pv, hier_ctx)/(norm(w_pv)*norm(hier_ctx)+1e-10)) : 0.0
    times["hierarchical"] += (time_ns() - t) / 1e6

    t = time_ns()
    entity_score = MirnanNew.Physics.EntityRegisterModule.entity_gravity(gen.entity_register, w_pv)
    times["entity"] += (time_ns() - t) / 1e6

    t = time_ns()
    trace_score = MirnanNew.Physics.InteractionTraceModule.get_context_boost(gen.interaction_trace, w_pv)
    times["trace"] += (time_ns() - t) / 1e6

    t = time_ns()
    bf_score = MirnanNew.Physics.AdvancedEngines.beamform_score(gen.beamformer, w_pv)
    times["beamform"] += (time_ns() - t) / 1e6

    t = time_ns()
    anchor_score = MirnanNew.Physics.RotatingAnchorModule.alignment(gen.rotating_anchor, w_pv)
    times["anchor"] += (time_ns() - t) / 1e6

    t = time_ns()
    tension_score = get!(gen.pv_cache, "TENSION\0" * word) do
        [1.0 - MirnanNew.Physics.FilamentTension.compute_tension(gen.filament_tension, word)]
    end[1]
    times["tension"] += (time_ns() - t) / 1e6
end

println("\n--- Profiling Breakdown (Total Time in ms for 100 Candidates) ---")
total_sum = sum(values(times))
for (k, v) in sort(collect(times), by=x->-x[2])
    pct = total_sum > 0 ? (v / total_sum * 100) : 0.0
    # Use standard Julia printing
    @printf("  %-18s: %8.3f ms (%5.1f%%)\n", k, v, pct)
end
@printf("Total profiled time: %.3f ms\n", total_sum)
