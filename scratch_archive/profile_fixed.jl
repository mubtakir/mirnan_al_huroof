push!(LOAD_PATH, joinpath(dirname(@__DIR__), "src"))
include(joinpath(@__DIR__, "src", "MirnanNew.jl")); using .MirnanNew, JSON, SparseArrays, FFTW, LinearAlgebra, Statistics

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
        return SparseMatrixCSC(Int(m), Int(n),
                               Vector{Int}(colptr),
                               Vector{Int}(rows),
                               vals)
    end
end

K_sem = _load_sparse_dat(joinpath(model_dir, "K_sem.dat"), V)
gen = MirnanGenerator(vocab, K_sem)

println("Getting candidates...")
prompt_tokens = ["غابة", "استوائية"]
us = Set(prompt_tokens)
prev = "استوائية"
cids = [get(gen.vocab, w, nothing) for w in prompt_tokens]
pvs = [MirnanNew.Physics.Generator._pv(gen, w) for w in prompt_tokens]
cands = MirnanNew.Physics.Generator._resonance_candidates(gen, cids, pvs, us; prev_word=prev)

println("Total candidates: $(length(cands))")

# ELEGANT CACHED FIXED OSCILLATOR SCORE
function score_fixed_cached(gen, word, used, all_pv, prompt_pv; gen_pos=0, total_pos=1, prev_word=nothing, context_ids=nothing, context_words=nothing)
    wid = get(gen.vocab, word, nothing)
    (wid===nothing || word in used || length(word)<2) && return -Inf
    w_pv = MirnanNew.Physics.Generator._pv(gen, word)

    target = isempty(all_pv) ? copy(w_pv) : Float64.(all_pv[end])
    align = MirnanNew.Physics.Generator._phase_align(gen, w_pv, target)

    prompt_align = 0.0
    if !isempty(prompt_pv)
        prompt_align = sum(MirnanNew.Physics.Generator._phase_align(gen,w_pv,pp) for pp in prompt_pv)/length(prompt_pv)
    end

    diversity=1.0; repulsion_score=0.0
    prev_used = collect(used)[max(1, end-6):end]
    if !isempty(prev_used)
        wc = count(x->x==word, prev_used)
        diversity = 1.0 - wc/length(prev_used)
        recent = prev_used[max(1, end-3):end]
        if !isempty(recent)
            sims = Float64[MirnanNew.Physics.Generator._phase_align(gen,w_pv,MirnanNew.Physics.Generator._pv(gen,w2)) for w2 in recent if w2!=word]
            repulsion_score = isempty(sims) ? 0.0 : -mean(sims)
        end
    end

    syntax_score = 0.0
    if prev_word !== nothing && gen.syntax_field !== nothing
        try; syntax_score = gen.syntax_field.transition_align(prev_word, word); catch; end
    end

    gravity_score = 0.0
    if context_words !== nothing
        m_word = MirnanNew.Physics.Generator._mass(gen, word)
        for (i, cw) in enumerate(context_words[max(1, end-5):end])
            c_pv = MirnanNew.Physics.Generator._pv(gen, cw); m_c = MirnanNew.Physics.Generator._mass(gen, cw)
            r = norm(w_pv .- c_pv) + 1e-6
            gravity_score += MirnanNew.Physics.Constants.GRAVITY_G * m_word * m_c / (r^2 + 0.01) * exp(-0.3*i)
        end
    end

    heterodyne_score = 0.0
    if context_words !== nothing && !isempty(context_words)
        heterodyne_score = MirnanNew.Physics.Heterodyne.score_candidate(gen.heterodyne, word, context_words)
    end

    # ELEGANT CACHED FIXED OSCILLATOR SCORE
    osc_score = 0.0
    if context_words !== nothing && length(context_words) >= 2
        key = join(context_words[max(1, end-5):end], "\0")
        evolved_phase = get!(gen.pv_cache, "OSC\0" * key) do
            try
                ctx_omega = [MirnanNew.Physics.WordPhysics.compute_word_frequency_with_irab(w) for w in context_words[max(1, end-5):end]]
                ctx_pv = [MirnanNew.Physics.Generator._pv(gen,w)[1:MirnanNew.Physics.Constants.PHASE_DIM] for w in context_words[max(1, end-5):end]]
                ctx_masses = [MirnanNew.Physics.Generator._mass(gen,w) for w in context_words[max(1, end-5):end]]
                n_ctx = length(ctx_omega)
                phases = zeros(n_ctx, MirnanNew.Physics.Constants.PHASE_DIM)
                for k in 1:n_ctx; phases[k,:] .= angle.(fft(ctx_pv[k])); end
                coupling = ones(n_ctx,n_ctx)*0.3
                evolved, _ = MirnanNew.Physics.Oscillator.simulate(gen.osc_engine, ctx_omega, phases, ctx_masses, ctx_pv, coupling; dt=0.01, steps=20, temperature=0.02)
                vec(mean(evolved; dims=1))
            catch e
                println("OSC RUN TIME ERROR: ", e)
                Float64[]
            end
        end
        if !isempty(evolved_phase)
            w_phase = get!(gen.pv_cache, "FFT\0" * word) do
                angle.(fft(w_pv[1:MirnanNew.Physics.Constants.PHASE_DIM]))
            end
            osc_score = MirnanNew.Physics.WordPhysics.phase_similarity(evolved_phase, w_phase)
        end
    end

    score = 0.0
    W = gen.scoring_weights
    score += get(W,"align",0.0)*align + get(W,"prompt_align",0.0)*prompt_align
    score += get(W,"diversity",0.0)*diversity + get(W,"repulsion",0.0)*repulsion_score
    score += get(W,"syntax",0.0)*syntax_score + get(W,"gravity",0.0)*gravity_score
    score += get(W,"heterodyne",0.0)*heterodyne_score + get(W,"oscillator",0.0)*osc_score
    
    return score
end

if length(cands) >= 2
    cand1 = cands[1]
    println("Profiling first candidate with cached fixed score (with compilation): '$cand1'")
    t0 = time_ns()
    s1 = score_fixed_cached(gen, cand1, us, pvs, pvs;
                     gen_pos=2, total_pos=10, prev_word=prev,
                     context_ids=cids, context_words=prompt_tokens)
    t1 = time_ns()
    dt1 = (t1 - t0) / 1e6 # ms
    println("Candidate 1 score: $s1")
    println("Time for candidate 1: $dt1 ms")

    cand2 = cands[2]
    println("Profiling second candidate with cached fixed score (pure runtime): '$cand2'")
    t2 = time_ns()
    s2 = score_fixed_cached(gen, cand2, us, pvs, pvs;
                     gen_pos=2, total_pos=10, prev_word=prev,
                     context_ids=cids, context_words=prompt_tokens)
    t3 = time_ns()
    dt2 = (t3 - t2) / 1e6 # ms
    println("Candidate 2 score: $s2")
    println("Time for candidate 2: $dt2 ms")

    cand3 = cands[3]
    println("Profiling third candidate with cached fixed score (pure runtime): '$cand3'")
    t4 = time_ns()
    s3 = score_fixed_cached(gen, cand3, us, pvs, pvs;
                     gen_pos=2, total_pos=10, prev_word=prev,
                     context_ids=cids, context_words=prompt_tokens)
    t5 = time_ns()
    dt3 = (t5 - t4) / 1e6 # ms
    println("Candidate 3 score: $s3")
    println("Time for candidate 3: $dt3 ms")
end
