"""
GlobalSpectralMemory — ذاكرة طيفية شاملة (توقيعات طيفية متماسكة لكل كلمة).
"""
module SpectralMemory
using LinearAlgebra, Statistics
using ..WordPhysics: compute_word_phase_vector

export GlobalSpectralMemory, build_contexts_map, global_resonance

function build_contexts_map(texts::Vector{String}, vocab::Dict{String,Int}; half_window=7)
    ctx_map = Dict{Int,Vector{Vector{Float64}}}()
    for text in texts
        words = split(text)
        for (i, w) in enumerate(words)
            wid = get(vocab, w, nothing)
            wid === nothing && continue
            start = max(1, i - half_window)
            stop_ = min(length(words), i + half_window)
            ctx_words = String[words[j] for j in start:stop_ if j != i && j <= length(words)]
            pvs = [Float64.(compute_word_phase_vector(cw)) for cw in ctx_words]
            if !haskey(ctx_map, wid); ctx_map[wid] = []; end
            append!(ctx_map[wid], pvs)
        end
    end
    return ctx_map
end

mutable struct GlobalSpectralMemory
    gss_cache::Dict{Int,Vector{Float64}}
end
GlobalSpectralMemory() = GlobalSpectralMemory(Dict{Int,Vector{Float64}}())

function build_global_signatures!(gss::GlobalSpectralMemory, vocab::Dict{String,Int}, ctx_map::Dict)
    for (wid, pvs) in ctx_map
        if isempty(pvs); continue; end
        mat = reduce(hcat, pvs)'
        signature = vec(mean(mat; dims=1))
        nrm = norm(signature)
        if nrm > 1e-10; signature ./= nrm; end
        # Coherence weighting: weight by alignment with mean
        aligns = [dot(pv, signature) for pv in pvs]
        # Refine
        final_sig = zeros(Float64, length(signature))
        total_w = 0.0
        for (pv, align) in zip(pvs, aligns)
            w = max(0.0, align)
            final_sig .+= w .* pv
            total_w += w
        end
        if total_w > 1e-10; final_sig ./= total_w; end
        nrm = norm(final_sig)
        if nrm > 1e-10; final_sig ./= nrm; end
        gss.gss_cache[wid] = final_sig
    end
end

function global_resonance(gss::GlobalSpectralMemory, wid::Int, candidate_pv)
    if !haskey(gss.gss_cache, wid); return 0.0; end
    sig = gss.gss_cache[wid]
    return max(0.0, dot(candidate_pv, sig) / (norm(candidate_pv) * norm(sig) + 1e-10))
end
end
