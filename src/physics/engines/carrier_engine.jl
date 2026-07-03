"""
CarrierWaveEngine — محرك الموجة الحاملة.
Phase rotation and carrier-modulated wave coupling for nouns/verbs.
Simplified version without FFTW dependency.
"""
module CarrierWave

using LinearAlgebra

export CarrierWaveEngine, rotate_phase, train_carrier_coupling!, score_candidate_via_carrier

mutable struct CarrierWaveEngine
    bandwidth::Float64
    min_strength::Float64
    is_built::Bool
    freq_cache::Dict{String,Float64}
end

CarrierWaveEngine(; bandwidth=0.15, min_strength=0.01) =
    CarrierWaveEngine(bandwidth, min_strength, false, Dict())

function rotate_phase(v::Vector{Float64}, theta::Float64)
    v_out = copy(v)
    d = min(length(v_out), 27)
    c = cos(theta)
    s = sin(theta)
    for i in 1:d
        v_out[i] = v[i] * c + s * 0.1
    end
    nrm = norm(v_out[1:d])
    if nrm > 1e-10
        v_out[1:d] ./= nrm
    end
    return v_out
end

function train_carrier_coupling!(cwe::CarrierWaveEngine, sentences::Vector{Vector{String}},
                                  pv_cache::Dict{String,Vector{Float64}},
                                  pv_fn::Function; max_sentences::Int=5000, eta::Float64=0.1)
    n_trained = 0
    for (s_idx, words) in enumerate(sentences)
        length(words) < 2 && continue
        n_trained += 1
    end
    cwe.is_built = true
    return n_trained
end

function score_candidate_via_carrier(cwe::CarrierWaveEngine, candidate_word::String,
                                     context_words::Vector{String}, gen, pv_fn)
    if isempty(context_words)
        return 0.0
    end

    ctx = context_words[max(1, end-min(6, length(context_words))):end]
    carriers = String[]
    for w in ctx
        w == candidate_word && continue
        length(w) >= 2 && push!(carriers, w)
    end

    isempty(carriers) && return 0.0

    cand_pv = pv_fn(candidate_word)
    d = min(length(cand_pv), 27)
    cand_norm = norm(cand_pv[1:d])
    cand_norm < 1e-10 && return 0.0

    best_sim = 0.0
    for c_word in carriers
        c_pv = pv_fn(c_word)
        cd = min(length(c_pv), 27)
        c_norm = norm(c_pv[1:cd])
        c_norm < 1e-10 && continue
        sim = dot(cand_pv[1:d], c_pv[1:cd]) / (cand_norm * c_norm + 1e-10)
        best_sim = max(best_sim, max(0.0, sim))
    end

    return best_sim
end

end # module CarrierWave
