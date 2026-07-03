"""CoherenceFeedback — تغذية راجعة للتماسك."""
module CoherenceFeedbackModule
using LinearAlgebra, Logging, Statistics
using ..WordPhysics: phase_similarity

export CoherenceFeedback

mutable struct CoherenceFeedback
    min_words::Int; threshold::Float64
    history::Vector{Dict}
end
CoherenceFeedback(; min_words=3, threshold=0.3) = CoherenceFeedback(min_words, threshold, Dict[])

function measure_sequential_resonance(cf::CoherenceFeedback, words, pvs)
    length(words) < 2 && return 1.0
    sims = Float64[]
    for i in 2:length(words)
        d = min(length(pvs[i-1]), length(pvs[i]))
        push!(sims, phase_similarity(view(pvs[i-1],1:d), view(pvs[i],1:d)))
    end
    return isempty(sims) ? 0.5 : mean(sims)
end

function measure_phase_coherence(cf::CoherenceFeedback, pvs)
    length(pvs) < cf.min_words && return 0.5
    mat = reduce(hcat, pvs)'
    mean_pv = vec(mean(mat; dims=1))
    nrm = norm(mean_pv); nrm < 1e-10 && return 0.5
    mean_pv ./= nrm
    return mean([phase_similarity(pv, mean_pv) for pv in pvs])
end

function evaluate(cf::CoherenceFeedback, words, pvs)
    sr = measure_sequential_resonance(cf, words, pvs)
    pc = measure_phase_coherence(cf, pvs)
    coherence = 0.5*sr + 0.5*pc
    report = Dict("sequential_resonance"=>sr, "phase_coherence"=>pc, "coherence"=>coherence)
    push!(cf.history, report)
    return report
end
end

