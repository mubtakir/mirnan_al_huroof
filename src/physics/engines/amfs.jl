"""
AMFS — Adaptive Mass & Frequency Shift.
Dynamically adapt word mass and frequency based on context.
Physical simulation of contextual embeddings.
"""
module AMFS

using LinearAlgebra, Statistics
using ..WordPhysics: compute_extended_phase_vector, compute_word_mass,
                     compute_word_frequency

export adapt_word

function adapt_word(word::String;
                    context_words::Union{Vector{String},Nothing}=nothing,
                    context_pvs::Union{Vector{<:AbstractVector},Nothing}=nothing,
                    w_pv::Union{AbstractVector,Nothing}=nothing,
                    base_mass::Union{Float64,Nothing}=nothing,
                    base_freq::Union{Float64,Nothing}=nothing)
    bm = base_mass !== nothing ? base_mass : 1.0
    bf = base_freq !== nothing ? base_freq : 1.0
    wpv = w_pv !== nothing ? Float64.(w_pv) : Float64.(compute_extended_phase_vector(word))
    if context_pvs === nothing && context_words !== nothing
        context_pvs = [compute_extended_phase_vector(w) for w in context_words]
        bm = base_mass !== nothing ? base_mass : compute_word_mass(word)
        bf = base_freq !== nothing ? base_freq : compute_word_frequency(word)
    end

    w_norm = norm(wpv)
    if w_norm < 1e-10
        return Dict("mass" => bm, "freq" => bf,
                    "phase_shift" => 0.0, "centrality" => 0.0)
    end

    if context_pvs === nothing || isempty(context_pvs)
        return Dict("mass" => bm, "freq" => bf,
                    "phase_shift" => 0.0, "centrality" => 0.0)
    end

    ctx_pvs = [Float64.(p) for p in context_pvs]

    aligns = Float64[]
    for cpv in ctx_pvs
        c_norm = norm(cpv)
        if c_norm > 1e-10
            push!(aligns, dot(wpv, cpv) / (w_norm * c_norm))
        end
    end

    centrality = isempty(aligns) ? 0.0 : mean(aligns)
    adapted_mass = bm * (1.0 + 0.3 * max(0.0, centrality))

    ctx_mean_mat = reduce(hcat, ctx_pvs)'
    ctx_mean = vec(mean(ctx_mean_mat; dims=1))
    ctx_norm = norm(ctx_mean)
    phase_shift = 0.0
    if ctx_norm > 1e-10
        phase_shift = dot(wpv, ctx_mean) / (w_norm * ctx_norm)
    end

    adapted_freq = bf * (1.0 + 0.15 * tanh(phase_shift))

    return Dict("mass" => adapted_mass, "freq" => adapted_freq,
                "phase_shift" => phase_shift, "centrality" => centrality)
end

end # module AMFS
