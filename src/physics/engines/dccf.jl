"""
DCCF — Dynamic Contextual Coupling Field.
Physical alternative to self-attention.
Weight long-range phase relationships with distance decay.
"""
module DCCF

using LinearAlgebra
using ..WordPhysics: compute_extended_phase_vector

export DynamicCCF, get_context_boost, build_coupling

mutable struct DynamicCCF
    decay_rate::Float64
    mass_threshold::Float64
    pv_cache::Dict{String,Vector{Float64}}

    function DynamicCCF(; decay_rate::Float64=0.5, mass_threshold::Float64=0.3)
        return new(decay_rate, mass_threshold, Dict{String,Vector{Float64}}())
    end
end

function build_coupling(dccf::DynamicCCF, context_pvs::Vector{<:AbstractVector})
    n = length(context_pvs)
    if n < 2
        return Matrix{Float64}(I, 1, 1), Dict()
    end

    pvs = [Float64.(pv) for pv in context_pvs]
    coupling = zeros(Float64, n, n)
    scores = Dict()

    for i in 1:n
        for j in (i+1):n
            norm_i = norm(pvs[i])
            norm_j = norm(pvs[j])
            if norm_i < 1e-10 || norm_j < 1e-10
                continue
            end
            phase_align = dot(pvs[i], pvs[j]) / (norm_i * norm_j)

            mass_sim = norm_i * norm_j
            dist_decay = exp(-dccf.decay_rate * (j - i))

            val = phase_align * mass_sim * dist_decay
            coupling[i, j] = val
            coupling[j, i] = val
            scores[(i, j)] = Dict(
                "phase_align" => phase_align,
                "mass_sim" => mass_sim,
                "dist_decay" => dist_decay,
                "total" => val,
            )
        end
    end

    return coupling, scores
end

function get_context_boost(dccf::DynamicCCF, candidate::String,
                           context_words::Vector{String})
    candidate_pv = compute_extended_phase_vector(candidate)
    context_pvs = [compute_extended_phase_vector(w) for w in context_words]
    return get_context_boost(dccf, candidate_pv, context_pvs)
end

function get_context_boost(dccf::DynamicCCF, candidate_pv::AbstractVector,
                           context_pvs::Vector{<:AbstractVector})
    n = length(context_pvs)
    if n < 1
        return 0.0
    end

    pw = Float64.(candidate_pv)
    norm_w = norm(pw)
    if norm_w < 1e-10
        return 0.0
    end

    total = 0.0
    weight_sum = 0.0
    for i in 1:n
        cpv = Float64.(context_pvs[i])
        norm_c = norm(cpv)
        if norm_c < 1e-10
            continue
        end
        phase_align = dot(pw, cpv) / (norm_w * norm_c)
        if phase_align < dccf.mass_threshold
            continue
        end
        dist_decay = exp(-dccf.decay_rate * (n - i + 1))
        w = dist_decay * norm_c
        total += phase_align * w
        weight_sum += w
    end

    return total / max(weight_sum, 1e-10)
end

end # module DCCF
