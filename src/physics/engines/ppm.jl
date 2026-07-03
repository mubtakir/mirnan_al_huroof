"""
PPM — Prompt Phase Modulation.
External temporary phase field that biases generation from prompt examples.
Physical simulation of In-Context Learning / Few-Shot.
"""
module PPM

using LinearAlgebra, Statistics
using ..Constants: TOTAL_DIM
using ..WordPhysics: compute_extended_phase_vector

export PromptField, score, absorb!, step!, reset!, modulate

mutable struct PromptField
    decay_rate::Float64
    strength::Float64
    max_examples::Int
    field::Vector{Float64}
    active::Bool
    pv_cache::Dict{String,Vector{Float64}}
    example_count::Int
    dim::Int

    function PromptField(; dim::Int=TOTAL_DIM, decay_rate::Float64=0.1, strength::Float64=1.0, max_examples::Int=10)
        return new(decay_rate, strength, max_examples,
                   zeros(Float64, dim), false,
                   Dict{String,Vector{Float64}}(), 0, dim)
    end
end

function absorb!(pf::PromptField, prompt::String)
    words = String[strip(w) for w in split(prompt) if !isempty(strip(w))]
    return absorb!(pf, [compute_extended_phase_vector(w) for w in words])
end

function absorb!(pf::PromptField, prompt_pvs::Vector{<:AbstractVector})
    if isempty(prompt_pvs)
        return
    end

    pvs = [Float64.(pv) for pv in prompt_pvs]
    pv_mat = reduce(hcat, pvs)'
    if length(pf.field) != size(pv_mat, 2)
        pf.field = zeros(Float64, size(pv_mat, 2))
        pf.dim = size(pv_mat, 2)
    end
    pf.field .= pf.strength .* vec(mean(pv_mat; dims=1))
    pf.active = true
    pf.example_count = length(pvs)
end

function modulate(pf::PromptField, candidate_pv::AbstractVector)
    if !pf.active
        return Float64.(candidate_pv)
    end
    d = min(pf.dim, length(candidate_pv))
    result = Float64.(candidate_pv)
    result[1:d] .+= pf.field[1:d] .* 0.15
    return result
end

function score(pf::PromptField, w_pv::AbstractVector)
    if !pf.active || norm(pf.field) < 1e-10
        return 0.0
    end
    d = min(pf.dim, length(w_pv))
    w_sub = Float64.(w_pv[1:d])
    f_sub = pf.field[1:d]
    w_norm = norm(w_sub)
    f_norm = norm(f_sub)
    if w_norm < 1e-10 || f_norm < 1e-10
        return 0.0
    end
    cos_val = dot(w_sub, f_sub) / (w_norm * f_norm)
    return max(0.0, cos_val)
end

function step!(pf::PromptField)
    if pf.active
        pf.field .*= (1.0 - pf.decay_rate)
        if norm(pf.field) < 0.01
            pf.field .= 0.0
            pf.active = false
        end
    end
end

function reset!(pf::PromptField)
    pf.field .= 0.0
    pf.active = false
    pf.example_count = 0
end

end # module PPM
