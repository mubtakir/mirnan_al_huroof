"""PromptConstraint — حقل قيد Dirichlet boundary."""
module PromptConstraint
using LinearAlgebra
using ..WordPhysics: phase_similarity

export PromptConstraintField

mutable struct PromptConstraintField
    constraints::Vector{Vector{Float64}}
    k::Float64; damping::Float64; progress::Float64
end
PromptConstraintField(; k=3.0, damping=0.15) = PromptConstraintField(Vector{Float64}[], k, damping, 0.0)

function set_prompt!(pf::PromptConstraintField, prompt_pvs)
    pf.constraints = [Float64.(pv) for pv in prompt_pvs]
end

function spring_force(pf::PromptConstraintField, candidate_pv, gen_pos, total_pos)
    isempty(pf.constraints) && return 0.0
    n = length(pf.constraints)
    idx = min(n, max(1, Int(floor(gen_pos / max(total_pos, 1) * (n-1)) + 1)))
    target = pf.constraints[idx]
    sim = phase_similarity(candidate_pv, target)
    direction = target .- Float64.(candidate_pv)
    dist = norm(direction)
    force = dist > 1e-10 ? pf.k * exp(-pf.damping * dist) : pf.k
    return sim * min(1.0, force/pf.k) * 0.3
end

function compute_goal_concept(pf::PromptConstraintField, prompt_pvs; alpha=0.6)
    isempty(prompt_pvs) && return nothing
    n = length(prompt_pvs)
    weights = exp.(-0.3 .* Float64(n-1:-1:0)); weights ./= sum(weights)
    center = sum(w .* pv for (w, pv) in zip(weights, prompt_pvs))
    nrm = norm(center); nrm > 1e-10 && (center ./= nrm)
    return center
end
end
