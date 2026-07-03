"""RotatingAnchor — مرساة دوارة (مرجع سياقي متحرك)."""
module RotatingAnchorModule
using LinearAlgebra
using ..Constants: TOTAL_DIM
export RotatingAnchor

mutable struct RotatingAnchor
    anchor::Vector{Float64}; alpha::Float64; gamma::Float64
end
RotatingAnchor(; dim=TOTAL_DIM, alpha=0.9, gamma=0.05) = RotatingAnchor(zeros(dim), alpha, gamma)

function update!(ra::RotatingAnchor, new_pv::AbstractVector)
    ra.anchor = ra.alpha .* ra.anchor .+ (1.0-ra.alpha) .* Float64.(new_pv)
    nrm = norm(ra.anchor); nrm > 1e-10 && (ra.anchor ./= nrm)
end

function alignment(ra::RotatingAnchor, pv::AbstractVector)
    a_nrm = norm(ra.anchor); a_nrm < 1e-10 && return 0.0
    return max(0.0, dot(pv, ra.anchor) / (norm(pv) * a_nrm + 1e-10))
end
end

