"""LocalThermoGate — بوابة حرارية محلية لكل كلمة (Θ_vec)."""
module LocalThermoGate
using LinearAlgebra
export local_entropy, local_temperature, compute_theta, adjust_beta, adjust_k_B

function local_entropy(pv_list::Vector{<:AbstractVector}, target_pv::AbstractVector)
    isempty(pv_list) && return 0.0
    sims = Float64[max(0.0, dot(pv, target_pv) / (norm(pv)*norm(target_pv) + 1e-10)) for pv in pv_list]
    s_sum = sum(sims); s_sum < 1e-10 && return 0.0
    p = sims ./ s_sum
    return -sum(p .* log.(p .+ 1e-10))
end

local_temperature(S::Float64; k_B=0.1) = S > 0 ? exp(S / k_B) : 0.0
compute_theta(pv::AbstractVector, context_pv::AbstractVector) = max(0.0, dot(pv, context_pv) / (norm(pv)*norm(context_pv) + 1e-10))
adjust_beta(beta::Float64, theta::Float64) = beta * (1.0 + 0.1 * theta)
adjust_k_B(k_B::Float64, S::Float64) = max(0.01, k_B * (1.0 - 0.05 * S))
end
