"""
Causal Flow Field — حقل التدفق السببي.
Physical alternative to multi-step logical inference.
Converts causal relationships to flow currents in phase space.

J(pv) = Σᵢ wᵢ · Cᵢ · (pv_target - pv)
"""
module CausalFlow

using LinearAlgebra, SparseArrays

export CausalFlowField, compute_flow, flow_alignment_score, compute_transitive_flow

mutable struct CausalFlowField
    dim::Int
    flow_strength::Float64
    flow_cache::Dict

    function CausalFlowField(; dim::Int=64, flow_strength::Float64=1.0)
        return new(dim, flow_strength, Dict())
    end
end

function _project_dim(v::AbstractVector, dim::Int)
    out = zeros(Float64, dim)
    n = min(length(v), dim)
    n > 0 && (out[1:n] .= Float64.(v[1:n]))
    return out
end

function compute_flow(cf::CausalFlowField, current_pv::AbstractVector,
                      context_pvs::Vector{<:AbstractVector},
                      context_ids::Vector{Int};
                      causal_matrix::Union{AbstractMatrix,Nothing}=nothing)
    n = length(context_ids)
    if n == 0 || causal_matrix === nothing
        return Dict("flow_vector" => zeros(cf.dim),
                    "flow_magnitude" => 0.0,
                    "causal_chain" => [],
                    "logical_score" => 0.0)
    end

    flow_vector = zeros(Float64, cf.dim)
    current_vec = _project_dim(current_pv, cf.dim)
    total_strength = 0.0
    causal_chain = []

    for i in 1:min(n, 6)
        cid = context_ids[end-(i-1)]
        if cid < 1 || cid > size(causal_matrix, 1)
            continue
        end

        row = causal_matrix[cid, :]
        if row isa SparseVector
            row = Vector(row)
        end

        sorted_idx = sortperm(row; rev=true)[1:min(5, end)]
        for tid in sorted_idx
            strength = float(row[tid])
            if strength <= 0.15
                continue
            end

            target_pv = if tid <= length(context_pvs)
                _project_dim(context_pvs[tid], cf.dim)
            else
                current_vec
            end

            direction = target_pv .- current_vec
            dir_norm = norm(direction)
            if dir_norm > 1e-10
                direction ./= dir_norm
            end

            weight = exp(-0.5 * (i - 1))
            flow_vector .+= cf.flow_strength .* weight .* strength .* direction
            total_strength += weight * strength

            push!(causal_chain, Dict(
                "cause_idx" => cid, "effect_idx" => tid,
                "strength" => round(strength; digits=3),
                "position_penalty" => round(weight; digits=3),
            ))
        end
    end

    flow_magnitude = norm(flow_vector)
    if flow_magnitude > 1e-10
        flow_vector ./= flow_magnitude
    end

    logical_score = tanh(total_strength * 2.0)

    return Dict("flow_vector" => flow_vector,
                "flow_magnitude" => flow_magnitude,
                "causal_chain" => causal_chain,
                "logical_score" => logical_score)
end

function flow_alignment_score(cf::CausalFlowField, candidate_pv::AbstractVector,
                               current_pv::AbstractVector, flow_vector::AbstractVector)
    if norm(flow_vector) < 1e-10
        return 0.0
    end

    candidate_direction = _project_dim(candidate_pv, cf.dim) .- _project_dim(current_pv, cf.dim)
    dir_norm = norm(candidate_direction)
    if dir_norm < 1e-10
        return 0.0
    end
    candidate_direction ./= dir_norm

    alignment = dot(candidate_direction, _project_dim(flow_vector, cf.dim))
    return max(0.0, alignment)
end

function compute_transitive_flow(cf::CausalFlowField, word_ids::Vector{Int},
                                  causal_matrix::AbstractMatrix)
    if length(word_ids) < 2
        return Dict("flow_strength" => 0.0, "chain_length" => 0, "chain" => [])
    end

    cumulative = 1.0
    chain = []
    chain_length = 0

    for i in 1:(length(word_ids)-1)
        cid = word_ids[i]
        eid = word_ids[i+1]
        if cid < 1 || cid > size(causal_matrix, 1)
            continue
        end
        if eid < 1 || eid > size(causal_matrix, 2)
            continue
        end

        strength = float(causal_matrix[cid, eid])
        if strength > 0.15
            cumulative *= min(strength, 1.0)
            chain_length += 1
            push!(chain, Dict("from" => cid, "to" => eid,
                              "strength" => round(strength; digits=3)))
        end
    end

    return Dict("flow_strength" => round(cumulative; digits=4),
                "chain_length" => chain_length, "chain" => chain)
end

end # module CausalFlow
