"""
Phase Density Matrix — مصفوفة الكثافة الطورية.
Represent context as a quantum mixed state:
  ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|

Physical alternative to vector averaging (which loses inter-relationships).
Resonance: R(v) = ⟨v| ρ |v⟩
"""
module DensityMatrix

using LinearAlgebra

export PhaseDensityMatrix, build!, resonance, get_trace, get_purity, get_spectral_entropy

mutable struct PhaseDensityMatrix
    dim::Int
    decay_rate::Float64
    rho::Union{Matrix{Float64},Nothing}
    n_words::Int
    word_count::Int
    _basis_pvs::Vector{Vector{Float64}}
    _basis_weights::Vector{Float64}

    function PhaseDensityMatrix(; dim::Int=64, decay_rate::Float64=0.8)
        return new(dim, decay_rate, nothing, 0, 0, Vector{Float64}[], Float64[])
    end
end

function build!(dm::PhaseDensityMatrix, pv_list::Vector{<:AbstractVector};
                dims::Union{Int,Tuple{Int,Int},Nothing}=nothing)
    if dims !== nothing
        if dims isa Tuple
            pvs = [Float64.(pv[dims[1]:dims[2]]) for pv in pv_list]
            d = dims[2] - dims[1] + 1
        else
            pvs = [Float64.(pv[1:dims]) for pv in pv_list]
            d = dims
        end
    else
        d = min(dm.dim, length(pv_list[1]))
        pvs = [Float64.(pv[1:d]) for pv in pv_list]
    end

    n = length(pvs)
    if n == 0
        dm.rho = zeros(d, d)
        dm.n_words = 0
        dm._basis_pvs = Vector{Float64}[]
        dm._basis_weights = Float64[]
        return dm.rho
    end

    weights = exp.(-dm.decay_rate .* Float64.(0:n-1))
    weights = reverse!(weights)
    weights ./= sum(weights)

    basis_pvs = Vector{Float64}[]
    basis_weights = Float64[]
    rho = zeros(Float64, d, d)
    for i in 1:n
        v = pvs[i][1:min(d, end)]
        v_norm = norm(v)
        if v_norm > 1e-10
            v ./= v_norm
        end
        push!(basis_pvs, copy(v))
        push!(basis_weights, weights[i])
        rho .+= weights[i] .* (v * v')
    end

    dm.rho = rho
    dm.n_words = n
    dm.word_count = n
    dm._basis_pvs = basis_pvs
    dm._basis_weights = basis_weights
    return dm.rho
end

function resonance(dm::PhaseDensityMatrix, candidate_pv::AbstractVector;
                   dims::Union{Int,Tuple{Int,Int},Nothing}=nothing)
    if dm.n_words == 0 || isempty(dm._basis_pvs)
        return 0.0
    end

    if dims !== nothing
        if dims isa Tuple
            v = Float64.(candidate_pv[dims[1]:dims[2]])
        else
            v = Float64.(candidate_pv[1:dims])
        end
    else
        d = length(dm._basis_pvs[1])
        v = Float64.(candidate_pv[1:min(d, end)])
    end

    v_norm = norm(v)
    if v_norm > 1e-10
        v ./= v_norm
    end

    resonance_val = 0.0
    for (i, bv) in enumerate(dm._basis_pvs)
        dv = min(length(v), length(bv))
        cos_sim = dot(view(v, 1:dv), view(bv, 1:dv))
        resonance_val += dm._basis_weights[i] * cos_sim * cos_sim
    end
    return clamp(resonance_val, 0.0, 1.0)
end

function get_trace(dm::PhaseDensityMatrix)
    dm.rho === nothing && return 0.0
    return tr(dm.rho)
end

function get_purity(dm::PhaseDensityMatrix)
    dm.rho === nothing && return 0.0
    return tr(dm.rho * dm.rho)
end

function get_spectral_entropy(dm::PhaseDensityMatrix)
    if dm.rho === nothing || dm.n_words == 0
        return 0.0
    end
    eigenvals = eigvals(Hermitian(dm.rho))
    eigenvals = max.(eigenvals, 1e-12)
    return -sum(eigenvals .* log.(eigenvals))
end

end # module DensityMatrix
