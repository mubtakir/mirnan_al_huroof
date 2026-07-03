"""SyntaxPhase — طور نحوي (SyntaxTransitionField)."""
module SyntaxPhase
using LinearAlgebra
export SyntaxTransitionField

mutable struct SyntaxTransitionField
    transition_matrix::Matrix{Float64}; counts::Matrix{Int}
end
SyntaxTransitionField(n_types::Int=6) = SyntaxTransitionField(ones(n_types,n_types)./n_types, zeros(Int,n_types,n_types))

function observe!(stf::SyntaxTransitionField, from_idx::Int, to_idx::Int)
    stf.counts[from_idx, to_idx] += 1
    row_total = sum(view(stf.counts, from_idx, :))
    if row_total > 0
        stf.transition_matrix[from_idx, :] .= stf.counts[from_idx, :] ./ row_total
    end
end

function gate(stf::SyntaxTransitionField, from_idx::Int, to_idx::Int)
    1 <= from_idx <= size(stf.transition_matrix,1) && 1 <= to_idx <= size(stf.transition_matrix,2) || return 0.5
    return stf.transition_matrix[from_idx, to_idx]
end
end
