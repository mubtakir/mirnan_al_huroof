"""ConceptMatrix — مصفوفة مفاهيم من التوقيعات الطيفية."""
module ConceptMatrixModule
using LinearAlgebra, SparseArrays

export ConceptMatrix, build_multi_k

mutable struct ConceptMatrix
    gss_cache::Dict{Int,Vector{Float64}}
    K_conc::Union{SparseMatrixCSC,Nothing}
end
ConceptMatrix(gss_cache=nothing) = ConceptMatrix(gss_cache === nothing ? Dict{Int,Vector{Float64}}() : gss_cache, nothing)

function build!(cm::ConceptMatrix, V::Int; top_k=100)
    cooc = spzeros(V, V)
    items = collect(cm.gss_cache)
    n = length(items)
    for a in 1:n
        (i, gss_i) = items[a]; i >= V && continue
        sims = Tuple{Int,Float64}[]
        for b in (a+1):n
            (j, gss_j) = items[b]; j >= V && continue
            sim = dot(gss_i, gss_j)
            sim > 0.05 && push!(sims, (j, sim))
        end
        sort!(sims; by=x->-x[2])
        for (j, sim) in sims[1:min(top_k, end)]
            cooc[i, j] = sim; cooc[j, i] = sim
        end
    end
    cm.K_conc = cooc
    return cm.K_conc
end
end

