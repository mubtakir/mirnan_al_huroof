"""CausalPhaseEngine — غرفة الرنين السببي (سببية موجهة من ترتيب الكلمات)."""
module CausalEngine
using SparseArrays, LinearAlgebra

export CausalPhaseEngine

mutable struct CausalPhaseEngine
    causal_K::Union{SparseMatrixCSC,Nothing}
    is_built::Bool
end
CausalPhaseEngine() = CausalPhaseEngine(nothing, false)

function build_from_corpus!(cpe::CausalPhaseEngine, texts::Vector{String}, vocab::Dict{String,Int}; window=5)
    V = length(vocab)
    before = spzeros(V, V)
    after = spzeros(V, V)
    for text in texts
        for line in split(text, '\n')
            tokens = [t for t in split(line) if haskey(vocab, t)]
            ids = [vocab[t] for t in tokens]
            for i in 1:length(ids)
                for j in (i+1):min(i+window, length(ids))
                    before[ids[i], ids[j]] += 1.0
                    after[ids[j], ids[i]] += 1.0
                end
            end
        end
    end
    I, J, Vals = Int[], Int[], Float64[]
    rows = rowvals(before)
    vals_b = nonzeros(before)
    for col in 1:V
        for p in nzrange(before, col)
            row = rows[p]; b = vals_b[p]
            row == col && continue
            a = after[row, col]
            total = b + a
            total < 1 && continue
            strength = (b - a) / total
            if abs(strength) > 0.15
                push!(I, row); push!(J, col); push!(Vals, strength)
            end
        end
    end
    cpe.causal_K = sparse(I, J, Vals, V, V)
    cpe.is_built = true
    return cpe.causal_K
end

function causal_strength(cpe::CausalPhaseEngine, cause_id::Int, effect_id::Int)
    !cpe.is_built && return 0.0
    (cause_id > size(cpe.causal_K,1) || effect_id > size(cpe.causal_K,2)) && return 0.0
    return max(0.0, cpe.causal_K[cause_id, effect_id])
end

function score_candidate(cpe::CausalPhaseEngine, word_id::Int, context_ids::Vector{Int})
    !cpe.is_built && return 0.0
    forward = 0.0; backward = 0.0; n_fwd=0; n_bwd=0
    for cid in context_ids[max(1, end-min(6, length(context_ids))):end]
        fwd = causal_strength(cpe, cid, word_id)
        if fwd > 0.3; forward += fwd; n_fwd += 1; end
        bwd = causal_strength(cpe, word_id, cid)
        if bwd > 0.3; backward += bwd; n_bwd += 1; end
    end
    n_fwd + n_bwd == 0 && return 0.0
    net = (forward - backward*0.5) / max(n_fwd + n_bwd, 1)
    return tanh(net * 2.0)
end
end
