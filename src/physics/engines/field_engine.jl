"""FieldEngine — حقل الجذب والتنافر (GeometricFieldEngine)."""
module FieldEngineModule
using LinearAlgebra, SparseArrays
using ..WordPhysics: compute_word_phase_vector
using ..Constants: PHASE_DIM

export FieldEngine, PHYSICAL_DIM

const PHYSICAL_DIM = 22

mutable struct FieldEngine
    vocab::Dict{String,Int}
    id2word::Dict{Int,String}
    all_pvs::Matrix{Float64}
    K_sem::Union{SparseMatrixCSC,Nothing}
end

function FieldEngine(vocab::Dict{String,Int})
    id2word = Dict{Int,String}(v=>k for (k,v) in vocab)
    n = length(vocab)
    all_pvs = zeros(Float64, n, PHYSICAL_DIM)
    words = String[id2word[i] for i in 0:n-1]
    for i in 0:n-1
        word = get(id2word, i, "")
        if !isempty(word)
            pv = compute_word_phase_vector(word)
            all_pvs[i+1, :] .= Float64.(pv[1:PHYSICAL_DIM])
        end
    end
    FieldEngine(vocab, id2word, all_pvs, nothing)
end

const _HARAKAT = ('ً','ٌ','ٍ','َ','ُ','ِ','ّ','ْ','ـ')

_strip_harakat(w::String) = String(filter(c -> c ∉ _HARAKAT, w))

function find_attraction_repulsion(fe::FieldEngine, target_word::String; top_k=15)
    tid = get(fe.vocab, target_word, nothing)
    tw = target_word
    if tid === nothing
        tw = _strip_harakat(target_word)
        tid = get(fe.vocab, tw, nothing)
    end
    tid === nothing && return Dict("attracted"=>[], "repelled"=>[])

    target_pv = view(fe.all_pvs, min(tid+1, size(fe.all_pvs,1)), :)  # 1-indexed
    n_target = norm(target_pv)
    n_target < 1e-10 && return Dict("attracted"=>[], "repelled"=>[])

    norms = [norm(view(fe.all_pvs, i, :)) for i in 1:size(fe.all_pvs,1)]
    sims = [(dot(view(fe.all_pvs,i,:), target_pv) / (max(norms[i],1e-10) * n_target), fe.id2word[i-1]) for i in 1:size(fe.all_pvs,1)]

    sort!(sims; by=x->-x[1])
    attracted = [(w,round(s; digits=4)) for (s,w) in sims[1:top_k] if w != tw]
    repelled = [(w,round(s; digits=4)) for (s,w) in reverse(sims[end-top_k:end]) if w != tw]
    return Dict("attracted"=>attracted, "repelled"=>repelled)
end
end

