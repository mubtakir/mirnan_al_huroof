"""
AssociativeDialogueMemory — ذاكرة ترابطية (intent+topic → response).

تخزين أزواج (مفتاح، قيمة) كمتجهات طورية مع تعزيز هيبياني.
"""
module AssociativeMemory
using LinearAlgebra

export AssociativeEntry, AssociativeDialogueMemory

mutable struct AssociativeEntry
    key_pv::Vector{Float64}
    value_pv::Vector{Float64}
    key_words::String
    value_words::String
    strength::Float64
    use_count::Int
end

mutable struct AssociativeDialogueMemory
    entries::Vector{AssociativeEntry}
    lr::Float64
    decay::Float64
    max_entries::Int
end

AssociativeDialogueMemory(; lr=0.1, decay=0.001, max_entries=500) =
    AssociativeDialogueMemory(AssociativeEntry[], lr, decay, max_entries)

function store!(mem::AssociativeDialogueMemory, key_pv, value_pv; key_words="", value_words="")
    # Check for near-duplicate
    for e in mem.entries
        if e.key_words == key_words
            e.value_pv = (e.value_pv + value_pv) / 2.0
            e.value_pv ./= norm(e.value_pv)
            e.strength = min(1.0, e.strength + mem.lr)
            e.use_count += 1
            return
        end
    end
    if length(mem.entries) >= mem.max_entries
        sort!(mem.entries; by=e -> e.strength)
        popfirst!(mem.entries)
    end
    push!(mem.entries, AssociativeEntry(Float64.(key_pv), Float64.(value_pv), key_words, value_words, mem.lr, 1))
end

function retrieve(mem::AssociativeDialogueMemory, query_pv; top_k=3)
    if isempty(mem.entries); return []; end
    scored = [(dot(e.key_pv, query_pv) * e.strength, e) for e in mem.entries]
    sort!(scored; by=x -> -x[1])
    return [(s, e.value_words, e.value_pv) for (s, e) in scored[1:min(top_k, end)] if s > 0.1]
end

function reinforce!(mem::AssociativeDialogueMemory, key_words::String, reward=0.5)
    for e in mem.entries
        if e.key_words == key_words
            e.strength = min(1.0, e.strength + reward * mem.lr)
            return
        end
    end
end

function penalize!(mem::AssociativeDialogueMemory, key_words::String, penalty=0.3)
    for e in mem.entries
        if e.key_words == key_words
            e.strength = max(0.0, e.strength - penalty * mem.lr)
            return
        end
    end
end
end
