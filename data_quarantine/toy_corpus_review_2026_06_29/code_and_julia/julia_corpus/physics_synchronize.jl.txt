"""Synchronize — مزامنة هيبيانية (بناء Vocab + K من النصوص)."""
module Synchronize
using SparseArrays, LinearAlgebra
export synchronize, Vocabulary

mutable struct Vocabulary
    word2id::Dict{String,Int}
    id2word::Dict{Int,String}
    next_id::Int
end
Vocabulary() = Vocabulary(Dict(), Dict(), 1)

function Base.get(vocab::Vocabulary, word::String)
    if !haskey(vocab.word2id, word)
        vocab.word2id[word] = vocab.next_id
        vocab.id2word[vocab.next_id] = word
        vocab.next_id += 1
    end
    return vocab.word2id[word]
end
Base.length(vocab::Vocabulary) = length(vocab.word2id)

function synchronize(texts::Vector{String}; mode::String="sem", window::Int=10, vocab=nothing)
    if vocab === nothing; vocab = Vocabulary(); end
    
    for text in texts
        for word in split(text)
            if length(word) >= 2; get(vocab, word); end  # Register word
        end
    end
    
    V = length(vocab)
    cooc = spzeros(V, V)
    
    w = mode == "syn" ? 4 : window
    for text in texts
        words = String[w for w in split(text) if length(w) >= 2]
        ids = [vocab.word2id[w] for w in words]
        for i in 1:length(ids)
            for j in max(1, i-w):min(length(ids), i+w)
                j == i && continue
                weight = 1.0 / abs(j - i)
                cooc[ids[i], ids[j]] += weight
            end
        end
    end
    
    return vocab, cooc, nothing  # vocab, K, syntax_field
end
end
