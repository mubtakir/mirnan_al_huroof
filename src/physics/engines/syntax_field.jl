"""
Syntax Field — الفضاء النحوي.
Arabic word type analysis and syntax vector computation.
6D syntax vectors for KANA/JAR/CONJ detection.
Supports English via _is_english_word detection.
"""
module SyntaxField

using LinearAlgebra, SparseArrays

using ..Constants: SYNTAX_DIMS
using ..WordPhysics: _parse_word_harakat, _is_english_word, _normalize_letters, IRAB_MAP

export SYNTAX_ANCHORS, compute_syntax_vector, expected_syntax, SyntaxFieldCache,
       KANA_VERBS, JAR_PREPS_LIST, CONJUNCTIONS_LIST, NEGATIONS_LIST

const SYNTAX_ANCHORS = Dict(
    "verb" => [1.00, 0.20, 0.00, 0.00, 0.15, 0.10],
    "noun" => [0.20, 1.00, 0.70, 0.00, 0.30, 0.40],
    "prep" => [0.00, 0.70, 1.00, 0.00, 0.10, 0.00],
    "part" => [0.10, 0.10, 0.00, 1.00, 0.10, 0.00],
    "conj" => [0.20, 0.20, 0.10, 0.05, 1.00, 0.05],
    "kana" => [0.10, 0.50, 0.00, 0.00, 0.05, 1.00],
)

const KANA_VERBS = Set([
    "كان", "كانت", "ليس", "ليست", "أصبح", "صار", "ما زال",
    "يكون", "تكون", "أمسى", "بات", "ليسوا",
])

const JAR_PREPS_LIST = Set([
    "في", "من", "على", "إلى", "عن", "بـ", "لـ", "كـ", "حتى",
    "منذ", "مذ", "رب", "واو", "تاء",
])

const CONJUNCTIONS_LIST = Set(["و", "فـ", "ثم", "أو", "أم", "بل", "لكن", "حتى"])
const NEGATIONS_LIST = Set(["لا", "لم", "لن", "ما", "ليس", "غير", "إن"])

function _l2(v::AbstractVector)
    n = norm(v)
    return n > 1e-10 ? v ./ n : v
end

function _get_syntax_anchor(word::String)
    if word in KANA_VERBS
        return SYNTAX_ANCHORS["kana"]
    end
    if word in JAR_PREPS_LIST
        return SYNTAX_ANCHORS["prep"]
    end
    if word in NEGATIONS_LIST
        return SYNTAX_ANCHORS["part"]
    end
    if word in CONJUNCTIONS_LIST
        return SYNTAX_ANCHORS["conj"]
    end
    return SYNTAX_ANCHORS["noun"]
end

const _EN_SYNTAX_ANCHORS = Dict(
    "DET" => [0.0, 0.8, 0.0, 0.0, 0.0, 0.0],
    "PREP" => [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "CONJ" => [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "PRON" => [0.0, 0.5, 0.0, 0.0, 0.0, 0.5],
    "VERB_BE" => [1.0, 0.2, 0.0, 0.0, 0.15, 0.1],
    "VERB_MODAL" => [0.8, 0.3, 0.0, 0.0, 0.2, 0.1],
    "NOUN" => [0.2, 1.0, 0.7, 0.0, 0.3, 0.4],
    "VERB_GERUND" => [0.9, 0.3, 0.0, 0.0, 0.1, 0.1],
    "VERB_PAST" => [0.85, 0.25, 0.0, 0.0, 0.1, 0.1],
    "ADV" => [0.4, 0.3, 0.1, 0.0, 0.1, 0.6],
)

const _EN_DET = Set(["the","a","an","this","that","these","those"])
const _EN_PREP = Set(["in","on","at","to","from","with","by","for","of","about","into","onto","upon","within","without","during","through","between","among","against","after","before","above","below","near","under","over"])
const _EN_CONJ = Set(["and","or","but","so","because","if","when","while","although","since","yet","nor","either","neither","both","whether"])
const _EN_PRON = Set(["i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","its","our","their","mine","yours","hers","ours","theirs","this","that","these","those","who","whom","which","what","whose","where","when","how","why"])
const _EN_BE = Set(["is","are","was","were","am","be","been","being"])
const _EN_MODAL = Set(["can","could","will","would","shall","should","may","might","must"])
const _EN_ADV = Set(["not","never","always","often","sometimes","usually","really","very","quite","just","still","already","soon","now","here","there","together","alone","well","quickly","slowly","easily","really","extremely"])

function _en_get_pos(word::String)
    w = lowercase(word)
    w in _EN_DET && return "DET"
    w in _EN_PREP && return "PREP"
    w in _EN_CONJ && return "CONJ"
    w in _EN_PRON && return "PRON"
    w in _EN_BE && return "VERB_BE"
    w in _EN_MODAL && return "VERB_MODAL"
    w in _EN_ADV && return "ADV"
    occursin(r"ing$", w) && return "VERB_GERUND"
    occursin(r"ed$", w) && length(w) > 4 && return "VERB_PAST"
    occursin(r"ly$", w) && length(w) > 4 && return "ADV"
    occursin(r"(tion|sion|ment|ness|ence|ance|ity|ism|ist|ure)$", w) && return "NOUN"
    occursin(r"(ful|less|able|ible|ous|ive|al|ial|ical|ent|ant)$", w) && return "NOUN"
    occursin(r"er$", w) && length(w) > 4 && return "NOUN"
    occursin(r"est$", w) && length(w) > 5 && return "NOUN"
    return "NOUN"
end

function _en_syntax_vector(word::String)
    pos = _en_get_pos(word)
    anchor = get(_EN_SYNTAX_ANCHORS, pos, [0.2, 1.0, 0.7, 0.0, 0.3, 0.4])
    n = norm(anchor)
    return n > 1e-10 ? anchor ./ n : anchor
end

function compute_syntax_vector(word::String)
    if _is_english_word(word)
        return _en_syntax_vector(word)
    end
    return _l2(_get_syntax_anchor(word))
end

function expected_syntax(prev_word::String, vocab::Dict{String,Int}, K::AbstractMatrix;
                         id2word=nothing)
    pid = get(vocab, prev_word, nothing)
    if pid === nothing || pid > size(K, 1)
        return zeros(Float64, SYNTAX_DIMS)
    end
    _id2word = id2word !== nothing ? id2word : Dict(v=>k for (k,v) in vocab)
    row = K[pid, :]
    if row isa AbstractSparseVector
        row = Vector(row)
    end
    total = zeros(Float64, SYNTAX_DIMS)
    weight_sum = 0.0
    sorted_idx = sortperm(row; rev=true)[1:min(30, end)]
    for tid in sorted_idx
        row[tid] <= 1e-3 && continue
        w = get(_id2word, tid, "")
        anchor = _get_syntax_anchor(w)
        total .+= row[tid] .* _l2(anchor)
        weight_sum += row[tid]
    end
    if weight_sum > 1e-10
        total ./= weight_sum
    end
    return _l2(total)
end

mutable struct SyntaxFieldCache
    cache::Dict{String,Vector{Float64}}
    exp_cache::Dict{String,Vector{Float64}}
    vocab::Dict{String,Int}
    K::AbstractMatrix

    function SyntaxFieldCache(vocab::Dict{String,Int}, K::AbstractMatrix)
        return new(Dict(), Dict(), vocab, K)
    end
end

function Base.get(cache::SyntaxFieldCache, word::String)
    if !haskey(cache.cache, word)
        if length(word) >= 2
            cache.cache[word] = compute_syntax_vector(word)
        else
            cache.cache[word] = zeros(Float64, SYNTAX_DIMS)
        end
    end
    return cache.cache[word]
end

function get_expected(cache::SyntaxFieldCache, prev_word::String)
    if !haskey(cache.exp_cache, prev_word)
        cache.exp_cache[prev_word] = expected_syntax(prev_word, cache.vocab, cache.K)
    end
    return cache.exp_cache[prev_word]
end

end # module SyntaxField
