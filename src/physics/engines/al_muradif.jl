module AlMuradif

using JSON
using SparseArrays

export MuradifCandidate, MuradifMemory,
       build_muradif_memory, muradif_terms,
       merge_muradif!,
       save_muradif, load_muradif, muradif_to_dict,
       has_muradif_records

const AL_MURADIF_VERSION = 1

mutable struct MuradifCandidate
    term::String
    score::Float64
    kind::String
    semantic_score::Float64
    syntactic_score::Float64
    causal_score::Float64
    direct_score::Float64
end

mutable struct MuradifMemory
    entries::Dict{String,Vector{MuradifCandidate}}
    max_candidates::Int
end

MuradifMemory(; max_candidates::Int=6) =
    MuradifMemory(Dict{String,Vector{MuradifCandidate}}(), max_candidates)

has_muradif_records(mem::MuradifMemory) = !isempty(mem.entries)

_clean_space(s::AbstractString) = replace(strip(String(s)), r"\s+" => " ")

function _norm_word(s::AbstractString)
    x = lowercase(_clean_space(s))
    x = replace(x, 'أ' => 'ا', 'إ' => 'ا', 'آ' => 'ا', 'ى' => 'ي', 'ة' => 'ه')
    x = replace(x, r"^[\W_]+|[\W_]+$" => "")
    x = replace(x, r"^(?:وال|فال|بال|كال|لل|ال)" => "")
    x = replace(x, r"[\.\,\،\؛\:\!\?\u061F\"'\(\)\[\]\{\}]" => "")
    return strip(x)
end

function _top_column(mat, idx::Int, id2word::Dict{Int,String}, limit::Int)
    mat === nothing && return Dict{String,Float64}()
    (idx < 1 || idx > size(mat, 2)) && return Dict{String,Float64}()
    col = mat[:, idx]
    rows = findnz(col)[1]
    vals = findnz(col)[2]
    pairs = Tuple{String,Float64}[]
    for (r, v) in zip(rows, vals)
        r == idx && continue
        w = get(id2word, Int(r), "")
        isempty(w) && continue
        push!(pairs, (w, Float64(v)))
    end
    sort!(pairs; by=x -> x[2], rev=true)
    out = Dict{String,Float64}()
    maxv = isempty(pairs) ? 1.0 : max(abs(pairs[1][2]), eps())
    for (w, v) in pairs[1:min(limit, length(pairs))]
        out[_norm_word(w)] = max(get(out, _norm_word(w), 0.0), abs(v) / maxv)
    end
    return out
end

function _direct_weight(mat, a::Int, b::Int)
    mat === nothing && return 0.0
    (a < 1 || b < 1 || a > size(mat, 1) || b > size(mat, 2)) && return 0.0
    return abs(Float64(mat[a, b]))
end

function _weighted_jaccard(a::Dict{String,Float64}, b::Dict{String,Float64})
    isempty(a) && isempty(b) && return 0.0
    keys_all = union(keys(a), keys(b))
    num = 0.0
    den = 0.0
    for k in keys_all
        av = get(a, k, 0.0)
        bv = get(b, k, 0.0)
        num += min(av, bv)
        den += max(av, bv)
    end
    return den <= eps() ? 0.0 : num / den
end

function _classify(sem::Float64, syn::Float64, causal::Float64, direct::Float64)
    if sem >= 0.28 && syn >= 0.12 && causal >= 0.08 && direct <= 0.35
        return "synonym"
    elseif sem >= 0.22 || causal >= 0.18 || direct >= 0.20
        return "thematic"
    elseif syn >= 0.25 && sem < 0.18
        return "syntactic"
    end
    return "weak"
end

function _combined_score(sem::Float64, syn::Float64, causal::Float64, direct::Float64)
    return clamp(0.50 * sem + 0.20 * syn + 0.20 * causal + 0.10 * min(direct, 1.0), 0.0, 1.0)
end

function build_muradif_memory(vocab::Dict{String,Int}, K_sem;
                              K_syn=nothing, K_causal=nothing,
                              max_words::Int=5000,
                              top_neighbors::Int=24,
                              max_candidates::Int=6,
                              min_score::Float64=0.16)
    id2word = Dict{Int,String}(v => k for (k, v) in vocab)
    ordered = sort(collect(vocab); by=x -> x[2])
    sampled = ordered[1:min(max_words, length(ordered))]
    sem_fp = Dict{String,Dict{String,Float64}}()
    syn_fp = Dict{String,Dict{String,Float64}}()
    causal_fp = Dict{String,Dict{String,Float64}}()

    for (word, idx) in sampled
        key = _norm_word(word)
        isempty(key) && continue
        sem_fp[key] = _top_column(K_sem, idx, id2word, top_neighbors)
        syn_fp[key] = _top_column(K_syn, idx, id2word, top_neighbors)
        causal_fp[key] = _top_column(K_causal, idx, id2word, top_neighbors)
    end

    mem = MuradifMemory(max_candidates=max_candidates)
    for (word, idx) in sampled
        key = _norm_word(word)
        isempty(key) && continue
        pool = Set{String}()
        union!(pool, keys(get(sem_fp, key, Dict{String,Float64}())))
        union!(pool, keys(get(syn_fp, key, Dict{String,Float64}())))
        union!(pool, keys(get(causal_fp, key, Dict{String,Float64}())))

        candidates = MuradifCandidate[]
        for cand in pool
            cand == key && continue
            cand_idx = get(vocab, cand, get(vocab, "ال" * cand, 0))
            cand_idx == 0 && continue
            sem = _weighted_jaccard(get(sem_fp, key, Dict{String,Float64}()),
                                    get(sem_fp, cand, Dict{String,Float64}()))
            syn = _weighted_jaccard(get(syn_fp, key, Dict{String,Float64}()),
                                    get(syn_fp, cand, Dict{String,Float64}()))
            causal = _weighted_jaccard(get(causal_fp, key, Dict{String,Float64}()),
                                       get(causal_fp, cand, Dict{String,Float64}()))
            direct = max(_direct_weight(K_sem, idx, cand_idx), _direct_weight(K_sem, cand_idx, idx))
            score = _combined_score(sem, syn, causal, direct)
            score >= min_score || continue
            kind = _classify(sem, syn, causal, direct)
            kind == "weak" && continue
            push!(candidates, MuradifCandidate(cand, score, kind, sem, syn, causal, direct))
        end
        sort!(candidates; by=x -> x.score, rev=true)
        !isempty(candidates) && (mem.entries[key] = candidates[1:min(max_candidates, length(candidates))])
    end
    return mem
end

function muradif_terms(mem::MuradifMemory, word::AbstractString;
                       kinds::Set{String}=Set(["synonym", "thematic"]),
                       min_score::Float64=0.20,
                       limit::Int=6)
    key = _norm_word(word)
    out = String[]
    for cand in get(mem.entries, key, MuradifCandidate[])
        cand.score >= min_score || continue
        cand.kind in kinds || continue
        push!(out, cand.term)
        length(out) >= limit && break
    end
    return out
end

function merge_muradif!(base::MuradifMemory, extra::MuradifMemory)
    for (word, incoming) in extra.entries
        bucket = get!(base.entries, word, MuradifCandidate[])
        by_term = Dict{String,Int}(c.term => i for (i, c) in enumerate(bucket))
        for cand in incoming
            if haskey(by_term, cand.term)
                old = bucket[by_term[cand.term]]
                if cand.score > old.score
                    bucket[by_term[cand.term]] = cand
                end
            else
                push!(bucket, cand)
            end
        end
        sort!(bucket; by=x -> x.score, rev=true)
        length(bucket) > base.max_candidates && resize!(bucket, base.max_candidates)
        base.entries[word] = bucket
    end
    return base
end

function _candidate_to_dict(c::MuradifCandidate)
    return Dict(
        "term" => c.term,
        "score" => c.score,
        "kind" => c.kind,
        "semantic_score" => c.semantic_score,
        "syntactic_score" => c.syntactic_score,
        "causal_score" => c.causal_score,
        "direct_score" => c.direct_score,
    )
end

function muradif_to_dict(mem::MuradifMemory)
    entries = Dict{String,Any}()
    for (word, cands) in mem.entries
        entries[word] = [_candidate_to_dict(c) for c in cands]
    end
    return Dict(
        "version" => AL_MURADIF_VERSION,
        "max_candidates" => mem.max_candidates,
        "entries" => entries,
    )
end

function save_muradif(mem::MuradifMemory, path::AbstractString)
    open(String(path), "w") do io
        JSON.print(io, muradif_to_dict(mem), 2)
    end
    return String(path)
end

function load_muradif(path::AbstractString)
    isfile(String(path)) || return MuradifMemory()
    data = JSON.parsefile(String(path))
    mem = MuradifMemory(max_candidates=Int(get(data, "max_candidates", 6)))
    for (word, raw_cands) in get(data, "entries", Dict{String,Any}())
        mem.entries[String(word)] = MuradifCandidate[
            MuradifCandidate(
                String(item["term"]),
                Float64(item["score"]),
                String(item["kind"]),
                Float64(get(item, "semantic_score", 0.0)),
                Float64(get(item, "syntactic_score", 0.0)),
                Float64(get(item, "causal_score", 0.0)),
                Float64(get(item, "direct_score", 0.0)),
            )
            for item in raw_cands
        ]
    end
    return mem
end

end # module
