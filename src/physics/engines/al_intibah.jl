module AlIntibah

using SparseArrays

using ..WordPhysics

export SemanticAttentionField, build_semantic_attention,
       has_semantic_attention, attention_bias_terms

struct SemanticAttentionField
    anchors::Vector{String}
    expanded_terms::Vector{String}
    bridge_terms::Vector{String}
    positive_bias::Dict{String,Float64}
    negative_bias::Dict{String,Float64}
    confidence::Float64
end

const AR_AL = "\u0627\u0644"
const STOP_KEYS = Set([
    "\u0645\u0627", "\u0645\u0627\u0630\u0627", "\u0647\u0644", "\u0643\u064a\u0641",
    "\u0644\u0645\u0627\u0630\u0627", "\u0645\u0646", "\u0645\u062a\u0649",
    "\u0627\u064a\u0646", "\u0623\u064a\u0646", "\u0647\u0648", "\u0647\u064a",
    "\u0647\u0630\u0627", "\u0647\u0630\u0647", "\u0630\u0644\u0643", "\u062a\u0644\u0643",
    "\u0641\u064a", "\u0639\u0644\u0649", "\u0639\u0646", "\u0627\u0644\u0649",
    "\u0625\u0644\u0649", "\u0648", "\u0641", "\u062b\u0645",
    "the", "a", "an", "is", "are", "what", "how", "why", "who", "when", "where",
])

_clean_token(t::AbstractString) =
    strip(lowercase(String(t)), [' ', '\t', '\n', '\r', '.', ',', '\u060C', '\u061B', ':', '?', '\u061F', '!'])

function _norm_key(t::AbstractString)
    s = _clean_token(t)
    s = replace(s, '\u0623' => '\u0627', '\u0625' => '\u0627', '\u0622' => '\u0627',
                   '\u0649' => '\u064a', '\u0629' => '\u0647')
    s = replace(s, r"^(?:\u0648|\u0641)+" => "")
    return strip(s)
end

function _content_terms(tokens)
    out = String[]
    seen = Set{String}()
    for token in tokens
        s = _clean_token(token)
        k = _norm_key(s)
        isempty(k) && continue
        k in STOP_KEYS && continue
        length(k) < 2 && continue
        k in seen && continue
        push!(out, s)
        push!(seen, k)
    end
    return out
end

function _candidate_ok(word::AbstractString)
    s = _clean_token(word)
    isempty(s) && return false
    k = _norm_key(s)
    isempty(k) && return false
    k in STOP_KEYS && return false
    length(k) < 2 && return false
    occursin(r"[#<>{}\[\]\(\)=+*/\\|@%^~`$]", s) && return false
    digit_count = count(isdigit, s)
    letter_count = count(isletter, s)
    digit_count > 0 && digit_count >= letter_count && return false
    return true
end

function _compact_similarity(a::AbstractString, b::AbstractString)
    try
        av = WordPhysics.compute_compact_phase_vector(String(a))
        bv = WordPhysics.compute_compact_phase_vector(String(b))
        return (WordPhysics.compact_phase_similarity(av, bv) + 1.0) / 2.0
    catch
        return 0.0
    end
end

function _lookup_id(vocab::Dict, anchor::AbstractString)
    raw = String(anchor)
    aid = get(vocab, raw, 0)
    aid > 0 && return aid
    key = _norm_key(raw)
    aid = get(vocab, key, 0)
    aid > 0 && return aid
    if startswith(key, AR_AL)
        return get(vocab, replace(key, r"^\u0627\u0644" => ""), 0)
    end
    return get(vocab, AR_AL * key, 0)
end

function _top_related(vocab::Dict, id2word::Dict, matrix, anchors::Vector{String};
                      per_anchor::Int=18)
    matrix === nothing && return Dict{String,Float64}()
    scores = Dict{String,Float64}()
    nrow, ncol = size(matrix)
    anchor_keys = Set(_norm_key.(anchors))
    raw_limit = max(per_anchor * 4, per_anchor)
    for anchor in anchors
        aid = _lookup_id(vocab, anchor)
        (aid <= 0 || aid > nrow) && continue
        row = matrix[aid, :]
        raw_pairs = Tuple{Float64,String}[]
        for (idx, val) in zip(row.nzind, row.nzval)
            (idx <= 0 || idx > ncol) && continue
            word = get(id2word, idx, "")
            _candidate_ok(word) || continue
            _norm_key(word) in anchor_keys && continue
            score = min(1.0, log1p(abs(Float64(val))) / 4.0)
            score <= 0 && continue
            push!(raw_pairs, (score, word))
        end
        sort!(raw_pairs; by=x -> -x[1])
        narrowed = raw_pairs[1:min(raw_limit, length(raw_pairs))]
        pairs = Tuple{Float64,String}[]
        for (score, word) in narrowed
            phase = _compact_similarity(anchor, word)
            push!(pairs, (0.82 * score + 0.18 * phase, word))
        end
        sort!(pairs; by=x -> -x[1])
        for (score, word) in pairs[1:min(per_anchor, length(pairs))]
            scores[word] = max(get(scores, word, 0.0), score)
        end
    end
    return scores
end

function _bridge_terms(vocab::Dict, matrix, anchors::Vector{String}, candidates::Vector{String})
    matrix === nothing && return Dict{String,Float64}()
    length(anchors) >= 2 || return Dict{String,Float64}()
    nrow, ncol = size(matrix)
    anchor_ids = Int[]
    for anchor in anchors
        aid = _lookup_id(vocab, anchor)
        aid > 0 && aid <= nrow && push!(anchor_ids, aid)
    end
    length(anchor_ids) >= 2 || return Dict{String,Float64}()
    out = Dict{String,Float64}()
    for word in candidates
        wid = _lookup_id(vocab, word)
        (wid <= 0 || wid > ncol) && continue
        hits = 0
        total = 0.0
        for aid in anchor_ids
            val = try
                abs(matrix[aid, wid])
            catch
                0.0
            end
            if val > 1e-9
                hits += 1
                total += min(1.0, log1p(val) / 4.0)
            end
        end
        hits >= 2 || continue
        out[word] = total / hits + 0.15 * hits
    end
    return out
end

function _sorted_keys(scores::Dict{String,Float64}; limit::Int=16)
    ordered = sort(collect(scores); by=x -> (-x[2], length(x[1]), x[1]))
    return String[x[1] for x in ordered[1:min(limit, length(ordered))]]
end

function build_semantic_attention(vocab::Dict, id2word::Dict,
                                  K_sem, K_causal,
                                  prompt_tokens;
                                  max_terms::Int=18)
    anchors = _content_terms(prompt_tokens)
    if isempty(anchors)
        return SemanticAttentionField(String[], String[], String[],
                                      Dict{String,Float64}(), Dict{String,Float64}(),
                                      0.0)
    end
    sem = _top_related(vocab, id2word, K_sem, anchors; per_anchor=max_terms)
    causal = _top_related(vocab, id2word, K_causal, anchors; per_anchor=max(8, div(max_terms, 2)))
    merged = Dict{String,Float64}()
    for (w, s) in sem
        merged[w] = max(get(merged, w, 0.0), 0.82 * s)
    end
    for (w, s) in causal
        merged[w] = max(get(merged, w, 0.0), 0.68 * s + 0.12)
    end
    candidates = _sorted_keys(merged; limit=max_terms * 2)
    bridges = _bridge_terms(vocab, K_sem, anchors, candidates)
    bridge_words = _sorted_keys(bridges; limit=min(8, max_terms))

    positive = Dict{String,Float64}()
    for (i, a) in enumerate(anchors)
        positive[a] = max(get(positive, a, 0.0), max(0.45, 0.9 - 0.08 * (i - 1)))
    end
    for (i, w) in enumerate(_sorted_keys(merged; limit=max_terms))
        positive[w] = max(get(positive, w, 0.0), max(0.18, 0.62 - 0.025 * (i - 1)))
    end
    for (i, w) in enumerate(bridge_words)
        positive[w] = max(get(positive, w, 0.0), max(0.28, 0.72 - 0.04 * (i - 1)))
    end
    negative = Dict{String,Float64}()
    for k in STOP_KEYS
        negative[k] = 0.25
    end
    confidence = clamp(0.35 + 0.08 * length(anchors) +
                       0.03 * min(length(merged), max_terms) +
                       0.05 * min(length(bridge_words), 4), 0.0, 1.0)
    return SemanticAttentionField(anchors, _sorted_keys(merged; limit=max_terms),
                                  bridge_words, positive, negative, confidence)
end

has_semantic_attention(field::SemanticAttentionField) =
    field.confidence >= 0.35 && (!isempty(field.expanded_terms) || !isempty(field.bridge_terms))

attention_bias_terms(field::SemanticAttentionField) =
    String[x[1] for x in sort(collect(field.positive_bias); by=x -> -x[2])]

end
