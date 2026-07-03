function _terms_overlap_keys(keys::Set{String}, terms::Vector{String})
    for term in terms
        tkeys = _generation_keys(term)
        any(k -> k in keys, tkeys) && return true
    end
    return false
end

function _contradiction_record_matches_prompt(rec::IstinbatAttentionRecord, keys::Set{String})
    left_hit = _terms_overlap_keys(keys, rec.before_terms)
    right_hit = _terms_overlap_keys(keys, rec.after_terms)
    focus_hit = _terms_overlap_keys(keys, rec.focus_terms)
    return left_hit && (right_hit || (rec.relation_type == "need" && focus_hit && length(intersect(keys, Set(["يكفي", "وحده", "وحدها", "بلا", "دون"]))) > 0))
end

function _strip_question_punct(text::AbstractString)
    return strip(String(text), [' ', '\t', '\n', '\r', '.', ',', '،', '؛', ':', '?', '؟', '!'])
end

function _definition_core(text::AbstractString)
    s = _strip_question_punct(text)
    s = replace(s, r"^\s*[^:：]+:\s*" => "")
    s = replace(s, r"^\s*[^،,؛;]+[،,]\s*" => "")
    return _strip_question_punct(s)
end

function _semantic_relation_knowledge_records()
    return Any[]
end

function _relation_norm_token(s::AbstractString)
    x = lowercase(strip(String(s)))
    x = replace(x, 'أ' => 'ا', 'إ' => 'ا', 'آ' => 'ا', 'ى' => 'ي', 'ة' => 'ه')
    x = replace(x, r"^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$" => "")
    x = replace(x, r"^(?:وال|فال|بال|كال|لل|ال)" => "")
    return strip(x)
end

function _relation_record_score(keys::Set{String}, prompt_tokens::Vector{String}, item)
    item isa AbstractDict || return 0.0
    terms = get(item, "terms", Any[])
    terms isa AbstractVector || return 0.0
    prompt_exact = Set(_relation_norm_token(t) for t in prompt_tokens)
    filter!(t -> !isempty(t), prompt_exact)
    hits = 0
    exact_hits = 0
    total = 0
    for term in terms
        term_s = string(term)
        tkeys = _generation_keys(term_s)
        isempty(tkeys) && continue
        total += 1
        any(k -> k in keys, tkeys) && (hits += 1)
        _relation_norm_token(term_s) in prompt_exact && (exact_hits += 1)
    end
    min_overlap = Int(get(item, "min_overlap", 2))
    max(hits, exact_hits) >= min_overlap || return 0.0
    return (hits / max(total, 1)) + 0.35 * exact_hits
end

function _semantic_relation_memory_answer(prompt_tokens::Vector{String})
    _strict_no_templates_enabled() && return ""
    keys = _token_keyset(prompt_tokens)
    isempty(keys) && return ""
    is_yesno = !isempty(prompt_tokens) && _is_question_token(first(prompt_tokens))
    is_yesno && return ""
    prompt_text = join(prompt_tokens, " ")
    preferred_type = if occursin("يحتاج", prompt_text) || occursin("تحتاج", prompt_text) || occursin("يكفي", prompt_text)
        "need"
    elseif occursin("يتحول", prompt_text) || occursin("تتحول", prompt_text) || occursin("تصبح", prompt_text) || occursin("تصير", prompt_text)
        "transform"
    elseif occursin("يشبه", prompt_text) || occursin("شبيها", prompt_text) || occursin("شبيهاً", prompt_text)
        "analogy"
    elseif occursin("يمنع", prompt_text) || occursin("تمنع", prompt_text) || occursin("يضبط", prompt_text) || occursin("تضبط", prompt_text) || occursin("يحمي", prompt_text)
        "prevention"
    else
        ""
    end
    best = nothing
    best_score = 0.0
    for item in _semantic_relation_knowledge_records()
        score = _relation_record_score(keys, prompt_tokens, item)
        !isempty(preferred_type) && string(get(item, "relation_type", "")) == preferred_type && (score += 0.45)
        if score > best_score
            best = item
            best_score = score
        end
    end
    best === nothing && return ""
    answer = strip(string(get(best, "answer", "")))
    isempty(answer) && return ""
    return (endswith(answer, ".") || endswith(answer, "؟") || endswith(answer, "?") || endswith(answer, "!")) ? answer : answer * "."
end

function _semantic_relation_gate_answer(prompt::AbstractString, prompt_tokens::Vector{String})
    negated_how = _negated_how_relation_answer(prompt_tokens)
    !isempty(strip(negated_how)) && return negated_how
    relation_memory_answer = _semantic_relation_memory_answer(prompt_tokens)
    !isempty(strip(relation_memory_answer)) && return relation_memory_answer
    return ""
end
