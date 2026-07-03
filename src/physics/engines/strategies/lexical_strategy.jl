function _extract_lexical_query(prompt::AbstractString)
    text = strip(String(prompt))
    patterns = [
        r"(?:حلل|تحليل|طيف|فيزياء)\s+(?:كلمة\s+)?([^\s؟?،,.]+)",
        r"(?:ما\s+معنى\s+كلمة|معنى\s+كلمة)\s+([^\s؟?،,.]+)",
        r"(?:ما\s+معنى|معنى|فسر|اشرح)\s+([^\s؟?،,.]+)",
        r"(?:analyze|analyse|profile)\s+(?:word\s+)?([A-Za-z\u0600-\u06FF_]+)",
    ]
    for pattern in patterns
        m = match(pattern, text)
        m !== nothing && return strip(String(m.captures[1]))
    end
    return ""
end

function _format_lexical_neighbors(items)
    isempty(items) && return "لا توجد كلمات قريبة كافية"
    parts = String[]
    for item in items[1:min(5, end)]
        word = string(get(item, "word", ""))
        score = get(item, "hybrid_score", 0.0)
        push!(parts, "$word ($(score))")
    end
    return join(parts, "، ")
end

function _lexical_oracle_answer(gen::MirnanGenerator, prompt::AbstractString)
    word = _extract_lexical_query(prompt)
    isempty(word) && return ""
    data = analyze_unknown_word(word, gen.vocab; top_k=5)
    profile = get(data, "profile", Dict{String,Any}())
    quality = get(data, "quality", Dict{String,Any}())
    neighbors = get(data, "nearest_words", Any[])
    root = string(get(profile, "root", ""))
    isempty(root) && (root = "غير محدد")
    mass = get(profile, "mass", 0.0)
    harmony = get(quality, "harmony", 0.0)
    stability = get(quality, "stability", 0.0)
    aesthetic = get(quality, "aesthetic_score", 0.0)
    confidence = get(data, "interpretation_confidence", 0.0)
    nearest = _format_lexical_neighbors(neighbors)
    return "فيزياء الحرف لكلمة [$(word)]: الجذر/الأثر الجذري $(root)، الكتلة $(mass)، التناغم $(harmony)، الثبات $(stability)، الجمال الطيفي $(aesthetic). أقرب كلمات طيفيا: $(nearest). ثقة الاستنباط $(confidence). هذا تقدير من بنية الحروف والسياق المعجمي، لا حكما نهائيا."
end

const _SENSE_QUERY_CUES = Set(["معنى", "كلمة", "فسر", "اشرح", "دلالة", "يقصد", "تعني", "يعني"])

function _single_lexical_word(prompt::AbstractString)
    text = strip(String(prompt))
    isempty(text) && return ""
    occursin(r"\s", text) && return ""
    occursin(r"[0-9+\-*/=<>#@{}()\[\]\\|]", text) && return ""
    any(isletter, text) || return ""
    return strip(text, [' ', '\t', '\n', '\r', '.', ',', ';', ':', '?', '!', '؟', '،', '؛'])
end

function _word_field_answer(gen::MirnanGenerator, word::AbstractString)
    w = strip(String(word))
    isempty(w) && return ""
    data = analyze_unknown_word(w, gen.vocab; top_k=8)
    profile = get(data, "profile", Dict{String,Any}())
    quality = get(data, "quality", Dict{String,Any}())
    neighbors = get(data, "nearest_words", Any[])
    root = string(get(profile, "root", ""))
    isempty(root) && (root = "غير محدد")
    mass = get(profile, "mass", 0.0)
    energy = get(profile, "energy", 0.0)
    harmony = get(quality, "harmony", get(profile, "harmony", 0.0))
    stability = get(quality, "stability", get(profile, "stability", 0.0))
    aesthetic = get(quality, "aesthetic_score", 0.0)
    nearest = _format_lexical_neighbors(neighbors)
    known = get(data, "known_in_vocab", false) ?
        "مسجلة في المعجم" :
        "غير مسجلة بصورة مباشرة"
    return join([
        "حقل الكلمة: $(w)",
        "الحالة: $(known)",
        "الجذر/الأثر الجذري: $(root)",
        "الكتلة: $(mass) | الطاقة: $(energy)",
        "التناغم: $(harmony) | الثبات: $(stability) | الجمال الطيفي: $(aesthetic)",
        "أقرب الكلمات طيفيا: $(nearest)",
    ], "\n")
end

function _lexical_words_from_prompt(prompt::AbstractString; max_words::Int=6)
    text = strip(String(prompt))
    isempty(text) && return String[]
    cleaned = replace(text, r"[0-9+\-*/=<>#@{}()\[\]\\|]" => " ")
    words = String[]
    for raw in split(cleaned)
        w = strip(String(raw), [' ', '\t', '\n', '\r', '.', ',', ';', ':', '?', '!', '؟', '،', '؛', '"', '\''])
        length(w) < 2 && continue
        any(isletter, w) || continue
        push!(words, w)
        length(words) >= max_words && break
    end
    return words
end

function _lexical_mode_answer(gen::MirnanGenerator, prompt::AbstractString)
    word = _single_lexical_word(prompt)
    if isempty(word)
        word = _extract_lexical_query(prompt)
    end
    if !isempty(word)
        return _word_field_answer(gen, word)
    end

    words = _lexical_words_from_prompt(prompt)
    isempty(words) && return "لا توجد كلمات قابلة للتحليل الحرفي."
    reports = String[]
    for w in words
        push!(reports, _word_field_answer(gen, w))
    end
    return join(reports, "\n\n---\n\n")
end

function _root_mode_answer(gen::MirnanGenerator, prompt::AbstractString, mode::AbstractString)
    word = _single_lexical_word(prompt)
    isempty(word) && (word = _extract_lexical_query(prompt))
    isempty(word) && return ""
    fmt = mode == "root_list" ? "list" : mode == "root_poetic" ? "poetic" : "detailed"
    return root_field_report(String(word), gen.vocab; format=fmt, max_results=30)
end

function _sense_query_word(prompt::AbstractString, prompt_tokens::Vector{String})
    word = _extract_lexical_query(prompt)
    if !isempty(word) && has_sense_inventory(word)
        return word
    end
    for t in prompt_tokens
        if has_sense_inventory(t)
            return t
        end
    end
    return ""
end

function _sense_context_tokens(prompt_tokens::Vector{String}, word::String)
    word_key = lowercase(strip(word))
    out = String[]
    for t in prompt_tokens
        clean = lowercase(strip(t))
        (isempty(clean) || clean == word_key || clean in _SENSE_QUERY_CUES) && continue
        push!(out, clean)
    end
    return out
end

function _sense_superposition_answer(gen::MirnanGenerator, prompt::AbstractString,
                                     prompt_tokens::Vector{String},
                                     cereb_policy::CerebellumPolicy,
                                     cereb_obs)
    cereb_policy.sense_mode == "measure" || return ""
    ("meaning_query" in cereb_obs.tags) || return ""
    word = _sense_query_word(prompt, prompt_tokens)
    isempty(word) && return ""
    context = _sense_context_tokens(prompt_tokens, word)
    measurement = measure_senses(word, context; pv_fn=w -> _pv(gen, String(w)))
    return explain_measurement(measurement)
end

function try_generate(::RootLexicalStrategy, gen::MirnanGenerator, prompt::String, prompt_tokens::Vector{String}, mode::String, cereb_obs, cereb_policy, response_plan, active_paras)
    if mode in ("root", "root_list", "root_poetic")
        answer = _root_mode_answer(gen, prompt, mode)
        !isempty(strip(answer)) && return String(answer)
    end

    single_word = _single_lexical_word(prompt)
    if mode in ("lexical", "word_field", "no_dialogue")
        return String(_lexical_mode_answer(gen, prompt))
    elseif mode == "standard" && !isempty(single_word)
        answer = _word_field_answer(gen, single_word)
        !isempty(strip(answer)) && return String(answer)
    end
    return nothing
end

function try_generate(::LexicalOracleStrategy, gen::MirnanGenerator, prompt::String, prompt_tokens::Vector{String}, mode::String, cereb_obs, cereb_policy, response_plan, active_paras)
    if mode == "auto" || mode == "lexical"
        lexical_answer = _lexical_oracle_answer(gen, prompt)
        if !isempty(strip(lexical_answer))
            return _finish_generation!(gen, prompt, prompt_tokens, lexical_answer,
                                       cereb_obs, cereb_policy;
                                       sanitize_output=false,
                                       apply_templates=false)
        end
    end
    return nothing
end

function try_generate(::SenseSuperpositionStrategy, gen::MirnanGenerator, prompt::String, prompt_tokens::Vector{String}, mode::String, cereb_obs, cereb_policy, response_plan, active_paras)
    sense_answer = _sense_superposition_answer(gen, prompt, prompt_tokens, cereb_policy, cereb_obs)
    if !isempty(strip(sense_answer))
        return _finish_generation!(gen, prompt, prompt_tokens, sense_answer,
                                   cereb_obs, cereb_policy)
    end
    return nothing
end

