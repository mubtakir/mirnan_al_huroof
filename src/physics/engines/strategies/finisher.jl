function _corpus_memory_answer(gen::MirnanGenerator,
                               prompt::AbstractString,
                               prompt_tokens::Vector{String};
                               intent::String="")
    enabled = _env_on("MIRNAN_CORPUS_MEMORY", "1")
    enabled || return ""
    short_declarative = !_is_question_prompt_safe(String(prompt)) && 1 <= length(prompt_tokens) <= 5
    (short_declarative || intent in ("mechanism", "causal", "descriptive")) || return ""
    sentences = _load_corpus_sentences_for_generator(gen)
    isempty(sentences) && return ""

    prompt_keys = reduce(union, (_generation_keys(t) for t in prompt_tokens); init=Set{String}())
    content_keys = setdiff(prompt_keys, union(GENERATION_STOPWORDS, _EXTRA_GENERATION_STOPWORDS, _QUESTION_TOOL_KEYS))
    filter!(k -> length(k) >= 3, content_keys)
    isempty(content_keys) && return ""
    prompt_phrase = join([_generation_family_key(t) for t in prompt_tokens
                          if !_any_key_in(_generation_keys(t), _QUESTION_TOOL_KEYS)], " ")

    best_words = String[]
    best_score = 0.0
    banned = Set(["سؤال", "جواب", "س", "ج", "question", "answer"])
    meta_banned = Set(["مبتدا", "مبتدأ", "خبر", "مرفوع", "منصوب", "مجرور",
                       "اعراب", "إعراب", "بالضمه", "بالضمة", "بالفتحه",
                       "بالفتحة", "بالكسره", "بالكسرة", "ظاهر", "الظاهره"])
    scan_limit = _env_int("MIRNAN_CORPUS_MEMORY_SCAN_LIMIT", 20000)
    scan_budget = _env_float("MIRNAN_CORPUS_MEMORY_BUDGET_SEC", 0.8)
    scan_started = time()
    scan_count = 0
    for ids in sentences
        scan_count += 1
        scan_count > scan_limit && break
        scan_budget > 0.0 && (time() - scan_started) > scan_budget && break
        words = _sentence_words(gen, ids)
        3 <= length(words) <= 28 || continue
        any(w -> _arabic_dialogue_word_key(w) in banned, words) && continue
        short_declarative && any(w -> _generation_family_key(w) in meta_banned, words) && continue
        sent_keys = reduce(union, (_generation_keys(w) for w in words); init=Set{String}())
        overlap = intersect(content_keys, sent_keys)
        isempty(overlap) && continue
        overlap_count = length(overlap)
        required_overlap = short_declarative ? length(content_keys) :
                           (intent in ("mechanism", "causal") && length(content_keys) >= 2 ? 2 : 1)
        overlap_count < required_overlap && continue
        anchor = overlap_count / max(length(content_keys), 1)
        sent_phrase = join(_generation_family_key.(words), " ")
        exact_phrase = !isempty(strip(prompt_phrase)) && occursin(prompt_phrase, sent_phrase)
        first_anchor = !isempty(words) &&
                       !isempty(intersect(_generation_keys(words[1]), content_keys))
        if short_declarative && length(words) > 14 && !first_anchor
            continue
        end
        brevity = 1.0 / (1.0 + max(0, length(words) - 14) / 10)
        subject_bonus = first_anchor ? 0.18 : 0.0
        phrase_bonus = exact_phrase ? 0.35 : 0.0
        question_leak = !isempty(intersect(sent_keys, _QUESTION_TOOL_KEYS)) ? 0.30 : 0.0
        score = 0.56 * anchor + 0.18 * min(overlap_count, 3) / 3 +
                0.08 * brevity + subject_bonus + phrase_bonus - question_leak
        score > best_score || continue
        best_score = score
        best_words = words
    end
    min_score = short_declarative ? 0.52 : 0.62
    best_score < min_score && return ""
    if short_declarative
        best_words = _trim_reported_speech_prefix(best_words, prompt_tokens)
    end
    text = join(best_words[1:min(length(best_words), 24)], " ")
    text = strip(text)
    isempty(text) && return ""
    endswith(text, ".") || endswith(text, "؟") || endswith(text, "?") || (text *= ".")
    return text
end

function _is_non_yesno_question_prompt(prompt::AbstractString)
    s = lowercase(strip(String(prompt)))
    isempty(s) && return false
    starts = (
        "\u0645\u062a\u0649", "\u0623\u064a\u0646", "\u0627\u064a\u0646",
        "\u0643\u064a\u0641",
        "\u0643\u0645", "\u0645\u0627 ", "\u0645\u0627\u0630\u0627", "\u0645\u0646 ",
        "when", "where", "how", "how many", "how much", "what ", "who ",
    )
    return any(p -> startswith(s, p), starts)
end

function _starts_with_yesno_answer(text::AbstractString)
    s = lowercase(strip(String(text)))
    return startswith(s, "\u0646\u0639\u0645") || startswith(s, "\u0644\u0627\u060c") ||
           startswith(s, "\u0644\u0627 ") || startswith(s, "yes") || startswith(s, "no,") ||
           s == "no"
end

function _non_yesno_question_fallback(prompt::AbstractString)
    s = lowercase(strip(String(prompt)))
    if startswith(s, "\u0645\u062a\u0649") || startswith(s, "when")
        return "\u0644\u0627 \u0623\u062c\u062f \u0632\u0645\u0646\u0627\u064b \u0645\u062d\u062f\u062f\u0627\u064b \u0641\u064a \u0627\u0644\u0630\u0627\u0643\u0631\u0629."
    elseif startswith(s, "\u0623\u064a\u0646") || startswith(s, "\u0627\u064a\u0646") || startswith(s, "where")
        return "\u0644\u0627 \u0623\u062c\u062f \u0645\u0643\u0627\u0646\u0627\u064b \u0645\u062d\u062f\u062f\u0627\u064b \u0641\u064a \u0627\u0644\u0630\u0627\u0643\u0631\u0629."
    elseif startswith(s, "\u0643\u0645") || startswith(s, "how many") || startswith(s, "how much")
        return "\u0644\u0627 \u0623\u062c\u062f \u0639\u062f\u062f\u0627\u064b \u0623\u0648 \u0645\u0642\u062f\u0627\u0631\u0627\u064b \u0645\u062d\u062f\u062f\u0627\u064b \u0641\u064a \u0627\u0644\u0630\u0627\u0643\u0631\u0629."
    end
    return ""
end

function _remove_wrong_yesno_for_question(prompt::AbstractString, text::AbstractString)
    _is_non_yesno_question_prompt(prompt) || return strip(String(text))
    _starts_with_yesno_answer(text) || return strip(String(text))
    return _non_yesno_question_fallback(prompt)
end

function _finish_generation!(gen::MirnanGenerator,
                             prompt::String,
                             prompt_tokens::Vector{String},
                             result::AbstractString,
                             cereb_obs,
                             cereb_policy::CerebellumPolicy;
                             observe_ram::Bool=true,
                             review=nothing,
                             sanitize_output::Bool=true,
                             apply_templates::Bool=true)
    text = sanitize_output ? _sanitize_generation_output(prompt_tokens, String(result)) : strip(String(result))
    if apply_templates
        templated = _simple_text_template(prompt_tokens)
        question_templated = _simple_question_template(prompt_tokens)
        if !isempty(strip(question_templated)) && _needs_simple_question_template(prompt_tokens, text)
            text = question_templated
        end
        if !isempty(strip(templated)) &&
           (length(split(text)) < 2 || _needs_simple_declarative_template(prompt_tokens, text))
            text = templated
        end
    end
    text = _repair_dialogue_yesno_word_order(prompt, text)
    text = _remove_wrong_yesno_for_question(prompt, text)
    text = polish_response(prompt, text; enabled=_env_on("MIRNAN_ENABLE_RESPONSE_POLISHER", "0"))
    final_review = review === nothing ?
        review_generation!(gen.self_review, prompt, text;
                           prompt_tokens=prompt_tokens,
                           pv_fn=w -> _pv(gen, String(w))) :
        review
    gen.self_review.last_review = final_review
    if observe_ram && !isempty(strip(text))
        RAMCore.observe!(gen.ram, String[vcat(prompt_tokens, split(text))...])
    end
    reward_val = learn_from_outcome!(gen.cerebellum, cereb_obs, cereb_policy, text;
                        reward=final_review.score)
    
    # ═══ حلقة التعلّم: تحديث ثقة المفاتيح النشطة ═══
    mem = _LEARNED_ISTINBAT_MEMORY[]
    if mem !== nothing
        active_markers = String[]
        for w in prompt_tokens
            if haskey(mem.discovered_markers, w)
                push!(active_markers, w)
            end
        end
        if !isempty(active_markers)
            update_marker_confidence!(mem, active_markers, reward_val)
        end
    end

    return String(text)
end

function _review_candidate!(gen::MirnanGenerator,
                            prompt::String,
                            prompt_tokens::Vector{String},
                            result::AbstractString)
    return review_generation!(gen.self_review, prompt, String(result);
                              prompt_tokens=prompt_tokens,
                              pv_fn=w -> _pv(gen, String(w)))
end

function _anchored_rejection_answer(gen::MirnanGenerator,
                                   prompt::String,
                                   prompt_tokens::Vector{String},
                                   review)
    getfield(review, :accepted) && return ""
    issues = Set(String.(getfield(review, :issues)))
    if isempty(intersect(issues, Set(["list_like_output", "missing_prompt_anchor", "weak_prompt_anchor"])))
        return ""
    end

    text = strip(prompt)
    isempty(text) && return ""
    if !_is_question_prompt_safe(text) && 1 <= length(prompt_tokens) <= 4
        active_paras = _get_active_paragraphs(gen, prompt_tokens)
        for tok in prompt_tokens
            keys = _generation_keys(tok)
            _any_key_in(keys, GENERATION_STOPWORDS) && continue
            _any_key_in(keys, _EXTRA_GENERATION_STOPWORDS) && continue
            ta3rif = answer_ta3rif(gen.ta3rif, "ما هو $(tok)"; active_paras=active_paras)
            !isempty(strip(ta3rif)) && return ta3rif
        end
        return endswith(text, ".") || endswith(text, "؟") || endswith(text, "?") ? text : text * "."
    end
    return "فشل التوليد الفيزيائي الحالي."
end

function _conservative_anchor_answer(prompt::String, prompt_tokens::Vector{String})
    text = strip(prompt)
    isempty(text) && return ""
    if !_is_question_prompt_safe(text) && 1 <= length(prompt_tokens) <= 5
        return endswith(text, ".") || endswith(text, "؟") || endswith(text, "?") ? text : text * "."
    end
    return ""
end

function _causal_anchor_answer(gen::MirnanGenerator, prompt_tokens::Vector{String}, active_paras)
    _strict_no_templates_enabled() && return ""
    istinbat_mem = _LEARNED_ISTINBAT_MEMORY[]
    istinbat_mem === nothing && return ""
    prompt = join(prompt_tokens, " ")
    rec = select_causal_anchor_attention(istinbat_mem, prompt; active_paras=active_paras)
    rec === nothing && return ""
    return causal_anchor_answer_from_attention(rec)
end
