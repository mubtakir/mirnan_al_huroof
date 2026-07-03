function _extract_benefit_subject(prompt::AbstractString)
    text = strip(_clean_aql_text(prompt))
    isempty(text) && return ""
    patterns = [
        r"^(?:ما\s+)?(?:فائدة|فوائد)\s+(.+?)[\u061F\?\.!]*$",
        r"^لماذا\s+(.+?)\s+(?:ضروري|ضرورية|مهم|مهمة|مفيد|مفيدة|نافع|نافعة)[\u061F\?\.!]*$",
    ]
    for pat in patterns
        m = match(pat, text)
        m === nothing && continue
        subject = strip(String(m.captures[1]))
        subject = replace(subject, r"^(?:هو|هي)\s+" => "")
        subject = strip(subject, [' ', '\t', '\n', '\r', '.', ',', ';', ':', '?', '!', '؟', '،', '؛'])
        !isempty(subject) && return subject
    end
    return ""
end

function _ta3rif_support_answer(gen::MirnanGenerator, prompt::AbstractString,
                                 active_paras::Union{Nothing,Dict{Tuple{String,Int},Float64}}=nothing)
    subject = _extract_benefit_subject(prompt)
    isempty(subject) && return ""
    return answer_ta3rif(gen.ta3rif, "ما هو $(subject)"; active_paras=active_paras)
end

function _explicit_ta3rif_prompt(prompt::AbstractString)
    s = lowercase(strip(String(prompt)))
    return occursin("ما معنى", s) ||
           occursin("ماذا معنى", s) ||
           occursin("تعريف", s) ||
           startswith(s, "عرف ") ||
           startswith(s, "عرّف ") ||
           startswith(s, "اشرح معنى")
end

function _ta3rif_definition_for_subject(gen::MirnanGenerator, subject::AbstractString)
    subject_key = _relation_norm_token(subject)
    target_keys = _identity_generation_keys(subject)
    best_rec = nothing
    best_score = -Inf
    for rec in values(gen.ta3rif.records)
        rec_key = _relation_norm_token(rec.subject)
        rec_keys = _identity_generation_keys(rec.subject)
        score = 0.0
        rec_key == subject_key && (score += 3.0)
        !isempty(subject_key) && !isempty(rec_key) && startswith(rec_key, subject_key * " ") && (score += 1.2)
        !isempty(intersect(target_keys, rec_keys)) && (score += 0.8)
        score > best_score && (best_rec = rec; best_score = score)
    end
    best_score <= 0 && return ""
    best_rec === nothing && return ""
    best_def = ""
    best_def_score = -Inf
    subject_is_light = _has_any_key(target_keys, ["نور", "النور", "ضوء", "الضوء"])
    for (definition, count) in best_rec.definitions
        core = _strip_question_punct(definition)
        isempty(core) && continue
        words = split(core)
        score = Float64(count)
        4 <= length(words) <= 18 && (score += 2.0)
        length(words) < 3 && (score -= 3.0)
        length(words) > 26 && (score -= 1.5)
        startswith(core, "من ") && (score -= 2.0)
        startswith(core, "من غير") && (score -= 4.0)
        startswith(core, "التي ") && (score -= 4.0)
        startswith(core, "الذي ") && (score -= 4.0)
        startswith(core, "بعيدا") && (score -= 4.0)
        startswith(core, "بعيداً") && (score -= 4.0)
        startswith(core, "في ") && (score -= 2.0)
        startswith(core, "عند ") && (score -= 2.0)
        occursin("سؤال", core) && (score -= 2.0)
        occursin("جواب", core) && (score -= 2.0)
        (!subject_is_light && (occursin("ضياء", core) || occursin("ضوء", core) || occursin("نور", core))) && (score -= 2.5)
        if score > best_def_score
            best_def = core
            best_def_score = score
        end
    end
    return best_def
end

function _difference_definition_quality_ok(definition::AbstractString)
    def = strip(String(definition))
    isempty(def) && return false
    words = split(def)
    length(words) >= 3 || return false
    bad_starts = Set(["و", "وأداء", "واداء",
                      "أما", "اما", "ثم", "لكن"])
    first_key = _generation_key(first(words))
    first_key in bad_starts && return false
    return any(p -> occursin(p, def), [" ", "ا", "ي"])
end

function _difference_focus_sentence(left::AbstractString, right::AbstractString,
                                    left_def::AbstractString, right_def::AbstractString)
    tail_stop = Set(["في", "من", "عن", "على", "إلى", "الى", "الي", "ي", "عند", "أو", "او", "و", "بما", "بين"])
    function hint(def::AbstractString)
        text = strip(String(def))
        clauses = split(text, r"[،,؛;]")
        if !isempty(clauses)
            first_clause = strip(String(first(clauses)))
            if length(split(first_clause)) >= 3
                return first_clause
            end
        end
        words = split(text)
        isempty(words) && return "معناه"
        selected = words[1:min(9, length(words))]
        while !isempty(selected) && _relation_norm_token(last(selected)) in tail_stop
            pop!(selected)
        end
        isempty(selected) && return "معناه"
        return join(selected, " ")
    end
    left_hint = hint(left_def)
    right_hint = hint(right_def)
    return "والفرق بينهما أن $(left) يدور حول $(left_hint)، بينما $(right) يدور حول $(right_hint)."
end

function _difference_concept_gloss(subject::AbstractString)
    _strict_no_templates_enabled() && return ""
    keys = _generation_keys(subject)
    _has_any_key(keys, ["علم", "العلم"]) &&
        return "إدراك الأشياء على ما هي عليه، وجمع المعرفة التي تكشف الخفاء"
    _has_any_key(keys, ["فهم", "الفهم"]) &&
        return "قدرة على ربط المعاني وتمييز العلاقات بينها"
    _has_any_key(keys, ["نور", "النور", "ضوء", "الضوء"]) &&
        return "ضياء يكشف الأشياء ويهدي الرؤية إلى الطريق"
    _has_any_key(keys, ["ظلام", "الظلام", "ظلمه", "الظلمه", "ظلمة", "الظلمة"]) &&
        return "غياب الضوء بما يحجب الرؤية ويزيد الحيرة"
    _has_any_key(keys, ["سلام", "السلام"]) &&
        return "أمن واستقرار يقل فيه الخوف والاعتداء"
    _has_any_key(keys, ["نزاع", "النزاع"]) &&
        return "خصومة واضطراب ينشأان من تضارب الحقوق أو الأهواء"
    _has_any_key(keys, ["قوة", "قوه", "القوة", "القوه"]) &&
        return "قدرة على الفعل والتأثير، وتحتاج إلى عقل ورحمة حتى لا تنحرف"
    _has_any_key(keys, ["ظلم", "الظلم"]) &&
        return "تعد على الحق ووضع الشيء في غير موضعه"
    _has_any_key(keys, ["عدل", "العدل"]) &&
        return "إعطاء كل ذي حق حقه ووضع الأمور في مواضعها"
    _has_any_key(keys, ["رحمة", "رحمه", "الرحمة", "الرحمه"]) &&
        return "رقة تدفع إلى الإحسان وتخفيف الأذى"
    _has_any_key(keys, ["جهل", "الجهل"]) &&
        return "غياب العلم بما يحجب الفهم ويزيد الخطأ"
    _has_any_key(keys, ["عقل", "العقل"]) &&
        return "ميزان يربط المعرفة بالعواقب ويمنع التهور"
    return ""
end

function _difference_answer(gen::MirnanGenerator, prompt::AbstractString)
    s = _strip_question_punct(prompt)
    _difference_prompt(s) || return ""
    between_parts = split(s, "بين"; limit=2)
    length(between_parts) == 2 || return ""
    pair_text = strip(between_parts[2])
    pair_parts = split(pair_text, r"\s+و"; limit=2)
    length(pair_parts) == 2 || return ""
    left = _strip_question_punct(pair_parts[1])
    right = _strip_question_punct(pair_parts[2])
    (isempty(left) || isempty(right)) && return ""
    left_def = _ta3rif_definition_for_subject(gen, left)
    right_def = _ta3rif_definition_for_subject(gen, right)
    !isempty(left_def) && !_difference_definition_quality_ok(left_def) && (left_def = "")
    !isempty(right_def) && !_difference_definition_quality_ok(right_def) && (right_def = "")
    if isempty(left_def) || isempty(right_def)
        evidence = _learned_pair_evidence_sentence(gen, left, right)
        !isempty(strip(evidence)) && return evidence
        return ""
    end
    focus = _difference_focus_sentence(left, right, left_def, right_def)
    return "$left: $left_def. أما $right: $right_def. $focus"
end

function _nisba_relation_allowed_prompt(prompt::AbstractString, response_plan)
    response_plan.intent == "dialogue" && return false
    s = lowercase(String(prompt))
    return occursin("هل", s) || occursin("ما ", s) || occursin("كيف", s) ||
           occursin("لماذا", s) || occursin("ماذا", s) || occursin("كيف", s) ||
           occursin("بين", s) || occursin("علاق", s) || occursin("ارتباط", s)
end

function _nisba_prompt_overlap(prompt_tokens::AbstractVector{<:AbstractString}, concepts::AbstractVector{<:AbstractString})
    c_keys = reduce(union, (_identity_generation_keys(c) for c in concepts); init=Set{String}())
    p_keys = reduce(union, (_identity_generation_keys(w) for w in prompt_tokens); init=Set{String}())
    return count(k -> k in p_keys, c_keys)
end

function _clean_nisba_evidence(text::AbstractString)
    s = strip(String(text))
    s = replace(s, r"^في فضاء العقل:\s*" => "")
    s = replace(s, r"^في فضاء العقل\s*" => "")
    return s
end

function _nisba_relation_answer(gen::MirnanGenerator, prompt::AbstractString,
                                prompt_tokens::Vector{String}, response_plan, active_paras=nothing)
    _strict_no_templates_enabled() && return ""
    if !isempty(prompt_tokens)
        first_tok = String(first(prompt_tokens))
        # Block why/how/what questions from matching Nisba evidence
        if first_tok in ("لماذا", "كيف", "why", "how", "لِمَ", "لما")
            return ""
        end
        if _is_question_token(first_tok)
            return ""
        end
    end
    _nisba_relation_allowed_prompt(prompt, response_plan) || return ""
    rec = select_nisba_relation(gen.nisba, prompt; min_score=0.38, active_paras=active_paras)
    rec === nothing && return ""
    allowed_types = _relationship_prompt(prompt) ?
                    ("causal", "prevention", "need", "transform", "analogy", "relation", "association", "support") :
                    ("causal", "prevention", "need", "transform", "analogy")
    rec.relation_type in allowed_types || return ""
    # A single shared concept is too weak: it lets a learned witness about "science"
    # answer an unrelated question that only happens to mention science.
    _nisba_prompt_overlap(prompt_tokens, rec.concepts) >= 2 || return ""
    isempty(rec.evidences) && return ""
    answer = _clean_nisba_evidence(rec.evidences[end])
    length(split(answer)) < 3 && return ""
    return answer
end


function try_generate(::DefinitionStrategy, gen::MirnanGenerator, prompt::String, prompt_tokens::Vector{String}, mode::String, cereb_obs, cereb_policy, response_plan, active_paras)
    if response_plan.intent != "dialogue" && _explicit_ta3rif_prompt(prompt)
        aql_meaning_answer = _aql_answer!(gen, prompt)
        structured_aql = occursin("في فضاء العقل", String(aql_meaning_answer)) ||
                         occursin("عملية", String(aql_meaning_answer))
        if !isempty(strip(aql_meaning_answer)) && structured_aql
            return _finish_generation!(gen, prompt, prompt_tokens, aql_meaning_answer,
                                       cereb_obs, cereb_policy;
                                       sanitize_output=false)
        end
        ta3rif_answer = answer_ta3rif(gen.ta3rif, prompt; active_paras=active_paras)
        if !isempty(strip(ta3rif_answer))
            return _finish_generation!(gen, prompt, prompt_tokens, ta3rif_answer,
                                       cereb_obs, cereb_policy;
                                       sanitize_output=false)
        end
    end

    difference_answer = response_plan.intent == "dialogue" ? "" : _difference_answer(gen, prompt)
    if !isempty(strip(difference_answer))
        return _finish_generation!(gen, prompt, prompt_tokens, difference_answer,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    nisba_answer = _nisba_relation_answer(gen, prompt, prompt_tokens, response_plan, active_paras)
    if !isempty(strip(nisba_answer))
        return _finish_generation!(gen, prompt, prompt_tokens, nisba_answer,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    ta3rif_support = response_plan.intent == "dialogue" ? "" : _ta3rif_support_answer(gen, prompt, active_paras)
    if !isempty(strip(ta3rif_support))
        return _finish_generation!(gen, prompt, prompt_tokens, ta3rif_support,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false)
    end

    ta3rif_answer = response_plan.intent == "dialogue" ? "" : answer_ta3rif(gen.ta3rif, prompt; active_paras=active_paras)
    if !isempty(strip(ta3rif_answer))
        return _finish_generation!(gen, prompt, prompt_tokens, ta3rif_answer,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false)
    end
    return nothing
end
