function _is_yesno_question_token(word::AbstractString)
    return String(word) in ("هل", "hal")
end

function _yesno_declarative_field_answer(gen::MirnanGenerator,
                                         prompt::String,
                                         prompt_tokens::Vector{String})
    _strict_no_templates_enabled() && return ""
    isempty(prompt_tokens) && return ""
    _is_yesno_question_token(first(prompt_tokens)) || return ""
    
    words = _yesno_content_words(prompt_tokens)
    2 <= length(words) <= 7 || return ""
    
    has_negation = any(_is_negation_word, words)
    positive_words = String[w for w in words if !_is_negation_word(w)]
    filter!(!isempty, positive_words)
    2 <= length(positive_words) <= 6 || return ""
    
    if !has_negation
        declarative = join(positive_words, " ")
        answer = "نعم، " * declarative
        return endswith(answer, ".") ? answer : answer * "."
    else
        subject = first(positive_words)
        target = last(positive_words)
        
        opposed = false
        istinbat_mem = _LEARNED_ISTINBAT_MEMORY[]
        if istinbat_mem !== nothing
            opposed = terms_are_opposed(istinbat_mem, subject, target) ||
                      terms_are_negated(istinbat_mem, subject, target) ||
                      _terms_share_negative_attention(istinbat_mem, subject, target)
        end
        
        preventative_keys = Set([
            "يمنع", "تمنع", "منع", "يزيل", "تزيل", "إزالة", "ازالة",
            "يبدد", "تبدد", "يطرد", "تطرد", "يحمي", "تحمي", "حماية",
            "يفسد", "تفسد", "يضعف", "تضعف", "يهدم", "تهدم"
        ])
        preventative = false
        for w in positive_words
            if !isempty(intersect(_light_verb_keys(w), preventative_keys))
                preventative = true
                break
            end
        end
        
        if opposed == preventative
            positive_statement = join(positive_words, " ")
            answer = "لا، " * positive_statement
            return endswith(answer, ".") ? answer : answer * "."
        else
            negated_statement = _yesno_negated_relation_statement(words, positive_words)
            answer = "نعم، " * negated_statement
            return endswith(answer, ".") ? answer : answer * "."
        end
    end
end

function _token_keyset(tokens::Vector{String})
    keys = Set{String}()
    for t in tokens
        union!(keys, _generation_keys(t))
    end
    return keys
end

function _has_any_key(keys::Set{String}, variants)
    return any(v -> v in keys, variants)
end

function _core_generation_keys(word::AbstractString)
    cleaned = strip(String(word), [' ', '\t', '\n', '\r', '.', ',', '،', '؛', ':', '?', '؟', '!', '"', '\''])
    raw = _generation_key(cleaned)
    fam = _generation_family_key(cleaned)
    projected = _generation_projection_key(cleaned)
    projected_fam = _generation_family_key(projected)
    return Set([raw, fam, projected, projected_fam])
end

function _identity_generation_keys(word::AbstractString)
    cleaned = strip(replace(String(word), r"^[\s\.,،؛:\??!\"']+|[\s\.,،؛:\??!\"']+$" => ""))
    raw = lowercase(cleaned)
    raw = replace(raw, 'أ' => 'ا', 'إ' => 'ا', 'آ' => 'ا',
                  'ى' => 'ي', 'ة' => 'ه')
    raw = replace(raw, r"[\u064B-\u065F\u0670]" => "")
    keys = Set([raw])
    for k in collect(keys)
        isempty(k) && continue
        push!(keys, replace(k, r"^(?:وال|فال|بال|كال|لل|ال)" => ""))
    end
    filter!(!isempty, keys)
    return keys
end

function _yesno_content_words(prompt_tokens::Vector{String})
    words = String[]
    for t in prompt_tokens[2:end]
        clean = strip(replace(t, r"[[:punct:]\u061F\u060C\u061B]" => ""))
        isempty(clean) || push!(words, clean)
    end
    return words
end

function _is_negation_word(word::AbstractString)
    w = strip(replace(String(word), r"[[:punct:]\u061F\u060C\u061B]" => ""))
    return w in ("ليس", "ليست", "غير", "لا")
end

function _canonical_relation_kind(subject::AbstractString, relation_words::Vector{String},
                                  target::AbstractString)
    subject_keys = _identity_generation_keys(subject)
    target_keys = _identity_generation_keys(target)
    has_subject(xs) = !isempty(intersect(subject_keys, Set(xs)))
    has_target(xs) = !isempty(intersect(target_keys, Set(xs)))
    has_action(xs) = _prompt_relation_has_action(relation_words, Set(xs))

    if has_subject(["علم"]) && has_target(["فهم"]) && has_action(["زاد", "يزيد", "تزيد", "فتح", "يفتح", "تفتح"])
        return (:true, "science_understanding")
    elseif has_subject(["فهم"]) && has_target(["علم"]) && has_action(["زاد", "يزيد", "تزيد"])
        return (:false, "understanding_science")
    elseif has_subject(["علم"]) && has_target(["جهل"]) && has_action(["منع", "يمنع", "تمنع", "يبدد", "تبدد"])
        return (:true, "science_ignorance")
    elseif has_subject(["جهل"]) && has_target(["فهم"]) && has_action(["زاد", "يزيد", "تزيد"])
        return (:false, "ignorance_understanding")
    elseif has_subject(["جهل"]) && has_target(["فهم"]) && has_action(["ضعف", "يضعف", "تضعف"])
        return (:true, "ignorance_weakens_understanding")
    elseif has_subject(["عدل"]) && has_target(["سلام"]) && has_action(["حفظ", "يحفظ", "تحفظ", "صان", "يصون"])
        return (:true, "justice_peace")
    elseif has_subject(["سلام"]) && has_target(["عدل"]) && has_action(["حفظ", "يحفظ", "تحفظ"])
        return (:false, "peace_justice")
    elseif has_subject(["ظلم"]) && has_target(["سلام"]) && has_action(["حفظ", "يحفظ", "تحفظ"])
        return (:false, "injustice_peace")
    elseif has_subject(["ظلم"]) && has_target(["سلام"]) && has_action(["فسد", "يفسد", "تفسد"])
        return (:true, "injustice_corrupts_peace")
    elseif has_subject(["رحمه"]) && has_target(["قوه"]) && has_action(["هذب", "يهذب", "تهذب"])
        return (:true, "mercy_power")
    elseif has_subject(["قوه"]) && has_target(["رحمه"]) && has_action(["هذب", "يهذب", "تهذب"])
        return (:false, "power_mercy")
    elseif has_subject(["رحمه"]) && has_target(["ثقه"]) && has_action(["بني", "يبني", "تبني"])
        return (:true, "mercy_trust")
    elseif has_subject(["ثقه"]) && has_target(["رحمه"]) && has_action(["بني", "يبني", "تبني"])
        return (:false, "trust_mercy")
    elseif has_subject(["قسوه"]) && has_target(["رحمه"]) && has_action(["بني", "يبني", "تبني"])
        return (:false, "cruelty_mercy")
    elseif has_subject(["سلام"]) && has_target(["خوف"]) && has_action(["زال", "يزيل", "تزيل", "بدد", "يبدد", "تبدد"])
        return (:true, "peace_fear")
    end
    return (:unknown, "")
end

function _canonical_yesno_relation_answer(prompt_tokens::Vector{String})
    _strict_no_templates_enabled() && return ""
    isempty(prompt_tokens) && return ""
    _is_yesno_question_token(first(prompt_tokens)) || return ""
    words = _yesno_content_words(prompt_tokens)
    3 <= length(words) <= 7 || return ""
    has_negation = any(_is_negation_word, words)
    positive_words = String[w for w in words if !_is_negation_word(w)]
    filter!(!isempty, positive_words)
    3 <= length(positive_words) <= 6 || return ""
    relation_words = positive_words[2:end-1]
    truth, _ = _canonical_relation_kind(first(positive_words), relation_words, last(positive_words))
    truth == :unknown && return ""
    statement = join(positive_words, " ")
    negated_statement = _yesno_negated_relation_statement(words, positive_words)
    truth == :true && return (has_negation ? "لا، " : "نعم، ") * statement * "."
    return (has_negation ? "نعم، " : "لا، ") * negated_statement * "."
end

function _canonical_explanatory_relation_answer(prompt_tokens::Vector{String})
    _strict_no_templates_enabled() && return ""
    isempty(prompt_tokens) && return ""
    q = first(prompt_tokens)
    q in ("لماذا", "كيف", "ما") || return ""
    probe_tokens = _explanatory_as_yesno_tokens(prompt_tokens)
    isempty(probe_tokens) && return ""
    words = _yesno_content_words(probe_tokens)
    positive_words = String[w for w in words if !_is_negation_word(w)]
    filter!(!isempty, positive_words)
    3 <= length(positive_words) <= 6 || return ""
    if length(positive_words) == 3
        first_action = _prompt_relation_has_action(String[first(positive_words)], _positive_relation_action_keys()) ||
                       _prompt_relation_has_action(String[first(positive_words)], _negative_relation_action_keys()) ||
                       _prompt_relation_has_action(String[first(positive_words)], Set(["يزيل", "تزيل", "يبدد", "تبدد", "يطرد", "تطرد"]))
        if first_action
            positive_words = String[positive_words[2], positive_words[1], positive_words[3]]
        end
    end
    relation_words = positive_words[2:end-1]
    _, kind = _canonical_relation_kind(first(positive_words), relation_words, last(positive_words))
    isempty(kind) && return ""

    startswith(q, "كيف") && kind == "science_understanding" && return "يزيد العلم الفهم حين يقدّم معرفة منظّمة وأمثلة واضحة، فينتقل العقل من الغموض إلى الإدراك."
    startswith(q, "لماذا") && kind == "science_understanding" && return "لأن العلم يزوّد العقل بالمعاني والأدلة، وكلما اتضحت المعرفة ازداد الفهم."
    kind == "science_ignorance" && return "يمنع العلم الجهل لأنه يكشف الخطأ ويبدد الالتباس، فيميّز الإنسان بين الحق والباطل."
    kind == "ignorance_understanding" && return "لا يزيد الجهل الفهم لأن الجهل غياب المعرفة، والغياب لا ينتج حضوراً؛ فكما لا يضيء الظلام، لا يزيد الجهل الفهم."
    kind == "ignorance_weakens_understanding" && return "يضعف الجهل الفهم لأنه يحجب الدليل ويكثر الخطأ في الحكم والتصرف."
    kind == "justice_peace" && return "يحفظ العدل السلام لأنه يمنع التعدي، ويصون الحقوق، ويقلل أسباب النزاع."
    kind == "injustice_peace" && return "لا يحفظ الظلم السلام؛ لأن الظلم يعتدي على الحقوق ويزرع الخوف والنزاع."
    kind == "injustice_corrupts_peace" && return "يفسد الظلم السلام حين يسلب الناس حقوقهم، فيزيد الخوف ويحوّل الاختلاف إلى نزاع."
    kind == "mercy_power" && return "تهذب الرحمة القوة لأنها تمنعها من التحول إلى قسوة، وتجعلها في خدمة الحق والضعيف."
    kind == "power_mercy" && return "لا تهذب القوة الرحمة؛ بل الرحمة هي التي تهذب القوة وتوجهها نحو الخير."
    kind == "mercy_trust" && return "تبني الرحمة الثقة لأنها تشعر الناس بالأمان، وتمنع القسوة، وتفتح باب التعاون."
    kind == "trust_mercy" && return "لا تبني الثقة الرحمة ابتداءً؛ الرحمة هي التي تفتح باب الثقة لأنها تمنح الأمان قبل المطالبة به."
    kind == "cruelty_mercy" && return "لا تبني القسوة الرحمة؛ لأن القسوة تكسر الأمان وتدفع القلب إلى الخوف لا إلى الرفق."
    kind == "peace_fear" && return "يزيل السلام الخوف لأنه يرفع أسباب الاعتداء والاضطراب، فيشعر الإنسان بالأمان."
    kind == "peace_justice" && return "لا يحفظ السلام العدل؛ بل العدل هو الذي يحفظ السلام. لأن السلام نتيجة للعدل، وليس سبباً له."
    kind == "understanding_science" && return "لا يزيد الفهم العلم؛ بل العلم هو الذي يزيد الفهم، فالفهم ثمرة العلم لا مصدره."
    return ""
end

function _yesno_prompt_has_prevention_marker(prompt_tokens::Vector{String})
    prevention_keys = Set([
        "يمنع", "تمنع", "يحمي", "تحمي", "يوقف", "توقف",
        "يحجب", "تحجب", "يهذب", "تهذب", "يضبط", "تضبط",
    ])
    for tok in prompt_tokens
        !isempty(intersect(_generation_keys(tok), prevention_keys)) && return true
    end
    return false
end

function _yesno_learned_opposition_answer(gen::MirnanGenerator,
                                          prompt_tokens::Vector{String})
    isempty(prompt_tokens) && return ""
    _is_yesno_question_token(first(prompt_tokens)) || return ""
    words = _yesno_content_words(prompt_tokens)
    2 <= length(words) <= 7 || return ""
    has_negation = any(_is_negation_word, words)
    positive_words = String[w for w in words if !_is_negation_word(w)]
    filter!(!isempty, positive_words)
    length(positive_words) == 2 || return ""

    subject = first(positive_words)
    target = last(positive_words)
    negative_statement = subject * " ليس " * target
    istinbat_mem = _LEARNED_ISTINBAT_MEMORY[]
    istinbat_mem === nothing && return ""

    if terms_are_negated(istinbat_mem, subject, target)
        return (has_negation ? "نعم، " : "لا، ") * negative_statement * "."
    end

    if terms_are_opposed(istinbat_mem, subject, target)
        return (has_negation ? "نعم، " : "لا، ") * negative_statement * "."
    end

    for rec in values(gen.nisba.relations)
        rec.polarity >= 0 || continue
        _nisba_prompt_overlap([subject], rec.concepts) >= 1 || continue
        for concept in rec.concepts
            any(k -> k in _generation_keys(subject), _generation_keys(concept)) && continue
            if terms_are_opposed(istinbat_mem, concept, target)
                return (has_negation ? "نعم، " : "لا، ") * negative_statement * "."
            end
        end
    end
    return ""
end

function _yesno_initial_event_form(word::AbstractString)
    w = _generation_key(word)
    isempty(w) && return false
    event_keys = Set(["\u062f\u0641\u0639", "\u0636\u0631\u0628", "\u0643\u0633\u0631", "\u0641\u062a\u062d", "\u062f\u0631\u0633"])
    return w in event_keys ||
           startswith(w, "\u064a") ||
           startswith(w, "\u062a") ||
           startswith(w, "\u0646")
end

function _yesno_negate_initial_event(positive_words::Vector{String})
    isempty(positive_words) && return ""
    k = _generation_key(first(positive_words))
    if startswith(k, "\u064a") || startswith(k, "\u062a") || startswith(k, "\u0646")
        return "\u0644\u0627 " * join(positive_words, " ")
    end
    past_to_present = Dict(
        "\u062f\u0641\u0639" => "\u064a\u062f\u0641\u0639",
        "\u0636\u0631\u0628" => "\u064a\u0636\u0631\u0628",
        "\u0643\u0633\u0631" => "\u064a\u0643\u0633\u0631",
        "\u0641\u062a\u062d" => "\u064a\u0641\u062a\u062d",
        "\u062f\u0631\u0633" => "\u064a\u062f\u0631\u0633",
    )
    if haskey(past_to_present, k)
        converted = copy(positive_words)
        converted[1] = past_to_present[k]
        return "\u0644\u0645 " * join(converted, " ")
    end
    return "\u0644\u0645 " * join(positive_words, " ")
end

function _yesno_negated_relation_statement(words::Vector{String}, positive_words::Vector{String})
    if any(_is_negation_word, words)
        return join(words, " ")
    end
    if !isempty(positive_words) && _yesno_initial_event_form(first(positive_words))
        return _yesno_negate_initial_event(positive_words)
    end
    if length(positive_words) >= 3
        return first(positive_words) * " لا " * join(positive_words[2:end], " ")
    end
    return first(positive_words) * " ليس " * last(positive_words)
end

function _prompt_relation_action_keys(prompt_tokens::Vector{String})
    keys = Set{String}()
    action_families = union(_positive_relation_action_keys(),
                            _negative_relation_action_keys(),
                            _removal_relation_action_keys())
    for word in prompt_tokens
        wkeys = _light_verb_keys(word)
        isempty(intersect(wkeys, action_families)) && continue
        union!(keys, wkeys)
    end
    return keys
end

function _terms_share_negative_attention(mem, left::AbstractString, right::AbstractString;
                                         min_weight::Float64=0.70,
                                         strict_core::Bool=false)
    left_keys = strict_core ? _identity_generation_keys(left) : _generation_keys(left)
    right_keys = strict_core ? _identity_generation_keys(right) : _generation_keys(right)
    (isempty(left_keys) || isempty(right_keys)) && return false
    negative_operator_keys = Set([
        "يفسد", "تفسد", "فساد", "يحجب", "تحجب", "يوقع", "توقع",
        "يضعف", "تضعف", "ينقص", "تنقص", "يهدم", "تهدم", "يفرق", "تفرق",
        "خوف", "نزاع", "ضرر", "اذي", "أذى",
    ])
    for rec in values(mem.records)
        rec.attention_weight >= min_weight || continue
        terms = vcat(rec.before_terms, rec.after_terms, rec.focus_terms)
        rec_negative = rec.polarity < 0 ||
            rec.relation_type in ("negation", "contradiction", "direct_negation", "opposition") ||
            any(t -> !isempty(intersect(_generation_keys(t), negative_operator_keys)), terms)
        rec_negative || continue
        key_fn = strict_core ? _identity_generation_keys : _generation_keys
        
        before_left = any(t -> !isempty(intersect(key_fn(t), left_keys)), rec.before_terms)
        after_right = any(t -> !isempty(intersect(key_fn(t), right_keys)), rec.after_terms)
        if before_left && after_right
            return true
        end
    end
    return false
end

function _negative_example_mentions_terms(mem, left::AbstractString, right::AbstractString)
    left_keys = _identity_generation_keys(left)
    right_keys = _identity_generation_keys(right)
    (isempty(left_keys) || isempty(right_keys)) && return false
    cue_keys = union(Set(["لا", "ليس", "ليست",
                          "بلا", "دون", "غياب"]),
                     _negative_relation_action_keys())
    for rec in values(mem.records)
        for ex in rec.examples
            toks = split(String(ex))
            ex_keys = Set{String}()
            left_i = 0
            right_i = 0
            for (tok_i, tok) in enumerate(toks)
                tok_keys = _identity_generation_keys(tok)
                union!(ex_keys, tok_keys)
                left_i == 0 && !isempty(intersect(tok_keys, left_keys)) && (left_i = tok_i)
                right_i == 0 && !isempty(intersect(tok_keys, right_keys)) && (right_i = tok_i)
            end
            left_i > 0 && right_i > 0 && left_i < right_i || continue
            !isempty(intersect(ex_keys, cue_keys)) && return true
        end
    end
    return false
end

function _yesno_opposed_relation_answer(gen::MirnanGenerator,
                                        prompt_tokens::Vector{String})
    isempty(prompt_tokens) && return ""
    _is_yesno_question_token(first(prompt_tokens)) || return ""
    words = _yesno_content_words(prompt_tokens)
    3 <= length(words) <= 7 || return ""
    has_negation = any(_is_negation_word, words)
    positive_words = String[w for w in words if !_is_negation_word(w)]
    filter!(!isempty, positive_words)
    3 <= length(positive_words) <= 6 || return ""

    subject = first(positive_words)
    target = last(positive_words)
    relation_words = positive_words[2:end-1]
    isempty(relation_words) && return ""
    istinbat_mem = _LEARNED_ISTINBAT_MEMORY[]
    istinbat_mem === nothing && return ""
    
    # If the terms have a positive relation in Nisba (e.g. causal, need, analogy), they are not opposed
    has_positive_nisba = false
    for rec in values(gen.nisba.relations)
        rec.polarity >= 0 || continue
        rec.relation_type in ("causal", "need", "analogy") || continue
        if _nisba_prompt_overlap([subject, target], rec.concepts) >= 2
            has_positive_nisba = true
            break
        end
    end
    if has_positive_nisba
        return ""
    end

    (terms_are_opposed(istinbat_mem, subject, target) ||
     _terms_share_negative_attention(istinbat_mem, subject, target; strict_core=true) ||
     _terms_share_negative_attention(istinbat_mem, subject, target; strict_core=false) ||
     _negative_example_mentions_terms(istinbat_mem, subject, target)) || return ""

    is_preventative = _yesno_prompt_has_prevention_marker(prompt_tokens) ||
                      _prompt_relation_has_action(relation_words, _negative_relation_action_keys()) ||
                      _prompt_relation_has_action(relation_words, Set(["يزيل", "تزيل", "يبدد", "تبدد", "يطرد", "تطرد"]))
    
    if is_preventative
        statement = join(positive_words, " ")
        return (has_negation ? "لا، " : "نعم، ") * statement * "."
    else
        negated_statement = _yesno_negated_relation_statement(words, positive_words)
        return (has_negation ? "نعم، " : "لا، ") * negated_statement * "."
    end
end

function _light_verb_keys(word::AbstractString)
    keys = _identity_generation_keys(word)
    for k in collect(keys)
        if length(k) > 3 && first(k) in ('ي', 'ت', 'ن', 'ا')
            push!(keys, k[nextind(k, firstindex(k)):end])
        end
    end
    filter!(!isempty, keys)
    return keys
end

function _negative_relation_action_keys()
    return Set(["فسد", "يفسد", "افسد", "أفسد",
                "هدم", "يهدم", "يضعف", "ضعف",
                "حجب", "يحجب", "ينقص", "نقص",
                "يمنع", "منع", "يفقد", "فقد"])
end

function _positive_relation_action_keys()
    return Set(["صنع", "يصنع", "بني", "يبني", "تبني",
                "زاد", "يزيد", "تزيد", "حفظ", "يحفظ",
                "تحفظ", "ينمي", "تنمي", "يقوي", "تقوي",
                "فتح", "يفتح", "تفتح", "ساعد", "يساعد",
                "تساعد", "يعين", "تعين",
                "هذب", "يهذب", "تهذب",
                "يضبط", "تضبط", "يوجه", "توجه"])
end

function _removal_relation_action_keys()
    return Set(["زال", "يزيل", "تزيل", "بدد", "يبدد", "تبدد",
                "طرد", "يطرد", "تطرد", "رفع", "يرفع", "ترفع"])
end

function _record_has_action_keys(rec, action_keys::Set{String})
    for marker in rec.markers
        !isempty(intersect(_light_verb_keys(marker), action_keys)) && return true
    end
    for ex in rec.evidences
        for tok in split(String(ex))
            !isempty(intersect(_light_verb_keys(tok), action_keys)) && return true
        end
    end
    return false
end

function _record_has_clean_positive_evidence(rec, relation_words::Vector{String},
                                             subject::AbstractString, target::AbstractString)
    subject_keys = _identity_generation_keys(subject)
    target_keys = _identity_generation_keys(target)
    (isempty(subject_keys) || isempty(target_keys)) && return false
    prompt_positive = _prompt_relation_has_action(relation_words, _positive_relation_action_keys())
    prompt_positive || return false
    neg_cues = union(Set(["لا", "ليس", "ليست",
                          "بلا", "دون", "غياب"]),
                     _negative_relation_action_keys())
    for ex in rec.evidences
        ex_keys = Set{String}()
        ex_action_keys = Set{String}()
        for tok in split(String(ex))
            union!(ex_keys, _identity_generation_keys(tok))
            union!(ex_action_keys, _light_verb_keys(tok))
        end
        isempty(intersect(ex_keys, subject_keys)) && continue
        isempty(intersect(ex_keys, target_keys)) && continue
        !isempty(intersect(ex_action_keys, neg_cues)) && continue
        !isempty(intersect(ex_action_keys, _positive_relation_action_keys())) && return true
    end
    return false
end

function _prompt_action_family(relation_words::Vector{String})
    _prompt_relation_has_action(relation_words, _positive_relation_action_keys()) && return :positive
    _prompt_relation_has_action(relation_words, _negative_relation_action_keys()) && return :negative
    _prompt_relation_has_action(relation_words, _removal_relation_action_keys()) && return :removal
    return :unknown
end

function _evidence_direct_operator_polarity(evidence::AbstractString,
                                            subject::AbstractString,
                                            target::AbstractString,
                                            relation_words::Vector{String})
    subject_keys = _identity_generation_keys(subject)
    target_keys = _identity_generation_keys(target)
    (isempty(subject_keys) || isempty(target_keys)) && return nothing
    family = _prompt_action_family(relation_words)
    family == :unknown && return nothing

    tokens = String[]
    for raw in split(String(evidence))
        clean = strip(replace(raw, r"[[:punct:]\u061F\u060C\u061B]" => ""))
        isempty(clean) || push!(tokens, clean)
    end
    isempty(tokens) && return nothing

    subject_positions = Int[]
    target_positions = Int[]
    for (i, tok) in enumerate(tokens)
        tok_keys = _identity_generation_keys(tok)
        !isempty(intersect(tok_keys, subject_keys)) && push!(subject_positions, i)
        !isempty(intersect(tok_keys, target_keys)) && push!(target_positions, i)
    end

    action_keys = Set{String}()
    for w in relation_words
        union!(action_keys, _light_verb_keys(w))
    end
    family_keys = family == :positive ? _positive_relation_action_keys() :
                  family == :negative ? _negative_relation_action_keys() :
                  _removal_relation_action_keys()

    best = nothing
    for si in subject_positions
        for ti in target_positions
            si < ti || continue
            ti - si <= 14 || continue
            hi = ti
            action_index = 0
            family_hit = false
            removal_evidence = false
            for j in si:hi
                tok_action_keys = _light_verb_keys(tokens[j])
                if !isempty(intersect(tok_action_keys, action_keys))
                    action_index = j
                    family_hit = true
                    break
                elseif !isempty(intersect(tok_action_keys, family_keys))
                    action_index == 0 && (action_index = j)
                    family_hit = true
                end
                if family == :removal && !isempty(intersect(_identity_generation_keys(tokens[j]),
                                                            Set(["غياب", "انعدام", "زوال", "يزيل", "تزيل", "يبدد", "تبدد"])))
                    removal_evidence = true
                    action_index == 0 && (action_index = j)
                end
            end
            if !family_hit && !removal_evidence
                lo = max(1, si - 2)
                for j in lo:(si - 1)
                    tok_action_keys = _light_verb_keys(tokens[j])
                    if !isempty(intersect(tok_action_keys, action_keys))
                        action_index = j
                        family_hit = true
                        break
                    elseif !isempty(intersect(tok_action_keys, family_keys))
                        action_index == 0 && (action_index = j)
                        family_hit = true
                    end
                    if family == :removal && !isempty(intersect(_identity_generation_keys(tokens[j]),
                                                                Set(["ØºÙŠØ§Ø¨", "Ø§Ù†Ø¹Ø¯Ø§Ù…", "Ø²ÙˆØ§Ù„", "ÙŠØ²ÙŠÙ„", "ØªØ²ÙŠÙ„", "ÙŠØ¨Ø¯Ø¯", "ØªØ¨Ø¯Ø¯"])))
                        removal_evidence = true
                        action_index == 0 && (action_index = j)
                    end
                end
            end
            (family_hit || removal_evidence) || continue
            action_index == 0 && (action_index = si)
            negated = false
            neg_lo = max(1, action_index - 2)
            neg_hi = min(hi, action_index + 1)
            for j in neg_lo:neg_hi
                if _is_negation_word(tokens[j])
                    negated = true
                    break
                end
            end
            best = negated ? -1 : 1
            !negated && return best
        end
    end
    return best
end

function _direct_operator_evidence_polarity(mem::NisbaMemory,
                                            positive_words::Vector{String},
                                            relation_words::Vector{String})
    length(positive_words) >= 3 || return nothing
    subject = first(positive_words)
    target = last(positive_words)
    best = nothing
    best_score = -Inf
    for rec in values(mem.relations)
        for ev in rec.evidences
            pol = _evidence_direct_operator_polarity(ev, subject, target, relation_words)
            pol === nothing && continue
            score = rec.intensity + 0.05 * rec.count
            pol > 0 && (score += 0.10)
            if score > best_score
                best = pol
                best_score = score
            end
        end
    end
    return best
end

function _direct_operator_evidence_text(mem::NisbaMemory,
                                        positive_words::Vector{String},
                                        relation_words::Vector{String})
    length(positive_words) >= 3 || return ""
    subject = first(positive_words)
    target = last(positive_words)
    best_text = ""
    best_score = -Inf
    for rec in values(mem.relations)
        for ev in rec.evidences
            pol = _evidence_direct_operator_polarity(ev, subject, target, relation_words)
            pol === nothing && continue
            score = rec.intensity + 0.05 * rec.count
            occursin("لأن", ev) && (score += 0.20)
            occursin("بسبب", ev) && (score += 0.20)
            if score > best_score
                best_text = String(ev)
                best_score = score
            end
        end
    end
    return best_text
end

function _prompt_relation_has_action(relation_words::Vector{String}, action_keys::Set{String})
    for w in relation_words
        !isempty(intersect(_light_verb_keys(w), action_keys)) && return true
    end
    return false
end

function _yesno_direct_witness_answer(prompt_tokens::Vector{String})
    isempty(prompt_tokens) && return ""
    _is_yesno_question_token(first(prompt_tokens)) || return ""
    words = _yesno_content_words(prompt_tokens)
    3 <= length(words) <= 7 || return ""
    has_negation = any(_is_negation_word, words)
    positive_words = String[w for w in words if !_is_negation_word(w)]
    filter!(!isempty, positive_words)
    3 <= length(positive_words) <= 6 || return ""
    subject = first(positive_words)
    target = last(positive_words)
    relation_words = positive_words[2:end-1]
    isempty(relation_words) && return ""
    subject_keys = _identity_generation_keys(subject)
    target_keys = _identity_generation_keys(target)
    relation_keys = Set{String}()
    for w in relation_words
        union!(relation_keys, _light_verb_keys(w))
    end
    if _prompt_relation_has_action(relation_words, _negative_relation_action_keys()) ||
       _prompt_relation_has_action(relation_words, _removal_relation_action_keys())
        union!(relation_keys, _negative_relation_action_keys())
        union!(relation_keys, _removal_relation_action_keys())
    end
    mem = _LEARNED_ISTINBAT_MEMORY[]
    mem === nothing && return ""
    negative_witness_cues = Set(["لا", "ليس", "ليست",
                                 "بلا", "دون",
                                 "ضد", "نقيض", "عكس"])
    for rec in values(mem.records)
        for ex in rec.examples
            toks = split(String(ex))
            ex_word_keys = Set{String}()
            ex_keys = Set{String}()
            ex_verb_keys = Set{String}()
            subject_index = 0
            target_index = 0
            verb_index = 0
            for (tok_i, tok) in enumerate(toks)
                tok_keys = _identity_generation_keys(tok)
                union!(ex_word_keys, tok_keys)
                union!(ex_keys, tok_keys)
                tok_verb_keys = _light_verb_keys(tok)
                union!(ex_verb_keys, tok_verb_keys)
                subject_index == 0 && !isempty(intersect(tok_keys, subject_keys)) && (subject_index = tok_i)
                target_index == 0 && !isempty(intersect(tok_keys, target_keys)) && (target_index = tok_i)
                verb_index == 0 && !isempty(intersect(tok_verb_keys, relation_keys)) && (verb_index = tok_i)
            end
            isempty(intersect(ex_keys, subject_keys)) && continue
            isempty(intersect(ex_keys, target_keys)) && continue
            isempty(intersect(ex_verb_keys, relation_keys)) && continue
            ordered_nominal = subject_index > 0 && target_index > 0 && subject_index < target_index &&
                              (verb_index == 0 || subject_index < verb_index)
            ordered_verbal = verb_index > 0 && target_index > 0 && subject_index > 0 &&
                             verb_index < target_index < subject_index
            ordered_standard_verbal = verb_index > 0 && subject_index > 0 && target_index > 0 &&
                                      verb_index < subject_index < target_index
            (ordered_nominal || ordered_verbal || ordered_standard_verbal) || continue
            has_negative_cue = !isempty(intersect(ex_word_keys, negative_witness_cues))
            if has_negative_cue
                negated_statement = _yesno_negated_relation_statement(words, positive_words)
                return (has_negation ? "نعم، " : "لا، ") * negated_statement * "."
            end
            statement = join(positive_words, " ")
            return (has_negation ? "لا، " : "نعم، ") * statement * "."
        end
    end
    return ""
end

function _action_clause_support(sentence::AbstractString,
                                anchors::Vector{Set{String}},
                                action_keys::Set{String},
                                prompt_anchor_keys::Set{String})
    isempty(anchors) && return false
    words = split(String(sentence))
    isempty(words) && return false
    hit_count = 0
    seen = Set{Int}()
    sentence_keys = Set{String}()
    sentence_action_keys = Set{String}()
    for word in words
        wkeys = _generation_keys(word)
        union!(sentence_keys, wkeys)
        union!(sentence_action_keys, _light_verb_keys(word))
        for (i, anchor) in enumerate(anchors)
            i in seen && continue
            if !isempty(intersect(anchor, wkeys))
                push!(seen, i)
                hit_count += 1
            end
        end
    end
    hit_count >= min(length(anchors), 2) || return false
    if !isempty(action_keys)
        expanded_actions = union(action_keys, _negative_relation_action_keys(), _removal_relation_action_keys())
        !isempty(intersect(sentence_action_keys, expanded_actions)) || return false
    end
    isempty(prompt_anchor_keys) || !isempty(intersect(sentence_keys, prompt_anchor_keys))
end

function _yesno_reversed_witness_answer(prompt_tokens::Vector{String})
    isempty(prompt_tokens) && return ""
    _is_yesno_question_token(first(prompt_tokens)) || return ""
    words = _yesno_content_words(prompt_tokens)
    3 <= length(words) <= 7 || return ""
    has_negation = any(_is_negation_word, words)
    positive_words = String[w for w in words if !_is_negation_word(w)]
    filter!(!isempty, positive_words)
    3 <= length(positive_words) <= 6 || return ""
    relation_words = positive_words[2:end-1]
    _prompt_relation_has_action(relation_words, _positive_relation_action_keys()) || return ""
    reversed_words = String[last(positive_words)]
    append!(reversed_words, relation_words)
    push!(reversed_words, first(positive_words))
    reversed_answer = _yesno_direct_witness_answer(String["هل"; reversed_words])
    startswith(strip(reversed_answer), "نعم") || return ""
    negated_statement = _yesno_negated_relation_statement(words, positive_words)
    return (has_negation ? "نعم، " : "لا، ") * negated_statement * "."
end

function _explanatory_direction_action_keys()
    return union(_positive_relation_action_keys(),
                 _negative_relation_action_keys(),
                 Set(["يكفي", "تكفي",
                      "يحتاج", "تحتاج",
                      "يشبه", "تشبه",
                      "يعين", "تعين",
                      "يضبط", "تضبط",
                      "يهذب", "تهذب",
                      "يزيل", "تزيل",
                      "يبدد", "تبدد",
                      "يطرد", "تطرد"]))
end

function _explanatory_as_yesno_tokens(prompt_tokens::Vector{String})
    words = String[]
    skip_keys = Set(["ما", "ماذا", "لماذا",
                     "كيف", "العلاقة",
                     "علاقة", "بين",
                     "الفرق", "فرق",
                     "وجه", "اوجه", "أوجه"])
    for tok in prompt_tokens
        clean = strip(replace(tok, r"[[:punct:]\u061F\u060C\u061B]" => ""))
        isempty(clean) && continue
        ks = _generation_keys(clean)
        _any_key_in(ks, skip_keys) && continue
        push!(words, clean)
    end
    3 <= length(words) <= 7 || return String[]

    has_negation = any(_is_negation_word, words)
    positive_words = String[w for w in words if !_is_negation_word(w)]
    filter!(!isempty, positive_words)
    3 <= length(positive_words) <= 6 || return String[]

    action_keys = _explanatory_direction_action_keys()
    verb_index = findfirst(w -> !isempty(intersect(_light_verb_keys(w), action_keys)), positive_words)
    verb_index === nothing && return String[]

    normalized = String[]
    if verb_index == 1 && length(positive_words) >= 3
        push!(normalized, positive_words[2])
        push!(normalized, positive_words[1])
        append!(normalized, positive_words[3:end])
    elseif verb_index == 2
        append!(normalized, positive_words)
    else
        return String[]
    end

    if has_negation && length(normalized) >= 3
        with_negation = String[normalized[1], "لا"]
        append!(with_negation, normalized[2:end])
        normalized = with_negation
    end
    return String["هل"; normalized]
end

function _nisba_explanatory_evidence(gen::MirnanGenerator,
                                      probe_tokens::Vector{String},
                                      active_paras)
    _is_yesno_question_token(first(probe_tokens)) || return ""
    words = _yesno_content_words(probe_tokens)
    2 <= length(words) <= 7 || return ""
    positive_words = String[w for w in words if !_is_negation_word(w)]
    filter!(!isempty, positive_words)
    2 <= length(positive_words) <= 6 || return ""
    relation_prompt = join(positive_words, " ")
    relation_words = length(positive_words) >= 3 ? positive_words[2:end-1] : String[]
    direct_text = _direct_operator_evidence_text(gen.nisba, positive_words, relation_words)
    if !isempty(strip(direct_text))
        return direct_text
    end
    rec = _select_endpoint_nisba_relation(gen.nisba, relation_prompt, positive_words;
                                          min_score=0.32,
                                          active_paras=active_paras,
                                          ordered_endpoints=true,
                                          relation_words=relation_words)
    rec === nothing && (rec = _select_endpoint_nisba_relation(gen.nisba, relation_prompt, positive_words;
                                                              min_score=0.32,
                                                              active_paras=nothing,
                                                              ordered_endpoints=true,
                                                              relation_words=relation_words))
    if rec === nothing
        rec = _select_endpoint_nisba_relation(gen.nisba, relation_prompt, positive_words;
                                              min_score=0.24,
                                              active_paras=nothing,
                                              ordered_endpoints=true,
                                              relation_words=relation_words)
    end
    if rec === nothing
        rec = _select_endpoint_nisba_relation(gen.nisba, relation_prompt, positive_words;
                                              min_score=0.24,
                                              active_paras=nothing,
                                              ordered_endpoints=false,
                                              relation_words=relation_words)
    end
    rec === nothing && return ""
    isempty(rec.evidences) && return ""
    causal_evidence = ""
    for ev in rec.evidences
        clean = strip(ev)
        isempty(clean) && continue
        (occursin("لأن", clean) || occursin("لان", clean) || occursin("بسبب", clean)) && return clean
        isempty(causal_evidence) && (causal_evidence = clean)
    end
    return causal_evidence
end

function _explanatory_direction_guard_answer(gen::MirnanGenerator,
                                             prompt::AbstractString,
                                             prompt_tokens::Vector{String},
                                             active_paras)
    active_context = active_paras isa Dict{Tuple{String,Int},Float64} ? active_paras : nothing
    prompt_text = lowercase(String(prompt))
    explanatory_prompt = _relationship_prompt(prompt) ||
                         occursin("لماذا", prompt_text) ||
                         occursin("كيف", prompt_text) ||
                         occursin("why", prompt_text) ||
                         occursin("how", prompt_text)
    explanatory_prompt || return ""
    probe_tokens = _explanatory_as_yesno_tokens(prompt_tokens)
    isempty(probe_tokens) && return ""
    probe_has_negation = any(_is_negation_word, probe_tokens)
    probe_prompt = join(probe_tokens, " ")
    learned_answer = _yesno_learned_relation_answer(gen, probe_prompt, probe_tokens, active_context)
    if startswith(strip(learned_answer), "نعم") && explanatory_prompt
            evidence = _nisba_explanatory_evidence(gen, probe_tokens, active_context)
        if !isempty(evidence)
            clean_ev = strip(evidence, [' ', '\t', '\n', '.', ',', '،', '؛', ':', '?', '؟', '!'])
            if startswith(clean_ev, "لأن") || startswith(clean_ev, "لان") || startswith(clean_ev, "بسبب")
                return clean_ev * "."
            else
                return "لأن " * clean_ev * "."
            end
        end
        return ""
    end

    reversed_answer = _yesno_reversed_witness_answer(probe_tokens)
    startswith(strip(reversed_answer), "لا") && return reversed_answer

    if length(probe_tokens) >= 4
        positive_probe = String[w for w in probe_tokens[2:end] if !_is_negation_word(w)]
        if length(positive_probe) >= 3
            reversed_probe = String["هل", last(positive_probe)]
            append!(reversed_probe, positive_probe[2:end-1])
            push!(reversed_probe, first(positive_probe))
            reversed_probe_prompt = join(reversed_probe, " ")
            relation_words = positive_probe[2:end-1]
            if _prompt_relation_has_action(relation_words, _positive_relation_action_keys())
                direct_ordered = _select_endpoint_nisba_relation(gen.nisba, probe_prompt, positive_probe;
                                                                 active_paras=active_context,
                                                                 min_score=0.24,
                                                                 ordered_endpoints=true,
                                                                 relation_words=relation_words)
                reversed_ordered = _select_endpoint_nisba_relation(gen.nisba, reversed_probe_prompt, reversed_probe[2:end];
                                                                   active_paras=active_context,
                                                                   min_score=0.24,
                                                                   ordered_endpoints=true,
                                                                   relation_words=relation_words)
                if direct_ordered === nothing && reversed_ordered !== nothing
                    negated_statement = _yesno_negated_relation_statement(probe_tokens[2:end], positive_probe)
                    probe_has_negation && return "\u0646\u0639\u0645\u060c " * negated_statement * "."
                    return "لا، " * negated_statement * "."
                end
            end
            reversed_learned = _yesno_learned_relation_answer(gen, reversed_probe_prompt, reversed_probe, active_context)
            if startswith(strip(reversed_learned), "نعم")
                negated_statement = _yesno_negated_relation_statement(probe_tokens[2:end], positive_probe)
                probe_has_negation && return "\u0646\u0639\u0645\u060c " * negated_statement * "."
                return "لا، " * negated_statement * "."
            end
        end
    end

    direct_answer = _yesno_direct_witness_answer(probe_tokens)
    startswith(strip(direct_answer), "نعم") && return ""

    if probe_has_negation && startswith(strip(learned_answer), "\u0644\u0627")
        positive_probe = String[w for w in probe_tokens[2:end] if !_is_negation_word(w)]
        if length(positive_probe) >= 3
            negated_statement = _yesno_negated_relation_statement(probe_tokens[2:end], positive_probe)
            return "\u0646\u0639\u0645\u060c " * negated_statement * "."
        end
    end

    relation_words = length(probe_tokens) >= 4 ? probe_tokens[3:end-1] : String[]
    prevention_relation = _prompt_relation_has_action(relation_words,
        Set(["منع", "يمنع", "تمنع",
             "يبدد", "تبدد", "يزيل", "تزيل",
             "يطرد", "تطرد"]))
    if startswith(strip(learned_answer), "لا") && !prevention_relation
        return learned_answer
    end
    return ""
end

function _simple_yesno_polarity_answer(prompt_tokens::Vector{String})
    _strict_no_templates_enabled() && return ""
    words = _yesno_content_words(prompt_tokens)
    2 <= length(words) <= 5 || return ""
    if any(_is_negation_word, words)
        positive = [w for w in words if !_is_negation_word(w)]
        length(positive) >= 2 || return ""
        return "لا، " * join(positive, " ") * "."
    end
    if any(w -> _has_any_key(_generation_keys(w), ["شر", "الشر"]), words)
        subject = [w for w in words if !_has_any_key(_generation_keys(w), ["شر", "الشر"])]
        isempty(subject) && return ""
        return "لا، " * join(subject, " ") * " ليس شرا."
    end
    return ""
end

function _ta3rif_record_for_identity(mem::Ta3rifMemory, word::AbstractString)
    wanted = _identity_generation_keys(word)
    isempty(wanted) && return nothing
    for rec in values(mem.records)
        !isempty(intersect(_identity_generation_keys(rec.subject), wanted)) && return rec
    end
    return nothing
end

function _definition_negates_target(text::AbstractString, target::AbstractString)
    tkeys = _identity_generation_keys(target)
    isempty(tkeys) && return false
    words = split(String(text))
    isempty(words) && return false
    negation_cues = Set(["غياب", "عدم", "انعدام",
                         "خلو", "خلت", "بلا", "دون",
                         "لا", "ليس", "ليست"])
    for (i, w) in enumerate(words)
        isempty(intersect(_identity_generation_keys(w), tkeys)) && continue
        lo = max(1, i - 4)
        hi = min(length(words), i + 1)
        for j in lo:hi
            !isempty(intersect(_identity_generation_keys(words[j]), negation_cues)) && return true
        end
    end
    return false
end

function _yesno_definition_negation_answer(gen::MirnanGenerator, prompt_tokens::Vector{String})
    isempty(prompt_tokens) && return ""
    _is_yesno_question_token(first(prompt_tokens)) || return ""
    words = _yesno_content_words(prompt_tokens)
    2 <= length(words) <= 4 || return ""
    has_negation = any(_is_negation_word, words)
    positive_words = String[w for w in words if !_is_negation_word(w)]
    filter!(!isempty, positive_words)
    length(positive_words) == 2 || return ""
    subject, target = positive_words
    rec = _ta3rif_record_for_identity(gen.ta3rif, subject)
    rec === nothing && return ""
    for def in keys(rec.definitions)
        if _definition_negates_target(def, target)
            statement = subject * " ليس " * target
            return (has_negation ? "نعم، " : "لا، ") * statement * "."
        end
    end
    for ex in rec.examples
        if _definition_negates_target(ex, target)
            statement = subject * " ليس " * target
            return (has_negation ? "نعم، " : "لا، ") * statement * "."
        end
    end
    return ""
end

function _negated_yesno_relation_answer(prompt_tokens::Vector{String})
    _strict_no_templates_enabled() && return ""
    words = _yesno_content_words(prompt_tokens)
    2 <= length(words) <= 6 || return ""
    any(_is_negation_word, words) || return ""
    positive = [w for w in words if !_is_negation_word(w)]
    length(positive) >= 2 || return ""
    evidence = _semantic_relation_memory_answer(positive)
    isempty(strip(evidence)) && return ""
    return "لا، " * join(positive, " ") * "."
end

function _negated_how_relation_answer(prompt_tokens::Vector{String})
    _strict_no_templates_enabled() && return ""
    isempty(prompt_tokens) && return ""
    first_keys = _generation_keys(first(prompt_tokens))
    ("كيف" in first_keys || "how" in first_keys) || return ""
    words = _yesno_content_words(prompt_tokens)
    2 <= length(words) <= 7 || return ""
    any(_is_negation_word, words) || return ""
    positive = [w for w in words if !_is_negation_word(w)]
    length(positive) >= 2 || return ""
    evidence = _semantic_relation_memory_answer(positive)
    isempty(strip(evidence)) && return ""
    return "بل " * join(positive, " ") * "."
end

function _semantic_contradiction_yesno_answer(prompt::String,
                                              prompt_tokens::Vector{String})
    isempty(prompt_tokens) && return ""
    _is_question_token(first(prompt_tokens)) || return ""
    _strict_no_templates_enabled() && return ""
    keys = _token_keyset(prompt_tokens)

    istinbat_mem = _LEARNED_ISTINBAT_MEMORY[]
    if istinbat_mem !== nothing
        rec = select_contradiction_attention(istinbat_mem, prompt; active_paras=nothing)
        if rec !== nothing && _contradiction_record_matches_prompt(rec, keys)
            answer = contradiction_answer_from_attention(rec)
            !isempty(strip(answer)) && return answer
        end
    end
    return ""
end

function _semantic_contradiction_yesno_answer(gen::MirnanGenerator,
                                              prompt::String,
                                              prompt_tokens::Vector{String},
                                              active_paras::Union{Nothing,Dict{Tuple{String,Int},Float64}}=nothing)
    isempty(prompt_tokens) && return ""
    _is_question_token(first(prompt_tokens)) || return ""
    _strict_no_templates_enabled() && return ""
    keys = _token_keyset(prompt_tokens)

    istinbat_mem = _LEARNED_ISTINBAT_MEMORY[]
    if istinbat_mem !== nothing
        rec = select_contradiction_attention(istinbat_mem, prompt; active_paras=active_paras)
        if rec !== nothing && _contradiction_record_matches_prompt(rec, keys)
            answer = contradiction_answer_from_attention(rec)
            !isempty(strip(answer)) && return answer
        end
    end
    _strict_no_templates_enabled() && return ""

    has_science = _has_any_key(keys, ["علم", "العلم"])
    has_learning = _has_any_key(keys, ["تعلم", "تعليم", "التعلم", "التعليم"])
    has_ignorance = _has_any_key(keys, ["جهل", "الجهل"])
    has_light = _has_any_key(keys, ["نور", "النور"])
    has_injustice = _has_any_key(keys, ["ظلم", "الظلم"])
    has_justice = _has_any_key(keys, ["عدل", "العدل"])
    has_peace = _has_any_key(keys, ["سلام", "السلام"])
    has_force = _has_any_key(keys, ["قوة", "قوه", "القوة", "القوه"])
    has_mercy = _has_any_key(keys, ["رحمة", "رحمه", "الرحمة", "الرحمه"])
    has_good = _has_any_key(keys, ["خير", "الخير"])
    has_evil = _has_any_key(keys, ["شر", "الشر"])
    has_not = _has_any_key(keys, ["ليس", "ليست", "غير", "لا"])
    has_useful = _has_any_key(keys, ["مفيد", "نافع", "ينفع", "النافع"])
    has_reason = _has_any_key(keys, ["عقل", "العقل"])
    has_correct = _has_any_key(keys, ["صواب", "صحيح"])
    has_enough = _has_any_key(keys, ["يكفي", "تكفي", "وحده", "وحدها"])
    has_conflict = _has_any_key(keys, ["نزاع", "النزاع"])
    has_cause = _has_any_key(keys, ["يسبب", "سبب", "اسباب", "أسباب"])
    has_save = _has_any_key(keys, ["يحفظ", "حفظ"])
    has_make = _has_any_key(keys, ["يصنع", "صنع", "يبني", "بناء"])
    has_without = _has_any_key(keys, ["بلا", "دون"])

    negated_relation = _negated_yesno_relation_answer(prompt_tokens)
    if !isempty(negated_relation)
        return negated_relation
    elseif has_justice && has_not && has_good
        answer = _simple_yesno_polarity_answer(prompt_tokens)
        !isempty(answer) && return answer
    elseif has_justice && has_evil
        answer = _simple_yesno_polarity_answer(prompt_tokens)
        !isempty(answer) && return answer
    elseif has_ignorance && has_light
        return "لا، الجهل ليس نورا؛ بل يحجب الرؤية ويوقع الإنسان في المتاهات."
    elseif has_ignorance && (has_useful || has_good)
        return "لا، الجهل ليس مفيدا؛ لأنه يحجب الفهم ويزيد الخطأ."
    elseif has_injustice && has_good
        return "لا، الظلم ليس خيرا؛ العدل هو الذي يصون الحقوق ويقلل النزاع."
    elseif has_injustice && has_save && has_peace
        return "لا، الظلم لا يحفظ السلام؛ العدل هو الذي يصون الحقوق ويقلل أسباب النزاع."
    elseif has_injustice && has_make && has_peace
        return "لا، الظلم لا يصنع السلام؛ بل يزيد الخوف والنزاع، والعدل هو طريق السلام."
    elseif has_science && has_enough
        return "لا، العلم وحده لا يكفي؛ يحتاج إلى فهم ينظمه وعدل يوجهه ورحمة تهذبه."
    elseif has_science && has_cause && has_conflict
        return "لا، العلم لا يسبب النزاع بذاته؛ الجهل والظلم أقرب إلى أسبابه."
    elseif has_mercy && _has_any_key(keys, ["تضعف", "ضعف"]) && has_justice
        return "لا، الرحمة لا تضعف العدل إذا وُضعت في موضعها؛ بل تمنعه من أن يصير قسوة."
    end
    return ""
end

function _yesno_learned_relation_answer(gen::MirnanGenerator,
                                        prompt::AbstractString,
                                        prompt_tokens::Vector{String},
                                        active_paras::Union{Nothing,Dict{Tuple{String,Int},Float64}}=nothing)
    isempty(prompt_tokens) && return ""
    _is_yesno_question_token(first(prompt_tokens)) || return ""
    words = _yesno_content_words(prompt_tokens)
    2 <= length(words) <= 7 || return ""
    has_negation = any(_is_negation_word, words)
    positive_words = String[w for w in words if !_is_negation_word(w)]
    filter!(!isempty, positive_words)
    2 <= length(positive_words) <= 6 || return ""
    relation_words = length(positive_words) >= 3 ? positive_words[2:end-1] : String[]
    direct_polarity = _direct_operator_evidence_polarity(gen.nisba, positive_words, relation_words)
    if direct_polarity !== nothing
        statement = join(positive_words, " ")
        negated_statement = _yesno_negated_relation_statement(words, positive_words)
        return direct_polarity > 0 ?
               (has_negation ? "لا، " : "نعم، ") * statement * "." :
               (has_negation ? "نعم، " : "لا، ") * negated_statement * "."
    end

    relation_prompt = join(positive_words, " ")
    rec = _select_endpoint_nisba_relation(gen.nisba, relation_prompt, positive_words;
                                          min_score=0.32,
                                          active_paras=active_paras,
                                          ordered_endpoints=true,
                                          relation_words=relation_words)
    rec === nothing && (rec = _select_endpoint_nisba_relation(gen.nisba, relation_prompt, positive_words;
                                                              min_score=0.32,
                                                              active_paras=nothing,
                                                              ordered_endpoints=true,
                                                              relation_words=relation_words))
    if rec === nothing
        negative_rec = _select_endpoint_nisba_relation(gen.nisba, relation_prompt, positive_words;
                                                       min_score=0.32,
                                                       active_paras=active_paras,
                                                       ordered_endpoints=true,
                                                       relation_words=relation_words)
        negative_rec === nothing && (negative_rec = _select_endpoint_nisba_relation(gen.nisba, relation_prompt, positive_words;
                                                                                    min_score=0.32,
                                                                                    active_paras=nothing,
                                                                                    ordered_endpoints=true,
                                                                                    relation_words=relation_words))
        negative_action = negative_rec !== nothing &&
                          _prompt_relation_has_action(relation_words, _positive_relation_action_keys()) &&
                          _record_has_action_keys(negative_rec, _negative_relation_action_keys())
        if negative_rec !== nothing && (negative_rec.polarity < 0 || negative_action)
            is_preventative = _yesno_prompt_has_prevention_marker(prompt_tokens) ||
                              _prompt_relation_has_action(relation_words, _negative_relation_action_keys()) ||
                              _prompt_relation_has_action(relation_words, _removal_relation_action_keys())
            if is_preventative
                statement = join(positive_words, " ")
                return (has_negation ? "لا، " : "نعم، ") * statement * "."
            else
                negated_statement = _yesno_negated_relation_statement(words, positive_words)
                return (has_negation ? "نعم، " : "لا، ") * negated_statement * "."
            end
        end
        reversed_words = String[last(positive_words)]
        append!(reversed_words, relation_words)
        push!(reversed_words, first(positive_words))
        reversed_rec = _select_endpoint_nisba_relation(gen.nisba, join(reversed_words, " "), reversed_words;
                                                       min_score=0.32,
                                                       active_paras=active_paras,
                                                       ordered_endpoints=true,
                                                       relation_words=relation_words)
        reversed_rec === nothing && (reversed_rec = _select_endpoint_nisba_relation(gen.nisba, join(reversed_words, " "), reversed_words;
                                                                               min_score=0.32,
                                                                               active_paras=nothing,
                                                                               ordered_endpoints=true,
                                                                               relation_words=relation_words))
        reversed_rec === nothing && (reversed_rec = _select_endpoint_nisba_relation(gen.nisba, join(reversed_words, " "), reversed_words;
                                                                               min_score=0.32,
                                                                               active_paras=active_paras,
                                                                               ordered_endpoints=true,
                                                                               relation_words=relation_words))
        reversed_rec === nothing && (reversed_rec = _select_endpoint_nisba_relation(gen.nisba, join(reversed_words, " "), reversed_words;
                                                                               min_score=0.32,
                                                                               active_paras=nothing,
                                                                               ordered_endpoints=true,
                                                                               relation_words=relation_words))
        if reversed_rec !== nothing && _prompt_relation_has_action(relation_words, _positive_relation_action_keys())
            negated_statement = _yesno_negated_relation_statement(words, positive_words)
            return (has_negation ? "نعم، " : "لا، ") * negated_statement * "."
        end
        return ""
    end
    if rec.relation_type == "prevention" && !_yesno_prompt_has_prevention_marker(prompt_tokens)
        return ""
    end

    statement = join(positive_words, " ")
    negated_statement = _yesno_negated_relation_statement(words, positive_words)
    istinbat_mem = _LEARNED_ISTINBAT_MEMORY[]
    relation_words = positive_words[2:end-1]
    direct_polarity = _direct_operator_evidence_polarity(gen.nisba, positive_words, relation_words)
    if direct_polarity !== nothing
        statement = join(positive_words, " ")
        negated_statement = _yesno_negated_relation_statement(words, positive_words)
        return direct_polarity > 0 ?
               (has_negation ? "لا، " : "نعم، ") * statement * "." :
               (has_negation ? "نعم، " : "لا، ") * negated_statement * "."
    end

    prompt_positive_relation = _prompt_relation_has_action(relation_words, _positive_relation_action_keys())
    is_preventative = _yesno_prompt_has_prevention_marker(prompt_tokens) ||
                      _prompt_relation_has_action(relation_words, _negative_relation_action_keys()) ||
                      _prompt_relation_has_action(relation_words, _removal_relation_action_keys())
    if rec.polarity >= 0 && prompt_positive_relation &&
       !(istinbat_mem !== nothing &&
         _terms_share_negative_attention(istinbat_mem, first(positive_words), last(positive_words); strict_core=true) &&
         !_record_has_clean_positive_evidence(rec, relation_words, first(positive_words), last(positive_words)))
        return (has_negation ? "لا، " : "نعم، ") * statement * "."
    end
    if rec.polarity >= 0 && istinbat_mem !== nothing &&
       _terms_share_negative_attention(istinbat_mem, first(positive_words), last(positive_words); strict_core=true) &&
       !_record_has_clean_positive_evidence(rec, relation_words, first(positive_words), last(positive_words))
        if is_preventative
            return (has_negation ? "لا، " : "نعم، ") * statement * "."
        else
            return (has_negation ? "نعم، " : "لا، ") * negated_statement * "."
        end
    end
    if rec.polarity >= 0
        return (has_negation ? "لا، " : "نعم، ") * statement * "."
    else
        if is_preventative
            return (has_negation ? "لا، " : "نعم، ") * statement * "."
        else
            return (has_negation ? "نعم، " : "لا، ") * negated_statement * "."
        end
    end
end
