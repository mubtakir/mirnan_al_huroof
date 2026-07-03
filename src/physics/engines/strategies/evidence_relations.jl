function _relationship_prompt(prompt::AbstractString)
    s = lowercase(String(prompt))
    return (occursin("علاقة", s) || occursin("العلاقة", s) ||
            occursin("علاقه", s) || occursin("العلاقه", s) ||
            occursin("صلة", s) || occursin("الصلة", s) ||
            occursin("أثر", s) || occursin("اثر", s) ||
            occursin("الأثر", s) || occursin("الاثر", s) ||
            occursin("ارتباط", s) || occursin("الارتباط", s)) &&
           (occursin("بين", s) || occursin("مع", s) ||
            occursin("اشرح", s) ||
            occursin("يرتبط", s) || occursin("ترتبط", s) ||
            occursin("relation", s) || occursin("relationship", s))
end

function _relationship_connector_token(tok::AbstractString)
    keys = _generation_keys(tok)
    return _has_any_key(keys, [
        "حفظ", "يحفظ", "تحفظ", "لحفظ",
        "صون", "يصون", "تصون",
        "يكفي", "تكفي", "كفاية",
    ])
    union!(relation_keys, Set([
        "\u0644\u0627",
        "\u064a\u0632\u064a\u062f", "\u062a\u0632\u064a\u062f",
        "\u064a\u0632\u064a\u0644", "\u062a\u0632\u064a\u0644",
    ]))
end

function _relationship_anchor_keys(prompt_tokens::Vector{String})
    anchors = Vector{Set{String}}()
    seen = Set{String}()
    relation_keys = Set([
        "علاقه", "العلاقه", "علاقة", "العلاقة", "بين", "ما", "هي", "هو",
        "صله", "الصله", "صلة", "الصلة", "اثر", "الأثر", "الاثر", "أثر",
        "اشرح", "شرح", "معني", "معنى", "باي", "بأي", "اي", "أي",
        "يمكن", "يكون", "تكون", "كان", "كانت", "يؤدي", "تؤدي", "يودي", "تودي",
        "يحتاج", "تحتاج", "استعمال", "استخدام", "يساعد", "تساعد", "يمنع", "تمنع",
        "يقلل", "تقلل", "يجتمع", "تجتمع", "يوازن", "توازن", "يتحول", "تتحول",
        "يكفي", "تكفي", "يثبت", "تثبت", "يستفيد", "تستفيد", "يجمع", "تجمع",
        "يخشى", "تخشى", "يخشي", "تخشي", "يخاف", "تخاف", "يبدد", "تبدد", "يطرد", "تطرد",
        "ينقلب", "تنقلب"
    ])
    for tok in prompt_tokens
        ks = _generation_keys(tok)
        _is_negation_word(tok) && continue
        action_keys = union(_positive_relation_action_keys(), _negative_relation_action_keys(), _removal_relation_action_keys())
        _prompt_relation_has_action([String(tok)], action_keys) && continue
        _any_key_in(ks, GENERATION_STOPWORDS) && continue
        _any_key_in(ks, _EXTRA_GENERATION_STOPWORDS) && continue
        _any_key_in(ks, _QUESTION_TOOL_KEYS) && continue
        _relationship_connector_token(tok) && continue
        !isempty(intersect(ks, relation_keys)) && continue
        clean = strip(replace(tok, r"[[:punct:]\u061F\u060C\u061B]" => ""))
        isempty(clean) && continue
        key = _generation_family_key(clean)
        isempty(key) && continue
        key in seen && continue
        push!(anchors, ks)
        push!(seen, key)
    end
    return anchors
end

_relationship_identity_anchor_keys(prompt_tokens::Vector{String}) = _relationship_anchor_keys(prompt_tokens)

function _relationship_endpoint_terms(prompt_tokens::Vector{String})
    terms = String[]
    after_between = false
    skip_keys = Set(["ما", "ماذا", "هي", "هو", "العلاقة", "علاقة", "الصلة", "صلة", "بين"])
    for tok in prompt_tokens
        ks = _generation_keys(tok)
        if _has_any_key(ks, ["بين"])
            after_between = true
            continue
        end
        after_between || continue
        clean = strip(replace(tok, r"[[:punct:]\u061F\u060C\u061B]" => ""))
        isempty(clean) && continue
        _any_key_in(ks, skip_keys) && continue
        push!(terms, clean)
    end
    return terms
end

function _personal_dialogue_leak(sentence::AbstractString)
    s = lowercase(String(sentence))
    return occursin("أقضي يومي", s) || occursin("اقضي يومي", s) ||
           occursin("يومي بين", s) || occursin("أجمل شيء", s) ||
           occursin("اجمل شيء", s)
end

function _dialogue_style_leak(sentence::AbstractString)
    s = lowercase(strip(String(sentence)))
    _personal_dialogue_leak(s) && return true
    occursin("؟", s) && return true
    occursin("?", s) && return true
    occursin("ج:", s) && return true
    occursin("س:", s) && return true
    startswith(s, "ج:") && return true
    startswith(s, "س:") && return true
    startswith(s, "سؤال:") && return true
    startswith(s, "جواب:") && return true
    startswith(s, "وعليكم") && return true
    startswith(s, "الى اللقاء") && return true
    startswith(s, "إلى اللقاء") && return true
    startswith(s, "انا ") && return true
    startswith(s, "أنا ") && return true
    startswith(s, "نعم، أحب") && return true
    startswith(s, "نعم احب") && return true
    startswith(s, "نعم، أؤمن") && return true
    startswith(s, "نعم اؤمن") && return true
    occursin("شكرا لسؤالك", s) && return true
    occursin("شكراً لسؤالك", s) && return true
    occursin("كيف يمكنني خدمتك", s) && return true
    return false
end

function _learned_pair_evidence_sentence(gen::MirnanGenerator,
                                         left::AbstractString,
                                         right::AbstractString)
    anchors = [_identity_generation_keys(left), _identity_generation_keys(right)]
    any(isempty, anchors) && return ""
    function pair_span_score(text::AbstractString)
        words = String[strip(w) for w in split(String(text)) if !isempty(strip(w))]
        isempty(words) && return 0.0, 999
        positions = Int[]
        for anchor in anchors
            idx = findfirst(w -> !isempty(intersect(anchor, _identity_generation_keys(w))), words)
            idx === nothing && return 0.0, 999
            push!(positions, idx)
        end
        span = maximum(positions) - minimum(positions)
        return 1.0 / (1.0 + span / 8), span
    end
    function focused_pair_clause(text::AbstractString)
        clean = strip(String(text))
        clauses = split(clean, r"[؛;]")
        best_clause = clean
        best_score = 0.0
        for clause in clauses
            c = strip(String(clause))
            isempty(c) && continue
            keys = reduce(union, (_identity_generation_keys(w) for w in split(c)); init=Set{String}())
            count(a -> !isempty(intersect(a, keys)), anchors) >= 2 || continue
            score, _ = pair_span_score(c)
            if score > best_score
                best_score = score
                best_clause = c
            end
        end
        best_clause = strip(best_clause)
        startswith(best_clause, "و") && (best_clause = strip(best_clause[nextind(best_clause, firstindex(best_clause)):end]))
        return best_clause
    end
    raw_sentences = _load_raw_training_sentences_for_generator(gen)
    relation_cues = Set([
        "يرتبط", "ترتبط", "يقود", "تقود", "يساعد", "تساعد", "يحفظ", "تحفظ",
        "يزيد", "تزيد", "ينتج", "تنتج", "يثمر", "تثمر", "يعزز", "تعزز",
        "yفتح", "تفتح", "يكشف", "تكشف", "يحجب", "تحجب", "يمنع", "تمنع",
        "يبدد", "تبدد", "يهذب", "تهذب", "يبني", "تبني", "يصنع", "تصنع"
    ])
    contrast_cues = Set([
        "أما", "اما", "لكن", "بل", "ليس", "ليست", "لا", "غياب", "ضد",
        "يفرق", "تختلف", "مختلف", "يقابل", "تقابل"
    ])
    best = ""
    best_score = 0.0
    for rec in values(gen.nisba.relations)
        for evidence in rec.evidences
            clean = focused_pair_clause(evidence)
            isempty(clean) && continue
            _dialogue_style_leak(clean) && continue
            words = String[strip(w) for w in split(clean) if !isempty(strip(w))]
            4 <= length(words) <= 32 || continue
            sent_keys = reduce(union, (_identity_generation_keys(w) for w in words); init=Set{String}())
            hits = count(a -> !isempty(intersect(a, sent_keys)), anchors)
            hits >= 2 || continue
            span_score, span = pair_span_score(clean)
            span <= 14 || continue
            cue_bonus = any(k -> k in sent_keys, relation_cues) ? 0.22 : 0.0
            contrast_bonus = any(k -> k in sent_keys, contrast_cues) ? 0.16 : 0.0
            score = 0.62 + cue_bonus + contrast_bonus + 0.24 * span_score + 0.06 * rec.count
            score > best_score || continue
            best_score = score
            best = clean
        end
    end
    for sentence in raw_sentences
        clean = focused_pair_clause(sentence)
        isempty(clean) && continue
        _dialogue_style_leak(clean) && continue
        startswith(clean, "نعم") && continue
        startswith(clean, "لا ") && continue
        words = String[strip(w) for w in split(clean) if !isempty(strip(w))]
        4 <= length(words) <= 32 || continue
        sent_keys = reduce(union, (_identity_generation_keys(w) for w in words); init=Set{String}())
        hits = count(a -> !isempty(intersect(a, sent_keys)), anchors)
        hits >= 2 || continue
        span_score, span = pair_span_score(clean)
        span <= 14 || continue
        cue_bonus = any(k -> k in sent_keys, relation_cues) ? 0.22 : 0.0
        contrast_bonus = any(k -> k in sent_keys, contrast_cues) ? 0.16 : 0.0
        punct_bonus = any(p -> occursin(p, clean), [".", "،", "؛", ":"]) ? 0.08 : 0.0
        brevity = 1.0 / (1.0 + max(0, length(words) - 18) / 10)
        score = 0.58 + cue_bonus + contrast_bonus + punct_bonus + 0.10 * brevity + 0.20 * span_score
        score > best_score || continue
        best_score = score
        best = clean
    end
    isempty(best) && return ""
    return (endswith(best, ".") || endswith(best, "؟") || endswith(best, "?") || endswith(best, "!")) ? best : best * "."
end

function _anchor_hit_count_in_text(text::AbstractString, anchors::Vector{Set{String}})
    isempty(strip(String(text))) && return 0
    words = String[strip(w) for w in split(String(text)) if !isempty(strip(w))]
    keys = reduce(union, (_generation_keys(w) for w in words); init=Set{String}())
    return count(anchor -> _anchor_matches_keys(anchor, keys), anchors)
end

function _compact_composed_evidence(first::AbstractString,
                                    second::AbstractString,
                                    anchors::Vector{Set{String}})
    summary = _summarize_composed_evidence(first, second, anchors)
    !isempty(strip(summary)) && return summary

    pieces = String[]
    for sentence in (String(first), String(second))
        clean = strip(sentence)
        isempty(clean) && continue
        clauses = split(clean, r"(?<=[.؟?])\s+|[؛;]")
        best_clause = ""
        best_hits = -1
        for clause in clauses
            c = strip(String(clause))
            isempty(c) && continue
            keys = reduce(union, (_generation_keys(w) for w in split(c)); init=Set{String}())
            hits = count(anchor -> _anchor_matches_keys(anchor, keys), anchors)
            if hits > best_hits
                best_hits = hits
                best_clause = c
            end
        end
        best_hits >= 1 || continue
        best_clause = strip(best_clause)
        isempty(best_clause) && continue
        if length(split(best_clause)) < 4
            best_clause = clean
        end
        endswith(best_clause, ".") || endswith(best_clause, "؟") || (best_clause *= ".")
        push!(pieces, best_clause)
    end
    isempty(pieces) && return strip(String(first) * " " * String(second))
    length(pieces) == 1 && return pieces[1]
    pieces[1] == pieces[2] && return pieces[1]
    return pieces[1] * " " * pieces[2]
end

function _anchor_label_from_text(anchor::Set{String}, texts::Tuple{String,String})
    best = ""
    for text in texts
        for raw in split(text)
            word = strip(String(raw), [' ', '\t', '\n', '\r', '.', ',', '،', '؛', ':', '?', '؟', '!', '"', '\''])
            isempty(word) && continue
            if !isempty(intersect(_generation_keys(word), anchor))
                clean = replace(word, r"^(?:و|ف)+" => "")
                clean = replace(clean, r"^(?:بال|فال|وال|كال|لل)" => "")
                return clean
            end
        end
    end
    for key in anchor
        length(key) >= 3 && return key
    end
    return best
end

function _best_relation_clause(text::AbstractString, anchors::Vector{Set{String}})
    relation_keys = Set(["يحتاج", "تحتاج", "يحفظ", "تحفظ", "يمنع", "تمنع",
                         "يكشف", "تكشف", "يرتبط", "ترتبط", "يؤدي", "تؤدي",
                         "يزيل", "تزيل", "يفتح", "تفتح", "ينفع", "تنفع",
                         "يهذب", "تهذب", "يوجه", "توجه", "يساعد", "تساعد",
                         "يقلل", "تقلل", "يرفع", "ترفع", "يقود", "تقود"])
    best = ""
    best_score = 0.0
    for raw_clause in split(String(text), r"(?<=[.؟?])\s+|[؛;]")
        clause = strip(String(raw_clause))
        isempty(clause) && continue
        words = String[strip(w) for w in split(clause) if !isempty(strip(w))]
        isempty(words) && continue
        keys = reduce(union, (_generation_keys(w) for w in words); init=Set{String}())
        hits = count(anchor -> _anchor_matches_keys(anchor, keys), anchors)
        rel = !isempty(intersect(keys, relation_keys)) ||
              occursin("لأن", clause) || occursin("لانه", clause)
        hits == 0 && !rel && continue
        brevity = 1.0 / (1.0 + max(0, length(words) - 14) / 8)
        score = hits + (rel ? 0.65 : 0.0) + 0.18 * brevity
        if score > best_score
            best_score = score
            best = clause
        end
    end
    return strip(best)
end

function _summarize_composed_evidence(first::AbstractString,
                                      second::AbstractString,
                                      anchors::Vector{Set{String}})
    return ""
end

function _anchor_matches_keys(anchor::Set{String}, keys::Set{String})
    !isempty(intersect(anchor, keys)) && return true

    learning_keys = Set(["تعلم", "التعلم", "متعلم", "المتعلم", "يتعلم", "التعليم"])
    science_keys = Set(["علم", "العلم"])
    if !isempty(intersect(anchor, learning_keys)) && !isempty(intersect(keys, science_keys))
        return true
    end
    if !isempty(intersect(anchor, science_keys)) && !isempty(intersect(keys, learning_keys))
        return true
    end

    useful_keys = Set(["نفع", "نافع", "ينفع", "النافع"])
    if !isempty(intersect(anchor, useful_keys)) && !isempty(intersect(keys, useful_keys))
        return true
    end
    return false
end

function _anchor_contrast_matches_keys(anchor::Set{String}, keys::Set{String})
    science_keys = Set(["علم", "العلم"])
    ignorance_keys = Set(["جهل", "الجهل", "جاهل", "الجاهل"])
    light_keys = Set(["نور", "النور", "ضوء", "الضوء"])
    darkness_keys = Set(["ظلام", "الظلام", "ظلمه", "الظلمه", "ظلمة", "الظلمة"])
    justice_keys = Set(["عدل", "العدل"])
    oppression_keys = Set(["ظلم", "الظلم", "ظالم", "الظالم"])
    mercy_keys = Set(["رحمه", "الرحمه", "رحمة", "الرحمة"])
    cruelty_keys = Set(["قسوه", "القسوه", "قسوة", "القسوة"])
    knowledge_keys = Set(["معرفه", "المعرفه", "معرفة", "المعرفة", "خبره", "الخبره", "خبرة", "الخبرة"])
    force_keys = Set(["قوه", "القوه", "قوة", "القوة"])
    society_keys = Set(["مجتمع", "المجتمع", "مدينه", "المدينه", "مدينة", "المدينة"])
    reason_keys = Set(["عقل", "العقل"])
    work_keys = Set(["عمل", "العمل"])

    if !isempty(intersect(anchor, science_keys)) && !isempty(intersect(keys, ignorance_keys))
        return true
    end
    if !isempty(intersect(anchor, ignorance_keys)) && !isempty(intersect(keys, science_keys))
        return true
    end
    if !isempty(intersect(anchor, light_keys)) && !isempty(intersect(keys, darkness_keys))
        return true
    end
    if !isempty(intersect(anchor, darkness_keys)) && !isempty(intersect(keys, light_keys))
        return true
    end
    if !isempty(intersect(anchor, justice_keys)) && !isempty(intersect(keys, oppression_keys))
        return true
    end
    if !isempty(intersect(anchor, oppression_keys)) && !isempty(intersect(keys, justice_keys))
        return true
    end
    if !isempty(intersect(anchor, mercy_keys)) && !isempty(intersect(keys, cruelty_keys))
        return true
    end
    if !isempty(intersect(anchor, cruelty_keys)) && !isempty(intersect(keys, mercy_keys))
        return true
    end
    if !isempty(intersect(anchor, knowledge_keys)) && !isempty(intersect(keys, force_keys))
        return true
    end
    if !isempty(intersect(anchor, force_keys)) && !isempty(intersect(keys, knowledge_keys))
        return true
    end
    if !isempty(intersect(anchor, society_keys)) && !isempty(intersect(keys, justice_keys))
        return true
    end
    if !isempty(intersect(anchor, justice_keys)) && !isempty(intersect(keys, society_keys))
        return true
    end
    if !isempty(intersect(anchor, force_keys)) && !isempty(intersect(keys, reason_keys))
        return true
    end
    if !isempty(intersect(anchor, reason_keys)) && !isempty(intersect(keys, force_keys))
        return true
    end
    if !isempty(intersect(anchor, work_keys)) && !isempty(intersect(keys, science_keys))
        return true
    end
    if !isempty(intersect(anchor, science_keys)) && !isempty(intersect(keys, work_keys))
        return true
    end
    return false
end

function _anchor_hit_indices_in_keys(anchors::Vector{Set{String}}, keys::Set{String})
    hit_indices = Set{Int}()
    for (i, anchor) in enumerate(anchors)
        _anchor_matches_keys(anchor, keys) && push!(hit_indices, i)
    end
    return hit_indices
end

function _anchor_strict_hit_indices_in_keys(anchors::Vector{Set{String}}, keys::Set{String})
    hit_indices = Set{Int}()
    for (i, anchor) in enumerate(anchors)
        !isempty(intersect(anchor, keys)) && push!(hit_indices, i)
    end
    return hit_indices
end

function _anchor_support_indices_in_keys(anchors::Vector{Set{String}}, keys::Set{String}; allow_contrast::Bool=false)
    hit_indices = _anchor_hit_indices_in_keys(anchors, keys)
    if allow_contrast
        for (i, anchor) in enumerate(anchors)
            i in hit_indices && continue
            _anchor_contrast_matches_keys(anchor, keys) && push!(hit_indices, i)
        end
    end
    return hit_indices
end

function _expanded_anchor_keys_for_penalty(anchors::Vector{Set{String}})
    keys = reduce(union, anchors; init=Set{String}())
    learning_keys = Set(["تعلم", "التعلم", "متعلم", "المتعلم", "يتعلم", "التعليم"])
    science_keys = Set(["علم", "العلم"])
    useful_keys = Set(["نفع", "نافع", "ينفع", "النافع"])
    if !isempty(intersect(keys, learning_keys))
        union!(keys, science_keys)
    end
    if !isempty(intersect(keys, science_keys))
        union!(keys, learning_keys)
    end
    if !isempty(intersect(keys, useful_keys))
        union!(keys, useful_keys)
    end
    return keys
end

function _composition_marker_score(sentence::AbstractString)
    s = String(sentence)
    score = 0.0
    occursin("، وإلى", s) && (score += 0.55)
    occursin("؛", s) && (score += 0.35)
    (occursin("لأن", s) || occursin("لانه", s)) && (score += 0.3)
    (occursin("حين", s) || occursin("إذا", s)) && (score += 0.22)
    (occursin("ينتج عن", s) || occursin("يرتبط", s) || occursin("يحتاج", s)) && (score += 0.28)
    (occursin("يتحول", s) || occursin("تتحول", s)) && (score += 0.5)
    count(ch -> ch == '،', s) >= 2 && (score += 0.18)
    return score
end

function _learned_composition_sentence(raw_sentences::Vector{String},
                                       anchors::Vector{Set{String}},
                                       prompt_anchor_keys::Set{String})
    length(anchors) >= 2 || return ""
    required_hits = min(length(anchors), 3)
    core_keys = Set(["علم", "العلم", "عدل", "العدل", "رحمه", "الرحمه", "سلام", "السلام",
                     "جهل", "الجهل", "ظلم", "الظلم", "نزاع", "النزاع", "تعلم", "التعلم",
                     "تعليم", "التعليم", "عمل", "العمل", "فهم", "الفهم", "حفظ", "الحفظ",
                     "قوه", "القوه", "خير", "الخير"])
    dialogue_leak_keys = Set(["اقضي", "أقضي", "يومي", "اجمل", "أجمل", "شيء", "نحميه"])
    best = ""
    best_score = 0.0
    for sentence in raw_sentences
        clean = strip(sentence)
        isempty(clean) && continue
        (occursin("؟", clean) || occursin("?", clean)) && continue
        _dialogue_style_leak(clean) && continue
        length(anchors) <= 2 && count(ch -> ch == '؛', clean) > 1 && continue
        startswith(clean, "س:") && continue
        startswith(clean, "سؤال:") && continue
        words = String[strip(w) for w in split(clean) if !isempty(strip(w))]
        6 <= length(words) <= 34 || continue

        keys = reduce(union, (_generation_keys(w) for w in words); init=Set{String}())
        !isempty(intersect(keys, dialogue_leak_keys)) && continue
        hit_indices = _anchor_hit_indices_in_keys(anchors, keys)
        length(hit_indices) >= required_hits || continue

        marker_score = _composition_marker_score(clean)
        marker_score >= 0.25 || continue

        early_anchor_bonus = 0.0
        for w in words[1:min(5, length(words))]
            if !isempty(intersect(_generation_keys(w), prompt_anchor_keys))
                early_anchor_bonus = 0.18
                break
            end
        end
        extra_core = setdiff(intersect(keys, core_keys), prompt_anchor_keys)
        extra_limit = (length(hit_indices) >= required_hits && marker_score >= 0.5) ? 3 : 2
        length(anchors) <= 2 && length(extra_core) > 1 && marker_score < 0.7 && continue
        length(extra_core) > extra_limit && continue
        extra_penalty = 0.16 * min(length(extra_core), 2)
        brevity = 1.0 / (1.0 + max(0, length(words) - 22) / 10)
        score = 1.25 * length(hit_indices) + marker_score + early_anchor_bonus +
                0.14 * brevity - extra_penalty
        score > best_score || continue
        best_score = score
        best = clean
    end
    best_score >= 1.25 * required_hits + 0.25 ? best : ""
end

function _raw_composed_evidence_answer(gen::MirnanGenerator,
                                       prompt::AbstractString,
                                       prompt_tokens::Vector{String})
    prompt_text = lowercase(String(prompt))
    composition_prompt = _relationship_prompt(prompt) ||
                         occursin("كيف", prompt_text) ||
                         occursin("لماذا", prompt_text) ||
                         occursin("why", prompt_text) ||
                         occursin("how", prompt_text)
    composition_prompt || return ""
    anchors = _relationship_anchor_keys(prompt_tokens)
    length(anchors) >= 2 || return ""
    prompt_anchor_keys = _expanded_anchor_keys_for_penalty(anchors)
    raw_sentences = _load_raw_training_sentences_for_generator(gen)
    isempty(raw_sentences) && return ""

    learned_composition = _learned_composition_sentence(raw_sentences, anchors, prompt_anchor_keys)
    !isempty(strip(learned_composition)) && return learned_composition

    cue_keys = Set(["لان", "lانه", "لأن", "لأنه", "بسبب", "سبب", "لذلك",
                    "يفتح", "يحفظ", "يمنع", "يكشف", "يحجب", "يؤدي", "ينتج",
                    "يثمر", "يزيد", "يساعد", "يقود", "تهذب", "يوازن", "يصلح",
                    "يحتاج", "تحتاج", "يعطى", "تعطى", "يتعلم", "تعلم", "مراجعه"])
    dialogue_leak_keys = Set(["اقضي", "أقضي", "يومي", "اجمل", "أجمل", "شيء", "نحميه"])
    candidates = Tuple{Float64,Set{Int},String}[]
    for sentence in raw_sentences
        (occursin("؟", sentence) || occursin("?", sentence)) && continue
        _dialogue_style_leak(sentence) && continue
        length(anchors) <= 2 && count(ch -> ch == '؛', sentence) > 1 && continue
        startswith(strip(sentence), "س:") && continue
        startswith(strip(sentence), "سؤال:") && continue
        sent_tokens = String[strip(w) for w in split(sentence) if !isempty(strip(w))]
        4 <= length(sent_tokens) <= 34 || continue
        sent_keys = reduce(union, (_generation_keys(w) for w in sent_tokens); init=Set{String}())
        cue_hit = !isempty(intersect(cue_keys, sent_keys)) ||
                  occursin("لأن", sentence) || occursin("لانه", sentence) ||
                  occursin("؛", sentence)
        hit_indices = _anchor_support_indices_in_keys(anchors, sent_keys; allow_contrast=cue_hit)
        direct_hit_indices = _anchor_hit_indices_in_keys(anchors, sent_keys)
        contrast_count = length(setdiff(hit_indices, direct_hit_indices))
        isempty(hit_indices) && continue
        length(hit_indices) >= 2 || cue_hit || continue
        first_bonus = 1 in hit_indices ? 0.35 : 0.0
        early_anchor_bonus = 0.0
        for w in sent_tokens[1:min(4, length(sent_tokens))]
            if !isempty(intersect(_generation_keys(w), prompt_anchor_keys))
                early_anchor_bonus = 0.22
                break
            end
        end
        cue_bonus = cue_hit ? 0.25 : 0.0
        core_keys = Set(["علم", "العلم", "عدل", "العدل", "رحمه", "الرحمه", "سلام", "السلام",
                         "جهل", "الجهل", "ظلم", "الظلم", "نزاع", "النزاع", "تعلم", "التعلم",
                         "عمل", "العمل", "فهم", "الفهم", "حفظ", "الحفظ"])
        extra_core = setdiff(intersect(sent_keys, core_keys), prompt_anchor_keys)
        length(anchors) <= 2 && length(extra_core) > 1 && continue
        length(extra_core) > 2 && continue
        extra_penalty = 0.18 * min(length(extra_core), 2)
        dialogue_penalty = !isempty(intersect(sent_keys, dialogue_leak_keys)) ? 0.65 : 0.0
        brevity = 1.0 / (1.0 + max(0, length(sent_tokens) - 22) / 10)
        score = length(hit_indices) + first_bonus + early_anchor_bonus + cue_bonus +
                0.12 * brevity - extra_penalty - dialogue_penalty - 0.18 * contrast_count
        push!(candidates, (score, hit_indices, sentence))
    end
    isempty(candidates) && return ""
    sort!(candidates; by=x -> x[1], rev=true)

    selected = Tuple{Float64,Set{Int},String}[]
    covered = Set{Int}()
    for cand in candidates
        gain = setdiff(cand[2], covered)
        if isempty(selected) || !isempty(gain)
            push!(selected, cand)
            union!(covered, cand[2])
        end
        length(selected) >= 2 && break
        length(covered) >= min(length(anchors), 3) && length(selected) >= 1 && break
    end

    length(covered) >= min(length(anchors), 2) || return ""
    if length(selected) == 1
        length(covered) >= min(length(anchors), 3) || return ""
        return selected[1][3]
    end

    first_part = strip(selected[1][3])
    second_part = strip(selected[2][3])
    (isempty(first_part) || isempty(second_part)) && return ""
    first_part == second_part && return first_part
    return _compact_composed_evidence(first_part, second_part, anchors)
end

function _direct_relation_evidence_answer(gen::MirnanGenerator,
                                          prompt::AbstractString,
                                          prompt_tokens::Vector{String})
    prompt_text = lowercase(String(prompt))
    is_yesno = !isempty(prompt_tokens) && _is_question_token(first(prompt_tokens))
    if !is_yesno
        probe_tokens = _explanatory_as_yesno_tokens(prompt_tokens)
        if !isempty(probe_tokens)
            evidence = _nisba_explanatory_evidence(gen, probe_tokens, nothing)
            !isempty(strip(evidence)) && return evidence
        end
    end
    (occursin("كيف", prompt_text) || occursin("لماذا", prompt_text) ||
     occursin("متى", prompt_text) || _relationship_prompt(prompt) || is_yesno) || return ""
    anchors = _relationship_anchor_keys(prompt_tokens)
    length(anchors) >= 2 || return ""
    raw_sentences = _load_raw_training_sentences_for_generator(gen)
    isempty(raw_sentences) && return ""
    relation_keys = Set(["لان", "لانه", "لأن", "لأنه", "بسبب", "سبب",
                         "yفتح", "يكشف", "يحجب", "يطرد", "يبدد", "يمنع",
                         "يحفظ", "يحتاج", "تحتاج", "يقود", "تلد", "يلد",
                         "ينتج", "يثمر", "yتحول", "تتحول", "yصير", "تصير",
                         "يساعد", "تهذب", "يرشد", "يميز", "يمكّن", "يمكن"])
    best = ""
    best_score = 0.0
    required_hits = min(length(anchors), 2)
    for sentence in raw_sentences
        clean = strip(sentence)
        isempty(clean) && continue
        (occursin("؟", clean) || occursin("?", clean)) && continue
        _dialogue_style_leak(clean) && continue
        startswith(clean, "س:") && continue
        startswith(clean, "سؤال:") && continue
        words = String[strip(w) for w in split(clean) if !isempty(strip(w))]
        4 <= length(words) <= 34 || continue
        keys = reduce(union, (_generation_keys(w) for w in words); init=Set{String}())
        direct_hits = is_yesno ? _anchor_strict_hit_indices_in_keys(anchors, keys) :
                      _anchor_hit_indices_in_keys(anchors, keys)
        length(direct_hits) >= required_hits || continue
        cue_hit = !isempty(intersect(keys, relation_keys)) ||
                  occursin("؛", clean) || occursin("لأن", clean) || occursin("لانه", clean)
        cue_hit || is_yesno || continue
        early_anchor_bonus = 0.0
        for w in words[1:min(5, length(words))]
            if any(anchor -> !isempty(intersect(anchor, _generation_keys(w))), anchors)
                early_anchor_bonus = 0.18
                break
            end
        end
        brevity = 1.0 / (1.0 + max(0, length(words) - 18) / 10)
        score = 1.0 * length(direct_hits) + 0.35 + early_anchor_bonus + 0.12 * brevity
        score > best_score || continue
        best_score = score
        best = clean
    end
    best_score >= required_hits + 0.35 || return ""
    if is_yesno
        words = _yesno_content_words(prompt_tokens)
        has_negation = any(_is_negation_word, words)
        filter!(w -> !_is_negation_word(w), words)
        2 <= length(words) <= 6 || return best
        has_negation && return "\u0644\u0627\u060c " * join(words, " ") * "."
        return "نعم، " * join(words, " ") * "."
    end
    return best
end

function _raw_explanatory_answer(gen::MirnanGenerator,
                                 prompt::AbstractString,
                                 prompt_tokens::Vector{String})
    plan = detect_response_intent(prompt)
    prompt_text = lowercase(String(prompt))
    explanatory_prompt = plan.intent in ("causal", "mechanism") ||
                         _relationship_prompt(prompt) ||
                         occursin("لماذا", prompt_text) ||
                         occursin("كيف", prompt_text) ||
                         occursin("why", prompt_text) ||
                         occursin("how", prompt_text)
    explanatory_prompt || return ""
    anchors = _relationship_anchor_keys(prompt_tokens)
    length(anchors) >= 2 || return ""
    first_anchor = anchors[1]
    required_hits = min(length(anchors), 3)
    prompt_negative_keys = reduce(union, (_generation_keys(t) for t in prompt_tokens); init=Set{String}())
    negative_prompt = any(k -> k in prompt_negative_keys, ["لا", "ليس", "بدون", "هل", "can"])
    mem = _LEARNED_ISTINBAT_MEMORY[]
    if mem !== nothing
        for t in prompt_tokens
            haskey(mem.discovered_markers, t) && (mem.discovered_markers[t] in ("negation", "question")) && (negative_prompt = true; break)
        end
    end

    pairs = _load_raw_dialogue_pairs_for_generator(gen)
    if !isempty(pairs)
        prompt_terms = _clean_prompt_terms(_clean_aql_text(prompt))
        prompt_keys = _aql_speech_content_keys(prompt_terms)
        prompt_norm = lowercase(join(prompt_terms, " "))
        best_pair = ""
        best_pair_score = 0.0
        for (q, a) in pairs
            _dialogue_style_leak(a) && continue
            answer_head = replace(strip(a), r"^(?:ج:|جواب:)\s*" => "")
            negative_answer = startswith(answer_head, "لا ") || startswith(answer_head, "لا يمكن")
            negative_answer && !negative_prompt && continue
            q_terms = _clean_prompt_terms(_clean_aql_text(q))
            q_keys = _aql_speech_content_keys(q_terms)
            q_anchor_hits = count(anchor -> !isempty(intersect(anchor, q_keys)), anchors)
            q_anchor_hits >= required_hits || continue
            !isempty(intersect(first_anchor, q_keys)) || continue
            score = lowercase(join(q_terms, " ")) == prompt_norm ? 1.0 :
                    (!isempty(prompt_keys) ? length(intersect(prompt_keys, q_keys)) / max(length(prompt_keys), 1) : 0.0)
            score > best_pair_score || continue
            best_pair_score = score
            best_pair = a
        end
        if best_pair_score >= 0.72 && !isempty(strip(best_pair))
            answer = strip(best_pair)
            return endswith(answer, ".") || endswith(answer, "؟") || endswith(answer, "?") ? answer : answer * "."
        end
    end

    raw_sentences = _load_raw_training_sentences_for_generator(gen)
    isempty(raw_sentences) && return ""

    best = ""
    best_score = 0.0
    causal_keys = Set(["لان", "lانه", "لأن", "لأنه", "بسبب", "سبب", "لذلك",
                       "yفتح", "يحفظ", "يمنع", "يكشف", "يحجب", "يؤدي",
                       "ينتج", "يثمر", "يزيد", "يساعد", "يقود", "تهذب"])
    core_keys = Set(["علم", "العلم", "عدل", "العدل", "رحمه", "الرحمه", "سلام", "السلام",
                     "جهل", "الجهل", "ظلم", "الظلم", "نزاع", "النزاع", "تعلم", "التعلم",
                     "تعليم", "التعليم", "عمل", "العمل", "فهم", "الفهم", "حفظ", "الحفظ",
                     "قوه", "القوه", "خير", "الخير"])
    dialogue_leak_keys = Set(["اقضي", "أقضي", "يومي", "اجمل", "أجمل", "شيء", "نحميه"])
    prompt_anchor_keys = _expanded_anchor_keys_for_penalty(anchors)
    prompt_phrase_terms = String[]
    for tok in prompt_tokens
        ks = _generation_keys(tok)
        _any_key_in(ks, GENERATION_STOPWORDS) && continue
        _any_key_in(ks, _EXTRA_GENERATION_STOPWORDS) && continue
        _any_key_in(ks, _QUESTION_TOOL_KEYS) && continue
        term = _generation_key(strip(replace(tok, r"[[:punct:]\u061F\u060C\u061B]" => "")))
        if length(term) > 3 && endswith(term, "ا")
            term = term[begin:prevind(term, lastindex(term))]
        end
        isempty(term) || push!(prompt_phrase_terms, term)
    end
    prompt_phrase = join(prompt_phrase_terms, " ")
    strong_candidate_exists = false
    for sentence in raw_sentences
        length(anchors) <= 2 && count(ch -> ch == '؛', sentence) > 1 && continue
        _dialogue_style_leak(sentence) && continue
        answer_head = replace(strip(sentence), r"^(?:ج:|جواب:)\s*" => "")
        negative_answer = startswith(answer_head, "لا ") || startswith(answer_head, "لا يمكن")
        negative_answer && !negative_prompt && continue
        sent_tokens = String[strip(w) for w in split(sentence) if !isempty(strip(w))]
        3 <= length(sent_tokens) <= 32 || continue
        sent_keys = reduce(union, (_generation_keys(w) for w in sent_tokens); init=Set{String}())
        !isempty(intersect(sent_keys, dialogue_leak_keys)) && continue
        hits = count(a -> _anchor_matches_keys(a, sent_keys), anchors)
        hits >= required_hits || continue
        extra_core = setdiff(intersect(sent_keys, core_keys), prompt_anchor_keys)
        length(anchors) <= 2 && length(extra_core) > 1 && continue
        length(extra_core) > 2 && continue
        causal_hit = !isempty(intersect(causal_keys, sent_keys)) ||
                     occursin("لأن", sentence) || occursin("لانه", sentence) ||
                     occursin("؛", sentence)
        if causal_hit
            strong_candidate_exists = true
            break
        end
    end
    for sentence in raw_sentences
        length(anchors) <= 2 && count(ch -> ch == '؛', sentence) > 1 && continue
        _dialogue_style_leak(sentence) && continue
        answer_head = replace(strip(sentence), r"^(?:ج:|جواب:)\s*" => "")
        negative_answer = startswith(answer_head, "لا ") || startswith(answer_head, "لا يمكن")
        negative_answer && !negative_prompt && continue
        sent_tokens = String[strip(w) for w in split(sentence) if !isempty(strip(w))]
        3 <= length(sent_tokens) <= 32 || continue
        sent_keys = reduce(union, (_generation_keys(w) for w in sent_tokens); init=Set{String}())
        !isempty(intersect(sent_keys, dialogue_leak_keys)) && continue
        hits = count(a -> _anchor_matches_keys(a, sent_keys), anchors)
        has_first_anchor = _anchor_matches_keys(first_anchor, sent_keys)
        if strong_candidate_exists
            hits >= required_hits || continue
        else
            (has_first_anchor && hits >= min(required_hits, 2)) || continue
        end
        causal_hit = !isempty(intersect(causal_keys, sent_keys)) ||
                     occursin("لأن", sentence) || occursin("لانه", sentence) ||
                     occursin("؛", sentence)
        causal_hit || continue
        extra_core = setdiff(intersect(sent_keys, core_keys), prompt_anchor_keys)
        length(anchors) <= 2 && length(extra_core) > 1 && continue
        length(extra_core) > 2 && continue
        # Direction reversal penalty: if question asks about A→B but answer talks about B→A
        direction_penalty = 0.0
        if length(anchors) >= 2
            c1_keys = first(anchors)
            c2_keys = last(anchors)
            c1_word = !isempty(c1_keys) ? first(c1_keys) : ""
            c2_word = !isempty(c2_keys) ? first(c2_keys) : ""
            if !isempty(c1_word) && !isempty(c2_word) && c1_word != c2_word
                c1_pos = findfirst(w -> _anchor_matches_keys(Set([c1_word]), _generation_keys(w)), sent_tokens)
                c2_pos = findfirst(w -> _anchor_matches_keys(Set([c2_word]), _generation_keys(w)), sent_tokens)
                if c1_pos !== nothing && c2_pos !== nothing && c2_pos < c1_pos
                    direction_penalty = 0.30
                end
            end
        end
        all_anchor_bonus = hits >= length(anchors) ? 0.10 : 0.0
        first_anchor_bonus = has_first_anchor ? 0.24 : -0.28
        because_bonus = (occursin("لأن", sentence) || occursin("لانه", sentence) ||
                         occursin("لان", sentence)) ? 0.22 : 0.0
        relation_form_bonus = occursin("؛", sentence) ? 0.08 : 0.0
        sent_phrase = join([begin
                                k = _generation_key(strip(replace(w, r"[[:punct:]\u061F\u060C\u061B]" => "")))
                                if length(k) > 3 && endswith(k, "ا")
                                    k = k[begin:prevind(k, lastindex(k))]
                                end
                                k
                            end for w in sent_tokens], " ")
        exact_phrase_bonus = (!isempty(prompt_phrase) && occursin(prompt_phrase, sent_phrase)) ? 0.26 : 0.0
        brevity = 1.0 / (1.0 + max(0, length(words) - 18) / 10)
        score = 0.60 * min(hits, 3) / min(length(anchors), 3) + 0.12 * brevity +
                all_anchor_bonus + first_anchor_bonus + because_bonus + relation_form_bonus + exact_phrase_bonus -
                0.14 * min(length(extra_core), 2) -
                direction_penalty
        score > best_score || continue
        best_score = score
        best = sentence
    end
    best_score < 0.65 && return ""
    return best
end

function _trim_reported_speech_prefix(words::Vector{String}, prompt_tokens::Vector{String})
    isempty(words) && return words
    first_keys = Set{String}()
    for tok in prompt_tokens
        ks = _generation_keys(tok)
        _any_key_in(ks, GENERATION_STOPWORDS) && continue
        _any_key_in(ks, _EXTRA_GENERATION_STOPWORDS) && continue
        _any_key_in(ks, _QUESTION_TOOL_KEYS) && continue
        first_keys = ks
        break
    end
    isempty(first_keys) && return words
    speech_keys = Set(["قال", "قالت", "قلت", "يقول", "تقول", "سأل", "سألت", "اجاب", "أجاب", "اجابت", "أجابت"])
    idx = findfirst(w -> !isempty(intersect(_generation_keys(w), first_keys)), words)
    idx === nothing && return words
    idx <= 1 && return words
    prefix = words[1:idx-1]
    any(w -> _generation_family_key(w) in speech_keys, prefix) || return words
    return words[idx:end]
end

function _relationship_answer(gen::MirnanGenerator,
                              prompt::AbstractString,
                              prompt_tokens::Vector{String})
    _relationship_prompt(prompt) || return ""
    anchors = _relationship_anchor_keys(prompt_tokens)
    if length(anchors) < 2
        endpoint_terms = _relationship_endpoint_terms(prompt_tokens)
        if length(endpoint_terms) >= 2
            evidence = _learned_pair_evidence_sentence(gen, first(endpoint_terms), last(endpoint_terms))
            !isempty(strip(evidence)) && return evidence
        end
        return ""
    end
    first_anchor = anchors[1]
    content_words = String[]
    for tok in prompt_tokens
        ks = _generation_keys(tok)
        _any_key_in(ks, GENERATION_STOPWORDS) && continue
        _any_key_in(ks, _EXTRA_GENERATION_STOPWORDS) && continue
        _any_key_in(ks, _QUESTION_TOOL_KEYS) && continue
        _relationship_connector_token(tok) && continue
        clean = strip(replace(tok, r"[[:punct:]\u061F\u060C\u061B]" => ""))
        isempty(clean) && continue
        !isempty(intersect(ks, Set(["علاقة", "العلاقة", "بين"]))) && continue
        push!(content_words, clean)
    end
    if length(content_words) >= 2
        rec = _select_endpoint_nisba_relation(gen.nisba, join(content_words, " "), content_words;
                                             min_score=0.30,
                                             ordered_endpoints=false)
        if rec !== nothing && !isempty(rec.evidences)
            answer = _clean_nisba_evidence(rec.evidences[end])
            if length(split(answer)) >= 3 && any(p -> occursin(p, answer), [".", "،", "؛", ":"])
                return answer
            end
        end
    end
    raw_sentences = _load_raw_training_sentences_for_generator(gen)
    best_raw = ""
    best_raw_score = 0.0
    for sentence in raw_sentences
        clean = strip(sentence)
        isempty(clean) && continue
        _dialogue_style_leak(clean) && continue
        startswith(clean, "نعم") && continue
        startswith(clean, "لا ") && continue
        sent_tokens = String[strip(w) for w in split(sentence) if !isempty(strip(w))]
        sent_keys = reduce(union, (_generation_keys(w) for w in sent_tokens); init=Set{String}())
        hits = count(a -> !isempty(intersect(a, sent_keys)), anchors)
        hits >= 2 || continue
        relation_bonus = any(k -> k in sent_keys,
                             ["يرتبط", "ترتبط", "يقود", "يساعد", "يحفظ", "يزيد",
                              "ينتج", "يثمر", "يعزز", "yفتح", "يهيئ", "تهذب",
                              "يكشف", "يحجب", "يمنع", "يبدد"]) ? 0.25 : 0.0
        brevity = 1.0 / (1.0 + max(0, length(sent_tokens) - 18) / 10)
        score = 0.64 * min(hits, 3) / min(length(anchors), 3) + 0.11 * brevity + relation_bonus
        score > best_raw_score || continue
        best_raw_score = score
        best_raw = sentence
    end
    best_raw_score >= 0.58 && return best_raw

    sentences = _load_corpus_sentences_for_generator(gen)
    isempty(sentences) && return ""

    best_words = String[]
    best_score = 0.0
    for ids in sentences
        words = _sentence_words(gen, ids)
        3 <= length(words) <= 26 || continue
        text_probe = join(words, " ")
        !any(p -> occursin(p, text_probe), [".", "،", "؛", ":"]) && length(words) >= 5 && continue
        startswith(strip(text_probe), "نعم") && continue
        startswith(strip(text_probe), "لا ") && continue
        sent_keys = reduce(union, (_generation_keys(w) for w in words); init=Set{String}())
        hits = count(a -> !isempty(intersect(a, sent_keys)), anchors)
        hits >= 2 || continue
        relation_bonus = any(k -> k in sent_keys,
                             ["يرتبط", "ترتبط", "يقود", "يساعد", "يحفظ", "يزيد",
                              "ينتج", "يثمر", "يعزز", "yفتح", "يهيئ",
                              "يكشف", "يحجب", "يمنع", "يبدد"]) ? 0.22 : 0.0
        brevity = 1.0 / (1.0 + max(0, length(words) - 16) / 10)
        score = 0.62 * min(hits, 3) / min(length(anchors), 3) + 0.16 * brevity + relation_bonus
        score > best_score || continue
        best_score = score
        best_words = words
    end
    if best_score < 0.58
        if length(content_words) >= 2
            evidence = _learned_pair_evidence_sentence(gen, first(content_words), last(content_words))
            !isempty(strip(evidence)) && return evidence
        end
        return ""
    end
    text = strip(join(best_words, " "))
    isempty(text) && return ""
    return endswith(text, ".") || endswith(text, "؟") || endswith(text, "?") ? text : text * "."
end
