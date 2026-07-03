const _AQL_SPEECH_GENERIC_KEYS = Set([
    "ما", "ماذا", "من", "اين", "أين",
    "متى", "كيف", "لماذا", "هل",
    "هو", "هي", "هذا", "هذه", "ذلك",
    "في", "ب", "بـ", "عن", "على", "الى", "إلى",
    "رايك", "رأيك", "قولك", "معنى",
    "مفيد", "مفيده", "مفيدة", "نافع", "نافعه", "نافعة",
    "يعمل", "تعمل", "يكون", "تكون",
    "يتعلم", "نتعلم", "ننام", "نحتاج", "نبحث",
    "what", "where", "when", "why", "how", "is", "are", "the", "a", "an", "of", "in", "about",
])

function _aql_speech_match_keys(terms::Vector{String})
    keys = Set{String}()
    for term in terms
        for key in _generation_keys(term)
            isempty(key) || push!(keys, key)
        end
    end
    return keys
end

function _aql_speech_content_keys(terms::Vector{String})
    keys = _aql_speech_match_keys(terms)
    filter!(k -> !(k in _AQL_SPEECH_GENERIC_KEYS) && length(k) >= 2, keys)
    return keys
end

function _aql_speech_act_score(prompt_terms::Vector{String}, prompt_norm::String,
                               prompt_keys::Set{String}, fact)
    content = strip(String(fact.content))
    response = strip(String(fact.response_content))
    (isempty(content) || isempty(response)) && return 0.0
    length(response) > 500 && return 0.0
    content_norm = lowercase(join(_clean_prompt_terms(content), " "))
    isempty(content_norm) && return 0.0
    prompt_norm == content_norm && return 1.0
    occursin(prompt_norm, content_norm) && return 0.92
    occursin(content_norm, prompt_norm) && return 0.88
    content_terms_vec = _clean_prompt_terms(content)
    content_terms = Set(content_terms_vec)
    content_keys = _aql_speech_match_keys(content_terms_vec)
    prompt_content_keys = _aql_speech_content_keys(prompt_terms)
    fact_content_keys = _aql_speech_content_keys(content_terms_vec)
    topic_overlap = count(k -> k in fact_content_keys, prompt_content_keys)
    isempty(content_terms) && isempty(content_keys) && return 0.0
    overlap = count(t -> t in content_terms, prompt_terms)
    key_overlap = count(k -> k in content_keys, prompt_keys)
    overlap = max(overlap, key_overlap)
    overlap == 0 && return 0.0
    if isempty(prompt_content_keys) || isempty(fact_content_keys)
        prompt_norm == content_norm || return 0.0
    elseif topic_overlap == 0
        return 0.0
    end
    coverage = overlap / max(length(prompt_terms), 1)
    precision = overlap / max(length(content_terms), 1)
    length(prompt_terms) <= 2 && coverage < 1.0 && return 0.0
    score = 0.72 * coverage + 0.28 * precision
    fact.act_type in ("تحية", "سؤال_حال", "سؤال", "question") && (score += 0.05)
    fact.response_act in ("رد_تحية", "جواب_حال", "جواب", "answer") && (score += 0.04)
    return min(score, 0.99)
end

function _aql_speech_act_answer(gen::MirnanGenerator, prompt::AbstractString; min_score::Float64=0.62)
    text = _clean_aql_text(prompt)
    isempty(text) && return ""
    prompt_terms = _clean_prompt_terms(text)
    isempty(prompt_terms) && return ""
    prompt_norm = lowercase(join(prompt_terms, " "))
    prompt_keys = _aql_speech_match_keys(prompt_terms)
    best_fact = nothing
    best_score = 0.0
    for fact in gen.aql_space.speech_acts
        score = _aql_speech_act_score(prompt_terms, prompt_norm, prompt_keys, fact)
        if score > best_score
            best_score = score
            best_fact = fact
        end
    end
    best_fact === nothing && return ""
    best_score < min_score && return ""
    answer = strip(String(best_fact.response_content))
    isempty(answer) && return ""
    dirty_markers = ("سؤال:", "جواب:", "سوال:", "السؤال:", "الجواب:", " question:", " answer:")
    any(m -> occursin(m, lowercase(answer)), dirty_markers) && return ""
    endswith(answer, ".") || endswith(answer, "؟") || endswith(answer, "?") || (answer *= ".")
    return answer
end

function _dialogue_answer_incompatible(kind::String, answer::AbstractString)
    a = lowercase(String(answer))
    if kind == "greeting"
        lacks_reply = !(occursin("وعليكم", a) || occursin("مرحبا", a) ||
                        occursin("مرحباً", a) || occursin("اهلا", a) ||
                        occursin("أهلا", a))
        definition_like = occursin("السلام هو", a) || occursin("سلام هو", a) ||
                          occursin("شعور بالامان", a) || occursin("شعور بالأمان", a)
        return definition_like || lacks_reply ||
               occursin("الى اللقاء", a) || occursin("إلى اللقاء", a) ||
               occursin("الي اللقاء", a) || occursin("وداع", a) ||
               occursin("استودع", a) || occursin("يوما سعيدا", a) ||
               occursin("يوماً سعيداً", a)
    elseif kind == "farewell"
        return occursin("وعليكم", a) || occursin("مرحبا", a) ||
               occursin("مرحباً", a) || occursin("اهلا", a) ||
               occursin("أهلا", a)
    end
    return false
end

function _dialogue_greeting_memory_answer(gen::MirnanGenerator, prompt::AbstractString)
    plan = detect_response_intent(prompt)
    plan.intent == "dialogue" && plan.subject == "greeting" || return ""
    candidates = String[String(prompt)]
    strip(String(prompt)) == "سلام" && pushfirst!(candidates, "السلام عليكم")
    for candidate in candidates
        answer = _aql_speech_act_answer(gen, candidate)
        isempty(strip(answer)) && continue
        _dialogue_answer_incompatible("greeting", answer) && continue
        return answer
    end
    return ""
end

function _aql_event_answer_from_plan!(gen::MirnanGenerator, plan::AqlGenerationPlan)
    isempty(plan.action) && return ""
    space = gen.aql_space
    source = isempty(plan.source) ? plan.target : plan.source
    target = isempty(plan.target) ? source : plan.target
    (isempty(source) || isempty(target)) && return ""

    haskey(space.entities, source) || AlAql.register_entity!(space, AlAql.Thing(source))
    haskey(space.entities, target) || AlAql.register_entity!(space, AlAql.Thing(target))

    frames = AlAql.CausalFrame[]
    for chain in values(space.event_chains)
        try
            append!(frames, AlAql.run_event_chain!(space, chain, source, target))
        catch e
            @debug "AQL event chain failed: $e"
        end
    end
    try
        append!(frames, AlAql.infer_event!(space, source, plan.action, target))
    catch e
        @debug "AQL template inference failed: $e"
    end

    if !isempty(frames)
        return "في فضاء العقل:\n" * _format_aql_frames(frames)
    end

    before = length(space.log)
    ok = try
        AlAql.evaluate_idea!(space, source, plan.action, target)
    catch e
        @debug "AQL dynamic evaluation failed: $e"
        false
    end
    ok || return ""
    new_log = space.log[(before + 1):end]
    isempty(new_log) && return "في فضاء العقل: وقع الحدث فتغيرت حالة الهدف."
    return "في فضاء العقل: " * join(new_log[1:min(5, end)], "؛ ")
end

function _format_aql_relations(relations)
    isempty(relations) && return ""
    parts = String[]
    for rel in relations[1:min(5, end)]
        push!(parts, "$(rel.source) $(rel.relation) $(rel.target)")
    end
    return join(parts, "؛ ")
end

function _aql_relation_answer(gen::MirnanGenerator, prompt::AbstractString)
    text = _clean_aql_text(prompt)
    m = match(r"^ما\s+علاق[ةه]\s+(.+?)\s+(?:ب|مع)\s*(.+?)$", text)
    m === nothing && return ""
    a = strip(m.captures[1])
    b = strip(m.captures[2])
    (isempty(a) || isempty(b)) && return ""
    rels = AlAql.relations_between(gen.aql_space, a, b)
    formatted = _format_aql_relations(rels)
    isempty(formatted) && return ""
    return "في فضاء العقل: $formatted."
end

function _aql_circumstance_answer(gen::MirnanGenerator, prompt::AbstractString)
    text = _clean_aql_text(prompt)
    m_place = match(r"^(?:أين|اين)\s+(.+?)$", text)
    if m_place !== nothing
        name = strip(m_place.captures[1])
        c = AlAql.get_circumstance(gen.aql_space, name)
        c.location != "غير_محدد" && return "في فضاء العقل: $name في $(c.location)."
    end

    m_time = match(r"^متى\s+(.+?)$", text)
    if m_time !== nothing
        name = strip(m_time.captures[1])
        c = AlAql.get_circumstance(gen.aql_space, name)
        c.time != "غير_محدد" && return "في فضاء العقل: زمن $name هو $(c.time)."
    end
    return ""
end

function _aql_event_answer(gen::MirnanGenerator, prompt::AbstractString)
    plan = _extract_event_plan(prompt)
    plan === nothing && return ""
    return _aql_event_answer_from_plan!(gen, plan)
end

function _find_named_process(space, prompt::AbstractString)
    text = _clean_aql_text(prompt)
    for process in values(space.processes)
        _contains_aql_term(text, process.name) && return process
    end
    m = match(r"^(?:ما\s+معنى|ما\s+هو|اشرح|عرّف|عرف)\s+(.+?)$", text)
    if m !== nothing
        name = strip(m.captures[1])
        return get(space.processes, name, nothing)
    end
    return nothing
end

function _aql_process_answer(gen::MirnanGenerator, prompt::AbstractString)
    process = _find_named_process(gen.aql_space, prompt)
    process === nothing && return ""
    parts = String[]
    push!(parts, "في فضاء العقل: $(_human_aql_text(process.name)) عملية")
    !isempty(process.parent) && push!(parts, "أصلها $(_human_aql_text(process.parent))")
    !isempty(process.domain) && push!(parts, "مجالها $(_human_aql_text(process.domain))")
    push!(parts, "شدتها $(round(process.intensity; digits=2))")
    if !isempty(process.steps)
        step_text = String[]
        for step in process.steps
            result = isempty(step.result) ? "" : " فتكون النتيجة $(_human_aql_text(step.result))"
            push!(step_text, "$(step.actor_role) يقوم بـ$(step.action) على $(step.target_role)$result")
        end
        push!(parts, "خطواتها: " * join(step_text, "؛ "))
    end
    return join(parts, "، ") * "."
end

function _aql_metaphor_answer(gen::MirnanGenerator, prompt::AbstractString)
    text = _clean_aql_text(prompt)
    facts = [fact for fact in gen.aql_space.metaphors
             if _aql_fact_matches_prompt(text, fact.expression, fact.literal_subject,
                                         fact.borrowed_actor, fact.action) ||
                occursin("مجاز", text)]
    isempty(facts) && return ""
    parts = String[]
    for fact in facts[1:min(3, end)]
        prop = isempty(fact.transferred_property) ? "خاصية من المجال المستعار" :
               _human_aql_text(fact.transferred_property)
        push!(parts, "$(fact.expression): تعبير مجازي؛ ليس $(fact.literal_subject) $(fact.borrowed_actor) حرفياً، بل نُقلت إليه خاصية $prop عبر فعل $(fact.action)")
    end
    return "في فضاء العقل: " * join(parts, "؛ ") * "."
end

function _aql_intent_answer(gen::MirnanGenerator, prompt::AbstractString)
    text = _clean_aql_text(prompt)
    facts = [fact for fact in gen.aql_space.intents
             if _aql_fact_matches_prompt(text, fact.actor, fact.action, fact.target,
                                         fact.intent, fact.goal)]
    isempty(facts) && return ""
    parts = String[]
    for fact in facts[1:min(4, end)]
        goal = isempty(fact.goal) ? fact.intent : fact.goal
        actual = isempty(fact.actual_result) ? "غير محددة" : fact.actual_result
        push!(parts, "$(fact.actor) فعل $(fact.action) بنية $(_human_aql_text(fact.intent)) وغاية $(_human_aql_text(goal))، والنتيجة الواقعة $(_human_aql_text(actual))")
    end
    return "في فضاء العقل: " * join(parts, "؛ ") * "."
end

function _aql_exception_answer(gen::MirnanGenerator, prompt::AbstractString)
    text = _clean_aql_text(prompt)
    facts = [rule for rule in gen.aql_space.exception_rules
             if _aql_fact_matches_prompt(text, rule.rule_name, rule.condition, rule.exception) ||
                occursin("استثناء", text)]
    isempty(facts) && return ""
    parts = String[]
    for rule in facts[1:min(4, end)]
        condition = isempty(rule.condition) ? "شرطه غير محدد" : "شرطه $(_human_aql_text(rule.condition))"
        push!(parts, "القاعدة $(_human_aql_text(rule.rule_name)) لها استثناء $(_human_aql_text(rule.exception))؛ $(condition)؛ أولوية $(rule.priority)")
    end
    return "في فضاء العقل: " * join(parts, "؛ ") * "."
end

function _aql_comparison_answer(gen::MirnanGenerator, prompt::AbstractString)
    text = _clean_aql_text(prompt)
    facts = [fact for fact in gen.aql_space.comparisons
             if _aql_fact_matches_prompt(text, fact.left, fact.property,
                                         fact.comparator, fact.right)]
    isempty(facts) && return ""
    parts = ["$(fact.left) $(_human_aql_text(fact.comparator)) من $(fact.right) في $(_human_aql_text(fact.property))" for fact in facts[1:min(4, end)]]
    return "في فضاء العقل: " * join(parts, "؛ ") * "."
end

function _aql_quantifier_answer(gen::MirnanGenerator, prompt::AbstractString)
    text = _clean_aql_text(prompt)
    facts = [fact for fact in gen.aql_space.quantified_facts
             if _aql_fact_matches_prompt(text, fact.quantifier, fact.subject,
                                         fact.predicate, fact.object)]
    isempty(facts) && return ""
    parts = String[]
    for fact in facts[1:min(4, end)]
        neg = fact.polarity < 0 ? "لا " : ""
        obj = isempty(fact.object) ? "" : " $(_human_aql_text(fact.object))"
        push!(parts, "$(fact.quantifier) $(fact.subject) $(neg)$(fact.predicate)$obj")
    end
    return "في فضاء العقل: " * join(parts, "؛ ") * "."
end

function _aql_public_attr_value(value)
    s = strip(_human_aql_text(string(value)))
    isempty(s) && return ""
    s in ("String[]", "Any[]", "[]", "nothing", "missing", "غير محدد", "غير_محدد") && return ""
    return s
end

function _aql_public_entity_facts(space, name::AbstractString, entity)
    facts = String[]
    try
        classes = AlAql.assigned_classes(space, name)
        if !isempty(classes)
            push!(facts, "$name ينتمي إلى " *
                         join(_human_aql_text.(classes[1:min(3, end)]), "، "))
        end
    catch e
        @debug "AQL entity classes failed for '$name': $e"
    end

    for rel in AlAql.relations_from(space, name)[1:min(3, end)]
        push!(facts, "$(_human_aql_text(rel.source)) $(_human_aql_text(rel.relation)) $(_human_aql_text(rel.target))")
    end

    c = AlAql.get_circumstance(space, name)
    c.location != "غير_محدد" && push!(facts, "$name مرتبط بمكان: $(_human_aql_text(c.location))")
    c.time != "غير_محدد" && push!(facts, "$name مرتبط بزمن: $(_human_aql_text(c.time))")

    if entity isa AlAql.Thing
        public_attrs = String[]
        for (k, v) in sort(collect(entity.attributes); by=x -> x[1])
            key = String(k)
            startswith(key, "__") && continue
            key in ("location", "time") && continue
            value = _aql_public_attr_value(v)
            isempty(value) && continue
            push!(public_attrs, "$(_human_aql_text(key)) $value")
        end
        for attr in public_attrs[1:min(3, end)]
            push!(facts, "$name له سمة: $attr")
        end
    end
    return facts
end

function _aql_entity_answer(gen::MirnanGenerator, prompt::AbstractString)
    text = _clean_aql_text(prompt)
    m = match(r"^(?:ما\s+صفات|ما\s+خصائص|ما\s+أحوال|ما\s+احوال|من\s+هو|ما\s+هو)\s+(.+?)$", text)
    m === nothing && return ""
    name = strip(m.captures[1])
    space = gen.aql_space
    haskey(space.entities, name) || return ""
    facts = _aql_public_entity_facts(space, name, space.entities[name])
    isempty(facts) && return ""
    return join(facts, "، ") * "."
end

function _aql_dialogue_leak(answer::AbstractString)
    s = strip(String(answer))
    startswith(s, "نعم، أحب") && return true
    startswith(s, "نعم احب") && return true
    startswith(s, "علامة فهمه") && return true
    startswith(s, "يفرق بينهما") && return true
    startswith(s, "يحتاج الإنسان") && return true
    startswith(s, "يحتاج الانسان") && return true
    return false
end

function _strip_arabic_al(w::AbstractString)
    s = String(w)
    if startswith(s, "ال") && length(s) > 2
        return s[nextind(s, 1, 2):end]
    end
    return s
end

function _aql_causal_graph_search_answer(gen::MirnanGenerator, prompt::AbstractString)
    space = gen.aql_space
    isempty(space.semantic_relations) && return ""
    
    clean_p = _clean_aql_text(prompt)
    prompt_words = Set(_clean_prompt_terms(clean_p))
    
    best_rel = nothing
    best_score = 0.0
    
    ignore_words = Set(["عند", "أي", "اي", "ما", "ماذا", "من", "كيف", "لماذا", "هل", "بين", "مع", "في", "على", "إلى", "الى", "ب", "ل", "عن", "لأن", "لان", "هو", "هي"])
    
    prompt_content = Set{String}()
    for w in prompt_words
        w_clean = _strip_arabic_al(w)
        w_clean in ignore_words || push!(prompt_content, w_clean)
    end
    
    isempty(prompt_content) && return ""
    
    for rel in space.semantic_relations
        rel_terms = vcat(
            _clean_prompt_terms(rel.source),
            _clean_prompt_terms(rel.relation),
            _clean_prompt_terms(rel.target)
        )
        
        rel_content = Set{String}()
        for w in rel_terms
            w_clean = _strip_arabic_al(w)
            w_clean in ignore_words || push!(rel_content, w_clean)
        end
        
        # Ignore giant dictionary/morphology entries
        length(rel_content) > 25 && continue
        
        # Jaccard similarity
        hits = length(intersect(prompt_content, rel_content))
        u_len = length(union(prompt_content, rel_content))
        score = u_len > 0 ? hits / u_len : 0.0
        
        if score > best_score
            best_score = score
            best_rel = rel
        end
    end
    
    if best_rel !== nothing && best_score >= 0.45
        src = _human_aql_text(best_rel.source)
        rel_name = _human_aql_text(best_rel.relation)
        tgt = _human_aql_text(best_rel.target)
        return "في فضاء العقل: $src $rel_name $tgt."
    end
    
    return ""
end

function _aql_answer!(gen::MirnanGenerator, prompt::AbstractString)
    stripped = strip(prompt)
    isempty(stripped) && return ""

    if startswith(stripped, "تعلم:") || startswith(stripped, "تعلم ")
        body = strip(replace(stripped, r"^تعلم\s*:?\s*" => ""))
        learned = _ingest_aql!(gen, body)
        return learned ? "تم إدخال المعرفة في فضاء العقل." : ""
    end

    response_intent = detect_response_intent(stripped)
    if response_intent.intent == "dialogue"
        speech_answer = response_intent.subject == "greeting" ?
                        _dialogue_greeting_memory_answer(gen, stripped) :
                        _aql_speech_act_answer(gen, stripped)
        if !isempty(strip(speech_answer))
            _dialogue_answer_incompatible(response_intent.subject, speech_answer) && return ""
            return speech_answer
        end
    end

    if !_is_question_prompt_safe(stripped)
        _ingest_aql!(gen, stripped)
        return ""
    end

    causal_search_ans = _aql_causal_graph_search_answer(gen, stripped)
    !isempty(causal_search_ans) && return causal_search_ans

    for answer_fn in (_aql_relation_answer, _aql_circumstance_answer,
                      _aql_process_answer, _aql_event_answer,
                      _aql_intent_answer, _aql_metaphor_answer,
                      _aql_exception_answer, _aql_comparison_answer,
                      _aql_quantifier_answer, _aql_entity_answer)
        answer = answer_fn(gen, stripped)
        !isempty(strip(answer)) && return answer
    end

    if response_intent.intent in ("mechanism", "causal", "descriptive", "definition")
        speech_answer = _aql_speech_act_answer(gen, stripped; min_score=0.94)
        if !isempty(strip(speech_answer))
            if response_intent.intent in ("causal", "mechanism") && _aql_dialogue_leak(speech_answer)
                return ""
            end
            return speech_answer
        end
    end
    return ""
end

function _raw_dialogue_answer(gen::MirnanGenerator, prompt::AbstractString)
    plan = detect_response_intent(prompt)
    plan.intent == "dialogue" || return ""
    pairs = _load_raw_dialogue_pairs_for_generator(gen)
    isempty(pairs) && return ""
    prompt_terms = _clean_prompt_terms(_clean_aql_text(prompt))
    prompt_keys = _aql_speech_content_keys(prompt_terms)
    prompt_norm = lowercase(join(prompt_terms, " "))
    best = ""
    best_score = 0.0
    for (q, a) in pairs
        q_terms = _clean_prompt_terms(_clean_aql_text(q))
        q_keys = _aql_speech_content_keys(q_terms)
        q_norm = lowercase(join(q_terms, " "))
        score = q_norm == prompt_norm ? 1.0 :
                (!isempty(prompt_keys) ? length(intersect(prompt_keys, q_keys)) / max(length(prompt_keys), 1) : 0.0)
        plan.subject == "greeting" && occursin("مرحبا", q_norm) && (score += 0.25)
        plan.subject == "love_learning" && occursin("مرحبا", q_norm) && (score += 0.20)
        score > best_score || continue
        best_score = score
        best = a
    end
    best_score < 0.55 && return ""
    answer = strip(best)
    isempty(answer) && return ""
    _dialogue_answer_incompatible(plan.subject, answer) && return ""
    return endswith(answer, ".") || endswith(answer, "؟") || endswith(answer, "?") ? answer : answer * "."
end

function try_generate(::DialogueStrategy, gen::MirnanGenerator, prompt::String, prompt_tokens::Vector{String}, mode::String, cereb_obs, cereb_policy, response_plan, active_paras)
    if response_plan.intent == "dialogue"
        raw_dialogue_answer = _raw_dialogue_answer(gen, prompt)
        if !isempty(strip(raw_dialogue_answer))
            return _finish_generation!(gen, prompt, prompt_tokens, raw_dialogue_answer,
                                       cereb_obs, cereb_policy;
                                       sanitize_output=false,
                                       apply_templates=false)
        end
        aql_dialogue_answer = _aql_answer!(gen, prompt)
        if !isempty(strip(aql_dialogue_answer))
            return _finish_generation!(gen, prompt, prompt_tokens, aql_dialogue_answer,
                                       cereb_obs, cereb_policy;
                                       sanitize_output=false)
        end
    end
    return nothing
end

function try_generate(::AqlStrategy, gen::MirnanGenerator, prompt::String, prompt_tokens::Vector{String}, mode::String, cereb_obs, cereb_policy, response_plan, active_paras)
    aql_answer = _aql_answer!(gen, prompt)
    if !isempty(strip(aql_answer))
        return _finish_generation!(gen, prompt, prompt_tokens, aql_answer,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false)
    end
    return nothing
end

