function _beam_step(gen::MirnanGenerator, beams, used_sets, prompt_pv; step=0, total=1)
    new_beams = Tuple{Float64,Vector{String},Set{String}}[]
    for (beam_idx, words) in enumerate(beams)
        us = used_sets[beam_idx]
        prev = isempty(words) ? nothing : words[end]
        cids = [get(gen.vocab, w, nothing) for w in words]
        pvs = [_pv(gen, w) for w in words]
        cands = _resonance_candidates(gen, cids, us)
        isempty(cands) && continue
        scored = Tuple{Float64,String,WaveContribution}[]
        for w in cands[1:min(40, end)]
            _reject_context_drift(w, words, words) && continue
            prob, wave, _ = _score(gen, w, us, pvs, prompt_pv;
                       prev_word=prev, context_words=words)
            coh = _context_coherence(gen, w, prompt_pv, words)
            anchor = _prompt_anchor_support(gen, w, words)
            drift = _drift_penalty(w, words, words)
            prob *= max(0.0, 0.20 + 0.80 * coh + 0.35 * anchor - 0.55 * drift)
            isfinite(prob) && prob > 0.0 && push!(scored, (prob, w, wave))
        end
        if !isempty(scored)
            total_prob = sum(s[1] for s in scored)
            if total_prob > 1e-10
                scored = [(s[1] / total_prob, s[2], s[3]) for s in scored]
            end
        end
        sort!(scored; by=x -> -x[1])
        for (s, w, _) in scored[1:min(gen.beam_width, end)]
            push!(new_beams, (s, vcat(words, [w]), union(us, [w])))
        end
    end
    sort!(new_beams; by=x -> -x[1])
    return new_beams[1:min(gen.beam_width, end)]
end

function _standard_generate(gen::MirnanGenerator, prompt_tokens::Vector{String}; max_words=15)
    prompt_pv = [_pv(gen, w) for w in prompt_tokens]
    absorb!(gen.prompt_field, prompt_pv)
    DensityMatrix.build!(gen.density_matrix, prompt_pv)
    beams = [copy(prompt_tokens)]
    used_sets = [Set(prompt_tokens)]
    for step in 1:max_words
        new = _beam_step(gen, beams, used_sets, prompt_pv; step=step, total=max_words)
        isempty(new) && break
        beams = [b[2] for b in new]
        used_sets = [b[3] for b in new]
        step!(gen.prompt_field)
    end
    return join(beams[1][length(prompt_tokens)+1:end], " ")
end

struct GeneratorState
    output::Vector{String}
    used::Set{String}
    used_families::Set{String}
    phase::Vector{Float64}
    forbidden::Set{String}
end

function _resonant_generate(gen::MirnanGenerator, prompt_tokens::Vector{String}; max_words=15,
                            shape_roles=String[], semantic_field=nothing,
                            syntax_multiplier::Float64=1.0,
                            semantic_multiplier::Float64=1.0,
                            causal_multiplier::Float64=1.0)
    K_sem = gen.K_sem
    K_sem === nothing && return _standard_generate(gen, prompt_tokens; max_words=max_words)
    prompt_pv = [_pv(gen, w) for w in prompt_tokens]
    absorb!(gen.prompt_field, prompt_pv)
    try; DensityMatrix.build!(gen.density_matrix, prompt_pv); catch; end
    V = length(gen.vocab)
    phase = zeros(Float64, V)
    for w in prompt_tokens
        wid = get(gen.vocab, w, 0)
        wid > 0 && (phase[wid] += 1.0)
    end
    topic = TopicDensityMatrix()
    for w in prompt_tokens
        update_topic_with_word!(topic, w, w -> _pv(gen, String(w)))
    end
    started_at = time()
    budget_sec = _env_float("MIRNAN_GENERATION_BUDGET_SEC", 8.0)
    output = String[]
    used = Set(prompt_tokens)
    used_families = Set(_generation_family_key.(prompt_tokens))
    repeat_count = 0

    states_stack = GeneratorState[]
    forbidden_current = Set{String}()

    step = 1
    while step <= max_words
        context_words = String[vcat(prompt_tokens, output)...]
        current_topic_10000 = zeros(Float32, Constants.TOTAL_DIM)
        n_topic = 0
        for cw in context_words[max(1, end-10):end]
            pv_cw = try _pv(gen, cw, nothing) catch; nothing end
            if pv_cw !== nothing
                current_topic_10000 .+= Float32.(pv_cw)
                n_topic += 1
            end
        end
        if n_topic > 0
            current_topic_10000 ./= n_topic
            nrm = norm(current_topic_10000)
            nrm > 1e-10 && (current_topic_10000 ./= nrm)
        else
            current_topic_10000 = nothing
        end
        budget_sec > 0.0 && (time() - started_at) > budget_sec && break
        if step <= 2
            syn_w = 0.12; sem_w = 0.88; damp = 0.55
        elseif step <= 5
            syn_w = 0.10; sem_w = 0.90; damp = 0.62
        else
            syn_w = 0.06; sem_w = 0.94; damp = 0.65
        end
        excited = zeros(Float64, V)
        sem_w_eff = sem_w * semantic_multiplier
        syn_w_eff = syn_w * syntax_multiplier
        causal_w_eff = causal_multiplier
        if sem_w_eff > 0
            k_exc = K_sem * phase
            ks = get(gen.k_sem_config, "strength", 1.0)
            kt = max(get(gen.k_sem_config, "temperature", 1.0), 0.01)
            excited .+= sem_w_eff .* ks .* (abs.(k_exc) .^ (1.0 / kt)) .* sign.(k_exc)
            if gen.K_causal !== nothing
                try
                    c_exc = gen.K_causal' * phase
                    excited .+= (0.30 * sem_w_eff * causal_w_eff) .* ks .* (abs.(c_exc) .^ (1.0 / kt)) .* sign.(c_exc)
                catch e
                    @debug "K_causal resonant excitation failed: $e"
                end
            end
        end
        if gen.K_syn !== nothing && syn_w_eff > 0
            try
                prev_word = isempty(output) ? (isempty(prompt_tokens) ? "" : prompt_tokens[end]) : output[end]
                prev_id = get(gen.vocab, prev_word, 0)
                if prev_id > 0 && prev_id <= size(gen.K_syn, 1)
                    excited .+= syn_w_eff .* gen.K_syn[prev_id, :]
                else
                    excited .+= syn_w_eff .* (gen.K_syn' * phase)
                end
            catch e
                @debug "K_syn resonant excitation failed: $e"
            end
        end
        for w in used
            wid2 = get(gen.vocab, w, 0)
            wid2 > 0 && wid2 <= V && (excited[wid2] = 0.0)
        end
        kt2 = get(gen.k_sem_config, "threshold", 0.0)
        kt2 > 0.0 && (excited[excited .< kt2] .= 0.0)
        max_exc = maximum(excited)
        max_exc < 1e-10 && break
        intent_boost = get(gen.scoring_weights, "intent_guidance_boost", 0.0)
        if intent_boost > 0.0
            _boost_guided_terms!(excited, gen, used, intent_boost)
            max_exc = maximum(excited)
        end
        soft_intent_boost = get(gen.scoring_weights, "soft_intent_guidance_boost", 0.0)
        if soft_intent_boost > 0.0
            _boost_guided_terms!(excited, gen, used, soft_intent_boost)
            max_exc = maximum(excited)
        end
        n_take = min(40, V)
        top_ids = Int[]
        if V <= 100000
            exc_copy = copy(excited)
            for _ in 1:n_take; _, idx = findmax(exc_copy); exc_copy[idx] = -Inf; push!(top_ids, idx); end
        else
            top_ids = partialsortperm(excited, 1:n_take; rev=true)
        end
        recent_window = 5
        recent_pvs = Vector{Float32}[]
        for rw in output[max(1, end-recent_window+1):end]
            pv_rw = _pv(gen, rw, current_topic_10000)
            if !isempty(pv_rw)
                push!(recent_pvs, pv_rw)
            end
        end
        avg_recent_pv = if !isempty(recent_pvs)
            s = zeros(Float32, length(first(recent_pvs)))
            for pv in recent_pvs
                s .+= pv
            end
            s ./= length(recent_pvs)
            nrm = norm(s)
            nrm > 1e-10 ? s ./ nrm : zeros(Float32, length(s))
        else
            Float32[]
        end
        candidates_scores = Float64[]
        candidates_words = String[]
        for wid in top_ids
            w = get(gen.id2word, wid, nothing)
            w === nothing && continue
            if w in forbidden_current
                continue
            end
            charged_answer = _aql_candidate_bias(gen, w)
            _is_generation_candidate(w, prompt_tokens) || continue
            family = _generation_family_key(w)
            family in used_families && charged_answer <= 0.0 && continue
            _reject_context_drift(w, prompt_tokens, output) && charged_answer <= 0.0 && continue
            inh_score = _aql_candidate_inhibition(gen, w)
            inh_score >= 0.995 && continue
            topic_score = try
                SemanticComprehension.topic_resonance(topic, w, w -> _pv(gen, String(w), current_topic_10000))
            catch e
                @debug "Topic resonance failed for '$w': $e"
                0.5
            end
            topic_score < 0.01 && continue
            en = clamp(excited[wid] / max_exc, 0.0, 1.0)
            shape_bonus = (!isempty(shape_roles) && step <= length(shape_roles)) ?
                _shape_syntax_bonus(gen, w, shape_roles[step]) : 0.0
            semantic_bonus = _semantic_candidate_bonus(w, semantic_field)
            context_words = String[vcat(prompt_tokens, output)...]
            coherence = _context_coherence(gen, w, prompt_pv, context_words, current_topic_10000)
            anchor = _prompt_anchor_support(gen, w, prompt_tokens)
            drift = _drift_penalty(w, prompt_tokens, output)
            answer_charge = charged_answer
            question_charge = _aql_candidate_inhibition(gen, w)
            intent_boost = get(gen.scoring_weights, "intent_guidance_boost", 0.0)
            if intent_boost > 0.0 && answer_charge <= 0.0
                continue
            end
            drift_gate = clamp(0.25 + 0.75 * coherence + 0.35 * anchor - 0.60 * drift, 0.0, 1.25)
            drift_gate <= 0.05 && continue
            diversity_penalty = 0.0
            if !isempty(avg_recent_pv) && length(output) >= 2
                w_pv = _pv(gen, w, current_topic_10000)
                if !isempty(w_pv)
                    w_nrm = norm(w_pv)
                    if w_nrm > 1e-10
                        sim = dot(w_pv, avg_recent_pv) / w_nrm
                        diversity_penalty = 0.20 * max(0.0, sim + 0.1)
                    end
                end
            end
            trajectory_bonus = trajectory_alignment_score(gen.trajectory_tracker, w, w -> _pv(gen, w))
            score = en * 0.32 + topic_score * 0.26 +
                    coherence * 0.16 +
                    anchor * 0.16 +
                    (w in used && answer_charge <= 0.0 ? -3.0 : 0.0) +
                    shape_bonus * 0.22 +
                    semantic_bonus * 0.10 +
                    trajectory_bonus * 0.15 +
                    answer_charge * (0.38 + 0.22 * soft_intent_boost) -
                    question_charge * 0.30 -
                    diversity_penalty
            score *= drift_gate
            push!(candidates_scores, score)
            push!(candidates_words, w)
        end
        best_w = isempty(candidates_scores) ? nothing : candidates_words[argmax(candidates_scores)]
        
        if best_w === nothing
            if isempty(states_stack)
                break
            end
            prev_state = pop!(states_stack)
            last_selected_word = output[end]
            output = copy(prev_state.output)
            used = copy(prev_state.used)
            used_families = copy(prev_state.used_families)
            phase = copy(prev_state.phase)
            forbidden_current = copy(prev_state.forbidden)
            push!(forbidden_current, last_selected_word)
            topic = TopicDensityMatrix()
            for w in prompt_tokens
                update_topic_with_word!(topic, w, w -> _pv(gen, String(w)))
            end
            for w in output
                update_topic_with_word!(topic, w, w -> _pv(gen, String(w)))
            end
            step -= 1
            continue
        end

        if gen.cerebellum.pid_enabled
            try
                _, _, quality = _score(gen, best_w, used, Vector{Float64}[_pv(gen, ow) for ow in output], prompt_pv;
                                       prev_word=isempty(output) ? (isempty(prompt_tokens) ? nothing : prompt_tokens[end]) : output[end],
                                       context_words=context_words)
                
                signal = Dict{String,Float64}("target" => gen.cerebellum.pid.setpoint, "current" => quality)
                pid_out = MirnanCerebellumModule.correct!(gen.cerebellum.pid, gen.scoring_weights, signal)
                MirnanCerebellumModule.update_integral!(gen.cerebellum.pid)
                
                kb_p_term = 0.0
                if !isempty(gen.retrieved_similarities)
                    avg_sim = sum(gen.retrieved_similarities) / length(gen.retrieved_similarities)
                    kb_err = avg_sim - 0.5
                    kb_p_term = MirnanCerebellumModule.correct_weight!(gen.cerebellum.pid, gen.scoring_weights, "kb_knowledge", kb_err)
                end
                
                active_marker_type = ""
                active_marker = ""
                mem = _LEARNED_ISTINBAT_MEMORY[]
                if mem !== nothing
                    prompt_str = join(prompt_tokens, " ")
                    active_marker_type, active_marker = AlIstinbat._marker_hit(prompt_str, mem.discovered_markers)
                end
                
                aql_p_term = 0.0
                if !isempty(active_marker_type)
                    aql_err = 0.3
                    aql_p_term = MirnanCerebellumModule.correct_weight!(gen.cerebellum.pid, gen.scoring_weights, "aql_guidance", aql_err)
                end
                
                push!(gen.cerebellum.integration_log, Dict{String,Any}(
                    "step" => length(output) + 1,
                    "word" => best_w,
                    "markers_active" => isempty(active_marker_type) ? String[] : [active_marker_type],
                    "active_marker" => active_marker,
                    "retrieval_similarity" => isempty(gen.retrieved_similarities) ? 0.0 : (sum(gen.retrieved_similarities) / length(gen.retrieved_similarities)),
                    "pid_output" => pid_out,
                    "kb_p_term" => kb_p_term,
                    "aql_p_term" => aql_p_term,
                    "weights" => copy(gen.scoring_weights),
                    "quality_signal" => quality,
                ))
            catch e
                @debug "Cerebellum PID correction failed: $e"
            end
        end

        push!(states_stack, GeneratorState(copy(output), copy(used), copy(used_families), copy(phase), copy(forbidden_current)))
        forbidden_current = Set{String}()

        push!(output, best_w); push!(used, best_w); push!(used_families, _generation_family_key(best_w))
        absorb_word!(gen.trajectory_tracker, best_w, w -> _pv(gen, w))
        update_topic_with_word!(topic, best_w, w -> _pv(gen, String(w), current_topic_10000))
        if length(output) >= 3
            last3 = output[max(1, end-2):end]
            if length(Set(last3)) <= 1
                repeat_count += 1
                repeat_count >= 2 && break
            else
                repeat_count = 0
            end
        end
        bwid = get(gen.vocab, best_w, 0)
        bwid > 0 && (phase[bwid] += 2.0)
        phase .*= damp
        step += 1
    end
    return join(output, " ")
end

function _fallback_generate(gen::MirnanGenerator, prompt_tokens::Vector{String}; max_words=10)
    used = Set{String}(prompt_tokens)
    output = String[]
    ctx_tokens = copy(prompt_tokens)
    function first_unused_vocab_word()
        for (w, _) in sort(collect(gen.vocab); by=x -> x[2])
            if !(w in used) && _is_generation_candidate(w, ctx_tokens) &&
               !_reject_context_drift(w, prompt_tokens, output)
                return w
            end
        end
        return nothing
    end
    function physics_vocab_step!()
        ppv = [_pv(gen, w) for w in prompt_tokens]
        cpv = [_pv(gen, w) for w in ctx_tokens[max(1, end-5):end]]
        bw = nothing; bs = -Inf
        for (w, _) in sort(collect(gen.vocab); by=x -> x[2])
            w in used && continue
            _is_generation_candidate(w, ctx_tokens) || continue
            _reject_context_drift(w, prompt_tokens, output) && continue
            prob, _, _ = _score(gen, w, used, cpv, ppv;
                prev_word=isempty(ctx_tokens) ? nothing : ctx_tokens[end],
                context_words=ctx_tokens)
            if isfinite(prob) && prob > bs; bs = prob; bw = w; end
        end
        bw === nothing && (bw = first_unused_vocab_word())
        bw === nothing && return false
        push!(output, bw); push!(used, bw); push!(ctx_tokens, bw)
        return true
    end
    if gen.K_sem === nothing
        for _ in 1:max_words; physics_vocab_step!() || break; end
        return join(output, " ")
    end
    for _ in 1:max_words
        bw = nothing; bs = -Inf
        for cw in ctx_tokens[max(1, end-3):end]
            cid = get(gen.vocab, cw, nothing)
            cid === nothing && continue; cid > size(gen.K_sem, 1) && continue
            row = Vector(gen.K_sem[cid, :])
            for tid in sortperm(row; rev=true)[1:min(30, end)]
                row[tid] > 1e-6 || break
                w = get(gen.id2word, tid, nothing)
                w === nothing && continue; w in used && continue; length(strip(w)) < 2 && continue
                _is_generation_candidate(w, ctx_tokens) || continue
                _reject_context_drift(w, prompt_tokens, output) && continue
                row[tid] > bs && (bs = row[tid]; bw = w)
            end
        end
        bw === nothing && (physics_vocab_step!() || break; continue)
        push!(output, bw); push!(used, bw); push!(ctx_tokens, bw)
    end
    return join(output, " ")
end

function _directed_regenerate!(gen::MirnanGenerator,
                               prompt_tokens::Vector{String},
                               effective_mode::String,
                               review,
                               cereb_policy::CerebellumPolicy;
                               max_words::Int=15,
                               used_fallback::Bool=false)
    target = getfield(review, :repair_target)
    target == "none" && return ""
    if target == "fallback"
        (used_fallback || length(gen.vocab) >= 50000) && return ""
        return _with_cerebellum_policy(gen, cereb_policy, () ->
            _fallback_generate(gen, prompt_tokens; max_words=max_words))
    end

    repair_policy = _repair_policy(cereb_policy, target)
    repair_mode = effective_mode
    if length(gen.vocab) < 50000 && target in ("diversity", "prompt_alignment", "syntax", "language")
        repair_mode = "standard"
    elseif target == "coherence" && gen.K_sem !== nothing
        repair_mode = "resonant"
    end

    return _with_cerebellum_policy(gen, repair_policy, () -> begin
        repair_mode == "resonant" ?
            _resonant_generate(gen, prompt_tokens; max_words=max_words) :
            _standard_generate(gen, prompt_tokens; max_words=max_words)
    end)
end

function _maybe_revise_generation!(gen::MirnanGenerator,
                                   prompt::String,
                                   prompt_tokens::Vector{String},
                                   result::AbstractString,
                                   effective_mode::String,
                                   cereb_policy::CerebellumPolicy;
                                   max_words::Int=15,
                                   used_fallback::Bool=false,
                                   allow_anchored_rejection::Bool=true)
    first_review = _review_candidate!(gen, prompt, prompt_tokens, result)
    intent_for_anchor = detect_response_intent(prompt).intent
    if _explicit_causal_prompt(prompt) && _semantic_anchor_weak(prompt_tokens, result, intent_for_anchor)
        mem = _corpus_memory_answer(gen, prompt, prompt_tokens; intent=intent_for_anchor)
        if !isempty(strip(mem))
            mem_review = _review_candidate!(gen, prompt, prompt_tokens, mem)
            return mem, mem_review
        end
        active_paras = _get_active_paragraphs(gen, prompt_tokens)
        causal = _causal_anchor_answer(gen, prompt_tokens, active_paras)
        if !isempty(strip(causal))
            causal_review = _review_candidate!(gen, prompt, prompt_tokens, causal)
            return causal, causal_review
        end
    end
    (!gen.self_review.enabled || first_review.accepted) && return String(result), first_review
    if !_env_on("MIRNAN_REVIEW_REGENERATE", "0")
        repaired = _non_template_repair_answer(gen, prompt, prompt_tokens, first_review)
        if !isempty(strip(repaired))
            repaired_review = _review_candidate!(gen, prompt, prompt_tokens, repaired)
            return repaired, repaired_review
        end
        return String(result), first_review
    end

    directed = _directed_regenerate!(gen, prompt_tokens, effective_mode,
                                     first_review, cereb_policy;
                                     max_words=max_words,
                                     used_fallback=used_fallback)
    if !isempty(strip(directed))
        directed_review = _review_candidate!(gen, prompt, prompt_tokens, directed)
        chosen = directed_review.score > first_review.score + 0.03
        learn_review_treatment!(gen.self_review, prompt_tokens;
                                prompt=prompt,
                                repair_target=first_review.repair_target,
                                before_score=first_review.score,
                                after_score=directed_review.score,
                                chosen=chosen,
                                source="directed")
        if chosen
            return String(directed), directed_review
        end
    end

    alternate = ""
    alternate_target = "none"
    if effective_mode == "resonant" && length(gen.vocab) < 50000
        alternate_target = "standard_mode"
        alternate = _with_cerebellum_policy(gen, cereb_policy, () ->
            _standard_generate(gen, prompt_tokens; max_words=max_words))
    elseif !used_fallback
        alternate_target = "fallback"
        alternate = _with_cerebellum_policy(gen, cereb_policy, () ->
            _fallback_generate(gen, prompt_tokens; max_words=max_words))
    end

    isempty(strip(alternate)) && return String(result), first_review
    second_review = _review_candidate!(gen, prompt, prompt_tokens, alternate)
    chosen = second_review.score > first_review.score + 0.03
    learn_review_treatment!(gen.self_review, prompt_tokens;
                            prompt=prompt,
                            repair_target=alternate_target,
                            before_score=first_review.score,
                            after_score=second_review.score,
                            chosen=chosen,
                            source="alternate")
    if chosen
        return String(alternate), second_review
    end
    gen.self_review.last_review = first_review
    allow_anchored_rejection || return String(result), first_review
    anchored = _anchored_rejection_answer(gen, prompt, prompt_tokens, first_review)
    if !isempty(strip(anchored))
        anchored_review = _review_candidate!(gen, prompt, prompt_tokens, anchored)
        return anchored, anchored_review
    end
    return String(result), first_review
end

function try_generate(::ResonantStrategy, gen::MirnanGenerator, prompt::String, prompt_tokens::Vector{String}, mode::String, cereb_obs, cereb_policy, response_plan, active_paras, max_words::Int)
    k_density_val = _cerebellum_k_density(gen)

    contextual = ContextualLearning.contextual_answer(gen.contextual_learning, prompt)
    if !isempty(strip(contextual))
        return _finish_generation!(gen, prompt, prompt_tokens, contextual,
                                   cereb_obs, cereb_policy)
    end
    
    yesno_field = _yesno_declarative_field_answer(gen, prompt, prompt_tokens)
    if response_plan.intent != "dialogue" && !isempty(strip(yesno_field))
        return _finish_generation!(gen, prompt, prompt_tokens, yesno_field,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    relation_answer = _relationship_answer(gen, prompt, prompt_tokens)
    if !isempty(strip(relation_answer))
        return _finish_generation!(gen, prompt, prompt_tokens, relation_answer,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    memory_answer = response_plan.intent == "dialogue" ? "" :
                    _corpus_memory_answer(gen, prompt, prompt_tokens; intent=response_plan.intent)
    if !isempty(strip(memory_answer))
        return _finish_generation!(gen, prompt, prompt_tokens, memory_answer,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end
    if response_plan.intent in ("causal", "mechanism") &&
       _explicit_causal_prompt(prompt) &&
       length(prompt_tokens) <= 8
        causal_anchor = _causal_anchor_answer(gen, prompt_tokens, active_paras)
        if !isempty(strip(causal_anchor))
            return _finish_generation!(gen, prompt, prompt_tokens, causal_anchor,
                                       cereb_obs, cereb_policy;
                                       sanitize_output=false,
                                       apply_templates=false)
        end
    end
    conservative_anchor = response_plan.intent == "dialogue" ? "" :
                          _conservative_anchor_answer(prompt, prompt_tokens)
    if !isempty(strip(conservative_anchor))
        return _finish_generation!(gen, prompt, prompt_tokens, conservative_anchor,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    planned_answer = _generate_intent_planned_response(gen, prompt, prompt_tokens)
    if !isempty(strip(planned_answer))
        return _finish_generation!(gen, prompt, prompt_tokens, planned_answer,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    sense_answer = _sense_superposition_answer(gen, prompt, prompt_tokens, cereb_policy, cereb_obs)
    if !isempty(strip(sense_answer))
        return _finish_generation!(gen, prompt, prompt_tokens, sense_answer,
                                   cereb_obs, cereb_policy)
    end

    gen.aql_bias = _aql_bias_from_prompt(gen, prompt)
    gen.aql_inhibition = _aql_inhibition_from_prompt(gen, prompt)
    gravity_profile = _intent_gravity_profile(prompt)

    intent = _detect_intent(prompt_tokens)
    plan!(gen.trajectory, prompt_tokens, w -> _pv(gen, w))
    try
        plan_response!(gen.response_planner, intent, w -> _pv(gen, w))
        plan_architect!(gen.architect, intent, prompt_tokens, w -> _pv(gen, w))
    catch e
        @warn "Response planning failed: $e"
    end

    effective_mode = mode
    if mode == "auto"
        effective_mode = cereb_policy.mode in ("standard", "resonant") ? cereb_policy.mode :
                         (gen.K_sem !== nothing && k_density_val > 0.001 ? "resonant" : "standard")
    end
    lisan_roles = _lisan_generate(gen, prompt_tokens)
    semantic_field = _hisban_prompt_guidance(gen, prompt_tokens)
    attention_field = _semantic_attention_field(gen, prompt_tokens)
    local_max_words = has_gravity_profile(gravity_profile) && gravity_profile.max_words > 0 ?
                      min(max_words, gravity_profile.max_words) : max_words
    result = _with_semantic_attention(gen, attention_field, () ->
        _with_intent_gravity(gen, gravity_profile, () ->
            _with_cerebellum_policy(gen, cereb_policy, () -> begin
            effective_mode == "resonant" ?
                _resonant_generate(gen, prompt_tokens; max_words=local_max_words, shape_roles=lisan_roles,
                                  semantic_field=semantic_field,
                                  syntax_multiplier=gravity_profile.syntax_multiplier,
                                  semantic_multiplier=gravity_profile.semantic_multiplier,
                                  causal_multiplier=gravity_profile.causal_multiplier) :
                _standard_generate(gen, prompt_tokens; max_words=local_max_words)
        end)))
    used_fallback = false
    if isempty(strip(result))
        if _relation_or_difference_prompt(prompt)
            return ""
        end
        if length(gen.vocab) < 50000
            result = _with_cerebellum_policy(gen, cereb_policy, () ->
                _fallback_generate(gen, prompt_tokens; max_words=max_words))
            used_fallback = true
        end
    end
    has_charge_profile = has_gravity_profile(gravity_profile)
    dialogue_charge_profile = has_charge_profile && gravity_profile.intent == "dialogue"
    dialogue_charge_profile && (result = _finalize_dialogue_charge_output(result, gravity_profile))
    result, review = _maybe_revise_generation!(gen, prompt, prompt_tokens, result,
                                               effective_mode, cereb_policy;
                                               max_words=local_max_words,
                                               used_fallback=used_fallback,
                                               allow_anchored_rejection=!has_charge_profile)
    if _relation_or_difference_prompt(prompt) && _list_like_generation_output(result)
        return ""
    end
    return _finish_generation!(gen, prompt, prompt_tokens, result,
                               cereb_obs, cereb_policy; review=review,
                               sanitize_output=!dialogue_charge_profile,
                               apply_templates=!dialogue_charge_profile)
end
