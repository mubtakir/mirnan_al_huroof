function _is_question_token(word::AbstractString)
    w = String(word)
    w in ("هل", "hal") && return true
    mem = _LEARNED_ISTINBAT_MEMORY[]
    if mem !== nothing
        haskey(mem.discovered_markers, w) && mem.discovered_markers[w] == "question" && return true
        for k in _generation_keys(w)
            haskey(mem.discovered_markers, k) && mem.discovered_markers[k] == "question" && return true
        end
    end
    return false
end

include("yesno_relations.jl")
include("nisba_relations.jl")
include("evidence_relations.jl")
include("difference_and_gate.jl")

function try_generate(::RelationStrategy, gen::MirnanGenerator, prompt::String, prompt_tokens::Vector{String}, mode::String, cereb_obs, cereb_policy, response_plan, active_paras)
    contextual = ContextualLearning.contextual_answer(gen.contextual_learning, prompt)
    if !isempty(strip(contextual))
        return _finish_generation!(gen, prompt, prompt_tokens, contextual,
                                   cereb_obs, cereb_policy)
    end

    # Canonical directional relations learned by the core corpus need to trump
    # noisy contradictory witnesses.
    canonical_yesno = _canonical_yesno_relation_answer(prompt_tokens)
    if !isempty(strip(canonical_yesno))
        return _finish_generation!(gen, prompt, prompt_tokens, canonical_yesno,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    canonical_explanation = _canonical_explanatory_relation_answer(prompt_tokens)
    if !isempty(strip(canonical_explanation))
        return _finish_generation!(gen, prompt, prompt_tokens, canonical_explanation,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    # Contradiction check
    contradiction_answer = _semantic_contradiction_yesno_answer(gen, prompt, prompt_tokens, active_paras)
    if !isempty(strip(contradiction_answer))
        return _finish_generation!(gen, prompt, prompt_tokens, contradiction_answer,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    # Learned relation evidence may contain a direct operator match such as
    # "العدل يحفظ السلام"; let that exact direction decide before noisier
    # witness paths inspect reversed or adjacent relations.
    yesno_relation = _yesno_learned_relation_answer(gen, prompt, prompt_tokens, active_paras)
    if response_plan.intent != "dialogue" && !isempty(strip(yesno_relation))
        return _finish_generation!(gen, prompt, prompt_tokens, yesno_relation,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    # Opposition check
    opposition_answer = _yesno_learned_opposition_answer(gen, prompt_tokens)
    if response_plan.intent != "dialogue" && !isempty(strip(opposition_answer))
        return _finish_generation!(gen, prompt, prompt_tokens, opposition_answer,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    # Semantic relation gate
    semantic_relation = _semantic_relation_gate_answer(prompt, prompt_tokens)
    if !isempty(strip(semantic_relation))
        return _finish_generation!(gen, prompt, prompt_tokens, semantic_relation,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    # Explanatory direction guard
    explanatory_direction_guard = _explanatory_direction_guard_answer(gen, prompt, prompt_tokens, active_paras)
    if response_plan.intent != "dialogue" && !isempty(strip(explanatory_direction_guard))
        return _finish_generation!(gen, prompt, prompt_tokens, explanatory_direction_guard,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    # Relationship answer early
    early_relation_answer = _relationship_answer(gen, prompt, prompt_tokens)
    if response_plan.intent != "dialogue" && !isempty(strip(early_relation_answer))
        return _finish_generation!(gen, prompt, prompt_tokens, early_relation_answer,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    # Direct evidence
    direct_evidence = _direct_relation_evidence_answer(gen, prompt, prompt_tokens)
    if !isempty(strip(direct_evidence))
        return _finish_generation!(gen, prompt, prompt_tokens, direct_evidence,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    # Direct witness
    witness_relation = _yesno_direct_witness_answer(prompt_tokens)
    if response_plan.intent != "dialogue" && !isempty(strip(witness_relation))
        return _finish_generation!(gen, prompt, prompt_tokens, witness_relation,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    # Reversed witness
    reversed_witness = _yesno_reversed_witness_answer(prompt_tokens)
    if response_plan.intent != "dialogue" && !isempty(strip(reversed_witness))
        return _finish_generation!(gen, prompt, prompt_tokens, reversed_witness,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    # Opposed relation
    opposed_relation = _yesno_opposed_relation_answer(gen, prompt_tokens)
    if response_plan.intent != "dialogue" && !isempty(strip(opposed_relation))
        return _finish_generation!(gen, prompt, prompt_tokens, opposed_relation,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    # Definition negation
    definition_negation = _yesno_definition_negation_answer(gen, prompt_tokens)
    if response_plan.intent != "dialogue" && !isempty(strip(definition_negation))
        return _finish_generation!(gen, prompt, prompt_tokens, definition_negation,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    # Explanatory evidence
    explanatory_answer = _raw_explanatory_answer(gen, prompt, prompt_tokens)
    composed_answer = _raw_composed_evidence_answer(gen, prompt, prompt_tokens)
    if !isempty(strip(explanatory_answer)) || !isempty(strip(composed_answer))
        anchors_for_choice = _relationship_anchor_keys(prompt_tokens)
        exp_hits = _anchor_hit_count_in_text(explanatory_answer, anchors_for_choice)
        comp_hits = _anchor_hit_count_in_text(composed_answer, anchors_for_choice)
        allow_composed_priority = length(anchors_for_choice) >= 3
        chosen_evidence = (allow_composed_priority && !isempty(strip(composed_answer)) && comp_hits > exp_hits) ?
                          composed_answer : explanatory_answer
        isempty(strip(chosen_evidence)) && (chosen_evidence = composed_answer)
        chosen_hits = _anchor_hit_count_in_text(chosen_evidence, anchors_for_choice)
        if chosen_hits >= min(length(anchors_for_choice), 2)
            return _finish_generation!(gen, prompt, prompt_tokens, chosen_evidence,
                                       cereb_obs, cereb_policy;
                                       sanitize_output=false,
                                       apply_templates=false)
        end
    end

    # YesNo declarative field
    yesno_field = _yesno_declarative_field_answer(gen, prompt, prompt_tokens)
    if response_plan.intent != "dialogue" && !isempty(strip(yesno_field))
        return _finish_generation!(gen, prompt, prompt_tokens, yesno_field,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    # Relationship answer fallback
    relation_answer = _relationship_answer(gen, prompt, prompt_tokens)
    if !isempty(strip(relation_answer))
        return _finish_generation!(gen, prompt, prompt_tokens, relation_answer,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end
    return nothing
end
