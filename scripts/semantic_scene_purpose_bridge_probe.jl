#!/usr/bin/env julia
# Live probe for the bridge between SemanticScene and RelationFrame purpose memory.

const MIRNAN_DIR = dirname(@__DIR__)
const MODEL_DIR = joinpath(MIRNAN_DIR, "model")

include(joinpath(MIRNAN_DIR, "src", "MirnanNew.jl"))

using .MirnanNew

const Physics = MirnanNew.Physics

function _env_int(name::String, default::Int)
    raw = get(ENV, name, "")
    isempty(strip(raw)) && return default
    try
        return parse(Int, raw)
    catch
        return default
    end
end

function _deadline(seconds::Int)
    seconds <= 0 && return Inf
    return time() + seconds
end

_timed_out(deadline::Float64) = time() > deadline

function _has_arabic(s::AbstractString)
    return occursin(r"[\u0600-\u06FF]", String(s))
end

function _scene_prompt(scene)
    action = strip(String(scene.action))
    patient = strip(String(scene.patient))
    isempty(action) && return ""
    if _has_arabic(action) || _has_arabic(patient)
        return "\u0644\u0645\u0627\u0630\u0627 $(action) $(patient)\u061f"
    end
    return "\u0644\u0645\u0627\u0630\u0627 $(action) $(patient)\u061f"
end

function _purpose_prompt(rec)
    left = strip(join(rec.before_terms, " "))
    right = strip(join(rec.after_terms, " "))
    core = isempty(left) ? right : left
    isempty(core) && return ""
    return "\u0644\u0645\u0627\u0630\u0627 $(core)\u061f"
end

function _strip_terminal_sentence_punc(s::AbstractString)
    return replace(strip(String(s)), r"[\.\!\?\u061F\u06D4]+$" => "")
end

function _scene_summary_fields(summary::AbstractString)
    fields = Dict{String,String}()
    for part in split(String(summary), " | ")
        idx = findfirst(==('='), part)
        idx === nothing && continue
        key = strip(part[1:prevind(part, idx)])
        val = strip(part[nextind(part, idx):end])
        isempty(key) && continue
        isempty(val) && continue
        fields[key] = val
    end
    return fields
end

function _scene_summary_sentence(summary::AbstractString)
    raw = strip(String(summary))
    isempty(raw) && return ""
    occursin("=", raw) || return raw
    fields = _scene_summary_fields(raw)
    action = get(fields, "action", "")
    patient = get(fields, "patient", "")
    before = get(fields, "before", "")
    after = get(fields, "after", "")
    effects = get(fields, "effects", "")
    isempty(action) && isempty(patient) && return raw
    if _has_arabic(raw)
        target = isempty(patient) ? "\u0627\u0644\u0645\u062a\u0623\u062b\u0631" : patient
        base = "\u0639\u0646\u062f $(action) $(target)"
        if !isempty(before) && !isempty(after)
            base *= " \u064a\u0646\u062a\u0642\u0644 \u0627\u0644\u0645\u062a\u0623\u062b\u0631 \u0645\u0646 $(before) \u0625\u0644\u0649 $(after)"
        end
        isempty(effects) || (base *= "\u060c \u0648\u062a\u0638\u0647\u0631 \u0622\u062b\u0627\u0631 \u0645\u062b\u0644 $(effects)")
        return base
    end
    target = isempty(patient) ? "the affected thing" : patient
    base = "When $(action) affects $(target)"
    if !isempty(before) && !isempty(after)
        base *= ", the affected thing shifts from $(before) to $(after)"
    end
    isempty(effects) || (base *= ", with effects such as $(effects)")
    return base
end

function _purpose_fragment_for_composite(purpose::AbstractString)
    p = _strip_terminal_sentence_punc(purpose)
    m = match(r"^الغاية\s+من\s+.+?\s+هي\s+(.+)$", p)
    m !== nothing && return strip(m.captures[1])
    m = match(r"^الغاية\s+هي\s+(.+)$", p)
    m !== nothing && return strip(m.captures[1])
    return p
end

function _arabic_list_separators(s::AbstractString)
    return replace(String(s), r"\s*,\s*" => "\u060c ")
end

function _natural_arabic_scene_purpose_sentence(summary::AbstractString,
                                                purpose_clean::AbstractString)
    fields = _scene_summary_fields(summary)
    action = get(fields, "action", "")
    actor = get(fields, "actor", "")
    patient = get(fields, "patient", "")
    before = get(fields, "before", "")
    after = get(fields, "after", "")
    effects = get(fields, "effects", "")
    scene_text = join([action, actor, patient, before, after, effects], " ")
    _has_arabic(scene_text) || return ""
    isempty(action) && isempty(patient) && return ""

    target = isempty(patient) ? "\u0627\u0644\u0645\u062a\u0623\u062b\u0631" : patient
    event = strip(isempty(actor) ? "$(action) $(target)" : "$(action) $(actor) $(target)")
    clauses = String[event]
    if !isempty(before) && !isempty(after)
        push!(clauses, "\u0641\u0627\u0646\u062a\u0642\u0644 $(target) \u0645\u0646 $(before) \u0625\u0644\u0649 $(after)")
    end
    if !isempty(effects)
        push!(clauses, "\u0648\u0638\u0647\u0631\u062a \u0622\u062b\u0627\u0631 \u0645\u062b\u0644 $(_arabic_list_separators(effects))")
    end
    purpose = _arabic_list_separators(purpose_clean)
    isempty(strip(purpose)) && return ""
    return join(clauses, "\u061b ") * "\u061b \u0648\u0643\u0627\u0646\u062a \u0627\u0644\u063a\u0627\u064a\u0629 $(purpose)."
end

function _composite_answer(result)
    result.agreement == "cooperative" || return ""
    scene_part = !isempty(strip(result.scene_answer)) ? result.scene_answer : _scene_summary_sentence(result.scene_summary)
    purpose_part = result.purpose_answer
    isempty(strip(scene_part)) && return ""
    isempty(strip(purpose_part)) && return ""
    scene_clean = _strip_terminal_sentence_punc(scene_part)
    purpose_clean = _purpose_fragment_for_composite(purpose_part)
    natural = _natural_arabic_scene_purpose_sentence(result.scene_summary, purpose_clean)
    isempty(natural) || return natural
    if _has_arabic(scene_clean) || _has_arabic(purpose_clean)
        _has_arabic(scene_clean) && (scene_clean = _arabic_list_separators(scene_clean))
        _has_arabic(purpose_clean) && (purpose_clean = _arabic_list_separators(purpose_clean))
        if _has_arabic(scene_clean)
            return "$(scene_clean)\u061b \u0648\u0627\u0644\u063a\u0627\u064a\u0629 $(purpose_clean)."
        end
        return "$(scene_clean). \u0648\u0627\u0644\u063a\u0627\u064a\u0629 $(purpose_clean)."
    end
    return "$(scene_clean). In terms of purpose: $(purpose_clean)."
end

function _print_result(label::AbstractString, result)
    println("="^72)
    println("CASE: $(label)")
    println("PROMPT: $(result.prompt)")
    println("AGREEMENT: $(result.agreement)")
    println("SCENE_HAS_EVENT: $(result.scene_has_event)")
    println("PURPOSE_HAS_MEMORY: $(result.memory_has_purpose)")
    println("SCENE_CONFIDENCE: $(round(result.scene_confidence; digits=3))")
    println("PURPOSE_CONFIDENCE: $(round(result.purpose_confidence; digits=3))")
    println("SCENE_RELATION: $(result.scene_relation)")
    println("PURPOSE_RELATION: $(result.purpose_relation)")
    println("-- scene summary --")
    println(isempty(result.scene_summary) ? "(empty)" : result.scene_summary)
    println("-- scene answer --")
    println(isempty(result.scene_answer) ? "(empty)" : result.scene_answer)
    println("-- purpose answer --")
    println(isempty(result.purpose_answer) ? "(empty)" : result.purpose_answer)
    println("-- composite answer --")
    composite = _composite_answer(result)
    println(isempty(composite) ? "(empty)" : composite)
end

function _first_purpose_record(mem)
    for rec in values(mem.records)
        rec.relation_type == "purpose" || continue
        return rec
    end
    return nothing
end

function _word_count(s::AbstractString)
    return length(split(strip(String(s))))
end

function _noisy_scene(result)
    text = lowercase(string(result.scene_summary, " ", result.scene_answer))
    noisy = ["\u0644\u0627", "\u062c\u0648\u0627\u0628", "\u0633\u0624\u0627\u0644", "\u0639\u0644\u064a",
             " no ", " answer ", " question "]
    return any(x -> occursin(x, text), noisy)
end

function _displayable_result(result)
    _word_count(result.prompt) > 18 && return false
    if result.memory_has_purpose
        isempty(strip(result.purpose_answer)) && return false
        _word_count(result.purpose_answer) > 30 && return false
    end
    if result.scene_has_event
        isempty(strip(result.scene_summary)) && return false
        _word_count(result.scene_summary) > 34 && return false
        _noisy_scene(result) && return false
    end
    return true
end

function _displayable_cooperative_result(result)
    result.agreement == "cooperative" || return false
    result.scene_has_event || return false
    result.memory_has_purpose || return false
    _word_count(result.prompt) > 18 && return false
    isempty(strip(result.scene_summary)) && return false
    isempty(strip(result.purpose_answer)) && return false
    _word_count(result.purpose_answer) > 30 && return false
    return true
end

function _single_record_memory(rec)
    mem = Physics.IstinbatAttentionMemory()
    mem.records[rec.record_id] = rec
    return mem
end

function _probe_terms(xs)
    out = Set{String}()
    for x in xs
        t = lowercase(strip(String(x)))
        isempty(t) && continue
        length(t) < 2 && continue
        push!(out, t)
    end
    return out
end

function _scene_terms(scene)
    return _probe_terms([scene.action, scene.patient, scene.actor])
end

function _record_left_terms(rec)
    return _probe_terms(rec.before_terms)
end

function _build_purpose_index(istinbat; limit::Int)
    index = Dict{String,Vector{Any}}()
    scanned = 0
    for rec in values(istinbat.records)
        scanned >= limit && break
        rec.relation_type == "purpose" || continue
        scanned += 1
        for term in _record_left_terms(rec)
            get!(index, term, Any[])
            push!(index[term], rec)
        end
    end
    return index
end

function _indexed_purpose_candidates(index, scene)
    seen = Set{String}()
    recs = Any[]
    for term in _scene_terms(scene)
        for rec in get(index, term, Any[])
            rec.record_id in seen && continue
            push!(seen, rec.record_id)
            push!(recs, rec)
        end
    end
    return recs
end

function _find_indexed_cooperative(scene_mem, hisban, istinbat; limit::Int, deadline::Float64)
    index = _build_purpose_index(istinbat; limit=limit)
    checked = 0
    for scene in scene_mem.scenes
        _timed_out(deadline) && return nothing
        checked >= limit && break
        isempty(_scene_terms(scene)) && continue
        checked += 1
        for rec in _indexed_purpose_candidates(index, scene)
            _timed_out(deadline) && return nothing
            small_mem = _single_record_memory(rec)
            for prompt in (_scene_prompt(scene), _purpose_prompt(rec))
                isempty(prompt) && continue
                result = Physics.compare_scene_purpose_strategies(scene_mem, hisban, small_mem, prompt)
                _displayable_cooperative_result(result) && return result
            end
        end
    end
    return nothing
end

function _find_indexed_scene_only(scene_mem, hisban; limit::Int, deadline::Float64)
    empty_istinbat = Physics.IstinbatAttentionMemory()
    checked = 0
    for scene in scene_mem.scenes
        _timed_out(deadline) && return nothing
        checked >= limit && break
        prompt = _scene_prompt(scene)
        isempty(prompt) && continue
        checked += 1
        result = Physics.compare_scene_purpose_strategies(scene_mem, hisban, empty_istinbat, prompt)
        (result.agreement == "scene_only" && _displayable_result(result)) && return result
    end
    return nothing
end

function _find_seed_cooperative(scene_mem, hisban, istinbat)
    prompts = [
        "\u0644\u0645\u0627\u0630\u0627 \u0636\u0631\u0628 \u062e\u0627\u0644\u062f \u0627\u0644\u0643\u0631\u0629\u061f",
        "\u0644\u0645\u0627\u0630\u0627 \u062f\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062d\u062c\u0631\u061f",
        "\u0644\u0645\u0627\u0630\u0627 \u0643\u0633\u0631 \u0627\u0644\u0637\u0641\u0644 \u0627\u0644\u0643\u0623\u0633\u061f",
        "\u0644\u0645\u0627\u0630\u0627 \u0623\u0636\u0627\u0621 \u0627\u0644\u0645\u0635\u0628\u0627\u062d \u0627\u0644\u063a\u0631\u0641\u0629\u061f",
        "\u0644\u0645\u0627\u0630\u0627 \u0641\u062a\u062d \u0627\u0644\u0639\u0627\u0645\u0644 \u0627\u0644\u0628\u0627\u0628\u061f",
        "\u0644\u0645\u0627\u0630\u0627 Khalid hit the ball\u061f",
        "\u0644\u0645\u0627\u0630\u0627 The child broke the cup\u061f",
    ]
    for prompt in prompts
        result = Physics.compare_scene_purpose_strategies(scene_mem, hisban, istinbat, prompt)
        _displayable_cooperative_result(result) && return result
    end
    return nothing
end

function _print_seed_diagnostics(scene_mem, hisban, istinbat)
    prompts = [
        "\u0644\u0645\u0627\u0630\u0627 \u0636\u0631\u0628 \u062e\u0627\u0644\u062f \u0627\u0644\u0643\u0631\u0629\u061f",
        "\u0644\u0645\u0627\u0630\u0627 \u062f\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062d\u062c\u0631\u061f",
        "\u0644\u0645\u0627\u0630\u0627 \u0643\u0633\u0631 \u0627\u0644\u0637\u0641\u0644 \u0627\u0644\u0643\u0623\u0633\u061f",
        "\u0644\u0645\u0627\u0630\u0627 \u0623\u0636\u0627\u0621 \u0627\u0644\u0645\u0635\u0628\u0627\u062d \u0627\u0644\u063a\u0631\u0641\u0629\u061f",
        "\u0644\u0645\u0627\u0630\u0627 Khalid hit the ball\u061f",
    ]
    println("="^72)
    println("SEED COOPERATIVE DIAGNOSTICS")
    for prompt in prompts
        result = Physics.compare_scene_purpose_strategies(scene_mem, hisban, istinbat, prompt)
        println("- prompt: $(prompt)")
        println("  agreement=$(result.agreement) scene=$(result.scene_has_event) purpose=$(result.memory_has_purpose) scene_conf=$(round(result.scene_confidence; digits=3)) purpose_conf=$(round(result.purpose_confidence; digits=3))")
        if !isempty(result.scene_summary)
            println("  scene: $(result.scene_summary)")
        end
        if !isempty(result.purpose_answer)
            println("  purpose: $(result.purpose_answer)")
        end
    end
end

function _controlled_scene_only()
    hisban = Physics.SemanticCalculusMemory()
    Physics.learn_semantic_calculus_from_pair!(
        hisban,
        "The child broke the cup",
        "The cup separated into pieces and lost its shape.",
    )
    scene_mem = Physics.SemanticSceneMemory()
    Physics.learn_semantic_scene_from_text!(
        scene_mem,
        hisban,
        "The child broke the cup.",
    )
    empty_istinbat = Physics.IstinbatAttentionMemory()
    return Physics.compare_scene_purpose_strategies(
        scene_mem, hisban, empty_istinbat,
        "\u0644\u0645\u0627\u0630\u0627 broke cup\u061f",
    )
end

function _find_trained_case(scene_mem, hisban, istinbat; wanted::String, limit::Int, deadline::Float64=Inf)
    if wanted == "none"
        empty_scene = Physics.SemanticSceneMemory()
        empty_hisban = Physics.SemanticCalculusMemory()
        empty_istinbat = Physics.IstinbatAttentionMemory()
        return Physics.compare_scene_purpose_strategies(
            empty_scene, empty_hisban, empty_istinbat,
            "\u0644\u0645\u0627\u0630\u0627 \u0647\u0630\u0627 \u0627\u0644\u0633\u0624\u0627\u0644 \u0628\u0644\u0627 \u062f\u0644\u064a\u0644\u061f",
        )
    end

    if wanted == "purpose_only"
        checked_purpose = 0
        for rec in values(istinbat.records)
            _timed_out(deadline) && return nothing
            checked_purpose >= limit && break
            rec.relation_type == "purpose" || continue
            checked_purpose += 1
            prompt = _purpose_prompt(rec)
            if !isempty(prompt)
                empty_scene = Physics.SemanticSceneMemory()
                result = Physics.compare_scene_purpose_strategies(empty_scene, hisban, _single_record_memory(rec), prompt)
                (result.agreement == wanted && _displayable_result(result)) && return result
            end
        end
        return nothing
    end

    if wanted == "cooperative"
        seed = _find_seed_cooperative(scene_mem, hisban, istinbat)
        seed !== nothing && return seed
        indexed = _find_indexed_cooperative(scene_mem, hisban, istinbat; limit=limit, deadline=deadline)
        indexed !== nothing && return indexed
    elseif wanted == "scene_only"
        indexed = _find_indexed_scene_only(scene_mem, hisban; limit=limit, deadline=deadline)
        indexed !== nothing && return indexed
        return nothing
    end

    checked = 0
    for scene in scene_mem.scenes
        _timed_out(deadline) && return nothing
        checked >= limit && break
        prompt = _scene_prompt(scene)
        isempty(prompt) && continue
        checked += 1
        result = Physics.compare_scene_purpose_strategies(scene_mem, hisban, istinbat, prompt)
        (result.agreement == wanted && _displayable_result(result)) && return result
    end

    if wanted == "cooperative"
        checked_purpose = 0
        for rec in values(istinbat.records)
            _timed_out(deadline) && return nothing
            checked_purpose >= limit && break
            rec.relation_type == "purpose" || continue
            checked_purpose += 1
            prompt = _purpose_prompt(rec)
            isempty(prompt) && continue
            result = Physics.compare_scene_purpose_strategies(scene_mem, hisban, istinbat, prompt)
            (result.agreement == wanted && _displayable_result(result)) && return result
        end
    end

    return nothing
end

function _controlled_cooperative()
    hisban = Physics.SemanticCalculusMemory()
    Physics.learn_semantic_calculus_from_pair!(
        hisban,
        "Khalid hit the ball",
        "The ball moved away and changed position.",
    )
    scene_mem = Physics.SemanticSceneMemory()
    Physics.learn_semantic_scene_from_text!(
        scene_mem,
        hisban,
        "Khalid hit the ball with bat in yard before sunset.",
    )
    istinbat = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(istinbat, "Khalid hit the ball \u0644\u0643\u064a \u064a\u062a\u062d\u0631\u0643 ball.")
    return Physics.compare_scene_purpose_strategies(
        scene_mem, hisban, istinbat,
        "\u0644\u0645\u0627\u0630\u0627 Khalid hit the ball\u061f",
    )
end

function main()
    limit = _env_int("MIRNAN_BRIDGE_PROBE_LIMIT", 200)
    seconds = _env_int("MIRNAN_BRIDGE_PROBE_SECONDS", 45)
    scene_only_limit = min(limit, _env_int("MIRNAN_BRIDGE_SCENE_ONLY_LIMIT", 60))

    scene_path = joinpath(MODEL_DIR, "semantic_scenes.json")
    hisban_path = joinpath(MODEL_DIR, "al_hisban_al_dalali.json")
    istinbat_path = joinpath(MODEL_DIR, "al_istinbat.json")

    scene_mem = Physics.load_semantic_scenes(scene_path)
    hisban = Physics.load_semantic_calculus(hisban_path)
    istinbat = Physics.load_istinbat(istinbat_path)

    purpose_count = count(rec -> rec.relation_type == "purpose", values(istinbat.records))

    println("SemanticScene/Purpose bridge probe")
    println("semantic_scenes: $(length(scene_mem.scenes))")
    println("istinbat_records: $(length(istinbat.records))")
    println("purpose_records: $(purpose_count)")
    println("scan_limit: $(limit)")
    println("scene_only_limit: $(scene_only_limit)")
    println("time_limit_seconds: $(seconds)")
    println("indexed_bridge_search: enabled")
    if purpose_count == 0
        println("NOTE: trained al_istinbat has no purpose records; trained cooperative/purpose_only cases require rebuilding al_istinbat or retraining after RelationFrame learning.")
    end

    for wanted in ("cooperative", "scene_only", "purpose_only", "none")
        deadline = wanted == "none" ? Inf : _deadline(seconds)
        case_limit = wanted == "scene_only" ? scene_only_limit : limit
        result = _find_trained_case(scene_mem, hisban, istinbat; wanted=wanted, limit=case_limit, deadline=deadline)
        if result === nothing
            println("="^72)
            println("CASE: trained $(wanted)")
            if _timed_out(deadline)
                println("Search stopped by time limit before finding a trained-memory example.")
            else
                println("No trained-memory example found in scan window.")
            end
            if wanted == "cooperative"
                _print_seed_diagnostics(scene_mem, hisban, istinbat)
                _print_result("controlled cooperative sanity", _controlled_cooperative())
            elseif wanted == "scene_only"
                _print_result("controlled scene_only sanity", _controlled_scene_only())
            end
        else
            _print_result("trained $(wanted)", result)
        end
    end
end

main()
