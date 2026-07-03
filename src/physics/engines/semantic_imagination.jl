module SemanticImagination

using JSON
using ..AlHisbanAlDalali: SemanticCalculusMemory, semantic_guidance, has_semantic_calculus

export SemanticScene, SemanticSceneMemory, SemanticSceneComparison,
       SemanticSceneAnswerComparison,
       extract_semantic_scene, scene_effect_terms,
       learn_semantic_scene_from_text!, train_semantic_scenes_from_texts!,
       has_semantic_scenes, select_semantic_scene, semantic_scene_diagnostic,
       semantic_scenes_to_dict, save_semantic_scenes, load_semantic_scenes,
       compare_semantic_scene_with_calculus, semantic_scene_comparison_diagnostic,
       semantic_scene_answer, compare_semantic_scene_strategies

const SEMANTIC_SCENES_VERSION = 1

const PHYSICAL_ACTIONS = Set([
    "ضرب", "دفع", "رمى", "حرك", "ركل", "صدم",
    "ضرب", "دفع", "رمى", "حرك", "ركل", "صدم",
    "\u0636\u0631\u0628", "\u062f\u0641\u0639", "\u0631\u0645\u0649", "\u062d\u0631\u0643", "\u0631\u0643\u0644", "\u0635\u062f\u0645",
    "hit", "push", "pushed", "throw", "threw", "kick", "kicked", "move", "moved",
])

const BREAK_ACTIONS = Set(["كسر", "حطم", "مزق", "break", "broke", "broken", "shatter", "shattered", "tear", "tore"])
const LIGHT_ACTIONS = Set(["أضاء", "أنار", "كشف", "يفتح", "illuminate", "illuminated", "lit", "reveal", "revealed"])

union!(BREAK_ACTIONS, Set([
    "\u0643\u0633\u0631", "\u062d\u0637\u0645", "\u0645\u0632\u0642",
]))
union!(LIGHT_ACTIONS, Set([
    "\u0627\u0636\u0627\u0621", "\u0627\u0646\u0627\u0631", "\u0643\u0634\u0641", "\u064a\u0641\u062a\u062d",
]))
setdiff!(LIGHT_ACTIONS, Set([
    "\u064a\u0641\u062a\u062d", "\u0641\u062a\u062d", "يفتح", "فتح", "ÙŠÙØªØ­",
]))

const STOPWORDS = Set([
    "في", "من", "عن", "على", "الى", "إلى", "مع", "ثم", "و", "ف",
    "هو", "هي", "هذا", "هذه", "ذلك", "تلك", "ان", "أن", "قد",
    "في", "من", "عن", "على", "الى", "إلى", "مع", "ثم", "و", "ف",
    "هو", "هي", "هذا", "هذه", "ذلك", "تلك", "ان", "أن", "قد",
    "\u0641\u064a", "\u0645\u0646", "\u0639\u0646", "\u0639\u0644\u0649", "\u0627\u0644\u0649", "\u0625\u0644\u0649", "\u0645\u0639", "\u062b\u0645", "\u0648", "\u0641",
    "\u0647\u0648", "\u0647\u064a", "\u0647\u0630\u0627", "\u0647\u0630\u0647", "\u0630\u0644\u0643", "\u062a\u0644\u0643", "\u0627\u0646", "\u0623\u0646", "\u0642\u062f",
    "the", "a", "an", "to", "of", "and", "then", "is", "are",
])

const EFFECT_BRIDGE_GROUPS = [
    ["move", "moved", "movement", "\u062d\u0631\u0643\u0629", "\u062a\u062d\u0631\u0643", "\u064a\u062a\u062d\u0631\u0643"],
    ["away", "far", "\u0627\u0628\u062a\u0639\u0627\u062f", "\u0627\u0628\u062a\u0639\u062f", "\u0627\u0628\u062a\u0639\u062f\u062a"],
    ["change", "changed", "position", "place", "\u062a\u063a\u064a\u0631", "\u0645\u0648\u0636\u0639", "\u0645\u0648\u0636\u0639\u0647\u0627", "\u0645\u0643\u0627\u0646"],
    ["clear", "visible", "visibility", "reveal", "revealed", "\u0648\u0636\u0648\u062d", "\u0638\u0647\u0648\u0631", "\u0627\u0646\u0643\u0634\u0627\u0641"],
    ["separate", "separated", "pieces", "piece", "lost", "shape", "\u0627\u0646\u0641\u0635\u0627\u0644", "\u062a\u0644\u0641", "\u0647\u064a\u0626\u0629", "\u062a\u063a\u064a\u0631", "\u0642\u0637\u0639"],
]

struct SemanticScene
    sentence::String
    actor::String
    action::String
    patient::String
    instrument::String
    place::String
    time_marker::String
    state_before::String
    state_after::String
    affect_tone::String
    effect_candidates::Vector{String}
    confidence::Float64
    guidance_relation::String
    source::String
end

SemanticScene(sentence::String, actor::String, action::String, patient::String,
              effect_candidates::Vector{String}, confidence::Float64,
              guidance_relation::String, source::String) =
    SemanticScene(sentence, actor, action, patient, "", "", "", "", "", "",
                  effect_candidates, confidence, guidance_relation, source)

mutable struct SemanticSceneMemory
    scenes::Vector{SemanticScene}
    max_scenes::Int
end

SemanticSceneMemory(; max_scenes::Int=5000) = SemanticSceneMemory(SemanticScene[], max_scenes)

struct SemanticSceneComparison
    prompt::String
    scene::Union{SemanticScene,Nothing}
    raw_guidance_terms::Vector{String}
    guidance_terms::Vector{String}
    scene_effect_terms::Vector{String}
    overlap_score::Float64
    agreement::String
    scene_confidence::Float64
    guidance_confidence::Float64
    guidance_relation::String
end

struct SemanticSceneAnswerComparison
    prompt::String
    scene_answer::String
    generate_answer::String
    memory_has_scene::Bool
    question_allowed::Bool
    agreement::String
    overlap_score::Float64
    scene_confidence::Float64
    guidance_confidence::Float64
end

const SCENE_INSTRUMENT_MARKERS = Set([
    "with", "using", "by",
    "\u0628\u0648\u0627\u0633\u0637\u0629", "\u0628\u0627\u0633\u062a\u062e\u062f\u0627\u0645",
])

const SCENE_PLACE_MARKERS = Set([
    "in", "inside", "at", "on", "near", "under", "above", "over",
    "\u0641\u064a", "\u062f\u0627\u062e\u0644", "\u0639\u0646\u062f", "\u0639\u0644\u0649", "\u0642\u0631\u0628", "\u062a\u062d\u062a", "\u0641\u0648\u0642",
])

const SCENE_TIME_MARKERS = Set([
    "before", "after", "during", "when", "while", "since",
    "\u0642\u0628\u0644", "\u0628\u0639\u062f", "\u0623\u062b\u0646\u0627\u0621", "\u0639\u0646\u062f\u0645\u0627", "\u062d\u064a\u0646", "\u0628\u064a\u0646\u0645\u0627", "\u0645\u0646\u0630",
])

const SCENE_PURPOSE_BOUNDARIES = Set([
    "to", "for",
    "\u0644\u0643\u064a", "\u0643\u064a", "\u0644\u0623\u062c\u0644", "\u0644\u0627\u062c\u0644",
    "\u0645\u0646", "\u0623\u062c\u0644", "\u0627\u062c\u0644", "\u0628\u063a\u064a\u0629", "\u0642\u0635\u062f",
])

const SCENE_DETAIL_BOUNDARIES = union(SCENE_INSTRUMENT_MARKERS, SCENE_PLACE_MARKERS,
                                      SCENE_TIME_MARKERS, SCENE_PURPOSE_BOUNDARIES)

const SCENE_EFFECT_NOISE = Set([
    "\u0639\u0644\u064a", "\u0639\u0644\u0649", "\u0644\u0627", "\u0648\u0644\u0627", "\u0643\u0644",
    "\u062c\u0648\u0627\u0628", "\u0633\u0648\u0627\u0644", "\u0633\u0624\u0627\u0644",
    "\u0628\u064a\u0646", "\u0627\u0644\u0627\u0646\u0633\u0627\u0646", "\u0627\u0646\u0633\u0627\u0646", "\u0627\u0644\u0625\u0646\u0633\u0627\u0646", "\u0625\u0646\u0633\u0627\u0646",
    "no", "not", "answer", "question", "all",
])

const SCENE_ATTACHED_B_PREFIX_EXCLUSIONS = Set([
    "\u0628\u0639\u062f", "\u0628\u064a\u0646", "\u0628\u064a\u0646\u0645\u0627", "\u0628\u0633\u0628\u0628",
    "\u0628\u0644\u0627", "\u0628\u062f\u0648\u0646", "\u0628\u063a\u064a\u0631", "\u0628\u0644",
])

function _clean_word(w::AbstractString)
    s = lowercase(strip(String(w)))
    s = replace(s, r"[\u064B-\u065F\u0670]" => "")
    s = replace(s, 'أ' => 'ا', 'إ' => 'ا', 'آ' => 'ا', 'ى' => 'ي')
    s = replace(s, 'أ' => 'ا', 'إ' => 'ا', 'آ' => 'ا', 'ى' => 'ي')
    s = replace(s, '\u0623' => '\u0627', '\u0625' => '\u0627', '\u0622' => '\u0627', '\u0649' => '\u064a')
    s = replace(s, r"^[\W_]+|[\W_]+$" => "")
    return strip(s)
end

function _attached_instrument_value(tok::AbstractString)
    t = _clean_word(tok)
    isempty(t) && return ""
    t in SCENE_ATTACHED_B_PREFIX_EXCLUSIONS && return ""
    startswith(t, "\u0628\u0627\u0644") && ncodeunits(t) > ncodeunits("\u0628\u0627\u0644") && return t[nextind(t, firstindex(t)):end]
    startswith(t, "\u0628") && ncodeunits(t) > ncodeunits("\u0628") + 1 && return t[nextind(t, firstindex(t)):end]
    return ""
end

function _tokens(text::AbstractString)
    out = String[]
    for raw in split(String(text))
        t = _clean_word(raw)
        isempty(t) && continue
        t in STOPWORDS && continue
        push!(out, t)
    end
    return out
end

function _bridge_expand_tokens(tokens)
    expanded = Set{String}()
    for tok in tokens
        clean = _clean_word(tok)
        isempty(clean) && continue
        push!(expanded, clean)
        for group in EFFECT_BRIDGE_GROUPS
            clean_group = Set(_clean_word(item) for item in group)
            if clean in clean_group
                union!(expanded, clean_group)
            end
        end
    end
    return expanded
end

function _sentences(text::AbstractString)
    parts = split(String(text), r"(?<=[\.\!\?\u061F\u06D4])\s+|[\r\n]+")
    return String[strip(p) for p in parts if !isempty(strip(p))]
end

function _is_action(w::AbstractString)
    t = _clean_word(w)
    t in PHYSICAL_ACTIONS && return true
    t in BREAK_ACTIONS && return true
    t in LIGHT_ACTIONS && return true
    startswith(t, "يضرب") && return true
    startswith(t, "تضرب") && return true
    startswith(t, "يدفع") && return true
    startswith(t, "تدفع") && return true
    startswith(t, "يتحرك") && return true
    startswith(t, "تتحرك") && return true
    startswith(t, "يضرب") && return true
    startswith(t, "تضرب") && return true
    startswith(t, "يدفع") && return true
    startswith(t, "تدفع") && return true
    startswith(t, "يتحرك") && return true
    startswith(t, "تتحرك") && return true
    startswith(t, "\u064a\u0636\u0631\u0628") && return true
    startswith(t, "\u062a\u0636\u0631\u0628") && return true
    startswith(t, "\u064a\u062f\u0641\u0639") && return true
    startswith(t, "\u062a\u062f\u0641\u0639") && return true
    startswith(t, "\u064a\u062a\u062d\u0631\u0643") && return true
    startswith(t, "\u062a\u062a\u062d\u0631\u0643") && return true
    startswith(t, "\u064a\u0643\u0633\u0631") && return true
    startswith(t, "\u062a\u0643\u0633\u0631") && return true
    startswith(t, "\u064a\u0636\u064a\u0621") && return true
    startswith(t, "\u062a\u0636\u064a\u0621") && return true
    startswith(t, "\u0627\u0636\u0627\u0621") && return true
    return false
end

function _parse_event(sentence::AbstractString)
    toks = _tokens(sentence)
    isempty(toks) && return "", "", ""
    action_index = findfirst(_is_action, toks)
    action_index === nothing && return "", "", ""

    action = toks[action_index]
    actor = action_index > 1 ? join(toks[1:action_index-1], " ") :
            (length(toks) >= action_index + 1 ? toks[action_index+1] : "")
    patient_start = action_index == 1 ? action_index + 2 : action_index + 1
    patient_terms = String[]
    if patient_start <= length(toks)
        for t in toks[patient_start:end]
            t in SCENE_DETAIL_BOUNDARIES && break
            !isempty(_attached_instrument_value(t)) && break
            _is_action(t) && break
            push!(patient_terms, t)
        end
    end
    patient = join(patient_terms, " ")
    return actor, action, patient
end

function _physical_priors(action::AbstractString, patient::AbstractString)
    a = _clean_word(action)
    isempty(a) && return String[]
    if occursin("\u0636\u0631\u0628", a) || occursin("\u062f\u0641\u0639", a)
        return ["\u062d\u0631\u0643\u0629", "\u0627\u0628\u062a\u0639\u0627\u062f", "\u062a\u063a\u064a\u0631 \u0645\u0648\u0636\u0639"]
    end
    if a in PHYSICAL_ACTIONS || occursin("ضرب", a) || occursin("دفع", a)
        return ["حركة", "ابتعاد", "تغير موضع"]
    elseif a in BREAK_ACTIONS
        return ["انفصال", "تلف", "تغير هيئة"]
    elseif a in LIGHT_ACTIONS
        return ["ظهور", "انكشاف", "وضوح"]
    end
    return isempty(strip(String(patient))) ? String[] : ["تغير في $(patient)"]
end

function _guidance_terms(mem::SemanticCalculusMemory, sentence::AbstractString)
    !has_semantic_calculus(mem) && return String[], "", 0.0
    guidance = semantic_guidance(mem, sentence; relation="semantic_continuation", limit=10)
    active = get(guidance, "active", false) == true
    active || return String[], "", 0.0
    terms = String[String(t) for t in get(guidance, "target_terms", String[])]
    relation = String(get(guidance, "relation", ""))
    confidence = Float64(get(guidance, "confidence", 0.0))
    return terms, relation, confidence
end

function _dedupe_effects(items::Vector{String}, action::String, patient::String)
    blocked = Set(vcat(_tokens(action), _tokens(patient)))
    out = String[]
    seen = Set{String}()
    for item in items
        clean = _clean_word(item)
        isempty(clean) && continue
        clean in STOPWORDS && continue
        clean in blocked && continue
        clean in SCENE_EFFECT_NOISE && continue
        clean in seen && continue
        push!(seen, clean)
        push!(out, clean)
    end
    return out
end

function _scene_terms(scene::SemanticScene)
    terms = String[]
    append!(terms, _tokens(scene.actor))
    append!(terms, _tokens(scene.action))
    append!(terms, _tokens(scene.patient))
    append!(terms, _tokens(scene.instrument))
    append!(terms, _tokens(scene.place))
    append!(terms, _tokens(scene.time_marker))
    append!(terms, _tokens(scene.state_before))
    append!(terms, _tokens(scene.state_after))
    append!(terms, _tokens(scene.affect_tone))
    append!(terms, scene.effect_candidates)
    return Set(_clean_word(t) for t in terms if !isempty(_clean_word(t)))
end

function _scene_overlap(query::Set{String}, scene::SemanticScene)
    isempty(query) && return 0.0
    terms = _scene_terms(scene)
    isempty(terms) && return 0.0
    return length(intersect(query, terms)) / max(1, length(query))
end

function _actions_compatible(prompt_action::AbstractString, scene_action::AbstractString)
    pa = _clean_word(prompt_action)
    sa = _clean_word(scene_action)
    isempty(pa) && return true
    isempty(sa) && return false
    pa == sa && return true
    pf = _action_family(pa)
    sf = _action_family(sa)
    (!isempty(pf) && pf == sf) && return true
    return occursin(pa, sa) || occursin(sa, pa)
end

function _action_family(action::AbstractString)
    a = _clean_word(action)
    isempty(a) && return ""
    if a in BREAK_ACTIONS || occursin("\u0643\u0633\u0631", a) || occursin("break", a) || occursin("broke", a)
        return "break"
    end
    if a in LIGHT_ACTIONS || occursin("\u0627\u0636\u0627", a) || occursin("\u0636\u064a\u0621", a) ||
       occursin("illuminat", a) || occursin("light", a) || occursin("reveal", a)
        return "light"
    end
    if a in PHYSICAL_ACTIONS || occursin("\u0636\u0631\u0628", a) || occursin("\u062f\u0641\u0639", a) ||
       occursin("hit", a) || occursin("push", a) || occursin("move", a)
        return "motion"
    end
    return ""
end

function _detail_after_marker(toks::Vector{String}, markers::Set{String}; max_terms::Int=3)
    for (i, tok) in enumerate(toks)
        if markers === SCENE_INSTRUMENT_MARKERS
            attached = _attached_instrument_value(tok)
            !isempty(attached) && return attached
        end
        tok in markers || continue
        vals = String[]
        for j in (i + 1):length(toks)
            t = toks[j]
            t in SCENE_DETAIL_BOUNDARIES && break
            _is_action(t) && break
            isempty(t) && continue
            push!(vals, t)
            length(vals) >= max_terms && break
        end
        isempty(vals) || return join(vals, " ")
    end
    return ""
end

function _state_pair_for_action(action::AbstractString, arabic::Bool)
    family = _action_family(action)
    if family == "motion"
        return arabic ? ("\u0633\u0643\u0648\u0646", "\u062d\u0631\u0643\u0629 \u0648\u062a\u063a\u064a\u0631 \u0645\u0648\u0636\u0639", "\u0645\u062d\u0627\u064a\u062f") :
                        ("stable", "moved and changed position", "neutral")
    elseif family == "break"
        return arabic ? ("\u0633\u0644\u0627\u0645\u0629", "\u062a\u0644\u0641 \u0648\u0627\u0646\u0641\u0635\u0627\u0644", "\u0627\u0636\u0637\u0631\u0627\u0628\u064a") :
                        ("whole", "damaged and separated", "disruptive")
    elseif family == "light"
        return arabic ? ("\u062e\u0641\u0627\u0621", "\u0638\u0647\u0648\u0631 \u0648\u0648\u0636\u0648\u062d", "\u0643\u0627\u0634\u0641") :
                        ("unclear", "visible and clear", "revealing")
    end
    return "", "", ""
end

function _scene_detail_fields(sentence::AbstractString, action::AbstractString, patient::AbstractString)
    toks = _tokens(sentence)
    instrument = _detail_after_marker(toks, SCENE_INSTRUMENT_MARKERS)
    place = _detail_after_marker(toks, SCENE_PLACE_MARKERS)
    time_marker = _detail_after_marker(toks, SCENE_TIME_MARKERS)
    arabic = _has_arabic(sentence) || _has_arabic(action) || _has_arabic(patient)
    state_before, state_after, affect_tone = _state_pair_for_action(action, arabic)
    return instrument, place, time_marker, state_before, state_after, affect_tone
end

function _term_overlap(left_terms::Vector{String}, right_terms::Vector{String})
    left = Set{String}()
    right = Set{String}()
    for term in left_terms
        union!(left, _bridge_expand_tokens(_tokens(term)))
    end
    for term in right_terms
        union!(right, _bridge_expand_tokens(_tokens(term)))
    end
    (isempty(left) || isempty(right)) && return 0.0
    return length(intersect(left, right)) / max(1, min(length(left), length(right)))
end

function _filter_guidance_terms_for_scene(scene::Union{SemanticScene,Nothing},
                                          guidance_terms::Vector{String})
    scene === nothing && return guidance_terms
    anchors = Set{String}()
    for term in vcat(_tokens(scene.actor), _tokens(scene.action), _tokens(scene.patient),
                     _tokens(scene.instrument), _tokens(scene.place), _tokens(scene.time_marker),
                     _tokens(scene.state_before), _tokens(scene.state_after),
                     _tokens(scene.affect_tone), scene.effect_candidates)
        for tok in _tokens(term)
            union!(anchors, _bridge_expand_tokens([tok]))
        end
    end
    isempty(anchors) && return guidance_terms

    filtered = String[]
    for term in guidance_terms
        toks = _bridge_expand_tokens(_tokens(term))
        !isempty(intersect(anchors, toks)) && push!(filtered, term)
    end
    return filtered
end

function extract_semantic_scene(mem::SemanticCalculusMemory, sentence::AbstractString)
    actor, action, patient = _parse_event(sentence)
    instrument, place, time_marker, state_before, state_after, affect_tone =
        _scene_detail_fields(sentence, action, patient)
    learned_terms, relation, confidence = _guidance_terms(mem, sentence)
    priors = _physical_priors(action, patient)
    effects = _dedupe_effects(vcat(learned_terms, priors), action, patient)
    source = !isempty(learned_terms) ? "al_hisban_al_dalali" :
             (!isempty(effects) ? "physical_semantic_prior" : "none")
    event_bonus = (!isempty(action) ? 0.15 : 0.0) + (!isempty(patient) ? 0.10 : 0.0)
    final_confidence = clamp(max(confidence, isempty(effects) ? 0.0 : 0.35) + event_bonus, 0.0, 1.0)
    return SemanticScene(String(sentence), actor, action, patient,
                         instrument, place, time_marker, state_before, state_after, affect_tone,
                         effects,
                         final_confidence, relation, source)
end

function scene_effect_terms(scene::SemanticScene)
    limit = max(1, length(scene.effect_candidates))
    return _clean_effect_list(scene.effect_candidates; limit=limit)
end

function has_semantic_scenes(mem::SemanticSceneMemory)
    return !isempty(mem.scenes)
end

function _store_scene!(mem::SemanticSceneMemory, scene::SemanticScene)
    isempty(scene.action) && return false
    isempty(scene.effect_candidates) && return false
    push!(mem.scenes, scene)
    if length(mem.scenes) > mem.max_scenes
        deleteat!(mem.scenes, 1:(length(mem.scenes) - mem.max_scenes))
    end
    return true
end

function learn_semantic_scene_from_text!(mem::SemanticSceneMemory,
                                         calculus::SemanticCalculusMemory,
                                         text::AbstractString)
    learned = 0
    for sentence in _sentences(text)
        scene = extract_semantic_scene(calculus, sentence)
        _store_scene!(mem, scene) && (learned += 1)
    end
    return learned
end

function train_semantic_scenes_from_texts!(mem::SemanticSceneMemory,
                                           calculus::SemanticCalculusMemory,
                                           texts;
                                           max_items::Int=typemax(Int))
    learned = 0
    for text in texts
        learned >= max_items && break
        learned += learn_semantic_scene_from_text!(mem, calculus, text)
    end
    return learned
end

function semantic_scenes_to_dict(mem::SemanticSceneMemory)
    return Dict{String,Any}(
        "version" => SEMANTIC_SCENES_VERSION,
        "max_scenes" => mem.max_scenes,
        "n_scenes" => length(mem.scenes),
        "scenes" => [Dict{String,Any}(
            "sentence" => scene.sentence,
            "actor" => scene.actor,
            "action" => scene.action,
            "patient" => scene.patient,
            "instrument" => scene.instrument,
            "place" => scene.place,
            "time_marker" => scene.time_marker,
            "state_before" => scene.state_before,
            "state_after" => scene.state_after,
            "affect_tone" => scene.affect_tone,
            "effect_candidates" => scene.effect_candidates,
            "confidence" => scene.confidence,
            "guidance_relation" => scene.guidance_relation,
            "source" => scene.source,
        ) for scene in mem.scenes],
    )
end

function save_semantic_scenes(mem::SemanticSceneMemory, path::String)
    mkpath(dirname(path))
    tmp = string(path, ".tmp")
    open(tmp, "w") do io
        JSON.print(io, semantic_scenes_to_dict(mem))
    end
    mv(tmp, path; force=true)
    return path
end

function load_semantic_scenes(path::String)
    mem = SemanticSceneMemory()
    isfile(path) || return mem
    filesize(path) == 0 && return mem
    data = try
        JSON.parsefile(path)
    catch e
        @warn "Could not load semantic scene memory: $path - $e"
        return mem
    end
    max_scenes = try
        Int(get(data, "max_scenes", mem.max_scenes))
    catch
        mem.max_scenes
    end
    mem = SemanticSceneMemory(max_scenes=max_scenes)
    for item in get(data, "scenes", Any[])
        item isa AbstractDict || continue
        effects = String[]
        for e in get(item, "effect_candidates", Any[])
            push!(effects, String(e))
        end
        scene = SemanticScene(
            String(get(item, "sentence", "")),
            String(get(item, "actor", "")),
            String(get(item, "action", "")),
            String(get(item, "patient", "")),
            String(get(item, "instrument", "")),
            String(get(item, "place", "")),
            String(get(item, "time_marker", "")),
            String(get(item, "state_before", "")),
            String(get(item, "state_after", "")),
            String(get(item, "affect_tone", "")),
            effects,
            Float64(get(item, "confidence", 0.0)),
            String(get(item, "guidance_relation", "")),
            String(get(item, "source", "")),
        )
        _store_scene!(mem, scene)
    end
    return mem
end

function select_semantic_scene(mem::SemanticSceneMemory, prompt::AbstractString;
                               min_score::Float64=0.18,
                               min_overlap::Float64=0.10)
    query = Set(_tokens(prompt))
    isempty(query) && return nothing
    best = nothing
    best_score = -Inf
    for scene in mem.scenes
        overlap = _scene_overlap(query, scene)
        overlap >= min_overlap || continue
        score = 0.72 * overlap + 0.28 * scene.confidence
        score >= min_score || continue
        if score > best_score
            best = scene
            best_score = score
        end
    end
    return best
end

function semantic_scene_diagnostic(mem::SemanticSceneMemory, prompt::AbstractString)
    scene = select_semantic_scene(mem, prompt)
    scene === nothing && return "لا يوجد مشهد دلالي مناسب."
    effects = isempty(scene.effect_candidates) ? "لا أثر مستخلص" : join(scene.effect_candidates, "، ")
    lines = String[]
    push!(lines, "الفاعل: $(scene.actor)")
    push!(lines, "الفعل: $(scene.action)")
    push!(lines, "المتأثر: $(scene.patient)")
    push!(lines, "الأثر الدلالي: $(effects)")
    push!(lines, "الثقة: $(round(scene.confidence; digits=3))")
    push!(lines, "المصدر: $(scene.source)")
    push!(lines, "المثال: $(scene.sentence)")
    return join(lines, "\n")
end

function compare_semantic_scene_with_calculus(scene_mem::SemanticSceneMemory,
                                              calculus::SemanticCalculusMemory,
                                              prompt::AbstractString)
    scene = select_semantic_scene(scene_mem, prompt)
    _, prompt_action, _ = _parse_event(prompt)
    if scene !== nothing && !_actions_compatible(prompt_action, scene.action)
        scene = nothing
    end
    if scene === nothing && isempty(prompt_action)
        return SemanticSceneComparison(String(prompt), nothing, String[], String[], String[],
                                       0.0, "none", 0.0, 0.0, "")
    end
    raw_guidance_terms, relation, guidance_confidence = _guidance_terms(calculus, prompt)
    if scene === nothing && isempty(prompt_action)
        raw_guidance_terms = String[]
        guidance_confidence = 0.0
        relation = ""
    end
    guidance_terms = _filter_guidance_terms_for_scene(scene, raw_guidance_terms)
    effects = scene === nothing ? String[] : scene_effect_terms(scene)
    overlap = _term_overlap(effects, guidance_terms)
    agreement = if scene === nothing && isempty(guidance_terms)
        "none"
    elseif scene === nothing
        "calculus_only"
    elseif isempty(guidance_terms)
        "scene_only"
    elseif overlap >= 0.34
        "aligned"
    elseif overlap > 0.0
        "partial"
    else
        "divergent"
    end
    scene_confidence = scene === nothing ? 0.0 : scene.confidence
    return SemanticSceneComparison(String(prompt), scene, raw_guidance_terms,
                                   guidance_terms, effects,
                                   overlap, agreement, scene_confidence,
                                   guidance_confidence, relation)
end

function semantic_scene_comparison_diagnostic(scene_mem::SemanticSceneMemory,
                                              calculus::SemanticCalculusMemory,
                                              prompt::AbstractString)
    cmp = compare_semantic_scene_with_calculus(scene_mem, calculus, prompt)
    lines = String[]
    push!(lines, "agreement: $(cmp.agreement)")
    push!(lines, "overlap: $(round(cmp.overlap_score; digits=3))")
    push!(lines, "scene_confidence: $(round(cmp.scene_confidence; digits=3))")
    push!(lines, "guidance_confidence: $(round(cmp.guidance_confidence; digits=3))")
    push!(lines, "guidance_relation: $(cmp.guidance_relation)")
    push!(lines, "scene_effects: $(join(cmp.scene_effect_terms, ", "))")
    push!(lines, "guidance_terms: $(join(cmp.guidance_terms, ", "))")
    push!(lines, "raw_guidance_terms: $(join(cmp.raw_guidance_terms, ", "))")
    if cmp.scene !== nothing
        push!(lines, "scene_sentence: $(cmp.scene.sentence)")
        push!(lines, "scene_instrument: $(cmp.scene.instrument)")
        push!(lines, "scene_place: $(cmp.scene.place)")
        push!(lines, "scene_time: $(cmp.scene.time_marker)")
        push!(lines, "scene_state_before: $(cmp.scene.state_before)")
        push!(lines, "scene_state_after: $(cmp.scene.state_after)")
        push!(lines, "scene_affect_tone: $(cmp.scene.affect_tone)")
    end
    return join(lines, "\n")
end

function _has_arabic(text::AbstractString)
    return occursin(r"[\u0600-\u06FF]", String(text))
end

function _is_semantic_scene_question(prompt::AbstractString)
    s = lowercase(String(prompt))
    occursin("what happens", s) && return true
    occursin("what happened", s) && return true
    occursin("what is the effect", s) && return true
    occursin("what effect", s) && return true
    occursin("how does", s) && occursin("affect", s) && return true
    occursin("\u0645\u0627\u0630\u0627 \u064a\u062d\u062f\u062b", s) && return true
    occursin("\u0645\u0627 \u0627\u062b\u0631", s) && return true
    occursin("\u0645\u0627 \u0623\u062b\u0631", s) && return true
    occursin("\u0643\u064a\u0641 \u064a\u0624\u062b\u0631", s) && return true
    return false
end

function _canonical_effects_for_action(action::AbstractString, arabic::Bool)
    family = _action_family(action)
    if family == "motion"
        return arabic ? ["\u062d\u0631\u0643\u0629", "\u0627\u0628\u062a\u0639\u0627\u062f", "\u062a\u063a\u064a\u0631 \u0645\u0648\u0636\u0639"] :
                        ["movement", "away", "position change"]
    elseif family == "break"
        return arabic ? ["\u0627\u0646\u0641\u0635\u0627\u0644", "\u062a\u0644\u0641", "\u062a\u063a\u064a\u0631 \u0647\u064a\u0626\u0629"] :
                        ["separation", "damage", "shape change"]
    elseif family == "light"
        return arabic ? ["\u0638\u0647\u0648\u0631", "\u0627\u0646\u0643\u0634\u0627\u0641", "\u0648\u0636\u0648\u062d"] :
                        ["visibility", "reveal", "clarity"]
    end
    return String[]
end

function _clean_effect_list(items::Vector{String}; limit::Int=3)
    blocked = Set([
        "its", "the", "and", "became", "from", "into",
        "no", "not", "answer", "all", "yes", "question",
        "\u0644\u0627", "\u0648\u0644\u0627", "\u0644\u064a\u0633", "\u062c\u0648\u0627\u0628", "\u0633\u0624\u0627\u0644", "\u0643\u0644", "\u0639\u0644\u0649", "\u0639\u0644\u064a",
        "Ù„Ø§", "ÙˆÙ„Ø§", "Ù„ÙŠØ³", "Ø¬ÙˆØ§Ø¨", "Ø³Ø¤Ø§Ù„", "ÙƒÙ„", "Ø¹Ù„Ù‰", "Ø¹Ù„ÙŠ",
    ])
    out = String[]
    seen = Set{String}()
    for item in items
        clean = _clean_word(item)
        isempty(clean) && continue
        clean in STOPWORDS && continue
        clean in blocked && continue
        clean in SCENE_EFFECT_NOISE && continue
        clean in seen && continue
        push!(seen, clean)
        push!(out, clean)
        length(out) >= limit && break
    end
    return out
end

function _scene_context_phrase(scene::SemanticScene, arabic::Bool)
    parts = String[]
    if !isempty(scene.instrument)
        push!(parts, arabic ? "\u0628\u0623\u062f\u0627\u0629 $(scene.instrument)" : "using $(scene.instrument)")
    end
    if !isempty(scene.place)
        push!(parts, arabic ? "\u0641\u064a $(scene.place)" : "in $(scene.place)")
    end
    if !isempty(scene.time_marker)
        push!(parts, arabic ? "\u0648\u0642\u062a $(scene.time_marker)" : "before/around $(scene.time_marker)")
    end
    return join(parts, arabic ? "\u060c " : ", ")
end

function _scene_state_phrase(scene::SemanticScene, arabic::Bool)
    (isempty(scene.state_before) || isempty(scene.state_after)) && return ""
    if arabic
        return "\u0648\u064a\u0646\u062a\u0642\u0644 \u0627\u0644\u0645\u062a\u0623\u062b\u0631 \u0645\u0646 $(scene.state_before) \u0625\u0644\u0649 $(scene.state_after)"
    end
    return "and the affected thing shifts from $(scene.state_before) to $(scene.state_after)"
end

function _natural_arabic_scene_answer(scene::SemanticScene, effects::Vector{String})
    action = strip(scene.action)
    patient = strip(scene.patient)
    actor = strip(scene.actor)
    scene_text = join([action, patient, actor], " ")
    _has_arabic(scene_text) || return ""
    isempty(action) && isempty(patient) && return ""

    target = isempty(patient) ? "\u0627\u0644\u0645\u062a\u0623\u062b\u0631" : patient
    event = strip(isempty(actor) ? "$(action) $(target)" : "$(action) $(actor) $(target)")
    clauses = String[event]

    if !isempty(scene.state_before) && !isempty(scene.state_after)
        push!(clauses, "\u0641\u062a\u063a\u064a\u0631 \u062d\u0627\u0644 $(target) \u0645\u0646 $(scene.state_before) \u0625\u0644\u0649 $(scene.state_after)")
    end
    if !isempty(effects)
        push!(clauses, "\u0648\u0638\u0647\u0631\u062a \u0622\u062b\u0627\u0631 \u0645\u062b\u0644 $(join(effects, "\u060c "))")
    end

    return join(clauses, "\u061b ") * "."
end

function _format_semantic_scene_answer(cmp::SemanticSceneComparison)
    scene = cmp.scene
    scene === nothing && return ""
    use_arabic = _has_arabic(cmp.prompt) || _has_arabic(scene.action) || _has_arabic(scene.patient)
    effects = _canonical_effects_for_action(scene.action, use_arabic)
    isempty(effects) && (effects = _clean_effect_list(scene_effect_terms(scene); limit=3))
    isempty(effects) && (effects = _clean_effect_list(cmp.guidance_terms; limit=3))
    isempty(effects) && return ""

    action = isempty(scene.action) ? "\u0627\u0644\u0641\u0639\u0644" : scene.action
    patient = isempty(scene.patient) ? "\u0627\u0644\u0634\u064a\u0621" : scene.patient
    context = _scene_context_phrase(scene, use_arabic)
    state = _scene_state_phrase(scene, use_arabic)
    if use_arabic
        natural = _natural_arabic_scene_answer(scene, effects)
        isempty(natural) || return natural
        prefix = "\u0639\u0646\u062f $(action) $(patient)"
        isempty(context) || (prefix *= "\u060c $(context)")
        suffix = isempty(state) ? "" : "\u060c $(state)"
        return "$(prefix) \u064a\u0638\u0647\u0631 \u0623\u062b\u0631 \u062f\u0644\u0627\u0644\u064a \u0645\u062b\u0644 $(join(effects, "\u060c "))$(suffix)."
    end
    prefix = "When $(action) affects $(patient)"
    isempty(context) || (prefix *= ", $(context)")
    suffix = isempty(state) ? "" : ", $(state)"
    return "$(prefix), the semantic effect includes $(join(effects, ", "))$(suffix)."
end

function semantic_scene_answer(scene_mem::SemanticSceneMemory,
                               calculus::SemanticCalculusMemory,
                               prompt::AbstractString)
    _is_semantic_scene_question(prompt) || return ""
    cmp = compare_semantic_scene_with_calculus(scene_mem, calculus, prompt)
    cmp.scene === nothing && return ""
    known_family = !isempty(_action_family(cmp.scene.action))
    (cmp.agreement in ("aligned", "partial") || (cmp.agreement == "scene_only" && known_family)) || return ""
    (cmp.overlap_score > 0.0 || (cmp.agreement == "scene_only" && known_family)) || return ""
    isempty(cmp.scene_effect_terms) && isempty(cmp.guidance_terms) && return ""
    return _format_semantic_scene_answer(cmp)
end

function compare_semantic_scene_strategies(scene_mem::SemanticSceneMemory,
                                           calculus::SemanticCalculusMemory,
                                           generate_func::Function,
                                           prompt::AbstractString)
    s = String(prompt)
    scene_ans = semantic_scene_answer(scene_mem, calculus, s)
    generate_ans = try
        String(generate_func(s))
    catch
        ""
    end
    cmp = compare_semantic_scene_with_calculus(scene_mem, calculus, s)
    return SemanticSceneAnswerComparison(
        s,
        scene_ans,
        generate_ans,
        cmp.scene !== nothing,
        _is_semantic_scene_question(s),
        cmp.agreement,
        cmp.overlap_score,
        cmp.scene_confidence,
        cmp.guidance_confidence,
    )
end

end # module SemanticImagination
