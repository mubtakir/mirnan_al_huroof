"""
demo_semantic_scene_from_text.jl
Reads event sentences, extracts semantic scenes, then renders visual wave images.
"""

using Pkg
Pkg.activate(@__DIR__)

const DEMO_DIR = @__DIR__
const OUTPUT_DIR = joinpath(DEMO_DIR, "output")
const MIRNAN_SRC = normpath(joinpath(DEMO_DIR, "..", "src", "MirnanNew.jl"))
const MIRNAN_EXTRACTOR = Ref{Union{Nothing,Function}}(nothing)
const MIRNAN_EXTRACTOR_ATTEMPTED = Ref(false)
const MIRNAN_EXTRACTOR_NOTICE_SHOWN = Ref(false)

include(joinpath(DEMO_DIR, "src", "HoloPRNN_Vision.jl"))
using .HoloPRNN_Vision
using Images

const ACTION_HINTS = Dict(
    "hit" => ["moved", "away", "changed"],
    "push" => ["moved", "away", "changed"],
    "pushed" => ["moved", "away", "changed"],
    "broke" => ["pieces", "separated", "lost shape"],
    "break" => ["pieces", "separated", "lost shape"],
    "illuminated" => ["clear", "visible", "appearance"],
    "illuminate" => ["clear", "visible", "appearance"],
    "\u0636\u0631\u0628" => ["\u062d\u0631\u0643\u0629", "\u0627\u0628\u062a\u0639\u0627\u062f", "\u062a\u063a\u064a\u0631 \u0645\u0648\u0636\u0639"],
    "\u062f\u0641\u0639" => ["\u062d\u0631\u0643\u0629", "\u0627\u0628\u062a\u0639\u0627\u062f", "\u062a\u063a\u064a\u0631 \u0645\u0648\u0636\u0639"],
    "\u0643\u0633\u0631" => ["\u0627\u0646\u0641\u0635\u0627\u0644", "\u062a\u0644\u0641", "\u062a\u063a\u064a\u0631 \u0647\u064a\u0626\u0629"],
    "\u0623\u0636\u0627\u0621" => ["\u0648\u0636\u0648\u062d", "\u0638\u0647\u0648\u0631", "\u0627\u0646\u0643\u0634\u0627\u0641"],
    "\u0627\u0636\u0627\u0621" => ["\u0648\u0636\u0648\u062d", "\u0638\u0647\u0648\u0631", "\u0627\u0646\u0643\u0634\u0627\u0641"],
    "ضرب" => ["حركة", "ابتعاد", "تغير موضع"],
    "دفع" => ["حركة", "ابتعاد", "تغير موضع"],
    "كسر" => ["انفصال", "تلف", "تغير هيئة"],
    "اضاء" => ["وضوح", "ظهور", "انكشاف"],
    "أضاء" => ["وضوح", "ظهور", "انكشاف"],
)

function _clean_token(s)
    return strip(lowercase(replace(String(s), r"^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$" => "")))
end

function _fold_arabic_demo_text(s::AbstractString)
    text = lowercase(String(s))
    text = replace(text, '\u0640' => "")
    text = replace(text, r"[\u064B-\u065F\u0670]" => "")
    return replace(text,
        '\u0623' => '\u0627',
        '\u0625' => '\u0627',
        '\u0622' => '\u0627',
        '\u0649' => '\u064A',
        '\u0629' => '\u0647',
    )
end

function _is_time_word(tok::AbstractString)
    t = _fold_arabic_demo_text(tok)
    return t in Set(["dawn", "morning", "evening", "night", "sunset", "day",
                     "\u0641\u062c\u0631", "\u0635\u0628\u0627\u062d", "\u0645\u0633\u0627\u0621", "\u0644\u064a\u0644", "\u063a\u0631\u0648\u0628", "\u0646\u0647\u0627\u0631"])
end

function _scene_context_from_tail(toks::AbstractVector{<:AbstractString})
    patient = String[]
    instrument = ""
    place = ""
    time_marker = ""
    i = 1
    while i <= length(toks)
        tok = toks[i]
        folded = _fold_arabic_demo_text(tok)
        nxt = i < length(toks) ? toks[i + 1] : ""
        if startswith(folded, "\u0628\u0627\u0644") && length(tok) > 2
            instrument = replace(tok, r"^\u0628\u0627\u0644" => "\u0627\u0644")
        elseif tok in ["with", "using", "\u0628", "\u0628\u0648\u0627\u0633\u0637\u0629", "\u0628\u0627\u0633\u062a\u062e\u062f\u0627\u0645"] && !isempty(nxt)
            instrument = nxt
            i += 1
        elseif tok in ["before", "after", "during", "\u0642\u0628\u0644", "\u0628\u0639\u062f", "\u0627\u062b\u0646\u0627\u0621"] && !isempty(nxt)
            time_marker = string(tok, " ", nxt)
            i += 1
        elseif tok in ["in", "at", "inside", "\u0641\u064a", "\u0639\u0646\u062f", "\u062f\u0627\u062e\u0644", "\u0639\u0644\u0649"] && !isempty(nxt)
            if _is_time_word(nxt)
                time_marker = string(tok, " ", nxt)
            else
                place = nxt
            end
            i += 1
        elseif tok in ["on", "\u0639\u0646"] && !isempty(nxt)
            place = nxt
            i += 1
        else
            push!(patient, tok)
        end
        i += 1
    end
    return (patient=join(patient, " "), instrument=instrument, place=place, time_marker=time_marker)
end

function _fallback_scene_from_text(text::AbstractString)
    toks = [_clean_token(t) for t in split(String(text))]
    toks = [t for t in toks if !isempty(t)]
    idx = findfirst(t -> haskey(ACTION_HINTS, t), toks)
    if idx === nothing
        return (actor="", action="", patient="", effect_candidates=String[], confidence=0.0)
    end
    action = toks[idx]
    actor = idx > 1 ? join(toks[1:idx-1], " ") : ""
    context = idx < length(toks) ? _scene_context_from_tail(toks[idx+1:end]) : (patient="", instrument="", place="", time_marker="")
    return (
        actor = actor,
        action = action,
        patient = context.patient,
        instrument = context.instrument,
        place = context.place,
        time_marker = context.time_marker,
        effect_candidates = ACTION_HINTS[action],
        confidence = 0.65,
    )
end

function _notice_mirnan_unavailable(err)
    MIRNAN_EXTRACTOR_NOTICE_SHOWN[] && return nothing
    MIRNAN_EXTRACTOR_NOTICE_SHOWN[] = true
    println("Mirnan extractor unavailable in this environment; using lightweight fallback. Reason: $(typeof(err))")
    return nothing
end

function _load_mirnan_extractor()
    MIRNAN_EXTRACTOR_ATTEMPTED[] && return MIRNAN_EXTRACTOR[]
    MIRNAN_EXTRACTOR_ATTEMPTED[] = true
    isfile(MIRNAN_SRC) || return nothing
    try
        include(MIRNAN_SRC)
        mirnan = Base.invokelatest(getproperty, Main, :MirnanNew)
        physics = Base.invokelatest(getproperty, mirnan, :Physics)
        memory_ctor = Base.invokelatest(getproperty, physics, :SemanticCalculusMemory)
        extractor = Base.invokelatest(getproperty, physics, :extract_semantic_scene)
        MIRNAN_EXTRACTOR[] = text -> begin
            calculus = Base.invokelatest(memory_ctor)
            Base.invokelatest(extractor, calculus, text)
        end
        return MIRNAN_EXTRACTOR[]
    catch err
        _notice_mirnan_unavailable(err)
        return nothing
    end
end

function _mirnan_scene_from_text(text::AbstractString)
    extractor = _load_mirnan_extractor()
    extractor === nothing && return nothing
    try
        return extractor(text)
    catch err
        _notice_mirnan_unavailable(err)
        return nothing
    end
end

function _scene_field(scene, field::Symbol, default="")
    scene === nothing && return default
    hasproperty(scene, field) || return default
    value = getproperty(scene, field)
    value === nothing && return default
    isempty(strip(string(value))) && return default
    return value
end

function _scene_effects(scene, default::Vector{String})
    scene === nothing && return default
    hasproperty(scene, :effect_candidates) || return default
    value = getproperty(scene, :effect_candidates)
    value isa AbstractVector || return default
    return String[string(v) for v in value if !isempty(strip(string(v)))]
end

function scene_from_text(text::AbstractString)
    scene = _mirnan_scene_from_text(text)
    fallback = _fallback_scene_from_text(text)
    scene === nothing && return fallback
    action = string(_scene_field(scene, :action, fallback.action))
    isempty(strip(action)) && return fallback
    return (
        actor = string(_scene_field(scene, :actor, fallback.actor)),
        action = action,
        patient = string(_scene_field(scene, :patient, fallback.patient)),
        instrument = string(_scene_field(scene, :instrument, fallback.instrument)),
        place = string(_scene_field(scene, :place, fallback.place)),
        time_marker = string(_scene_field(scene, :time_marker, fallback.time_marker)),
        effect_candidates = _scene_effects(scene, fallback.effect_candidates),
        confidence = _scene_field(scene, :confidence, fallback.confidence),
    )
end

function _safe_name(i, text)
    base = lowercase(replace(String(text), r"[^\p{L}\p{N}]+" => "_"))
    base = replace(base, r"^_+|_+$" => "")
    isempty(base) && (base = "scene")
    length(base) > 32 && (base = first(base, 32))
    return "text_scene_$(i)_$(base).png"
end

function run_demo()
    println("=========================================================================")
    println("      HoloPRNN Vision - Semantic Scene From Text Demo")
    println("=========================================================================")

    mkpath(OUTPUT_DIR)
    prompts = isempty(ARGS) ? [
        "Khalid hit the ball",
        "The child broke the cup",
        "The lamp illuminated the room",
        "\u0636\u0631\u0628 \u062e\u0627\u0644\u062f \u0627\u0644\u0643\u0631\u0629",
        "\u0636\u0631\u0628 \u062e\u0627\u0644\u062f \u0627\u0644\u0643\u0631\u0629 \u0628\u0627\u0644\u0645\u0636\u0631\u0628 \u0641\u064a \u0627\u0644\u062d\u062f\u064a\u0642\u0629 \u0642\u0628\u0644 \u0627\u0644\u0641\u062c\u0631",
        "\u062f\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062d\u062c\u0631",
        "\u062f\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062d\u062c\u0631 \u0639\u0646 \u0627\u0644\u0637\u0631\u064a\u0642 \u0639\u0646\u062f \u0627\u0644\u0641\u062c\u0631",
        "\u0643\u0633\u0631 \u0627\u0644\u0637\u0641\u0644 \u0627\u0644\u0643\u0623\u0633",
        "\u0623\u0636\u0627\u0621 \u0627\u0644\u0645\u0635\u0628\u0627\u062d \u0627\u0644\u063a\u0631\u0641\u0629",
    ] : ARGS

    extractor = _load_mirnan_extractor()
    for (i, text) in enumerate(prompts)
        visual_scene = semantic_text_to_visual_scene(text; extractor=extractor, width=64, height=48)
        img = render_visual_scene(visual_scene)
        path = joinpath(OUTPUT_DIR, _safe_name(i, text))
        save(path, img)
        println("TEXT: $(text)")
        println("  action=$(visual_scene.action) objects=$(length(visual_scene.objects)) effects=$(join(visual_scene.effect_terms, ", "))")
        println("  visual=$(join([string(o.kind, ":", o.shape) for o in visual_scene.objects], ", "))")
        println("  saved $(path)")
    end

    println("=========================================================================")
end

run_demo()
