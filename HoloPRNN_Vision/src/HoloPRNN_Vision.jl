"""
HoloPRNN_Vision (مرنان البصر)
Core module implementing Phase-Resonant Neural Network (PRNN) physical computing
for image processing, classification, inpainting, style transfer, and crystallization.
"""
module HoloPRNN_Vision

using LinearAlgebra, Random, Statistics, Images, FileIO, Serialization

export OscillatorParams, StyleProfile,
       VisualObject, VisualScene,
       image_to_wavefield, wavefield_to_image,
       simulate_wave_field!, run_inpainting,
       run_style_transfer, run_pattern_crystallization,
       PhasNetClassifier, train_classifier!, predict_classifier!,
       extract_style_profile, apply_style,
       save_style_profile, load_style_profile,
       compute_classification_loss, predict_batch,
       semantic_scene_to_visual_scene, visual_scene_to_wavefield,
       render_visual_scene, semantic_text_to_visual_scene

# --- Parameters ---
Base.@kwdef struct OscillatorParams
    μ::Float64 = 1.0       # Saturation limit
    g_inh::Float64 = 0.5   # Global competitive inhibition strength
    γ::Float64 = 2.0       # Local synaptic fatigue strength
    τ_a::Float64 = 1.5     # Fatigue decay time constant
    dt::Float64 = 0.03     # Time step
end

# --- Style Profile (Reusable Texture/Phase Coupling) ---

"""
    StyleProfile

Captures the local phase-relationship pattern of a style image.
- `K_h[y, x] = exp(i * (θ[y, x+1] - θ[y, x]))`  — horizontal phase coupling
- `K_v[y, x] = exp(i * (θ[y+1, x] - θ[y, x]))`  — vertical phase coupling
- `params` — oscillator parameters used during extraction

This profile can be saved, loaded, and applied to any content wave field
via `apply_style`.
"""
struct StyleProfile
    H::Int
    W::Int
    K_h::Matrix{ComplexF64}
    K_v::Matrix{ComplexF64}
    params::OscillatorParams
end

# --- Semantic Scene -> Visual Scene bridge ---

Base.@kwdef struct VisualObject
    label::String
    kind::String
    x::Float64
    y::Float64
    radius::Float64
    hue::Float64
    amplitude::Float64 = 1.0
    shape::String = "disc"
end

Base.@kwdef struct VisualScene
    width::Int = 32
    height::Int = 32
    objects::Vector{VisualObject} = VisualObject[]
    action::String = ""
    effect_terms::Vector{String} = String[]
    motion::Tuple{Float64, Float64} = (0.0, 0.0)
    confidence::Float64 = 0.0
end

function _assert_same_size(name::AbstractString, value, expected::Tuple{Int, Int})
    size(value) == expected || throw(DimensionMismatch("$name has size $(size(value)); expected $expected"))
    return nothing
end

function _assert_positive_dim(name::AbstractString, value::Int)
    value > 0 || throw(ArgumentError("$name must be positive, got $value"))
    return nothing
end

function _assert_style_profile(profile::StyleProfile)
    _assert_positive_dim("StyleProfile.H", profile.H)
    _assert_positive_dim("StyleProfile.W", profile.W)
    _assert_same_size("StyleProfile.K_h", profile.K_h, (profile.H, profile.W))
    _assert_same_size("StyleProfile.K_v", profile.K_v, (profile.H, profile.W))
    return nothing
end

_is_finite_field(z) = all(isfinite, real.(z)) && all(isfinite, imag.(z))

function _complex_noise(rng::AbstractRNG, dims...)
    return (randn(rng, Float64, dims...) .+ im .* randn(rng, Float64, dims...)) ./ sqrt(2.0)
end

function _scene_get(scene, field::Symbol, default="")
    hasproperty(scene, field) || return default
    value = getproperty(scene, field)
    value === nothing && return default
    return value
end

function _scene_string(scene, field::Symbol)
    return String(strip(string(_scene_get(scene, field, ""))))
end

function _scene_effects(scene)
    effects = _scene_get(scene, :effect_candidates, String[])
    effects isa AbstractVector || return String[]
    return String[string(e) for e in effects if !isempty(strip(string(e)))]
end

function _fold_arabic_visual_text(s::AbstractString)
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

function _action_family_visual(action::AbstractString, effects::Vector{String})
    s = _fold_arabic_visual_text(string(action, " ", join(effects, " ")))
    if occursin("break", s) || occursin("broke", s) ||
       occursin("\u0643\u0633\u0631", s) || occursin("\u062a\u0644\u0641", s) ||
       occursin("\u0627\u0646\u0641\u0635\u0627\u0644", s) || occursin("\u0634\u0638\u0627\u064a\u0627", s) ||
       occursin("\u0642\u0637\u0639", s) || occursin("\u0632\u062c\u0627\u062c", s)
        return "break"
    elseif occursin("light", s) || occursin("illumin", s) || occursin("clear", s) ||
           occursin("\u0627\u0636\u0627\u0621", s) || occursin("\u0627\u0636\u0627\u0621\u0647", s) ||
           occursin("\u0646\u0648\u0631", s) || occursin("\u0636\u0648\u0621", s) ||
           occursin("\u0648\u0636\u0648\u062d", s) || occursin("\u0638\u0647\u0648\u0631", s) ||
           occursin("\u0627\u0646\u0643\u0634\u0627\u0641", s) || occursin("\u0645\u0635\u0628\u0627\u062d", s)
        return "light"
    elseif occursin("hit", s) || occursin("push", s) || occursin("move", s) ||
           occursin("\u0636\u0631\u0628", s) || occursin("\u062f\u0641\u0639", s) ||
           occursin("\u062d\u0631\u0643", s) || occursin("\u062a\u062d\u0631\u0643", s) ||
           occursin("\u0627\u0628\u062a\u0639\u0627\u062f", s) || occursin("\u0627\u0628\u0639\u0627\u062f", s)
        return "motion"
    end
    return "generic"
end

function _visual_object(label::AbstractString, kind::AbstractString, x::Float64, y::Float64;
                        radius::Float64=0.12, hue::Float64=0.0, amplitude::Float64=1.0,
                        shape::AbstractString="disc")
    return VisualObject(String(label), String(kind), clamp(x, 0.0, 1.0), clamp(y, 0.0, 1.0),
                        clamp(radius, 0.01, 0.5), mod(hue, 360.0), clamp(amplitude, 0.0, 2.0),
                        String(shape))
end

function _shape_for_patient(patient::AbstractString, family::AbstractString)
    p = lowercase(String(patient))
    if occursin("ball", p) || occursin("كرة", p) || occursin("ÙƒØ±Ø©", p)
        return "ball"
    elseif occursin("cup", p) || occursin("كأس", p) || occursin("كاس", p) || occursin("ÙƒØ£Ø³", p) || occursin("ÙƒØ§Ø³", p)
        return "cup"
    elseif occursin("room", p) || occursin("غرفة", p) || occursin("ØºØ±ÙØ©", p)
        return "room"
    elseif family == "break"
        return "fragile"
    elseif family == "light"
        return "area"
    end
    return "disc"
end

function _shape_for_actor(actor::AbstractString, family::AbstractString)
    a = _fold_arabic_visual_text(actor)
    if occursin("lamp", a) || occursin("\u0645\u0635\u0628\u0627\u062d", a)
        return "lamp"
    elseif occursin("stone", a) || occursin("\u062d\u062c\u0631", a)
        return "stone"
    elseif family == "light"
        return "source"
    end
    return "person"
end

function _shape_for_patient(patient::AbstractString, family::AbstractString)
    p = _fold_arabic_visual_text(patient)
    if occursin("ball", p) || occursin("\u0643\u0631\u0647", p) || occursin("ÙƒØ±Ø©", p) || occursin("Ã™Æ’Ã˜Â±Ã˜Â©", p)
        return "ball"
    elseif occursin("cup", p) || occursin("\u0643\u0627\u0633", p) || occursin("ÙƒØ£Ø³", p) || occursin("ÙƒØ§Ø³", p) || occursin("Ã™Æ’Ã˜Â£Ã˜Â³", p) || occursin("Ã™Æ’Ã˜Â§Ã˜Â³", p)
        return "cup"
    elseif occursin("room", p) || occursin("\u063a\u0631\u0641\u0647", p) || occursin("ØºØ±ÙØ©", p) || occursin("Ã˜ÂºÃ˜Â±Ã™ÂÃ˜Â©", p)
        return "room"
    elseif occursin("stone", p) || occursin("\u062d\u062c\u0631", p)
        return "stone"
    elseif occursin("lamp", p) || occursin("\u0645\u0635\u0628\u0627\u062d", p)
        return "lamp"
    elseif family == "break"
        return "fragile"
    elseif family == "light"
        return "area"
    end
    return "disc"
end

function _shape_for_instrument(instrument::AbstractString)
    i = _fold_arabic_visual_text(instrument)
    if occursin("bat", i) || occursin("\u0645\u0636\u0631\u0628", i) || occursin("\u0639\u0635\u0627", i)
        return "bat"
    elseif occursin("pen", i) || occursin("\u0642\u0644\u0645", i)
        return "pen"
    elseif occursin("key", i) || occursin("\u0645\u0641\u062a\u0627\u062d", i)
        return "key"
    elseif occursin("knife", i) || occursin("\u0633\u0643\u064a\u0646", i)
        return "blade"
    end
    return "tool"
end

function _place_visual_profile(place::AbstractString)
    p = _fold_arabic_visual_text(place)
    if occursin("garden", p) || occursin("\u062d\u062f\u064a\u0642\u0647", p) || occursin("\u0634\u062c\u0631", p)
        return ("garden", 118.0, 0.28)
    elseif occursin("road", p) || occursin("path", p) || occursin("\u0637\u0631\u064a\u0642", p)
        return ("road", 32.0, 0.28)
    elseif occursin("room", p) || occursin("house", p) || occursin("\u063a\u0631\u0641\u0647", p) || occursin("\u0628\u064a\u062a", p)
        return ("room", 210.0, 0.24)
    elseif occursin("school", p) || occursin("\u0645\u062f\u0631\u0633\u0647", p)
        return ("school", 48.0, 0.26)
    elseif occursin("sea", p) || occursin("\u0628\u062d\u0631", p)
        return ("water", 195.0, 0.26)
    elseif occursin("mountain", p) || occursin("\u062c\u0628\u0644", p)
        return ("mountain", 92.0, 0.26)
    end
    return ("place", 150.0, 0.22)
end

function _time_visual_profile(time_marker::AbstractString)
    t = _fold_arabic_visual_text(time_marker)
    if occursin("dawn", t) || occursin("\u0641\u062c\u0631", t) || occursin("\u0634\u0631\u0648\u0642", t)
        return ("dawn", 42.0, 0.34)
    elseif occursin("morning", t) || occursin("\u0635\u0628\u0627\u062d", t)
        return ("morning", 55.0, 0.32)
    elseif occursin("sunset", t) || occursin("evening", t) || occursin("\u063a\u0631\u0648\u0628", t) || occursin("\u0645\u0633\u0627\u0621", t)
        return ("evening", 26.0, 0.32)
    elseif occursin("night", t) || occursin("\u0644\u064a\u0644", t)
        return ("night", 235.0, 0.30)
    elseif occursin("day", t) || occursin("\u0646\u0647\u0627\u0631", t)
        return ("day", 58.0, 0.30)
    end
    return ("time", 52.0, 0.24)
end

const TEXT_SCENE_ACTION_HINTS = Dict(
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
    "\u0627\u0636\u0627\u0621" => ["\u0648\u0636\u0648\u062d", "\u0638\u0647\u0648\u0631", "\u0627\u0646\u0643\u0634\u0627\u0641"],
)

function _clean_text_scene_token(s)
    return strip(_fold_arabic_visual_text(replace(String(s), r"^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$" => "")))
end

function _text_scene_time_word(tok::AbstractString)
    t = _fold_arabic_visual_text(tok)
    return t in Set(["dawn", "morning", "evening", "night", "sunset", "day",
                     "\u0641\u062c\u0631", "\u0635\u0628\u0627\u062d", "\u0645\u0633\u0627\u0621", "\u0644\u064a\u0644", "\u063a\u0631\u0648\u0628", "\u0646\u0647\u0627\u0631"])
end

function _text_scene_context_from_tail(toks::AbstractVector{<:AbstractString})
    patient = String[]
    instrument = ""
    place = ""
    time_marker = ""
    i = 1
    while i <= length(toks)
        tok = String(toks[i])
        folded = _fold_arabic_visual_text(tok)
        nxt = i < length(toks) ? String(toks[i + 1]) : ""
        if startswith(folded, "\u0628\u0627\u0644") && length(tok) > 2
            instrument = replace(tok, r"^\u0628\u0627\u0644" => "\u0627\u0644")
        elseif tok in ["with", "using", "\u0628", "\u0628\u0648\u0627\u0633\u0637\u0629", "\u0628\u0627\u0633\u062a\u062e\u062f\u0627\u0645"] && !isempty(nxt)
            instrument = nxt
            i += 1
        elseif tok in ["before", "after", "during", "\u0642\u0628\u0644", "\u0628\u0639\u062f", "\u0627\u062b\u0646\u0627\u0621"] && !isempty(nxt)
            time_marker = string(tok, " ", nxt)
            i += 1
        elseif tok in ["in", "at", "inside", "\u0641\u064a", "\u0639\u0646\u062f", "\u062f\u0627\u062e\u0644", "\u0639\u0644\u0649"] && !isempty(nxt)
            if _text_scene_time_word(nxt)
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

function _fallback_semantic_scene_from_text(text::AbstractString)
    toks = [_clean_text_scene_token(t) for t in split(String(text))]
    toks = [t for t in toks if !isempty(t)]
    idx = findfirst(t -> haskey(TEXT_SCENE_ACTION_HINTS, t), toks)
    if idx === nothing
        return (actor="", action="", patient="", instrument="", place="", time_marker="",
                effect_candidates=String[], confidence=0.0)
    end
    action = toks[idx]
    actor = idx > 1 ? join(toks[1:idx-1], " ") : ""
    context = idx < length(toks) ? _text_scene_context_from_tail(toks[idx+1:end]) :
                                   (patient="", instrument="", place="", time_marker="")
    return (
        actor = actor,
        action = action,
        patient = context.patient,
        instrument = context.instrument,
        place = context.place,
        time_marker = context.time_marker,
        effect_candidates = TEXT_SCENE_ACTION_HINTS[action],
        confidence = 0.65,
    )
end

function _scene_field_or(scene, field::Symbol, default="")
    scene === nothing && return default
    hasproperty(scene, field) || return default
    value = getproperty(scene, field)
    value === nothing && return default
    isempty(strip(string(value))) && return default
    return value
end

function _scene_effects_or(scene, default::Vector{String})
    scene === nothing && return default
    hasproperty(scene, :effect_candidates) || return default
    value = getproperty(scene, :effect_candidates)
    value isa AbstractVector || return default
    effects = String[string(v) for v in value if !isempty(strip(string(v)))]
    return isempty(effects) ? default : effects
end

"""
    semantic_text_to_visual_scene(text; extractor=nothing, width=32, height=32)

Builds a `VisualScene` directly from text.  If `extractor` is supplied, it is
called with the text and may return a Mirnan `SemanticScene`; any missing visual
context is filled by the lightweight text parser.
"""
function semantic_text_to_visual_scene(text::AbstractString; extractor=nothing,
                                       width::Int=32, height::Int=32)
    fallback = _fallback_semantic_scene_from_text(text)
    scene = nothing
    if extractor !== nothing
        try
            scene = extractor(text)
        catch
            scene = nothing
        end
    end
    action = string(_scene_field_or(scene, :action, fallback.action))
    isempty(strip(action)) && (scene = nothing; action = fallback.action)
    merged = (
        actor = string(_scene_field_or(scene, :actor, fallback.actor)),
        action = action,
        patient = string(_scene_field_or(scene, :patient, fallback.patient)),
        instrument = string(_scene_field_or(scene, :instrument, fallback.instrument)),
        place = string(_scene_field_or(scene, :place, fallback.place)),
        time_marker = string(_scene_field_or(scene, :time_marker, fallback.time_marker)),
        effect_candidates = _scene_effects_or(scene, fallback.effect_candidates),
        confidence = _scene_field_or(scene, :confidence, fallback.confidence),
    )
    return semantic_scene_to_visual_scene(merged; width=width, height=height)
end

# --- Image <-> Wavefield Conversions ---

"""
    semantic_scene_to_visual_scene(scene; width=32, height=32) -> VisualScene

Converts a semantic event-like object into a small visual layout.
The input only needs fields named like `actor`, `action`, `patient`,
`effect_candidates`, and `confidence`; this keeps the bridge independent from
Mirnan's concrete `SemanticScene` type.
"""
function semantic_scene_to_visual_scene(scene; width::Int=32, height::Int=32)
    _assert_positive_dim("width", width)
    _assert_positive_dim("height", height)

    actor = _scene_string(scene, :actor)
    action = _scene_string(scene, :action)
    patient = _scene_string(scene, :patient)
    instrument = _scene_string(scene, :instrument)
    place = _scene_string(scene, :place)
    time_marker = _scene_string(scene, :time_marker)
    effects = _scene_effects(scene)
    confidence_raw = _scene_get(scene, :confidence, 0.0)
    confidence = confidence_raw isa Number ? Float64(confidence_raw) : 0.0

    family = _action_family_visual(action, effects)
    objects = VisualObject[]
    if !isempty(place)
        place_shape, place_hue, place_amp = _place_visual_profile(place)
        push!(objects, _visual_object(place, "place", 0.50, 0.80; radius=0.26, hue=place_hue, amplitude=place_amp, shape=place_shape))
    end
    if !isempty(time_marker)
        time_shape, time_hue, time_amp = _time_visual_profile(time_marker)
        push!(objects, _visual_object(time_marker, "time", 0.14, 0.14; radius=0.12, hue=time_hue, amplitude=time_amp, shape=time_shape))
    end
    if !isempty(actor)
        actor_shape = _shape_for_actor(actor, family)
        actor_hue = actor_shape == "lamp" ? 55.0 : actor_shape == "stone" ? 35.0 : 215.0
        push!(objects, _visual_object(actor, "actor", 0.28, 0.55; radius=0.11, hue=actor_hue, amplitude=0.85, shape=actor_shape))
    end
    if !isempty(instrument)
        instrument_shape = _shape_for_instrument(instrument)
        instrument_hue = instrument_shape == "bat" ? 28.0 : instrument_shape == "pen" ? 260.0 : 44.0
        push!(objects, _visual_object(instrument, "instrument", 0.42, 0.48; radius=0.055, hue=instrument_hue, amplitude=0.78, shape=instrument_shape))
    end
    if !isempty(patient)
        patient_hue = family == "break" ? 35.0 : family == "light" ? 55.0 : 12.0
        patient_radius = family == "light" ? 0.22 : 0.12
        patient_shape = _shape_for_patient(patient, family)
        push!(objects, _visual_object(patient, "patient", 0.70, 0.55; radius=patient_radius, hue=patient_hue, amplitude=1.0, shape=patient_shape))
    end
    if family == "light"
        push!(objects, _visual_object("light", "effect", 0.50, 0.35; radius=0.18, hue=58.0, amplitude=0.65, shape="glow"))
    elseif family == "break"
        push!(objects, _visual_object("fragment", "effect", 0.62, 0.42; radius=0.045, hue=25.0, amplitude=0.8, shape="fragment"))
        push!(objects, _visual_object("fragment", "effect", 0.78, 0.68; radius=0.045, hue=25.0, amplitude=0.8, shape="fragment"))
    end

    motion = family == "motion" ? (0.30, 0.0) :
             family == "break" ? (0.12, 0.10) :
             family == "light" ? (0.0, -0.12) : (0.0, 0.0)
    return VisualScene(width, height, objects, action, effects, motion, clamp(confidence, 0.0, 1.0))
end

function _paint_disc!(z::Matrix{ComplexF64}, obj::VisualObject)
    H, W = size(z)
    cx = 1 + obj.x * (W - 1)
    cy = 1 + obj.y * (H - 1)
    sigma = max(obj.radius * min(H, W), 1.0)
    phase = obj.hue * pi / 180.0
    wave = obj.amplitude * exp(im * phase)
    for y in 1:H, x in 1:W
        d2 = (x - cx)^2 + (y - cy)^2
        amp = exp(-d2 / (2 * sigma^2))
        z[y, x] += amp * wave
    end
    return z
end

function _paint_line!(z::Matrix{ComplexF64}, x1::Float64, y1::Float64, x2::Float64, y2::Float64,
                      hue::Float64, amplitude::Float64; width::Float64=1.8)
    H, W = size(z)
    phase = hue * pi / 180.0
    wave = amplitude * exp(im * phase)
    steps = max(ceil(Int, hypot(x2 - x1, y2 - y1) * max(W, H) * 2), 1)
    for i in 0:steps
        t = i / steps
        cx = 1 + ((1 - t) * x1 + t * x2) * (W - 1)
        cy = 1 + ((1 - t) * y1 + t * y2) * (H - 1)
        for y in 1:H, x in 1:W
            d2 = (x - cx)^2 + (y - cy)^2
            z[y, x] += exp(-d2 / (2 * width^2)) * wave
        end
    end
    return z
end

function _paint_ring!(z::Matrix{ComplexF64}, obj::VisualObject; thickness::Float64=1.2)
    H, W = size(z)
    cx = 1 + obj.x * (W - 1)
    cy = 1 + obj.y * (H - 1)
    radius_px = max(obj.radius * min(H, W), 1.0)
    phase = obj.hue * pi / 180.0
    wave = obj.amplitude * exp(im * phase)
    for y in 1:H, x in 1:W
        d = abs(hypot(x - cx, y - cy) - radius_px)
        z[y, x] += exp(-(d^2) / (2 * thickness^2)) * wave
    end
    return z
end

function _paint_person!(z::Matrix{ComplexF64}, obj::VisualObject)
    _paint_disc!(z, _visual_object(obj.label, obj.kind, obj.x, obj.y - obj.radius * 0.9;
                                   radius=obj.radius * 0.45, hue=obj.hue, amplitude=obj.amplitude, shape="disc"))
    _paint_line!(z, obj.x, obj.y - obj.radius * 0.35, obj.x, obj.y + obj.radius * 0.75, obj.hue, obj.amplitude; width=1.5)
    _paint_line!(z, obj.x, obj.y + obj.radius * 0.05, obj.x - obj.radius * 0.75, obj.y + obj.radius * 0.30, obj.hue, obj.amplitude * 0.8; width=1.2)
    _paint_line!(z, obj.x, obj.y + obj.radius * 0.05, obj.x + obj.radius * 0.75, obj.y + obj.radius * 0.30, obj.hue, obj.amplitude * 0.8; width=1.2)
    return z
end

function _paint_cup!(z::Matrix{ComplexF64}, obj::VisualObject)
    r = obj.radius
    _paint_line!(z, obj.x - r, obj.y - r * 0.75, obj.x - r * 0.55, obj.y + r * 0.85, obj.hue, obj.amplitude; width=1.5)
    _paint_line!(z, obj.x + r, obj.y - r * 0.75, obj.x + r * 0.55, obj.y + r * 0.85, obj.hue, obj.amplitude; width=1.5)
    _paint_line!(z, obj.x - r * 0.55, obj.y + r * 0.85, obj.x + r * 0.55, obj.y + r * 0.85, obj.hue, obj.amplitude; width=1.5)
    _paint_line!(z, obj.x - r * 1.05, obj.y - r * 0.75, obj.x + r * 1.05, obj.y - r * 0.75, obj.hue, obj.amplitude * 0.7; width=1.0)
    return z
end

function _paint_room!(z::Matrix{ComplexF64}, obj::VisualObject)
    r = obj.radius
    _paint_line!(z, obj.x - r, obj.y - r, obj.x + r, obj.y - r, obj.hue, obj.amplitude * 0.75; width=1.2)
    _paint_line!(z, obj.x + r, obj.y - r, obj.x + r, obj.y + r, obj.hue, obj.amplitude * 0.75; width=1.2)
    _paint_line!(z, obj.x + r, obj.y + r, obj.x - r, obj.y + r, obj.hue, obj.amplitude * 0.75; width=1.2)
    _paint_line!(z, obj.x - r, obj.y + r, obj.x - r, obj.y - r, obj.hue, obj.amplitude * 0.75; width=1.2)
    _paint_disc!(z, _visual_object(obj.label, obj.kind, obj.x, obj.y; radius=r * 0.65, hue=obj.hue, amplitude=obj.amplitude * 0.25, shape="disc"))
    return z
end

function _paint_fragment!(z::Matrix{ComplexF64}, obj::VisualObject)
    r = obj.radius
    _paint_line!(z, obj.x - r, obj.y + r, obj.x, obj.y - r, obj.hue, obj.amplitude; width=1.0)
    _paint_line!(z, obj.x, obj.y - r, obj.x + r, obj.y + r * 0.4, obj.hue, obj.amplitude; width=1.0)
    _paint_line!(z, obj.x + r, obj.y + r * 0.4, obj.x - r, obj.y + r, obj.hue, obj.amplitude; width=1.0)
    return z
end

function _paint_glow!(z::Matrix{ComplexF64}, obj::VisualObject)
    _paint_disc!(z, obj)
    _paint_ring!(z, _visual_object(obj.label, obj.kind, obj.x, obj.y;
                                   radius=obj.radius * 1.15, hue=obj.hue, amplitude=obj.amplitude * 0.65, shape="ring"))
    return z
end

function _paint_visual_object!(z::Matrix{ComplexF64}, obj::VisualObject)
    obj.shape == "person" && return _paint_person!(z, obj)
    obj.shape == "ball" && return _paint_ring!(z, obj)
    obj.shape == "cup" && return _paint_cup!(z, obj)
    obj.shape == "room" && return _paint_room!(z, obj)
    obj.shape == "fragment" && return _paint_fragment!(z, obj)
    obj.shape == "glow" && return _paint_glow!(z, obj)
    return _paint_disc!(z, obj)
end

function _paint_motion!(z::Matrix{ComplexF64}, scene::VisualScene)
    isempty(scene.objects) && return z
    actor_idx = findfirst(o -> o.kind == "actor", scene.objects)
    patient_idx = findfirst(o -> o.kind == "patient", scene.objects)
    (actor_idx === nothing || patient_idx === nothing) && return z

    H, W = size(z)
    actor = scene.objects[actor_idx]
    patient = scene.objects[patient_idx]
    phase = 190.0 * pi / 180.0
    wave = 0.42 * exp(im * phase)
    x1, y1 = 1 + actor.x * (W - 1), 1 + actor.y * (H - 1)
    x2 = 1 + (patient.x + scene.motion[1]) * (W - 1)
    y2 = 1 + (patient.y + scene.motion[2]) * (H - 1)
    steps = max(W, H)
    for i in 0:steps
        t = i / steps
        cx = (1 - t) * x1 + t * x2
        cy = (1 - t) * y1 + t * y2
        for y in 1:H, x in 1:W
            d2 = (x - cx)^2 + (y - cy)^2
            z[y, x] += exp(-d2 / 3.0) * wave
        end
    end
    return z
end

"""
    visual_scene_to_wavefield(scene::VisualScene) -> Matrix{ComplexF64}

Renders a `VisualScene` into a complex wave field suitable for HoloPRNN
post-processing or conversion to an image.
"""
function visual_scene_to_wavefield(scene::VisualScene)
    _assert_positive_dim("scene.width", scene.width)
    _assert_positive_dim("scene.height", scene.height)
    z = zeros(ComplexF64, scene.height, scene.width)
    for obj in scene.objects
        _paint_visual_object!(z, obj)
    end
    _paint_motion!(z, scene)
    for obj in scene.objects
        if obj.shape == "ball" && scene.motion != (0.0, 0.0)
            ghost = _visual_object(obj.label, obj.kind, obj.x + scene.motion[1], obj.y + scene.motion[2];
                                   radius=obj.radius * 0.75, hue=obj.hue, amplitude=obj.amplitude * 0.45, shape="ball")
            _paint_ring!(z, ghost)
        end
    end
    max_amp = maximum(abs.(z))
    max_amp > 1.0 && (z ./= max_amp)
    return z
end

"""
    render_visual_scene(scene::VisualScene) -> Matrix{RGB{N0f8}}

Converts a `VisualScene` directly to an RGB image.
"""
render_visual_scene(scene::VisualScene) = wavefield_to_image(visual_scene_to_wavefield(scene))

"""
    image_to_wavefield(img::AbstractMatrix{<:Colorant}) -> Matrix{ComplexF64}

Converts an RGB or Grayscale image to a 2D complex wave field.
- Amplitude represents brightness (HSV Value).
- Phase represents color (HSV Hue in radians).
"""
function image_to_wavefield(img::AbstractMatrix{<:Colorant})
    H, W = size(img)
    z = zeros(ComplexF64, H, W)
    for y in 1:H, x in 1:W
        c = HSV(img[y, x])
        amp = Float64(c.v)
        # Hue is in [0, 360], convert to [0, 2π]
        phase = Float64(c.h) * pi / 180.0
        z[y, x] = amp * exp(im * phase)
    end
    return z
end

"""
    wavefield_to_image(z::Matrix{ComplexF64}; saturation=1.0) -> Matrix{RGB{N0f8}}

Converts a 2D complex wave field back to an RGB image.
- Hue is extracted from the angle/phase.
- Value (brightness) is extracted from the amplitude.
"""
function wavefield_to_image(z::Matrix{ComplexF64}; saturation=1.0)
    H, W = size(z)
    img = Matrix{RGB{N0f8}}(undef, H, W)
    for y in 1:H, x in 1:W
        val = clamp(abs(z[y, x]), 0.0, 1.0)
        hue_rad = angle(z[y, x])
        hue_deg = mod(hue_rad * 180.0 / pi, 360.0)
        c_hsv = HSV(hue_deg, saturation, val)
        img[y, x] = RGB(c_hsv)
    end
    return img
end

# --- 2D Laplacian (Spatial Coupling) ---

function laplacian2d(z::Matrix{ComplexF64})
    H, W = size(z)
    L = zeros(ComplexF64, H, W)
    for y in 1:H, x in 1:W
        val = 0.0 + 0.0im
        count = 0
        if y > 1; val += z[y-1, x]; count += 1; end
        if y < H; val += z[y+1, x]; count += 1; end
        if x > 1; val += z[y, x-1]; count += 1; end
        if x < W; val += z[y, x+1]; count += 1; end
        L[y, x] = val - count * z[y, x]
    end
    return L
end

# --- Physical Wave Simulator ---

"""
    simulate_wave_field!(z::Matrix{ComplexF64}, a::Matrix{Float64}, omega::Matrix{Float64},
                         params::OscillatorParams; D_spatial=0.1, steps=50, clamped_mask=nothing)

Simulates the 2D wave field using Stuart-Landau non-linear dynamics with:
- Local spatial coupling (Laplacian diffusion)
- Global competitive inhibition
- Local synaptic fatigue (adaptation)
- Boundary/Pixel clamping support
"""
function simulate_wave_field!(z::Matrix{ComplexF64}, a::Matrix{Float64}, omega::Matrix{Float64},
                             params::OscillatorParams; D_spatial::Float64=0.1, steps::Int=50,
                             clamped_mask::Union{AbstractMatrix{Bool}, Nothing}=nothing)
    H, W = size(z)
    _assert_same_size("a", a, (H, W))
    _assert_same_size("omega", omega, (H, W))
    clamped_mask !== nothing && _assert_same_size("clamped_mask", clamped_mask, (H, W))
    steps >= 0 || throw(ArgumentError("steps must be non-negative, got $steps"))
    dz = zeros(ComplexF64, H, W)
    
    for _ in 1:steps
        L = laplacian2d(z)
        global_activity = sum(abs2.(z)) / (H * W)
        
        for y in 1:H, x in 1:W
            if clamped_mask !== nothing && clamped_mask[y, x]
                continue
            end
            
            # Stuart-Landau oscillator + competitive inhibition + spatial coupling
            dz[y, x] = (params.μ - a[y, x] - params.g_inh * global_activity - abs2(z[y, x]) + im * omega[y, x]) * z[y, x] + D_spatial * L[y, x]
        end
        
        z .+= params.dt .* dz
        _is_finite_field(z) || throw(DomainError(z, "wave field became non-finite; reduce dt, coupling, or steps"))
        # Update local fatigue
        @. a += params.dt * (-a + params.γ * abs2(z)) / params.τ_a
    end
    return z
end

# --- Inpainting Engine ---

"""
    run_inpainting(img_corrupted::Matrix{<:Colorant}, mask::Matrix{Bool}, params::OscillatorParams;
                   D_spatial=0.2, steps=300) -> Matrix{RGB{N0f8}}

Reconstructs missing image pixels (where mask is true) using wave propagation.
Clamps the visible pixels (where mask is false) and lets the missing region evolve.
"""
function run_inpainting(img_corrupted::Matrix{<:Colorant}, mask::Matrix{Bool}, params::OscillatorParams;
                        D_spatial::Float64=0.2, steps::Int=300, seed::Int=42)
    H, W = size(img_corrupted)
    _assert_same_size("mask", mask, (H, W))
    z = image_to_wavefield(img_corrupted)
    a = zeros(Float64, H, W)
    
    # Intrinsic frequencies (gradient based to encourage edge continuation)
    omega = zeros(Float64, H, W)
    # Add minor noise to break symmetry
    rng = MersenneTwister(seed)
    omega .+= (rand(rng, H, W) .- 0.5) .* 0.01
    
    # Clamped mask is the inverse of the missing mask (visible pixels are clamped)
    clamped_mask = .!mask
    
    # Run simulation
    simulate_wave_field!(z, a, omega, params; D_spatial=D_spatial, steps=steps, clamped_mask=clamped_mask)
    
    return wavefield_to_image(z)
end

# --- Style Transfer Engine ---

"""
    extract_style_profile(img::Matrix{<:Colorant}, params=OscillatorParams()) -> StyleProfile

Extracts a reusable `StyleProfile` from a style reference image.
Computes the local horizontal and vertical phase-relationship matrices
(anisotropic coupling) without running any simulation.

The profile can be saved, loaded, and applied to any content wave field
via `apply_style`.
"""
function extract_style_profile(img::Matrix{<:Colorant}, params::OscillatorParams=OscillatorParams())
    H, W = size(img)
    H > 0 && W > 0 || throw(ArgumentError("img must not be empty"))
    z = image_to_wavefield(img)
    
    K_h = zeros(ComplexF64, H, W)
    K_v = zeros(ComplexF64, H, W)
    for y in 1:H, x in 1:W
        if x < W
            K_h[y, x] = exp(im * (angle(z[y, x+1]) - angle(z[y, x])))
        end
        if y < H
            K_v[y, x] = exp(im * (angle(z[y+1, x]) - angle(z[y, x])))
        end
    end
    return StyleProfile(H, W, K_h, K_v, params)
end

"""
    apply_style(z_content::Matrix{ComplexF64}, profile::StyleProfile;
                D_spatial=0.1, steps=200, inplace=false) -> Matrix{ComplexF64}

Evolves a content wave field under a style profile's anisotropic phase coupling.
Returns the styled complex wave field (call `wavefield_to_image` to render).

Set `inplace=true` to mutate `z_content` directly and avoid allocation.
"""
function apply_style(z_content::Matrix{ComplexF64}, profile::StyleProfile;
                     D_spatial::Float64=0.1, steps::Int=200, inplace::Bool=false)
    _assert_style_profile(profile)
    H, W = size(z_content)
    H == profile.H && W == profile.W || throw(DimensionMismatch(
        "content wave field is $(size(z_content)), but StyleProfile expects ($(profile.H), $(profile.W))"))
    steps >= 0 || throw(ArgumentError("steps must be non-negative, got $steps"))
    
    z = inplace ? z_content : copy(z_content)
    a = zeros(Float64, H, W)
    dz = zeros(ComplexF64, H, W)
    params = profile.params
    
    for _ in 1:steps
        global_activity = sum(abs2.(z)) / (H * W)
        
        for y in 1:H, x in 1:W
            coupling = 0.0 + 0.0im
            count = 0
            if x < W
                coupling += profile.K_h[y, x] * z[y, x+1]
                count += 1
            end
            if x > 1
                coupling += conj(profile.K_h[y, x-1]) * z[y, x-1]
                count += 1
            end
            if y < H
                coupling += profile.K_v[y, x] * z[y+1, x]
                count += 1
            end
            if y > 1
                coupling += conj(profile.K_v[y-1, x]) * z[y-1, x]
                count += 1
            end
            
            if count > 0
                coupling = (coupling / count) - z[y, x]
            end
            
            dz[y, x] = (params.μ - a[y, x] - params.g_inh * global_activity - abs2(z[y, x])) * z[y, x] + D_spatial * coupling
        end
        
        z .+= params.dt .* dz
        _is_finite_field(z) || throw(DomainError(z, "style-transfer wave field became non-finite; reduce dt, coupling, or steps"))
        @. a += params.dt * (-a + params.γ * abs2(z)) / params.τ_a
    end
    
    return z
end

"""
    save_style_profile(profile::StyleProfile, path::String)

Serialises a StyleProfile to a Julia binary-serialisation file for later reuse.
"""
function save_style_profile(profile::StyleProfile, path::String)
    _assert_style_profile(profile)
    dir = dirname(path)
    if !isdir(dir) && !isempty(dir)
        mkpath(dir)
    end
    open(path, "w") do io
        serialize(io, (profile.H, profile.W, profile.K_h, profile.K_v, profile.params))
    end
    return path
end

"""
    load_style_profile(path::String) -> StyleProfile

Loads a StyleProfile previously saved with `save_style_profile`.
"""
function load_style_profile(path::String)
    isfile(path) || throw(ArgumentError("StyleProfile file not found: $path"))
    open(path, "r") do io
        data = deserialize(io)
        profile = StyleProfile(data[1], data[2], data[3], data[4], data[5])
        _assert_style_profile(profile)
        return profile
    end
end

"""
    run_style_transfer(img_content::Matrix{<:Colorant}, img_style::Matrix{<:Colorant},
                       params::OscillatorParams; D_spatial=0.1, steps=200) -> Matrix{RGB{N0f8}}

Performs wave-based style transfer.
Delegates to `extract_style_profile` + `apply_style` for reusability.
"""
function run_style_transfer(img_content::Matrix{<:Colorant}, img_style::Matrix{<:Colorant},
                            params::OscillatorParams=OscillatorParams(); D_spatial::Float64=0.1, steps::Int=200)
    H, W = size(img_content)
    H > 0 && W > 0 || throw(ArgumentError("img_content must not be empty"))
    !isempty(img_style) || throw(ArgumentError("img_style must not be empty"))
    
    profile = extract_style_profile(Images.imresize(img_style, (H, W)), params)
    z_content = image_to_wavefield(img_content)
    z_styled = apply_style(z_content, profile; D_spatial=D_spatial, steps=steps)
    return wavefield_to_image(z_styled)
end

# --- Pattern Crystallization (Generation from Noise) ---

"""
    run_pattern_crystallization(target_shape::Matrix{Bool}, params::OscillatorParams;
                                steps=250, noise_level=0.5) -> Matrix{RGB{N0f8}}

Demonstrates pattern formation from random noise.
Starts with random phase/amplitude noise, and converges to a target shape
under the influence of a shape-directed coupling matrix K.
"""
function run_pattern_crystallization(target_shape::Matrix{Bool}, params::OscillatorParams;
                                     steps::Int=250, noise_level::Float64=0.5, seed::Int=42)
    H, W = size(target_shape)
    H > 0 && W > 0 || throw(ArgumentError("target_shape must not be empty"))
    steps >= 0 || throw(ArgumentError("steps must be non-negative, got $steps"))
    noise_level >= 0 || throw(ArgumentError("noise_level must be non-negative, got $noise_level"))
    rng = MersenneTwister(seed)
    
    # Start with random complex noise
    z = (rand(rng, Float64, H, W) .* noise_level) .* exp.(im .* (rand(rng, Float64, H, W) .* 2pi))
    a = zeros(Float64, H, W)
    
    # Construct shape attraction field K
    # K pulls oscillators inside the target shape into phase synchronization (e.g. Red, phase 0)
    # and pushes background pixels to zero amplitude or opposite phase (e.g. Green, phase 2π/3)
    K_target = zeros(ComplexF64, H, W)
    for y in 1:H, x in 1:W
        if target_shape[y, x]
            # Red color (phase 0)
            K_target[y, x] = 1.5 * exp(im * 0.0)
        else
            # Green color (phase 2π/3) with low amplitude (background)
            K_target[y, x] = 0.2 * exp(im * 2pi/3)
        end
    end
    
    dz = zeros(ComplexF64, H, W)
    for step in 1:steps
        L = laplacian2d(z)
        global_activity = sum(abs2.(z)) / (H * W)
        
        # Annealing noise (cool down)
        current_noise = noise_level * (1.0 - step / steps)
        thermal_noise = current_noise * (_complex_noise(rng, H, W) ./ sqrt(H*W))
        
        for y in 1:H, x in 1:W
            # Coupling pulls pixel towards target state
            coupling = K_target[y, x] - z[y, x]
            
            dz[y, x] = (params.μ - a[y, x] - params.g_inh * global_activity - abs2(z[y, x])) * z[y, x] + 0.1 * L[y, x] + 1.2 * coupling
        end
        
        z .+= params.dt .* dz .+ thermal_noise
        _is_finite_field(z) || throw(DomainError(z, "crystallization wave field became non-finite; reduce dt, coupling, or noise"))
        @. a += params.dt * (-a + params.γ * abs2(z)) / params.τ_a
    end
    
    return wavefield_to_image(z)
end

# --- PhasNet Classifier (CHL Image Classifier) ---

mutable struct PhasNetClassifier
    input_dim::Int
    hidden_dim::Int
    output_dim::Int
    K1::Matrix{ComplexF64} # Weights Input -> Hidden
    K2::Matrix{ComplexF64} # Weights Hidden -> Output
    params::OscillatorParams
    loss_history::Vector{Float64}
end

function PhasNetClassifier(input_dim::Int, hidden_dim::Int, output_dim::Int, params=OscillatorParams(); seed::Int=42)
    _assert_positive_dim("input_dim", input_dim)
    _assert_positive_dim("hidden_dim", hidden_dim)
    _assert_positive_dim("output_dim", output_dim)
    rng = MersenneTwister(seed)
    # Initialize complex weights with small random amplitudes
    K1 = 0.1 .* _complex_noise(rng, hidden_dim, input_dim)
    K2 = 0.1 .* _complex_noise(rng, output_dim, hidden_dim)
    return PhasNetClassifier(input_dim, hidden_dim, output_dim, K1, K2, params, Float64[])
end

"""
    train_classifier!(model::PhasNetClassifier, X::Vector{Vector{Float64}}, Y::Vector{Vector{Float64}};
                      epochs=100, lr=0.05)

Trains the PhasNet classifier using Contrastive Hebbian Learning (CHL).
"""
function train_classifier!(model::PhasNetClassifier, X::Vector{Vector{Float64}}, Y::Vector{Vector{Float64}};
                           epochs::Int=100, lr::Float64=0.05, weight_decay::Float64=0.999,
                           seed::Int=42, early_stopping::Bool=false, patience::Int=10,
                           min_delta::Float64=1e-6, record_loss::Bool=true,
                           reset_loss_history::Bool=false)
    in_dim, hid_dim, out_dim = model.input_dim, model.hidden_dim, model.output_dim
    params = model.params
    rng = MersenneTwister(seed)
    length(X) == length(Y) || throw(DimensionMismatch("X and Y must contain the same number of samples"))
    epochs >= 0 || throw(ArgumentError("epochs must be non-negative, got $epochs"))
    0.0 <= weight_decay <= 1.0 || throw(ArgumentError("weight_decay must be in [0, 1], got $weight_decay"))
    patience > 0 || throw(ArgumentError("patience must be positive, got $patience"))
    min_delta >= 0.0 || throw(ArgumentError("min_delta must be non-negative, got $min_delta"))
    reset_loss_history && empty!(model.loss_history)
    best_loss = isempty(model.loss_history) ? Inf : minimum(model.loss_history)
    stagnant_epochs = 0
    
    for epoch in 1:epochs
        total_loss = 0.0
        
        for (img_flat, target_class) in zip(X, Y)
            length(img_flat) == in_dim || throw(DimensionMismatch("input sample has length $(length(img_flat)); expected $in_dim"))
            length(target_class) == out_dim || throw(DimensionMismatch("target sample has length $(length(target_class)); expected $out_dim"))
            # Map input brightness directly to real parts of input vector
            in_vec = ComplexF64.(img_flat)
            
            # --- POSITIVE PHASE (Target Clamped) ---
            z_hid_pos = zeros(ComplexF64, hid_dim)
            # Output clamped: target=1 => 1.0, target=0 => 0.0
            z_out_pos = [target_class[i] == 1.0 ? 1.0+0.0im : 0.0+0.0im for i in 1:out_dim]
            a_hid = zeros(Float64, hid_dim)
            
            # Simulate hidden layer
            for _ in 1:30
                act_in = sum(abs2.(z_hid_pos)) / hid_dim
                z_hid_pos .+= params.dt .* ((params.μ .- a_hid .- params.g_inh * act_in .- abs2.(z_hid_pos)) .* z_hid_pos .+ (model.K1 * in_vec))
                a_hid .+= params.dt .* (-a_hid .+ params.γ .* abs2.(z_hid_pos)) ./ params.τ_a
            end
            
            # --- NEGATIVE PHASE (Target Free) ---
            z_hid_neg = copy(z_hid_pos)
            z_out_neg = zeros(ComplexF64, out_dim) .+ 0.1 .* _complex_noise(rng, out_dim)
            a_hid_neg = copy(a_hid)
            a_out_neg = zeros(Float64, out_dim)
            
            # Simulate hidden and output layers together
            for _ in 1:30
                act_in_neg = (sum(abs2.(z_hid_neg)) + sum(abs2.(z_out_neg))) / (hid_dim + out_dim)
                
                # Hidden
                z_hid_neg .+= params.dt .* ((params.μ .- a_hid_neg .- params.g_inh * act_in_neg .- abs2.(z_hid_neg)) .* z_hid_neg .+ (model.K1 * in_vec) .+ (model.K2' * z_out_neg))
                a_hid_neg .+= params.dt .* (-a_hid_neg .+ params.γ .* abs2.(z_hid_neg)) ./ params.τ_a
                
                # Output (Free)
                z_out_neg .+= params.dt .* ((params.μ .- a_out_neg .- params.g_inh * act_in_neg .- abs2.(z_out_neg)) .* z_out_neg .+ (model.K2 * z_hid_neg))
                a_out_neg .+= params.dt .* (-a_out_neg .+ params.γ .* abs2.(z_out_neg)) ./ params.τ_a
            end
            
            # --- Contrastive Hebbian Learning Update ---
            for i in 1:hid_dim, j in 1:in_dim
                pos_corr = real(z_hid_pos[i] * conj(in_vec[j]))
                neg_corr = real(z_hid_neg[i] * conj(in_vec[j]))
                model.K1[i, j] += lr * (pos_corr - neg_corr)
            end
            
            for i in 1:out_dim, j in 1:hid_dim
                pos_corr = real(z_out_pos[i] * conj(z_hid_pos[j]))
                neg_corr = real(z_out_neg[i] * conj(z_hid_neg[j]))
                model.K2[i, j] += lr * (pos_corr - neg_corr)
            end
            
            # Compute amplitude-based loss
            total_loss += sum(abs2.(abs.(z_out_neg) .- target_class))
        end
        # General weight decay to keep weights stable
        model.K1 .*= weight_decay
        model.K2 .*= weight_decay
        if record_loss
            push!(model.loss_history, total_loss)
        end
        if early_stopping
            if total_loss < best_loss - min_delta
                best_loss = total_loss
                stagnant_epochs = 0
            else
                stagnant_epochs += 1
                stagnant_epochs >= patience && break
            end
        end
    end
    return model.loss_history
end

"""
    predict_classifier!(model::PhasNetClassifier, x::Vector{Float64}) -> Int

Classifies a single flattened image.
Runs free simulation and decodes the index of the output node with the highest amplitude.
"""
function predict_classifier!(model::PhasNetClassifier, x::Vector{Float64})
    in_dim, hid_dim, out_dim = model.input_dim, model.hidden_dim, model.output_dim
    params = model.params
    length(x) == in_dim || throw(DimensionMismatch("input sample has length $(length(x)); expected $in_dim"))
    in_vec = ComplexF64.(x)
    
    z_hid = zeros(ComplexF64, hid_dim)
    z_out = zeros(ComplexF64, out_dim)
    a = zeros(Float64, hid_dim + out_dim)
    
    # Run free simulation
    for _ in 1:60
        act = (sum(abs2.(z_hid)) + sum(abs2.(z_out))) / (hid_dim + out_dim)
        z_hid .+= params.dt .* ((params.μ .- a[1:hid_dim] .- params.g_inh * act .- abs2.(z_hid)) .* z_hid .+ (model.K1 * in_vec) .+ (model.K2' * z_out))
        z_out .+= params.dt .* ((params.μ .- a[hid_dim+1:end] .- params.g_inh * act .- abs2.(z_out)) .* z_out .+ (model.K2 * z_hid))
        a[1:hid_dim] .+= params.dt .* (-a[1:hid_dim] .+ params.γ .* abs2.(z_hid)) ./ params.τ_a
        a[hid_dim+1:end] .+= params.dt .* (-a[hid_dim+1:end] .+ params.γ .* abs2.(z_out)) ./ params.τ_a
    end
    
    return argmax(abs.(z_out))
end

"""
    compute_classification_loss(model::PhasNetClassifier, x::Vector{Float64},
                                target::Vector{Float64}) -> Float64

Runs a single negative-phase pass and returns the amplitude‑based loss
without mutating model weights.  Useful for validation / monitoring during
training without duplicating the forward pass logic.
"""
function compute_classification_loss(model::PhasNetClassifier, x::Vector{Float64},
                                     target::Vector{Float64})
    in_dim, hid_dim, out_dim = model.input_dim, model.hidden_dim, model.output_dim
    params = model.params
    length(x) == in_dim || throw(DimensionMismatch("input length $(length(x)); expected $in_dim"))
    length(target) == out_dim || throw(DimensionMismatch("target length $(length(target)); expected $out_dim"))

    in_vec = ComplexF64.(x)
    z_hid = zeros(ComplexF64, hid_dim)
    z_out = zeros(ComplexF64, out_dim) .+ 0.1 .* _complex_noise(MersenneTwister(42), out_dim)
    a_hid = zeros(Float64, hid_dim)
    a_out = zeros(Float64, out_dim)

    for _ in 1:30
        act = (sum(abs2.(z_hid)) + sum(abs2.(z_out))) / (hid_dim + out_dim)

        z_hid .+= params.dt .* ((params.μ .- a_hid .- params.g_inh * act .- abs2.(z_hid)) .* z_hid
                                .+ (model.K1 * in_vec) .+ (model.K2' * z_out))
        a_hid .+= params.dt .* (-a_hid .+ params.γ .* abs2.(z_hid)) ./ params.τ_a

        z_out .+= params.dt .* ((params.μ .- a_out .- params.g_inh * act .- abs2.(z_out)) .* z_out
                                .+ (model.K2 * z_hid))
        a_out .+= params.dt .* (-a_out .+ params.γ .* abs2.(z_out)) ./ params.τ_a
    end

    return sum(abs2.(abs.(z_out) .- target))
end

"""
    predict_batch(model::PhasNetClassifier, X::Vector{Vector{Float64}}) -> Vector{Int}

Classifies a batch of flattened images.
Convenience wrapper around `predict_classifier!`.
"""
function predict_batch(model::PhasNetClassifier, X::Vector{Vector{Float64}})
    return [predict_classifier!(model, x) for x in X]
end

end # module HoloPRNN_Vision
