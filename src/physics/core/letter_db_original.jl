"""
قاعدة بيانات الحروف الفيزيائية — Letter Database.

30 حرفاً عربياً + 26 حرفاً إنجليزياً، لكل حرف:
- متجه طوري 9958D (عشوائي حتمي)
- operator (±1, 0), activation, spin, articulation, manner, meaning
"""
module LetterDB

using ..Constants: PHASE_DIM
using JSON, Random, LinearAlgebra

export LetterDatabase, DIM_NAMES, get_vector, get_omega_0, get_raw_norm, get_operator, has

const DIM_NAMES = [
    "concentration", "internal_external", "stability_motion",
    "density", "temperature", "time_accumulation", "time_peak",
    "time_discharge", "motion_linear", "motion_rotary",
    "motion_pulse", "motion_stretch", "motion_slip", "motion_air",
    "axis_v", "mass", "hardness_solid", "penetration", "charge",
    "reference_self", "space_extensionality", "time_causality",
]

"""
    LetterEntry

بيانات حرف واحد: متجه طوري، operator، activation، spin، articulation، manner، meaning.
"""
struct LetterEntry
    vector::Vector{Int8}
    operator::String
    activation::Float64
    spin::Float64
    articulation::String
    manner::String
    meaning::String
end

"""
    LetterDatabase

قاعدة بيانات الحروف. تحمّل من ملف JSON أو تبني متجهات عشوائية حتمية.
"""
mutable struct LetterDatabase
    data::Dict{String, LetterEntry}
    dim::Int
    dim_names::Vector{String}
end

const _DEFAULT_MATRIX_FILE = joinpath(@__DIR__, "..", "..", "..", "data", "letter_physics_matrix.json")

function _operator_value(op)
    s = string(op)
    s == "+1" && return 1.0
    s == "-1" && return -1.0
    return 0.0
end

function _text_feature(text, scale::Float64)
    isempty(text) && return 0.0
    total = 0.0
    for (i, b) in enumerate(codeunits(text))
        total += Float64(b) * (0.37 + 0.11 * mod(i, 7))
    end
    return sin(total * scale)
end

function _feature_vector(letter::String, info)::Vector{Float64}
    raw_v = get(info, "v", Any[])
    features = Float64[]
    for x in raw_v
        x isa Number && push!(features, Float64(x))
    end
    push!(features, _operator_value(get(info, "operator", "0")))
    push!(features, Float64(get(info, "a", 0.0)))
    push!(features, Float64(get(info, "s", 0.0)))
    push!(features, _text_feature(string(get(info, "articulation", "")), 0.013))
    push!(features, _text_feature(string(get(info, "manner", "")), 0.017))
    push!(features, _text_feature(string(get(info, "meaning", "")), 0.019))
    push!(features, _text_feature(letter, 0.023))

    if isempty(features)
        for b in codeunits(letter)
            push!(features, sin(Float64(b) * 0.031))
            push!(features, cos(Float64(b) * 0.047))
        end
    end
    return features
end

function _feature_based_phase_vector(letter::String, info, dim::Int)::Vector{Int8}
    features = _feature_vector(letter, info)
    n = length(features)
    seed = 42.0 + sum(Float64, codeunits(letter))
    v = Vector{Int8}(undef, dim)

    for i in 1:dim
        acc = 0.0
        for k in 1:n
            angle = (seed + 17.0 * i + 31.0 * k + 0.07 * i * k) * 0.013
            acc += features[k] * (sin(angle) + 0.5 * cos(angle * 1.61803398875))
        end
        acc /= sqrt(Float64(max(n, 1)))
        identity = sin((seed + i * 53.0) * 0.071) +
                   0.7 * cos((seed * 1.61803398875 + i * 97.0) * 0.043)
        v[i] = (acc + 1.35 * identity) >= 0 ? Int8(1) : Int8(-1)
    end

    return v
end

"""
    LetterDatabase(; path::Union{String,Nothing}=_DEFAULT_MATRIX_FILE)

بناء قاعدة بيانات الحروف. إذا لم يُعطَ مسار، يبني متجهات عشوائية حتمية فارغة.
"""
function LetterDatabase(; path::Union{String,Nothing}=_DEFAULT_MATRIX_FILE)
    db = LetterDatabase(Dict{String, LetterEntry}(), PHASE_DIM, DIM_NAMES)
    if path !== nothing && isfile(path)
        load_json!(db, path)
    end
    return db
end

function load_json!(db::LetterDatabase, path::String)
    raw = JSON.parsefile(path)
    letters = get(raw, "letters", Dict{String,Any}())
    for (ch, info) in letters
        v = _feature_based_phase_vector(ch, info, db.dim)
        db.data[ch] = LetterEntry(
            v,
            get(info, "operator", "0"),
            get(info, "a", 0.0),
            get(info, "s", 0.0),
            get(info, "articulation", ""),
            get(info, "manner", ""),
            get(info, "meaning", ""),
        )
    end
    return db
end

"""
    get_vector(db::LetterDatabase, letter::String) -> Vector{Int8}

استرجاع المتجه الطوري لحرف. يُنشئ متجهاً عشوائياً حتمياً إن لم يوجد.
"""
function get_vector(db::LetterDatabase, letter::String)
    if !haskey(db.data, letter)
        v = _feature_based_phase_vector(letter, Dict{String,Any}(), db.dim)
        db.data[letter] = LetterEntry(v, "0", 0.0, 0.0, "", "", "")
    end
    return db.data[letter].vector
end

"""
    get_operator(db::LetterDatabase, letter::String) -> String

استرجاع تصنيف operator الحرف (+1, -1, 0).
"""
function get_operator(db::LetterDatabase, letter::String)
    entry = get(db.data, letter, nothing)
    return entry === nothing ? "0" : entry.operator
end

"""
    get_omega_0(db::LetterDatabase, letter::String) -> Float64

التردد الذاتي للحرف: ω₀ = 0.5 + 2.0 * |v| / √dim
"""
function get_omega_0(db::LetterDatabase, letter::String)
    v = get_vector(db, letter)
    return 0.5 + 2.0 * norm(v) / sqrt(db.dim)
end

"""
    get_raw_norm(db::LetterDatabase, letter::String) -> Float64

المقدار الأصلي لمتجه الحرف.
"""
function get_raw_norm(db::LetterDatabase, letter::String)
    return Float64(norm(get_vector(db, letter)))
end

"""
    has(db::LetterDatabase, letter::String) -> Bool

وجود الحرف في القاعدة (دائماً true — يُنشئ افتراضياً).
"""
has(db::LetterDatabase, letter::String) = true

end # module LetterDB
