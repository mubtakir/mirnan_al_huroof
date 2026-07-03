module Entities

using LinearAlgebra

export AqlEntity, Thing, Lion, Gazelle, Earth, Furniture,
       get_property, set_property!, get_attribute, set_attribute!,
       register_action!, get_action, has_action, print_entity, PROPERTY_MAP

# رسم خرائط الصفات والخصائص (Properties Subspaces)
const PROPERTY_MAP = Dict{String, Int}(
    "temperature"   => 1, # درجة الحرارة
    "motion"        => 2, # الحركة/السرعة
    "awareness"     => 3, # الوعي/الانتباه
    "fear"          => 4, # الخوف/التوجس
    "stability"     => 5, # الاستقرار/الاتزان
    "integrity"     => 6, # التماسك/الحياة
    "light"         => 7, # الضياء/النور
    "energy"        => 8  # الطاقة العامة
)

# النوع المجرد للكيانات (كلاس الكيان الرئيسي)
abstract type AqlEntity end

mutable struct Thing <: AqlEntity
    name::String
    kind::String
    mass::Float64
    properties::Vector{ComplexF64}
    attributes::Dict{String,Any}
    actions::Dict{String,Function}
end

function Thing(name::String; kind::String="thing", mass::Float64=1.0,
               attributes::Dict{String,<:Any}=Dict{String,Any}())
    props = ones(ComplexF64, 8) .* 0.1
    return Thing(name, kind, mass, props, Dict{String,Any}(attributes), Dict{String,Function}())
end

# كلاس الأسد
mutable struct Lion <: AqlEntity
    name::String
    mass::Float64
    properties::Vector{ComplexF64}
end
function Lion(name::String, mass::Float64=1.0)
    props = ones(ComplexF64, 8) .* 0.1
    return Lion(name, mass, props)
end

# كلاس الغزال
mutable struct Gazelle <: AqlEntity
    name::String
    mass::Float64
    properties::Vector{ComplexF64}
end
function Gazelle(name::String, mass::Float64=1.0)
    props = ones(ComplexF64, 8) .* 0.1
    return Gazelle(name, mass, props)
end

# كلاس الأرض
mutable struct Earth <: AqlEntity
    name::String
    mass::Float64
    properties::Vector{ComplexF64}
end
function Earth(name::String, mass::Float64=1.0)
    props = ones(ComplexF64, 8) .* 0.1
    return Earth(name, mass, props)
end

# كلاس الأثاث
mutable struct Furniture <: AqlEntity
    name::String
    mass::Float64
    properties::Vector{ComplexF64}
end
function Furniture(name::String, mass::Float64=1.0)
    props = ones(ComplexF64, 8) .* 0.1
    return Furniture(name, mass, props)
end

"""
    get_property(e::AqlEntity, prop) -> Tuple{Float64, Float64}

الحصول على شدة (Amplitude) وطور (Phase) صفة معينة للكيان.
"""
function get_property(e::AqlEntity, prop::String)
    idx = get(PROPERTY_MAP, prop, 0)
    idx == 0 && return (0.0, 0.0) # سعة صفر، طور صفر
    val = e.properties[idx]
    return abs(val), angle(val)
end

"""
    set_property!(e::AqlEntity, prop, amplitude, phase)

تعيين شدة وطور صفة معينة للكيان.
"""
function set_property!(e::AqlEntity, prop::String, amplitude::Float64, phase::Float64)
    idx = get(PROPERTY_MAP, prop, 0)
    idx == 0 && return false
    # الحفاظ على القيم ضمن الحدود
    amplitude = clamp(amplitude, 0.0, 1.0)
    # حصر الطور بين 0 و 2pi
    phase = mod2pi(phase)
    e.properties[idx] = amplitude * exp(im * phase)
    return true
end

function get_attribute(e::Thing, key::String, default=nothing)
    return get(e.attributes, key, default)
end

function set_attribute!(e::Thing, key::String, value)
    e.attributes[key] = value
    return value
end

function register_action!(e::Thing, name::String, action::Function)
    e.actions[name] = action
    return action
end

function get_action(e::Thing, name::String)
    return get(e.actions, name, nothing)
end

has_action(e::Thing, name::String) = haskey(e.actions, name)
has_action(e::AqlEntity, name::String) = false

"""
    print_entity(e::AqlEntity)

طباعة تفصيلية لحالة الكيان وخصائصه الحالية.
"""
function print_entity(e::AqlEntity)
    println("👤 الكيان: ", e.name, " (الفئة: ", typeof(e), " | القصور الذاتي: ", e.mass, ")")
    for (prop, idx) in sort(collect(PROPERTY_MAP); by=x->x[2])
        val = e.properties[idx]
        amp = abs(val)
        ph = angle(val)
        deg = round(ph * 180 / π; digits=1)
        amp_round = round(amp; digits=3)
        println("   - ", rpad(prop, 15), " -> الشدة: ", rpad(amp_round, 6), " | الطور: ", deg, "°")
    end
end

function print_entity(e::Thing)
    println("👤 الكيان: ", e.name, " (النوع: ", e.kind, " | القصور الذاتي: ", e.mass, ")")
    for (prop, idx) in sort(collect(PROPERTY_MAP); by=x->x[2])
        val = e.properties[idx]
        amp = abs(val)
        ph = angle(val)
        deg = round(ph * 180 / π; digits=1)
        amp_round = round(amp; digits=3)
        println("   - ", rpad(prop, 15), " -> الشدة: ", rpad(amp_round, 6), " | الطور: ", deg, "°")
    end
    if !isempty(e.attributes)
        println("   السمات:")
        for (key, value) in sort(collect(e.attributes); by=x->x[1])
            println("   - ", key, ": ", value)
        end
    end
    if !isempty(e.actions)
        println("   الأفعال: ", join(sort(collect(keys(e.actions))), ", "))
    end
end

end # module Entities
