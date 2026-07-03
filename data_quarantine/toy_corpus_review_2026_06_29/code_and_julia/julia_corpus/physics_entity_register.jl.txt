"""
EntityRegister — سجل الكيانات وتتبع الضمائر في فضاء الطور.
"""
module EntityRegisterModule
using LinearAlgebra

export Entity, EntityRegister, PRONOUN_GENDERS

const PRONOUN_GENDERS = Dict(
    "هو" => "MASC", "هي" => "FEM", "هما" => "DUAL",
    "هم" => "MASC_PL", "هن" => "FEM_PL",
    "هذا" => "MASC", "هذه" => "FEM", "ذلك" => "MASC", "تلك" => "FEM",
    "he" => "MASC", "she" => "FEM", "it" => "NEUT",
    "they" => "PL", "this" => "NEUT", "that" => "NEUT",
)

struct Entity
    name::String
    pv::Vector{Float64}
    mass::Float64
    age::Int
end

mutable struct EntityRegister
    entities::Vector{Entity}
    max_entities::Int
end
EntityRegister(; max_entities=20) = EntityRegister(Entity[], max_entities)

function extract!(reg::EntityRegister, words::Vector{String}, pv_fn)
    for w in words
        if haskey(PRONOUN_GENDERS, w); continue; end
        if length(w) >= 3 && !all(c -> isletter(c) && c <= 'z', w) # Arabic or mixed
            pv = Float64.(pv_fn(w))
            reg.entities = [e for e in reg.entities if e.name != w] # Replace
            push!(reg.entities, Entity(w, pv, norm(pv), 0))
            if length(reg.entities) > reg.max_entities; popfirst!(reg.entities); end
        end
    end
end

function resolve_pronoun(reg::EntityRegister, pronoun::String, context_pv=nothing)
    gender = get(PRONOUN_GENDERS, pronoun, "MASC")
    if isempty(reg.entities); return nothing; end
    # Return most recent matching entity
    for e in reverse(reg.entities)
        if context_pv !== nothing
            sim = max(0.0, dot(e.pv, context_pv) / (norm(e.pv) * norm(context_pv) + 1e-10))
            if sim > 0.3; return e; end
        end
    end
    return last(reg.entities)
end

function entity_gravity(reg::EntityRegister, candidate_pv)
    if isempty(reg.entities); return 0.0; end
    grav = 0.0
    for e in reg.entities
        sim = max(0.0, dot(candidate_pv, e.pv) / (norm(candidate_pv) * norm(e.pv) + 1e-10))
        grav += e.mass * sim * exp(-0.1 * e.age)
    end
    return grav / length(reg.entities)
end
end

