"""
DialogueMemory — ذاكرة حوار طورية (آخر 20 جملة كمسارات طورية).
"""
module DialogueMemoryModule
using LinearAlgebra

export DialogueTurn, DialogueMemory

mutable struct DialogueTurn
    text::String
    pv::Vector{Float64}
    mass::Float64
    age::Int
end

mutable struct DialogueMemory
    turns::Vector{DialogueTurn}
    max_turns::Int
    pv_fn::Any
end

DialogueMemory(; max_turns=20, pv_fn=nothing) = DialogueMemory(DialogueTurn[], max_turns, pv_fn)

function add_turn!(dm::DialogueMemory, text::String, pv=nothing)
    if dm.pv_fn !== nothing && pv === nothing
        pv = Float64.(dm.pv_fn(text))
    end
    if pv !== nothing; pv = Float64.(pv); end
    pv_mass = pv === nothing ? 0.0 : norm(pv)
    push!(dm.turns, DialogueTurn(text, pv === nothing ? zeros(1) : pv, pv_mass, 0))
    if length(dm.turns) > dm.max_turns
        popfirst!(dm.turns)
    end
end

function tick!(dm::DialogueMemory)
    for t in dm.turns; t.age += 1; end
end

function get_context_vector(dm::DialogueMemory)
    if isempty(dm.turns); return nothing; end
    ctx = zeros(Float64, length(dm.turns[1].pv))
    total_w = 0.0
    for t in dm.turns
        w = exp(-0.1 * t.age)
        ctx .+= w .* t.pv
        total_w += w
    end
    ctx ./= total_w
    return ctx
end

function get_dialogue_gravity(dm::DialogueMemory, candidate_pv)
    if isempty(dm.turns); return 0.0; end
    grav = 0.0
    for t in dm.turns
        w = exp(-0.1 * t.age)
        sim = dot(candidate_pv, t.pv) / (norm(candidate_pv) * norm(t.pv) + 1e-10)
        grav += w * max(0.0, sim)
    end
    return grav / max(length(dm.turns), 1)
end

function is_repeat(dm::DialogueMemory, text::String; threshold=0.95)
    for t in dm.turns
        if text == t.text; return true; end
    end
    return false
end
end

