"""
IntentLandscape — مشهد القصد كآبار جهد محتمل في فضاء الطور.
10 آبار افتراضية: GREETING, QUESTION, COMMAND, REQUEST, etc.
"""
module IntentLandscapeModule
using LinearAlgebra, Random
using ..Constants: TOTAL_DIM

export IntentWell, IntentLandscape

struct IntentWell
    name::String
    center::Vector{Float64}
    depth::Float64
    width::Float64
end

mutable struct IntentLandscape
    wells::Vector{IntentWell}
end

function IntentLandscape(; pv_fn=nothing)
    wells = IntentWell[]
    names = ["GREETING","QUESTION","COMMAND","REQUEST","FAREWELL","STATEMENT","OPINION","SUGGESTION","COMPLAINT","THANK"]
    for (i, name) in enumerate(names)
        seed = UInt64(i * 42)
        rng = MersenneTwister(seed)
        center = randn(rng, TOTAL_DIM)
        center ./= norm(center)
        push!(wells, IntentWell(name, center, 0.8 + 0.1*i, 0.5))
    end
    IntentLandscape(wells)
end

function potential(well::IntentWell, pv::AbstractVector)
    d = min(length(well.center), length(pv))
    dist2 = sum((well.center[1:d] .- Float64.(pv[1:d])).^2)
    return -well.depth * exp(-dist2 / (2 * well.width^2))
end

function detect(land::IntentLandscape, pv::AbstractVector)
    best_well = land.wells[1]
    best_V = potential(best_well, pv)
    for w in land.wells[2:end]
        V = potential(w, pv)
        if V < best_V; best_V = V; best_well = w; end
    end
    confidence = clamp(abs(best_V), 0.0, 1.0)
    return Dict("intent" => best_well.name, "confidence" => confidence)
end

function adapt!(land::IntentLandscape, pv::AbstractVector, intent_name::String)
    for w in land.wells
        if w.name == intent_name
            d = min(length(w.center), length(pv))
            w.center[1:d] = 0.9 .* w.center[1:d] .+ 0.1 .* Float64.(pv[1:d])
            w.center ./= norm(w.center)
            return
        end
    end
end
end

