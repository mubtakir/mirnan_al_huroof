"""
InteractionTrace — أثر التفاعل (حقل طوري مستمر من تفاعلات المستخدم).
"""
module InteractionTraceModule
using LinearAlgebra
using ..Constants: TOTAL_DIM

export InteractionTrace

mutable struct InteractionTrace
    field::Vector{Float64}
    decay::Float64
    max_attractors::Int
    lr::Float64
    attractors::Vector{Vector{Float64}}
    intent_freq::Dict{String,Int}
    user_style::Dict{String,Float64}
end

InteractionTrace(; decay=0.95, max_attractors=50, lr=0.05) =
    InteractionTrace(zeros(TOTAL_DIM), decay, max_attractors, lr, Vector{Float64}[], Dict{String,Int}(), Dict{String,Float64}())

function record!(trace::InteractionTrace, prompt::String, response::String, pv_fn=nothing)
    trace.field .*= trace.decay
    if pv_fn !== nothing
        pv = Float64.(pv_fn(prompt * " " * response))
        trace.field .+= trace.lr .* pv
        push!(trace.attractors, copy(pv))
        if length(trace.attractors) > trace.max_attractors
            popfirst!(trace.attractors)
        end
    end
    trace.user_style["avg_prompt_len"] = 0.9 * get(trace.user_style, "avg_prompt_len", 10.0) + 0.1 * length(prompt)
end

function get_context_boost(trace::InteractionTrace, candidate_pv)
    if norm(trace.field) < 1e-10; return 0.0; end
    return max(0.0, dot(candidate_pv, trace.field) / (norm(candidate_pv) * norm(trace.field) + 1e-10))
end

function recommend_beta(trace::InteractionTrace)
    avg_len = get(trace.user_style, "avg_prompt_len", 10.0)
    if avg_len < 5; return 1.5; elseif avg_len > 30; return 2.5; else return 2.0; end
end
end

