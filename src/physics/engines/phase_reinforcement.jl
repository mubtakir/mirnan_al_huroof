"""
Phase Reinforcement — تعزيز طوري ذاتي.
Learn without backprop in phase space:
- Reinforce successful phase paths
- Weaken weak paths
- Hebbian learning in phase space
"""
module PhaseReinforcement

using LinearAlgebra

export PhaseReinforcer, apply, reinforce!, weaken!, decay_all!, reset!, get_strength

mutable struct PhaseReinforcer
    lr::Float64
    decay_rate::Float64
    max_traces::Int
    traces::Dict{String,Vector{Float64}}
    strengths::Dict{String,Float64}
    step_count::Int

    function PhaseReinforcer(; lr::Float64=0.15, decay_rate::Float64=0.005, max_traces::Int=200)
        return new(lr, decay_rate, max_traces,
                   Dict{String,Vector{Float64}}(),
                   Dict{String,Float64}(), 0)
    end
end

function reinforce!(pr::PhaseReinforcer, word::String, pv::AbstractVector;
                    reward::Float64=1.0)
    if norm(pv) < 1e-10
        return
    end

    if !haskey(pr.traces, word)
        pr.traces[word] = Float64.(pv)
        pr.strengths[word] = 0.0
    end

    old = pr.traces[word]
    reinforced = old .+ pr.lr .* reward .* (Float64.(pv) .- old)
    nrm = norm(reinforced)
    if nrm > 1e-10
        reinforced ./= nrm
    end
    pr.traces[word] = reinforced
    pr.strengths[word] = min(1.0, pr.strengths[word] + reward * pr.lr)

    pr.step_count += 1
    _prune!(pr)
end

function weaken!(pr::PhaseReinforcer, word::String; penalty::Float64=0.3)
    if !haskey(pr.strengths, word)
        return
    end
    pr.strengths[word] = max(0.0, pr.strengths[word] - penalty * pr.lr)
    if pr.strengths[word] < 0.01
        delete!(pr.traces, word)
        delete!(pr.strengths, word)
    end
end

function apply(pr::PhaseReinforcer, word::String, pv::AbstractVector)
    if !haskey(pr.traces, word)
        return Float64.(pv)
    end
    strength = pr.strengths[word]
    if strength < 0.01
        return Float64.(pv)
    end
    alpha = strength * 0.5
    return (1.0 - alpha) .* Float64.(pv) .+ alpha .* pr.traces[word]
end

get_strength(pr::PhaseReinforcer, word::String) = get(pr.strengths, word, 0.0)

function decay_all!(pr::PhaseReinforcer)
    for w in collect(keys(pr.strengths))
        pr.strengths[w] *= (1.0 - pr.decay_rate)
        if pr.strengths[w] < 0.01
            delete!(pr.traces, w)
            delete!(pr.strengths, w)
        end
    end
end

function reset!(pr::PhaseReinforcer)
    empty!(pr.traces)
    empty!(pr.strengths)
    pr.step_count = 0
end

function _prune!(pr::PhaseReinforcer)
    if length(pr.traces) <= pr.max_traces
        return
    end
    sorted = sort(collect(pr.strengths); by=x -> -x[2])
    keep = Set{String}(first.(sorted[1:pr.max_traces]))
    for w in setdiff(collect(keys(pr.traces)), keep)
        delete!(pr.traces, w)
        delete!(pr.strengths, w)
    end
end

end # module PhaseReinforcement
