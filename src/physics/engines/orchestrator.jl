"""
Orchestrator — منسق فيزيائي (PhysicsOrchestrator).
Coordinates generation modes and physics state.
"""
module Orchestrator

export PhysicsOrchestrator, PhysicsState, generate!

mutable struct PhysicsState
    mode::String
    entropy::Float64
    beta::Float64
    k_B::Float64
    temperature::Float64
    phase_coherence::Float64
    cascade_enabled::Bool
    cascade_strength::Float64
end

PhysicsState() = PhysicsState("standard", 0.0, 2.0, 0.1, 0.5, 0.5, false, 1.8)

mutable struct PhysicsOrchestrator
    gen::Any
    state::PhysicsState
    history::Vector{Dict}
end

PhysicsOrchestrator(gen) = PhysicsOrchestrator(gen, PhysicsState(), Dict[])

function adjust_beta!(orch::PhysicsOrchestrator, val::Float64)
    orch.state.beta = clamp(val, 0.1, 10.0)
end

function set_cascade!(orch::PhysicsOrchestrator, enabled::Bool, strength::Float64=1.8)
    orch.state.cascade_enabled = enabled
    orch.state.cascade_strength = strength
end

function generate!(orch::PhysicsOrchestrator, prompt::String; kwargs...)
    result = generate!(orch.gen, prompt; mode=orch.state.mode, kwargs...)
    push!(orch.history, Dict("prompt" => prompt, "result" => result, "state" => Dict("mode" => orch.state.mode)))
    return result
end

function get_report(orch::PhysicsOrchestrator)
    return Dict("mode" => orch.state.mode, "beta" => orch.state.beta,
                "k_B" => orch.state.k_B, "temperature" => orch.state.temperature)
end

function reset!(orch::PhysicsOrchestrator)
    orch.state = PhysicsState()
    empty!(orch.history)
end

function get_history(orch::PhysicsOrchestrator, n::Int=10)
    return orch.history[max(1, end-n+1):end]
end

end # module Orchestrator
