"""
clifford_trajectory_tracker - Clifford Cognitive Trajectory Tracker (Semantic State Space Model).

This module tracks the cumulative semantic state of the conversation (inputs and outputs)
recursively in 10000D phase vector space, computing trajectory alignment scores
for candidate words to ensure coherent thematic progression.
"""
module CliffordTrajectoryTrackerModule

using LinearAlgebra
using ..WordPhysics: compute_extended_phase_vector

export CognitiveTrajectoryTracker, reset_tracker!, absorb_input!, absorb_word!, trajectory_alignment_score

const SIGNATURE_DIM = 10000

mutable struct CognitiveTrajectoryTracker
    state::Vector{Float64}
    decay::Float64
    strength::Float64
end

function CognitiveTrajectoryTracker(; decay::Real=0.85, strength::Real=1.0)
    return CognitiveTrajectoryTracker(zeros(Float64, SIGNATURE_DIM), Float64(decay), Float64(strength))
end

function reset_tracker!(tracker::CognitiveTrajectoryTracker)
    fill!(tracker.state, 0.0)
    return tracker
end

function get_sentence_vector(text::AbstractString)
    words = String[m.match for m in eachmatch(r"[\p{L}\p{N}_]+", String(text))]
    vec = zeros(Float64, SIGNATURE_DIM)
    for w in words
        length(w) < 2 && continue
        v = Float64.(compute_extended_phase_vector(String(w)))
        vec .+= v
    end
    n = norm(vec)
    n > 1e-10 && (vec ./= n)
    return vec
end

function absorb_input!(tracker::CognitiveTrajectoryTracker, text::AbstractString)
    v = get_sentence_vector(text)
    tracker.state .= tracker.state .* tracker.decay .+ v .* tracker.strength
    n = norm(tracker.state)
    n > 1e-10 && (tracker.state ./= n)
    return tracker
end

function absorb_word!(tracker::CognitiveTrajectoryTracker, word::AbstractString, pv_fn::Function)
    w_pv = try
        Float64.(pv_fn(String(word)))
    catch
        zeros(Float64, SIGNATURE_DIM)
    end
    if norm(w_pv) > 1e-5
        w_pv_norm = w_pv ./ norm(w_pv)
        tracker.state .= tracker.state .* tracker.decay .+ w_pv_norm
        n = norm(tracker.state)
        n > 1e-10 && (tracker.state ./= n)
    end
    return tracker
end

function trajectory_alignment_score(tracker::CognitiveTrajectoryTracker, word::AbstractString, pv_fn::Function)
    n_state = norm(tracker.state)
    n_state < 1e-5 && return 0.0
    
    w_pv = try
        Float64.(pv_fn(String(word)))
    catch
        zeros(Float64, SIGNATURE_DIM)
    end
    n_w = norm(w_pv)
    n_w < 1e-5 && return 0.0
    
    sim = dot(tracker.state, w_pv) / (n_state * n_w)
    return sim
end

end # module CliffordTrajectoryTrackerModule
