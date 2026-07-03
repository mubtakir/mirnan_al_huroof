"""MizanEngine — محرك الميزان للاتساق الفيزيائي (SVO coherence)."""
module MizanEngineModule
using LinearAlgebra
using ..WordPhysics: compute_word_phase_vector

export MizanEngine

const MIZAN_OVERRIDES = Dict{String,Dict{Int,Float64}}(
    "سكين" => Dict(16=>0.8, 17=>0.9), "السكين" => Dict(16=>0.8, 17=>0.9),
    "حجر" => Dict(3=>0.5, 16=>0.4), "الحجر" => Dict(3=>0.5, 16=>0.4),
    "قلم" => Dict(16=>0.1, 17=>0.1), "القلم" => Dict(16=>0.1, 17=>0.1),
    "حديد" => Dict(3=>0.9, 16=>0.9), "الحديد" => Dict(3=>0.9, 16=>0.9),
)

struct MizanEngine end

function calculate_coherence(eng::MizanEngine, verb::String, subject::String, obj::Union{String,Nothing}=nothing)
    v_verb = Float64.(compute_word_phase_vector(verb))
    v_subj = Float64.(compute_word_phase_vector(subject))
    v_dim = length(v_verb)
    if verb in keys(MIZAN_OVERRIDES)
        for (idx, val) in MIZAN_OVERRIDES[verb]
            idx+1 <= v_dim && (v_verb[idx+1] = val)
        end
    end
    if subject in keys(MIZAN_OVERRIDES)
        for (idx, val) in MIZAN_OVERRIDES[subject]
            idx+1 <= v_dim && (v_subj[idx+1] = val)
        end
    end

    carrier_force = min(1.0, (norm(v_verb) + norm(v_subj)) / 4.0)
    if obj === nothing
        return Dict("coherence"=>carrier_force, "carrier_force"=>carrier_force, "recipient_resistance"=>0.0)
    end

    v_obj = Float64.(compute_word_phase_vector(obj))
    if obj in keys(MIZAN_OVERRIDES)
        for (idx, val) in MIZAN_OVERRIDES[obj]
            idx+1 <= length(v_obj) && (v_obj[idx+1] = val)
        end
    end
    idx_18 = min(18, v_dim); idx_17 = min(17, v_dim); idx_4 = min(4, length(v_obj))
    p_force = (abs(v_verb[idx_18]) + abs(v_subj[idx_18])) * abs(v_subj[idx_17])
    o_resistance = abs(v_obj[idx_4]) + abs(v_obj[idx_17])

    coherence = 1.0
    if p_force < o_resistance * 0.8; coherence -= 0.6
    elseif p_force < o_resistance; coherence -= 0.25; end
    return Dict("coherence"=>max(0.0, coherence), "carrier_force"=>carrier_force,
                "recipient_resistance"=>o_resistance)
end

function get_mizan_score(eng::MizanEngine, context_words::Vector{String}, candidate_word::String)
    length(context_words) < 1 && return 1.0
    recent = vcat(context_words[max(1, end-3):end], [candidate_word])
    length(recent) < 2 && return 1.0
    verb, subject, obj = recent[1], recent[2], length(recent) >= 3 ? recent[3] : nothing
    res = calculate_coherence(eng, verb, subject, obj)
    return res["coherence"]
end
end

