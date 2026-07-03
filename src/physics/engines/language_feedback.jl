"""
LanguageFeedback — حلقة تغذية راجعة مغلقة بين التوليد والتعلم.
Evaluates generated sentences and feeds back to learning engines.
"""
module LanguageFeedback

using LinearAlgebra, Statistics

export LanguageFeedbackEngine, evaluate_sentence!, apply_feedback!,
       reinforce_good_sentence!, get_feedback_report

mutable struct LanguageFeedbackEngine
    enabled::Bool
    sentence_history::Vector{Dict{String,Any}}
    avg_coherence::Float64
    total_generations::Int
    good_count::Int
    poor_count::Int
    lr::Float64
    min_words_for_feedback::Int
    reinforcement_threshold::Float64
    penalty_threshold::Float64
end

function LanguageFeedbackEngine(;
    enabled::Bool=true,
    lr::Float64=0.05,
    min_words_for_feedback::Int=3,
    reinforcement_threshold::Float64=0.4,
    penalty_threshold::Float64=0.15,
)
    return LanguageFeedbackEngine(
        enabled, Dict{String,Any}[], 0.0, 0, 0, 0,
        lr, min_words_for_feedback,
        reinforcement_threshold, penalty_threshold,
    )
end

function _phase_similarity(a, b)
    d = min(length(a), length(b))
    na = norm(view(a, 1:d))
    nb = norm(view(b, 1:d))
    (na < 1e-10 || nb < 1e-10) && return 0.0
    return max(0.0, dot(view(a, 1:d), view(b, 1:d)) / (na * nb))
end

function evaluate_sentence!(lfe::LanguageFeedbackEngine, words::Vector{String},
                             pv_fn::Function, mass_fn::Function)
    n = length(words)
    dict = Dict{String,Any}(
        "words" => words,
        "n_words" => n,
    )

    if n >= 2
        pvs = [try pv_fn(w) catch; nothing end for w in words]
        valid_pvs = [pv for pv in pvs if pv !== nothing]
        if length(valid_pvs) >= 2
            seq_sims = Float64[]
            for i in 1:length(valid_pvs)-1
                push!(seq_sims, _phase_similarity(valid_pvs[i], valid_pvs[i+1]))
            end
            dict["phase_coherence"] = mean(seq_sims)
        else
            dict["phase_coherence"] = 0.0
        end
    else
        dict["phase_coherence"] = 0.0
    end

    quality = clamp(dict["phase_coherence"], 0.0, 1.0)
    dict["quality"] = quality
    dict["is_good"] = quality >= lfe.reinforcement_threshold
    dict["is_poor"] = quality <= lfe.penalty_threshold

    push!(lfe.sentence_history, dict)
    if length(lfe.sentence_history) > 200
        popfirst!(lfe.sentence_history)
    end

    lfe.total_generations += 1
    if dict["is_good"]; lfe.good_count += 1; end
    if dict["is_poor"]; lfe.poor_count += 1; end
    lfe.avg_coherence = (lfe.avg_coherence * (lfe.total_generations - 1) + quality) / lfe.total_generations

    return dict
end

function apply_feedback!(lfe::LanguageFeedbackEngine, gen, eval_dict::Dict{String,Any})
    !lfe.enabled && return
    n = get(eval_dict, "n_words", 0)
    n < lfe.min_words_for_feedback && return

    words = get(eval_dict, "words", String[])
    quality = get(eval_dict, "quality", 0.5)
    is_good = get(eval_dict, "is_good", false)
    is_poor = get(eval_dict, "is_poor", false)

    lr = lfe.lr

    if isdefined(gen, :reinforcement) && gen.reinforcement !== nothing && n >= 2
        for w in words
            haskey(gen.pv_cache, w) || continue
            pv = gen.pv_cache[w]
            if is_good && quality >= 0.5
                reinforce!(gen.reinforcement, w, pv; reward=quality)
            elseif is_poor
                weaken!(gen.reinforcement, w; penalty=0.3 * (1.0 - quality))
            end
        end
    end

    if n >= 2 && isdefined(gen, :K_sem) && gen.K_sem !== nothing
        if is_good
            for i in 1:n-1
                a = get(gen.vocab, words[i], 0)
                b = get(gen.vocab, words[i+1], 0)
                if a > 0 && b > 0 && a <= size(gen.K_sem, 1) && b <= size(gen.K_sem, 2)
                    delta = lr * quality * 0.1
                    gen.K_sem[a, b] = min(gen.K_sem[a, b] + delta, 100.0)
                end
            end
        end
    end
end

function reinforce_good_sentence!(lfe::LanguageFeedbackEngine, gen,
                                   words::Vector{String}, pv_fn::Function, mass_fn::Function)
    !lfe.enabled && return
    eval_dict = evaluate_sentence!(lfe, words, pv_fn, mass_fn)
    apply_feedback!(lfe, gen, eval_dict)
end

function get_feedback_report(lfe::LanguageFeedbackEngine)
    return Dict(
        "total_generations" => lfe.total_generations,
        "good_count" => lfe.good_count,
        "poor_count" => lfe.poor_count,
        "avg_coherence" => round(lfe.avg_coherence, digits=4),
    )
end

end # module LanguageFeedback
