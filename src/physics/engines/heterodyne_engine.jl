"""
محرك التغاير الترددي الطيفي — Heterodyne Engine (محسّن).

المبدأ الفيزيائي: كل كلمة لها طيف ترددي (متجهها المحسّن).
التغاير: لكل قناة f_carrier[ch] ± f_mod[ch]، والمرشح الرنيني هو من
تقع مكوناته الطيفية ضمن نطاقات التغاير.
"""

module Heterodyne

using LinearAlgebra, Statistics

using ..Constants: ENHANCED_DIM
using ..WordPhysics: compute_word_enhanced_vector, phase_similarity_enhanced

export HeterodyneEngine, score_candidate, get_word_spectrum,
       compute_sidebands, compute_candidate_resonance, clear_cache!

"""
    HeterodyneEngine

محرك تغاير ترددي بطيف متعدد القنوات.

الحقول:
- `bandwidth`: عرض نطاق التطابق الترددي
- `context_window`: نافذة السياق
- `spectrum_cache`: تخزين مؤقت للأطياف
"""
mutable struct HeterodyneEngine
    bandwidth::Float64
    context_window::Int
    spectrum_cache::Dict{String,Vector{Float64}}

    function HeterodyneEngine(; bandwidth::Float64=0.15, context_window::Int=8)
        return new(bandwidth, context_window, Dict{String,Vector{Float64}}())
    end
end

"""
    get_word_spectrum(he::HeterodyneEngine, word::String) -> Vector{Float64}

الطيف الترددي للكلمة = المتجه الطوري المحسّن لها.
"""
function get_word_spectrum(he::HeterodyneEngine, word::String)
    if !haskey(he.spectrum_cache, word)
        pv = compute_word_enhanced_vector(word)
        spectrum = abs.(Float64.(pv))
        he.spectrum_cache[word] = spectrum
    end
    return he.spectrum_cache[word]
end

"""
    compute_sidebands(he::HeterodyneEngine, carrier_word, context_words) -> Vector

النطاقات الجانبية الطيفية. لكل كلمة سياق:
  f_plus[ch]  = f_carrier[ch] + f_context[ch]
  f_minus[ch] = |f_carrier[ch] - f_context[ch]|
"""
function compute_sidebands(he::HeterodyneEngine, carrier_word::String,
                           context_words::Vector{String})
    f_carrier = get_word_spectrum(he, carrier_word)
    sidebands = Tuple{String,Vector{Float64},Vector{Float64},Float64}[]
    ctx = isempty(context_words) ? String[] : context_words[max(1, end-he.context_window+1):end]

    rank = 0
    for ctx_word in reverse(ctx)
        rank += 1
        ctx_word == carrier_word && continue
        f_ctx = get_word_spectrum(he, ctx_word)
        f_plus = f_carrier .+ f_ctx
        f_minus = abs.(f_carrier .- f_ctx)
        weight = 1.0 / rank
        push!(sidebands, (ctx_word, f_plus, f_minus, weight))
    end

    return sidebands
end

"""
    compute_candidate_resonance(he::HeterodyneEngine, candidate_word, candidate_spectrum, sidebands) -> Float64

رنين طيفي: كم قناة من المرشح تقع ضمن نطاقات التغاير؟
"""
function compute_candidate_resonance(he::HeterodyneEngine, candidate_word::String,
                                     candidate_spectrum::Vector{Float64},
                                     sidebands::Vector)
    if isempty(sidebands)
        return 0.0
    end

    total_resonance = 0.0
    total_weight = 0.0

    for (_, f_plus, f_minus, weight) in sidebands
        channel_resonance = zeros(Float64, length(candidate_spectrum))
        for ch in 1:length(candidate_spectrum)
            dist_plus = abs(candidate_spectrum[ch] - f_plus[ch]) / max(f_plus[ch], 1e-10)
            dist_minus = abs(candidate_spectrum[ch] - f_minus[ch]) / max(f_minus[ch], 1e-10)
            closest = min(dist_plus, dist_minus)
            if closest < he.bandwidth
                channel_resonance[ch] = 1.0 - (closest / he.bandwidth)
            end
        end
        avg_resonance = mean(channel_resonance)
        total_resonance += avg_resonance * weight
        total_weight += weight
    end

    return total_weight == 0.0 ? 0.0 : total_resonance / total_weight
end

"""
    score_candidate(he::HeterodyneEngine, candidate_word, context_words) -> Float64

تسجيل مرشح عبر مقارنة طيفه مع نطاقات تغاير السياق.
"""
function score_candidate(he::HeterodyneEngine, candidate_word::String,
                         context_words::Vector{String})
    if isempty(context_words)
        return 0.0
    end
    candidate_spectrum = get_word_spectrum(he, candidate_word)
    sidebands = compute_sidebands(he, candidate_word, context_words)
    return compute_candidate_resonance(he, candidate_word, candidate_spectrum, sidebands)
end

"""
    clear_cache!(he::HeterodyneEngine)

مسح ذاكرة التخزين المؤقتة.
"""
function clear_cache!(he::HeterodyneEngine)
    empty!(he.spectrum_cache)
end

end # module Heterodyne
