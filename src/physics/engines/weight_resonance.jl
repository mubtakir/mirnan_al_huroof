"""
محرك الرنين الصرفي — Weight Resonance Engine.

15 وزناً صرفياً عربياً (فعَل، فعَّل، فاعَل، أفعَل، تفعَّل، تفاعَل،
انفعل، افتعل، افعلَّ، مفعول، فعيل، فعَّال، فعول، أفعال، مُفعِّل)
مع متجهات طورية 22D وجداول انتقالية ومصفوفة رنين.
"""
module WeightResonance

using LinearAlgebra, Statistics

export WeightResonanceEngine, WEIGHT_NAMES

const WEIGHT_EMBEDDINGS = Dict{String,Vector{Float64}}(
    "فَعَلَ"       => [0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "فَعَّلَ"      => [0.30, 0.10, 0.00, 0.00, 0.20, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.30, 0.10, 0.00, 0.00, 0.00, 0.10, 0.00],
    "فَاعَلَ"      => [0.20, 0.00, 0.00, 0.00, 0.10, 0.20, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00],
    "أَفْعَلَ"     => [0.00, 0.00, 0.20, 0.10, 0.00, 0.00, 0.30, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.00],
    "تَفَعَّلَ"    => [0.10, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.00, 0.00, 0.00, 0.10, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00],
    "تَفَاعَلَ"    => [0.10, 0.00, 0.00, 0.10, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "اِنْفَعَلَ"   => [0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00],
    "اِفْتَعَلَ"   => [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "اِفْعَلَّ"    => [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "مَفْعُول"     => [0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "فَعِيل"       => [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "فَعَّال"      => [0.20, 0.00, 0.00, 0.00, 0.00, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "فُعُول"       => [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00],
    "أَفْعَال"     => [0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00],
    "مُفَعِّل"     => [0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
)

const WEIGHT_NAMES = collect(keys(WEIGHT_EMBEDDINGS))

const POS_WEIGHTS = Dict("verb" => "فَعَلَ", "adj" => "فَعِيل", "noun" => "مَفْعُول")

const WEIGHT_TRANSITIONS = Dict{String,Dict{String,Float64}}(
    "فَعَلَ"      => Dict("فَعَلَ" => 0.5, "فَعَّلَ" => 0.6, "فَاعَلَ" => 0.4, "أَفْعَلَ" => 0.5, "مَفْعُول" => 0.9, "فَعِيل" => 0.7, "فَعَّال" => 0.6, "أَفْعَال" => 0.3),
    "فَعَّلَ"     => Dict("فَعَلَ" => 0.4, "فَعَّلَ" => 0.3, "تَفَعَّلَ" => 0.8, "مَفْعُول" => 0.6, "فَعَّال" => 0.8, "مُفَعِّل" => 0.9),
    "فَاعَلَ"     => Dict("فَاعَلَ" => 0.4, "فَعَلَ" => 0.5, "مَفْعُول" => 0.6, "فَعِيل" => 0.5, "تَفَاعَلَ" => 0.7),
    "أَفْعَلَ"    => Dict("فَعَلَ" => 0.4, "أَفْعَلَ" => 0.3, "مَفْعُول" => 0.7, "أَفْعَال" => 0.8, "فَعِيل" => 0.5),
    "تَفَعَّلَ"   => Dict("فَعَلَ" => 0.6, "تَفَعَّلَ" => 0.3, "مَفْعُول" => 0.5, "فَعَّال" => 0.5),
    "تَفَاعَلَ"   => Dict("فَاعَلَ" => 0.7, "فَعَلَ" => 0.5, "مَفْعُول" => 0.4),
    "اِنْفَعَلَ"  => Dict("فَعَلَ" => 0.7, "اِنْفَعَلَ" => 0.3, "مَفْعُول" => 0.5),
    "اِفْتَعَلَ"  => Dict("فَعَلَ" => 0.7, "اِفْتَعَلَ" => 0.3, "مَفْعُول" => 0.6, "مُفَعِّل" => 0.4),
    "مَفْعُول"    => Dict("فَعَلَ" => 0.3, "مَفْعُول" => 0.2, "فَعِيل" => 0.4, "فُعُول" => 0.6, "أَفْعَال" => 0.7),
    "فَعِيل"      => Dict("فَعَلَ" => 0.4, "فَعِيل" => 0.3, "فَعَّال" => 0.5, "أَفْعَال" => 0.6),
    "فَعَّال"     => Dict("فَعَلَ" => 0.3, "فَعَّال" => 0.3, "أَفْعَال" => 0.7, "مَفْعُول" => 0.4),
    "فُعُول"      => Dict("مَفْعُول" => 0.3, "فُعُول" => 0.2, "فَعِيل" => 0.4),
    "أَفْعَال"    => Dict("مَفْعُول" => 0.3, "أَفْعَال" => 0.2, "فَعِيل" => 0.4, "فَعَّال" => 0.3),
    "مُفَعِّل"    => Dict("فَعَّلَ" => 0.5, "مُفَعِّل" => 0.3, "مَفْعُول" => 0.7),
)

"""
    WeightResonanceEngine

محرك الرنين الصرفي — يحسب resonance و transition_score للأوزان العربية.

الحقول:
- `morpho`: المحلل الصرفي (اختياري)
- `weight_pv_cache`: متجهات طورية مُطبَّعة للأوزان الـ15
"""
struct WeightResonanceEngine
    morpho::Any
    weight_pv_cache::Dict{String,Vector{Float64}}
end

WeightResonanceEngine(; morpho=nothing) = WeightResonanceEngine(morpho, _build_cache())

function _build_cache()
    cache = Dict{String,Vector{Float64}}()
    for (name, vec) in WEIGHT_EMBEDDINGS
        nrm = norm(vec)
        cache[name] = nrm > 1e-10 ? vec ./ nrm : copy(vec)
    end
    return cache
end

"""
    get_weight(engine, word) -> Union{String,Nothing}

استخراج الوزن الصرفي لكلمة (يتطلب morpho analyzer).
"""
function get_weight(engine::WeightResonanceEngine, word::String)
    if engine.morpho === nothing
        return nothing
    end
    analysis = engine.morpho.analyze(word)
    if !get(analysis, "has_morph", false)
        return nothing
    end
    return get(analysis, "weight", nothing)
end

"""
    get_weight_pv(engine, weight_name) -> Vector{Float64}

متجه طوري للوزن الصرفي (22D، مُطبَّع).
"""
function get_weight_pv(engine::WeightResonanceEngine, weight_name::String)
    return get(engine.weight_pv_cache, weight_name, zeros(Float64, 22))
end

"""
    resonance(engine, w1_weight, w2_weight) -> Float64

رنين صرفي بين وزنين = mean(cos(pv1 - pv2)).
"""
function resonance(engine::WeightResonanceEngine, w1_weight::String, w2_weight::String)
    if !haskey(WEIGHT_EMBEDDINGS, w1_weight) || !haskey(WEIGHT_EMBEDDINGS, w2_weight)
        return 0.0
    end
    pv1 = engine.weight_pv_cache[w1_weight]
    pv2 = engine.weight_pv_cache[w2_weight]
    n1 = norm(pv1); n2 = norm(pv2)
    if n1 < 1e-10 || n2 < 1e-10; return 0.0; end
    return dot(pv1, pv2) / (n1 * n2)
end

"""
    transition_score(engine, prev_weight, word_weight) -> Float64

درجة الانتقال الصرفي بين وزنين من الجدول الانتقالي (0..0.9).
"""
function transition_score(engine::WeightResonanceEngine, prev_weight::String, word_weight::String)
    trans = get(WEIGHT_TRANSITIONS, prev_weight, Dict())
    return get(trans, word_weight, 0.0)
end

"""
    weight_density(engine, words) -> Float64

كثافة الأوزان الصرفية في قائمة كلمات = تردد الوزن الأكثر شيوعاً / العدد الكلي.
"""
function weight_density(engine::WeightResonanceEngine, words::Vector{String})
    counts = Dict{String,Int}()
    for w in words
        weight = get_weight(engine, w)
        if weight !== nothing
            counts[weight] = get(counts, weight, 0) + 1
        end
    end
    if isempty(counts)
        return 0.0
    end
    max_count = maximum(values(counts))
    return max_count / max(length(words), 1)
end

end # module WeightResonance
