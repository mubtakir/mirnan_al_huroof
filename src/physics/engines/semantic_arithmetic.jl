"""
Semantic Arithmetic — الحساب الدلالي.
Operations: difference, analogy, midpoint, blending, antonym prediction.
Simplified version using 27D phase vectors.
"""
module SemanticArithmetic

using LinearAlgebra, Statistics
using ..WordPhysics: compute_extended_phase_vector

export semantic_difference, solve_analogy, semantic_midpoint, blend_words,
       predict_antonym, FAST_ANTONYMS, analyze_semantic_affinity,
       classify_word_field, build_general_antinomy_operator,
       build_field_specific_antinomy_operators

const FAST_ANTONYMS = Dict{String,String}(
    "قوة" => "ضعف", "ضعف" => "قوة",
    "نور" => "ظلام", "ظلام" => "نور",
    "علم" => "جهل", "جهل" => "علم",
    "كبر" => "صغر", "صغر" => "كبر",
    "صدق" => "كذب", "كذب" => "صدق",
    "حر" => "برد", "برد" => "حر",
    "حياة" => "موت", "موت" => "حياة",
    "حق" => "باطل", "باطل" => "حق",
    "غنى" => "فقر", "فقر" => "غنى",
    "طول" => "قصر", "قصر" => "طول",
    "شجاعة" => "جبن", "جبن" => "شجاعة",
    "كرم" => "بخل", "بخل" => "كرم",
    "حكمة" => "حمق", "حمق" => "حكمة",
    "عدل" => "ظلم", "ظلم" => "عدل",
    "سماء" => "أرض", "أرض" => "سماء",
    "شمس" => "قمر", "قمر" => "شمس",
    "صيف" => "شتاء", "شتاء" => "صيف",
    "ليل" => "نهار", "نهار" => "ليل",
    "خير" => "شر", "شر" => "خير",
    "حرب" => "سلام", "سلام" => "حرب",
    "صديق" => "عدو", "عدو" => "صديق",
    "قريب" => "بعيد", "بعيد" => "قريب",
    "سريع" => "بطيء", "بطيء" => "سريع",
    "سهل" => "صعب", "صعب" => "سهل",
    "كثير" => "قليل", "قليل" => "كثير",
    "جميل" => "قبيح", "قبيح" => "جميل",
    "جديد" => "قديم", "قديم" => "جديد",
    "سعيد" => "حزين", "حزين" => "سعيد",
    "واسع" => "ضيق", "ضيق" => "واسع",
    "ثقيل" => "خفيف", "خفيف" => "ثقيل",
    "حلو" => "مر", "مر" => "حلو",
    "صحيح" => "خطأ", "خطأ" => "صحيح",
    "حي" => "ميت", "ميت" => "حي",
    "قوي" => "ضعيف", "ضعيف" => "قوي",
    "كبير" => "صغير", "صغير" => "كبير",
    "غني" => "فقير", "فقير" => "غني",
    "حزن" => "فرح", "فرح" => "حزن",
)

function classify_word_field(word::String)
    epistemology = Set(["علم", "جهل", "حكمة", "فكر", "عقل"])
    sensory = Set(["نور", "ظلام", "صوت", "لون", "حر", "برد"])
    state = Set(["حياة", "موت", "قوة", "ضعف", "سلام", "حرب"])
    word in epistemology && return "epistemology"
    word in sensory && return "sensory"
    word in state && return "state"
    return "state"
end

function build_general_antinomy_operator(pairs)
    ops = Dict{String,Any}()
    for (a, b) in pairs
        ops["$a::$b"] = Dict("from" => a, "to" => b, "field" => classify_word_field(a))
    end
    return ops
end

function build_field_specific_antinomy_operators(pairs)
    grouped = Dict{String,Vector{Tuple{String,String}}}()
    for (a, b) in pairs
        field = classify_word_field(a)
        push!(get!(grouped, field, Tuple{String,String}[]), (a, b))
    end
    return Dict(field => build_general_antinomy_operator(ps) for (field, ps) in grouped)
end

function semantic_difference(word_from::String, word_to::String, pv_fn::Function)
    v_from = Float64.(pv_fn(word_from))
    v_to = Float64.(pv_fn(word_to))
    delta = v_to .- v_from
    mag = norm(delta)
    return Dict("method" => "linear", "word_from" => word_from, "word_to" => word_to,
                "magnitude" => round(mag; digits=4))
end

function solve_analogy(a::String, b::String, c::String, pv_fn::Function;
                       vocab::Union{Dict,Nothing}=nothing)
    v_a = Float64.(pv_fn(a))
    v_b = Float64.(pv_fn(b))
    v_c = Float64.(pv_fn(c))
    delta_v = v_b .- v_a
    v_d = v_c .+ delta_v
    v_d_norm = v_d ./ (norm(v_d) + 1e-10)

    candidates = Tuple{Float64,String}[]
    if vocab !== nothing
        for (word, _) in vocab
            length(word) < 2 && continue
            word in (a, b, c) && continue
            wv = Float64.(pv_fn(word))
            sim = dot(v_d_norm, wv) / (norm(wv) + 1e-10)
            push!(candidates, (sim, word))
        end
        sort!(candidates; by=x -> -x[1])
    end

    nearest = isempty(candidates) ? "?" : candidates[1][2]
    conf = isempty(candidates) ? 0.0 : candidates[1][1]
    return Dict("analogy" => "$a : $b :: $c : ?", "method" => "linear",
                "predicted" => nearest, "confidence" => round(conf; digits=4),
                "delta_norm" => round(norm(delta_v); digits=4))
end

function semantic_midpoint(word1::String, word2::String, pv_fn::Function)
    v1 = Float64.(pv_fn(word1))
    v2 = Float64.(pv_fn(word2))
    mid = (v1 .+ v2) ./ 2
    nrm = norm(mid)
    if nrm > 1e-10; mid ./= nrm; end
    return Dict("word1" => word1, "word2" => word2, "midpoint_norm" => round(norm(mid); digits=4))
end

function blend_words(word1::String, word2::String, pv_fn::Function; ratio::Float64=0.5)
    v1 = Float64.(pv_fn(word1))
    v2 = Float64.(pv_fn(word2))
    blend = v1 .* (1 - ratio) .+ v2 .* ratio
    nrm = norm(blend)
    if nrm > 1e-10; blend ./= nrm; end
    return Dict("word1" => word1, "word2" => word2, "ratio" => ratio,
                "blend_norm" => round(norm(blend); digits=4))
end

function predict_antonym(word::String)
    if haskey(FAST_ANTONYMS, word)
        return [(FAST_ANTONYMS[word], 1.0)]
    end
    return Tuple{String,Float64}[]
end

function analyze_semantic_affinity(word1::String, word2::String, pv_fn::Function)
    v1 = Float64.(pv_fn(word1))
    v2 = Float64.(pv_fn(word2))
    sim = dot(v1, v2) / (norm(v1) * norm(v2) + 1e-10)

    affinity_class = if sim > 0.4
        "close_synonym"
    elseif sim > 0.15
        "semantic_affinity"
    elseif sim < -0.25
        "divergence_antonym"
    else
        "nuanced_relation"
    end

    return Dict(
        "word1" => word1, "word2" => word2,
        "similarity" => round(sim; digits=4),
        "affinity_class" => affinity_class,
    )
end

function analyze_semantic_affinity(word1::String, word2::String)
    return analyze_semantic_affinity(word1, word2, compute_extended_phase_vector)
end

end # module SemanticArithmetic
