"""
محرك المؤثرات السياقية — Contextual Operators Engine.

198 مؤثراً سياقياً (نفي، تضخيم، تباين، شرط) تؤثر على المتجهات
الطورية وتخلق جسوراً سببية بين المتضادات.

المؤثرات تُحمّل من JSON خارجي (contextual_operators.json).
الأنواع: negation, amplify, contrast, condition, conjecture.
"""
module ContextualOperators

using JSON, LinearAlgebra

export ContextualOperatorEngine, OPERATOR_TYPES

const OPERATOR_TYPES = Set(["negation", "amplify", "contrast", "condition", "conjecture"])

"""
    ContextualOperatorEngine

محرك المؤثرات السياقية — يطبق تعديلات على المتجهات الطورية
حسب نوع الكلمة ودورها السياقي.

الحقول:
- `operators`: op_type → {strength, description}
- `word_to_op`: word → (op_type, strength)
"""
mutable struct ContextualOperatorEngine
    operators::Dict{String,Dict{String,Any}}
    word_to_op::Dict{String,Tuple{String,Float64}}
end

"""
    ContextualOperatorEngine(; json_path=nothing)

تحميل المؤثرات من ملف JSON. إذا لم يوجد، يُنشئ محركاً فارغاً.
"""
function ContextualOperatorEngine(; json_path::Union{String,Nothing}=nothing)
    ops = Dict{String,Dict{String,Any}}()
    w2o = Dict{String,Tuple{String,Float64}}()

    if json_path !== nothing && isfile(json_path)
        data = JSON.parsefile(json_path)
        ops_data = get(data, "operators", Dict())
        for (op_type, op_info) in ops_data
            strength = get(op_info, "strength", 0.0)
            words = get(op_info, "words", [])
            ops[op_type] = Dict(
                "description" => get(op_info, "description", ""),
                "strength" => strength,
                "words" => Set{String}(words),
            )
            for w in words
                w2o[w] = (op_type, strength)
            end
        end
    end

    return ContextualOperatorEngine(ops, w2o)
end

# ─── 198 مؤثراً مضمّناً ───
const BUILTIN_OPERATORS = Dict{String,Dict}(
    "negation" => Dict(
        "description" => "نفي/إلغاء", "strength" => -0.8,
        "words" => [
            "لا", "لم", "لن", "ما", "ليس", "غير", "إن", "لات",
            "غياب", "عدم", "انعدام", "فقدان", "غيـب", "غاب", "غائب",
            "اختفى", "زوال", "تلاشى", "اضمحل", "انقض", "ذهب",
            "انتهى", "خلا", "فرغ", "فنى", "زال", "انقطع",
            "نفى", "ينفي", "منفي", "ينكر", "أنكر", "نكران",
            "بلا", "بدون", "دون", "سوى", "عدا", "إلا",
            "أبداً", "قط", "عوض", "ما عاد", "ما زال",
            "حرمان", "منع", "حظر", "امتناع", "كف",
        ],
    ),
    "amplify" => Dict(
        "description" => "تضخيم/تقوية", "strength" => 1.5,
        "words" => [
            "جداً", "كثيراً", "شديد", "قوي", "عظيم", "هائل", "جبار",
            "يطغى", "يشتد", "يقوى", "يعظم", "يتضخم", "يتفاقم", "يستفحل",
            "يزداد", "يتزايد", "يتكاثر", "يتنامى", "يتصاعد",
            "ضخم", "كبير", "عظيم", "مضاعف", "معزز", "مكبر",
            "أكثر", "أشد", "أقوى", "أعظم", "أكبر", "أضخم",
            "بإفراط", "بإسراف", "بشدة", "بقوة", "بعنف",
            "فائق", "مفرط", "مبالغ", "استثنائي", "خارق",
        ],
    ),
    "contrast" => Dict(
        "description" => "تباين/مقابلة", "strength" => -0.4,
        "words" => [
            "لكن", "لكنَّ", "بل", "رغم", "بالرغم", "على_الرغم",
            "بينما", "في_حين", "إلا_أن", "غير_أن", "مع_أن",
            "بيد_أن", "على_أن", "إذ", "فيما", "حين",
            "بعكس", "بخلاف", "عكس", "مقابل", "نقيض",
            "ومع_ذلك", "وبرغم_ذلك", "وبالرغم", "لا_كن",
            "في_المقابل", "بالعكس", "على_العكس", "على_النقيض",
            "رغم_أنف", "شئت_أم_أبيت", "طوعاً_أو_كرهاً",
            "ولكن", "فبينما", "حيث_أن", "في_حين_أن",
            "أما", "فأما", "إما", "سواء",
        ],
    ),
    "condition" => Dict(
        "description" => "شرط/احتمال", "strength" => 0.6,
        "words" => [
            "إذا", "إن", "لو", "لولا", "كلما", "متى", "أيان",
            "حينما", "أينما", "كيفما", "أي", "أيما", "مهما",
            "متى_ما", "أنى", "حيثما", "إذما", "ما_دام",
            "شرط", "بشرط", "شريطة", "بافتراض", "بفرض",
            "إذا_ما", "لو_أن", "لأن", "بسبب", "نتيجة",
            "ربما", "لعل", "عسى", "قد", "يمكن", "احتمال",
            "من_الممكن", "من_المحتمل", "يجوز", "قد_يكون",
        ],
    ),
    "conjecture" => Dict(
        "description" => "تخمين/استفهام", "strength" => 0.3,
        "words" => [
            "هل", "أ", "كيف", "لماذا", "ماذا", "من", "أين", "متى",
            "كم", "أي", "أنى", "أيان", "علام", "فيم", "بم",
            "عم", "مم", "إلام", "حتام", "لِم", "أفلا",
            "أليس", "أولم", "أفلم", "ألن", "ألم",
            "ربما", "لعل", "عسى", "قد", "قد_يكون",
            "أترى", "أرأيت", "ألا_ترى", "أو_لا_ترى",
            "ما_رأيك", "ما_قولك", "أظن", "أحسب",
        ],
    ),
)

"""
    init_builtin!(engine::ContextualOperatorEngine)

تحميل المؤثرات الـ 198 المضمنة.
"""
function init_builtin!(engine::ContextualOperatorEngine)
    for (op_type, op_data) in BUILTIN_OPERATORS
        strength = op_data["strength"]
        words = Set{String}(op_data["words"])
        engine.operators[op_type] = Dict(
            "description" => op_data["description"],
            "strength" => strength,
            "words" => words,
        )
        for w in words
            engine.word_to_op[w] = (op_type, strength)
        end
    end
    return engine
end

"""
    get_operator(engine, word) -> Union{Tuple{String,Float64},Nothing}

الاستعلام عن مؤثر الكلمة.
"""
function get_operator(engine::ContextualOperatorEngine, word::String)
    return get(engine.word_to_op, word, nothing)
end

"""
    has_operator(engine, word) -> Bool
"""
has_operator(engine::ContextualOperatorEngine, word::String) = haskey(engine.word_to_op, word)

"""
    detect_chain(engine, context_words) -> Vector{Tuple{Int,String,String,Float64}}

كشف سلسلة المؤثرات في السياق.
تُرجع: [(word_index, word, operator_type, strength), ...]
"""
function detect_chain(engine::ContextualOperatorEngine, context_words::Vector{String})
    chain = Tuple{Int,String,String,Float64}[]
    for (i, w) in enumerate(context_words)
        op = get(engine.word_to_op, w, nothing)
        if op !== nothing
            push!(chain, (i, w, op[1], op[2]))
        end
    end
    return chain
end

"""
    apply_negation(engine, pv, strength) -> Vector{Float64}

تطبيق مؤثر النفي — عكس الأبعاد المهيمنة (أعلى 5 قيم مطلقة).
"""
function apply_negation(engine::ContextualOperatorEngine, pv::AbstractVector, strength::Float64)
    indices = sortperm(abs.(pv); rev=true)[1:min(5, end)]
    adjusted = Float64.(pv)
    adjusted[indices] .*= (1.0 + strength)  # strength سالب → عكس
    nrm = norm(adjusted)
    return nrm > 1e-10 ? adjusted ./ nrm : adjusted
end

"""
    apply_amplify(engine, pv, strength) -> Vector{Float64}

تطبيق مؤثر التضخيم — تقوية المتجه بالكامل.
"""
function apply_amplify(engine::ContextualOperatorEngine, pv::AbstractVector, strength::Float64)
    adjusted = Float64.(pv) .* strength
    nrm = norm(adjusted)
    return nrm > 1e-10 ? adjusted ./ nrm : adjusted
end

"""
    apply(engine, candidate_pv, context_words, operator_pvs) -> Pair

تطبيق كل المؤثرات في السياق على المتجه المرشح.
تُرجع: (adjusted_pv, boost)
"""
function apply(engine::ContextualOperatorEngine, candidate_pv::AbstractVector,
               context_words::Vector{String}, operator_pvs=nothing)
    chain = detect_chain(engine, context_words)
    if isempty(chain)
        return candidate_pv, 0.0
    end

    adjusted = Float64.(candidate_pv)
    boost = 0.0

    for (_, _, op_type, strength) in chain
        if op_type == "negation"
            adjusted = apply_negation(engine, adjusted, strength)
            boost += abs(strength) * 0.3
        elseif op_type == "amplify"
            adjusted = apply_amplify(engine, adjusted, strength)
            boost += strength * 0.2
        elseif op_type == "contrast"
            boost += abs(strength) * 0.15
        elseif op_type == "condition"
            boost += strength * 0.1
        elseif op_type == "conjecture"
            boost += strength * 0.05
        end
    end

    return adjusted, boost
end

"""
    compute_boost(engine, candidate_word, context_words) -> Float64

حساب تعزيز المؤثرات السياقية لكلمة مرشحة.
"""
function compute_boost(engine::ContextualOperatorEngine, candidate_word::String,
                       context_words::Vector{String})
    chain = detect_chain(engine, context_words)
    if isempty(chain)
        return 0.0
    end

    boost = 0.0
    for (_, _, op_type, strength) in chain
        if op_type == "negation";       boost += abs(strength) * 0.3
        elseif op_type == "amplify";    boost += strength * 0.2
        elseif op_type == "contrast";   boost += abs(strength) * 0.15
        elseif op_type == "condition";  boost += strength * 0.1
        elseif op_type == "conjecture"; boost += strength * 0.05
        end
    end

    return clamp(boost, 0.0, 1.0)
end

end # module ContextualOperators
