"""
clifford_enhanced.jl - دمج جبر كليفورد مع النموذج المعزز
كل حرف = Multivector22 (متجه كليفورد 22D)
الكلمة = جداء هندسي للحروف (Geometric Product)

الفكرة:
1. كل حرف له متجه كليفورد فريد (22D vector + 231D bivector)
2. الكلمة = ح1 × ح2 × ح3 ... (geometric product)
3. الجداء الهندسي يحافظ على الترتيب ويُنتج متجهاً فريداً
"""

module CliffordMath

using LinearAlgebra

using ..WordPhysics

include("clifford_math_base.jl")
using .CliffordMath

export letter_to_multivector, word_to_multivector, 
       clifford_distance, clifford_similarity,
       enhanced_geometric_product, compose_word_clifford,
       from_vector, get_scalar_essence

# ─────────────────────────────────────────────────────────
# تحويل الحرف إلى Multivector22
# ─────────────────────────────────────────────────────────

"""
    letter_to_multivector(letter::Char; alpha::Float64=1.0) -> Multivector22
تحويل الحرف إلى متجه كليفورد
المتجه يحتوي على:
- s: 0 (سلمي)
- v: 22D vector (من المعاملات بعد الأسي)
- b: 231D bivector (الجداء الخارجي بين أجزاء المتجه)
- p: 0 (pseudoscalar)
"""
function letter_to_multivector(letter::Char; alpha::Float64=1.0)
    # حساب المعاملات بعد العامل الأسي (27D)
    params = WordPhysics.get_letter_params(letter)
    if params === nothing
        return Multivector22(0.0, zeros(22), zeros(231), 0.0)
    end
    
    # تحويل المعاملات إلى متجه 27D مع العامل الأسي
    raw = zeros(27)
    for (i, p) in enumerate(params)
        for (j, val) in enumerate(p)
            raw[(i-1)*9 + j] = val
        end
    end
    exp_params = WordPhysics.apply_exponential_factor(raw, alpha=alpha)
    
    # تحويل 27D إلى 22D (取前22个维度)
    # نحتاج 22 بُعداً فقط لـ Multivector22
    vec_22 = zeros(22)
    vec_22[1:min(22, length(exp_params))] .= exp_params[1:min(22, length(exp_params))]
    
    # إنشاء bivector من الأبعاد المتبقية
    # نستخدم آخر 5 أبعاد لإنشاء bivector
    bivector = zeros(231)
    if length(exp_params) > 22
        extra = exp_params[23:min(27, length(exp_params))]
        # إنشاء bivector بسيط من الأبعاد الإضافية
        for i in 1:min(5, length(extra))
            bivector[i] = extra[i]
        end
    end
    
    return Multivector22(0.0, vec_22, bivector, 0.0)
end

# ─────────────────────────────────────────────────────────
# الجداء الهندسي المحسّن
# ─────────────────────────────────────────────────────────

"""
    enhanced_geometric_product(mv1::Multivector22, mv2::Multivector22) -> Multivector22
الجداء الهندسي المحسّن مع تطبيع
"""
function enhanced_geometric_product(mv1::Multivector22, mv2::Multivector22)
    result = mv1 * mv2
    
    # تطبيع النتيجة لمنع الانفجار
    n = norm(result)
    if n > 1e10
        normalize!(result)
    end
    
    return result
end

"""
    compose_word_clifford(letters::Vector{Char}; alpha::Float64=1.0) -> Multivector22
تجميع الكلمة باستخدام جداء كليفورد
الكلمة = ح1 × ح2 × ح3 × ... (geometric product)
"""
function compose_word_clifford(letters::Vector{Char}; alpha::Float64=1.0)
    if isempty(letters)
        return Multivector22(0.0, zeros(22), zeros(231), 0.0)
    end
    
    # تحويل الحرف الأول
    result = letter_to_multivector(letters[1], alpha=alpha)
    
    # الجداء مع باقي الحروف
    for i in 2:length(letters)
        mv = letter_to_multivector(letters[i], alpha=alpha)
        result = enhanced_geometric_product(result, mv)
    end
    
    return result
end

# ─────────────────────────────────────────────────────────
# تحويل الكلمة إلى متجه كليفورد
# ─────────────────────────────────────────────────────────

"""
    word_to_multivector(word::String; alpha::Float64=1.0) -> Multivector22
تحويل الكلمة إلى متجه كليفورد
"""
function word_to_multivector(word::String; alpha::Float64=1.0)
    letters = collect(filter(c -> !isspace(c), word))
    return compose_word_clifford(letters, alpha=alpha)
end

# ─────────────────────────────────────────────────────────
# حساب المسافة والتشابه بين كلمتين
# ─────────────────────────────────────────────────────────

"""
    clifford_distance(mv1::Multivector22, mv2::Multivector22) -> Float64
المسافة بين متجهي كليفورد (Euclidean distance)
"""
function clifford_distance(mv1::Multivector22, mv2::Multivector22)
    # حساب الفرق
    diff = mv1 - mv2
    return norm(diff)
end

"""
    clifford_similarity(mv1::Multivector22, mv2::Multivector22) -> Float64
التشابه بين متجهي كليفورد (cosine similarity)
"""
function clifford_similarity(mv1::Multivector22, mv2::Multivector22)
    # حساب الجداء الداخلي
    dot_val = mv1.s * mv2.s + dot(mv1.v, mv2.v) + dot(mv1.b, mv2.b) + mv1.p * mv2.p
    n1 = norm(mv1)
    n2 = norm(mv2)
    
    if n1 < 1e-10 || n2 < 1e-10
        return 0.0
    end
    
    return clamp(dot_val / (n1 * n2), -1.0, 1.0)
end

# ─────────────────────────────────────────────────────────
# استخراج متجه من Multivector22
# ─────────────────────────────────────────────────────────

"""
    multivector_to_vector(mv::Multivector22) -> Vector{Float64}
تحويل Multivector22 إلى متجه عادي (454D = 1 + 22 + 231)
"""
function multivector_to_vector(mv::Multivector22)
    return vcat([mv.s], mv.v, mv.b, [mv.p])
end

"""
    multivector_to_vector_22(mv::Multivector22) -> Vector{Float64}
تحويل Multivector22 إلى متجه 22D فقط (المتجه الأساسي)
"""
function multivector_to_vector_22(mv::Multivector22)
    return mv.v
end

end # module CliffordMath
