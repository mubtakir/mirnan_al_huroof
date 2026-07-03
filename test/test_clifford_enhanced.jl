"""
test_clifford_enhanced.jl - اختبار النموذج المتكامل مع Clifford algebra
"""

# ─────────────────────────────────────────────────────────
# تعريف Multivector22 مباشرة لتجنب مشاكل المسار
# ─────────────────────────────────────────────────────────
module CliffordLocal

using LinearAlgebra

export Multivector22, norm, normalize!, geometric_product, 
       clifford_distance, clifford_similarity

const CLIFFORD_DIM = 22
const BIVECTOR_DIM = CLIFFORD_DIM * (CLIFFORD_DIM - 1) ÷ 2

mutable struct Multivector22
    s::Float64
    v::Vector{Float64}
    b::Vector{Float64}
    p::Float64
    
    function Multivector22(scalar::Real=0.0, vector=nothing, bivector=nothing, pseudoscalar::Real=0.0)
        v_arr = zeros(Float64, CLIFFORD_DIM)
        if vector !== nothing
            vec_v = Float64.(collect(vector))
            n = min(length(vec_v), CLIFFORD_DIM)
            v_arr[1:n] .= vec_v[1:n]
        end
        
        b_arr = zeros(Float64, BIVECTOR_DIM)
        if bivector !== nothing
            vec_b = Float64.(collect(bivector))
            n = min(length(vec_b), BIVECTOR_DIM)
            b_arr[1:n] .= vec_b[1:n]
        end
        
        return new(Float64(scalar), v_arr, b_arr, Float64(pseudoscalar))
    end
end

function LinearAlgebra.norm(mv::Multivector22)
    return sqrt(mv.s^2 + sum(mv.v .^ 2) + sum(mv.b .^ 2) + mv.p^2)
end

function normalize!(mv::Multivector22)
    n = norm(mv)
    if n > 0
        mv.s /= n
        mv.v ./= n
        mv.b ./= n
        mv.p /= n
    end
    return mv
end

function reverse(mv::Multivector22)
    return Multivector22(mv.s, copy(mv.v), -copy(mv.b), 0.0)
end

function wedge_product(v1::AbstractVector, v2::AbstractVector)
    b = zeros(Float64, BIVECTOR_DIM)
    k = 0
    for i in 1:CLIFFORD_DIM
        for j in (i+1):CLIFFORD_DIM
            k += 1
            b[k] = v1[i] * v2[j] - v1[j] * v2[i]
        end
    end
    return b
end

function contract_bv(b::Vector{Float64}, v::Vector{Float64})
    res = zeros(Float64, CLIFFORD_DIM)
    k = 0
    for i in 1:CLIFFORD_DIM
        for j in (i+1):CLIFFORD_DIM
            k += 1
            val = b[k]
            res[i] -= val * v[j]
            res[j] += val * v[i]
        end
    end
    return res
end

function geometric_product(a::Multivector22, b::Multivector22)
    # Scalar
    res_s = a.s * b.s + dot(a.v, b.v)
    # Vector
    res_v = a.s .* b.v .+ b.s .* a.v
    res_v .+= contract_bv(a.b, b.v)
    res_v .-= contract_bv(b.b, a.v)
    # Bivector
    res_b = a.s .* b.b .+ b.s .* a.b
    res_b .+= wedge_product(a.v, b.v)
    # Pseudoscalar
    res_p = a.s * b.p + b.s * a.p
    
    return Multivector22(res_s, res_v, res_b, res_p)
end

function Base.:(+)(a::Multivector22, b::Multivector22)
    return Multivector22(a.s + b.s, a.v + b.v, a.b + b.b, a.p + b.p)
end

function Base.:(-)(a::Multivector22, b::Multivector22)
    return Multivector22(a.s - b.s, a.v - b.v, a.b - b.b, a.p - b.p)
end

function clifford_distance(mv1::Multivector22, mv2::Multivector22)
    diff = mv1 - mv2
    return norm(diff)
end

function clifford_similarity(mv1::Multivector22, mv2::Multivector22)
    dot_val = mv1.s * mv2.s + dot(mv1.v, mv2.v) + dot(mv1.b, mv2.b) + mv1.p * mv2.p
    n1 = norm(mv1)
    n2 = norm(mv2)
    
    if n1 < 1e-10 || n2 < 1e-10
        return 0.0
    end
    
    return clamp(dot_val / (n1 * n2), -1.0, 1.0)
end

end # module CliffordLocal

# ─────────────────────────────────────────────────────────
# قاعدة بيانات الحروف العربية (من letter_equations.jl)
# ─────────────────────────────────────────────────────────
module LetterDB

export get_letter_params, apply_exponential, list_all_letters

# المعاملات الأساسية للحروف العربية (3 حدود × 9 معاملات)
const LETTER_DB = Dict{Char, Vector{Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64}}}(
    # ═══ حروف مفخّمة ═══
    'ض' => [
        (3.0,  2.5,  30.0, -0.8,  3.5, 2.0, 35.0, -0.5,  6.0),
        (2.0,  2.0,  20.0,  0.9,  2.5, 1.5, 25.0,  0.4,  5.0),
        (1.0,  1.0,  10.0,  0.0,  1.2, 0.8, 12.0,  0.0,  3.5),
    ],
    'ص' => [
        (2.8,  2.3,  28.0, -0.7,  3.2, 1.8, 32.0, -0.45, 5.5),
        (1.8,  1.8,  18.0,  0.8,  2.2, 1.3, 22.0,  0.35, 4.5),
        (0.9,  0.9,   9.0,  0.0,  1.0, 0.7, 10.0,  0.0,  3.0),
    ],
    'ط' => [
        (2.9,  2.4,  29.0, -0.75, 3.4, 1.9, 34.0, -0.48, 5.8),
        (1.9,  1.9,  19.0,  0.85, 2.4, 1.4, 24.0,  0.38, 4.8),
        (0.95, 0.95, 9.5,   0.0,  1.1, 0.75,11.0,  0.0,  3.2),
    ],
    'ظ' => [
        (2.85, 2.35, 27.0, -0.72, 3.3, 1.85,31.0, -0.42, 5.6),
        (1.85, 1.85, 17.0,  0.82, 2.3, 1.35,21.0,  0.32, 4.6),
        (0.92, 0.92, 8.5,   0.0,  1.05,0.72, 9.5,  0.0,  3.1),
    ],
    
    # ═══ حروف حلقية/حنجرية ═══
    'ع' => [
        (2.5,  2.0,  26.0,  1.8,  3.0, 1.5, 15.0,  0.8,  4.0),
        (1.6,  1.3,  16.0,  1.2,  2.0, 1.0,  9.0,  1.0,  3.0),
        (0.8,  0.65, 8.0,   0.0,  0.9, 0.5,  4.5,  1.2,  2.0),
    ],
    'غ' => [
        (2.6,  2.1,  25.0,  1.9,  3.1, 1.6, 16.0,  0.85, 4.2),
        (1.7,  1.4,  15.0,  1.3,  2.1, 1.1, 10.0,  1.05, 3.2),
        (0.85, 0.7,  7.5,   0.0,  0.95,0.55, 5.0,  1.3,  2.2),
    ],
    'ح' => [
        (2.3,  1.9,  24.0,  1.7,  2.8, 1.4, 14.0,  0.7,  3.8),
        (1.5,  1.2,  14.0,  1.1,  1.9, 0.95, 8.5,  0.95, 2.8),
        (0.75, 0.6,  7.0,   0.0,  0.85,0.45, 4.0,  1.1,  1.8),
    ],
    'خ' => [
        (2.4,  2.0,  23.0,  1.75, 2.9, 1.45,14.5,  0.75, 3.9),
        (1.55, 1.25, 14.5,  1.15, 1.95,1.0,  8.8,  1.0,  2.9),
        (0.78, 0.62, 7.2,   0.0,  0.88,0.48, 4.2,  1.15, 1.9),
    ],
    'ع' => [
        (2.5,  2.0,  26.0,  1.8,  3.0, 1.5, 15.0,  0.8,  4.0),
        (1.6,  1.3,  16.0,  1.2,  2.0, 1.0,  9.0,  1.0,  3.0),
        (0.8,  0.65, 8.0,   0.0,  0.9, 0.5,  4.5,  1.2,  2.0),
    ],
    'غ' => [
        (2.6,  2.1,  25.0,  1.9,  3.1, 1.6, 16.0,  0.85, 4.2),
        (1.7,  1.4,  15.0,  1.3,  2.1, 1.1, 10.0,  1.05, 3.2),
        (0.85, 0.7,  7.5,   0.0,  0.95,0.55, 5.0,  1.3,  2.2),
    ],
    'ه' => [
        (1.2,  1.0,  12.0,  0.5,  1.5, 0.8,  8.0,  0.3,  2.5),
        (0.8,  0.6,   6.0,  0.3,  1.0, 0.5,  4.0,  0.2,  2.0),
        (0.4,  0.3,   3.0,  0.0,  0.5, 0.25, 2.0,  0.1,  1.5),
    ],
    
    # ═══ حروف أنفية ═══
    'م' => [
        (1.8,  1.5,  18.0,  0.2,  2.2, 1.2, 20.0,  0.1,  3.5),
        (1.2,  1.0,  12.0,  0.15, 1.5, 0.8, 14.0,  0.08, 3.0),
        (0.6,  0.5,   6.0,  0.0,  0.8, 0.4,  7.0,  0.05, 2.5),
    ],
    'ن' => [
        (1.9,  1.6,  19.0,  0.25, 2.3, 1.3, 21.0,  0.12, 3.6),
        (1.3,  1.1,  13.0,  0.18, 1.6, 0.85,15.0,  0.1,  3.1),
        (0.65, 0.55, 6.5,   0.0,  0.85,0.45, 7.5,  0.06, 2.6),
    ],
    
    # ═══ حروف شفوية ═══
    'ب' => [
        (1.5,  1.3,  15.0,  0.0,  2.0, 1.0, 18.0,  0.0,  3.0),
        (1.0,  0.8,  10.0,  0.0,  1.3, 0.7, 12.0,  0.0,  2.5),
        (0.5,  0.4,   5.0,  0.0,  0.7, 0.35, 6.0,  0.0,  2.0),
    ],
    'ف' => [
        (1.6,  1.4,  16.0,  0.05, 2.1, 1.1, 19.0,  0.02, 3.1),
        (1.1,  0.9,  11.0,  0.03, 1.4, 0.75,13.0,  0.01, 2.6),
        (0.55, 0.45, 5.5,   0.0,  0.75,0.38, 6.5,  0.005,2.1),
    ],
    'و' => [
        (1.4,  1.2,  14.0, -0.1,  1.9, 0.9, 17.0, -0.05, 2.9),
        (0.9,  0.7,   9.0, -0.08, 1.2, 0.6, 11.0, -0.03, 2.4),
        (0.45, 0.35, 4.5,   0.0,  0.6, 0.3,  5.5, -0.01, 1.9),
    ],
    'ل' => [
        (2.0,  1.7,  20.0,  0.3,  2.5, 1.4, 22.0,  0.15, 3.8),
        (1.4,  1.2,  14.0,  0.2,  1.8, 1.0, 16.0,  0.1,  3.3),
        (0.7,  0.6,   7.0,  0.0,  0.9, 0.5,  8.0,  0.05, 2.8),
    ],
    'ر' => [
        (1.7,  1.5,  17.0, -0.2,  2.2, 1.2, 19.5, -0.1,  3.4),
        (1.1,  0.9,  11.0, -0.15, 1.5, 0.8, 13.0, -0.08, 2.9),
        (0.55, 0.45, 5.5,   0.0,  0.8, 0.4,  6.5, -0.04, 2.4),
    ],
    'ز' => [
        (1.75, 1.55, 17.5, -0.18, 2.25,1.25,20.0, -0.12, 3.5),
        (1.15, 0.95, 11.5, -0.13, 1.55,0.85,13.5, -0.09, 3.0),
        (0.58, 0.48, 5.8,   0.0,  0.82,0.42, 6.8, -0.045,2.5),
    ],
    'ج' => [
        (1.65, 1.45, 16.5,  0.1,  2.15,1.15,19.2,  0.05, 3.3),
        (1.05, 0.85, 10.5,  0.08, 1.45,0.78,12.8,  0.03, 2.8),
        (0.52, 0.42, 5.2,   0.0,  0.72,0.37, 6.3,  0.01, 2.3),
    ],
    'ش' => [
        (2.1,  1.8,  21.0,  0.35, 2.6, 1.5, 23.0,  0.18, 3.9),
        (1.45, 1.25, 14.5,  0.25, 1.85,1.05,16.5,  0.12, 3.4),
        (0.72, 0.62, 7.2,   0.0,  0.92,0.52, 8.2,  0.06, 2.9),
    ],
    'س' => [
        (2.05, 1.75, 20.5,  0.32, 2.55,1.45,22.5,  0.16, 3.85),
        (1.42, 1.22, 14.2,  0.22, 1.82,1.02,16.2,  0.11, 3.35),
        (0.71, 0.61, 7.1,   0.0,  0.91,0.51, 8.1,  0.055,2.85),
    ],
    
    # ═══ حروف لسانية ═══
    'ت' => [
        (1.55, 1.35, 15.5,  0.02, 2.05,1.05,18.5,  0.01, 3.05),
        (1.05, 0.85, 10.5,  0.01, 1.35,0.72,12.5,  0.005,2.55),
        (0.52, 0.42, 5.2,   0.0,  0.72,0.36, 6.2,  0.002,2.05),
    ],
    'د' => [
        (1.45, 1.25, 14.5, -0.05, 1.95,0.95,17.5, -0.02, 2.95),
        (0.95, 0.75,  9.5, -0.03, 1.25,0.65,11.5, -0.01, 2.45),
        (0.48, 0.38, 4.8,   0.0,  0.65,0.32, 5.8, -0.005,1.95),
    ],
    'ك' => [
        (2.2,  1.9,  22.0,  0.4,  2.7, 1.55,24.0,  0.2,  4.0),
        (1.5,  1.3,  15.0,  0.28, 1.9, 1.1, 17.0,  0.14, 3.5),
        (0.75, 0.65, 7.5,   0.0,  0.95,0.55, 8.5,  0.07, 3.0),
    ],
    'ق' => [
        (2.3,  2.0,  23.0,  0.45, 2.8, 1.6, 25.0,  0.22, 4.1),
        (1.6,  1.4,  16.0,  0.3,  2.0, 1.15,18.0,  0.15, 3.6),
        (0.8,  0.7,   8.0,  0.0,  1.0, 0.6,  9.0,  0.08, 3.1),
    ],
    
    # ═══ حروف أخرى ═══
    'أ' => [
        (1.3,  1.1,  13.0,  0.0,  1.8, 0.85,16.0,  0.0,  2.8),
        (0.85, 0.65, 8.5,   0.0,  1.1, 0.55,10.0,  0.0,  2.3),
        (0.42, 0.32, 4.2,   0.0,  0.55,0.28, 5.0,  0.0,  1.8),
    ],
    'إ' => [
        (1.35, 1.15, 13.5,  0.0,  1.85,0.9, 16.5,  0.0,  2.85),
        (0.88, 0.68, 8.8,   0.0,  1.15,0.58,10.5,  0.0,  2.35),
        (0.44, 0.34, 4.4,   0.0,  0.58,0.29, 5.2,  0.0,  1.85),
    ],
    'آ' => [
        (1.4,  1.2,  14.0,  0.0,  1.9, 0.95,17.0,  0.0,  2.9),
        (0.9,  0.7,   9.0,  0.0,  1.2, 0.6, 11.0,  0.0,  2.4),
        (0.45, 0.35, 4.5,   0.0,  0.6, 0.3,  5.5,  0.0,  1.9),
    ],
    'ا' => [
        (1.3,  1.1,  13.0,  0.0,  1.8, 0.85,16.0,  0.0,  2.8),
        (0.85, 0.65, 8.5,   0.0,  1.1, 0.55,10.0,  0.0,  2.3),
        (0.42, 0.32, 4.2,   0.0,  0.55,0.28, 5.0,  0.0,  1.8),
    ],
    'ة' => [
        (1.25, 1.05, 12.5,  0.0,  1.75,0.88,15.5,  0.0,  2.75),
        (0.82, 0.62, 8.2,   0.0,  1.08,0.52,9.8,   0.0,  2.25),
        (0.41, 0.31, 4.1,   0.0,  0.54,0.27, 4.9,  0.0,  1.75),
    ],
)

function get_letter_params(letter::Char)
    return get(LETTER_DB, letter, nothing)
end

function apply_exponential(v::Vector{Float64}; alpha::Float64=1.0)
    return exp.(alpha .* v)
end

function list_all_letters()
    return collect(keys(LETTER_DB))
end

end # module LetterDB

# ─────────────────────────────────────────────────────────
# الاختبار الرئيسي
# ─────────────────────────────────────────────────────────
using .CliffordLocal
using .LetterDB
using LinearAlgebra
using Statistics

println("=" ^ 70)
println("  اختبار النموذج المتكامل مع Clifford algebra")
println("=" ^ 70)

# ─────────────────────────────────────────────────────────
# 1. اختبار تحويل الحرف إلى Multivector22
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  1. اختبار تحويل الحروف إلى Multivector22             │")
println("└─────────────────────────────────────────────────────────┘")

alpha = 1.0
test_letters = list_all_letters()

# حساب المعاملات لكل حرف
letter_params = Dict{Char, Vector{Float64}}()
for letter in test_letters
    params = get_letter_params(letter)
    if params !== nothing
        raw = zeros(27)
        for (i, p) in enumerate(params)
            for (j, val) in enumerate(p)
                raw[(i-1)*9 + j] = val
            end
        end
        exp_params = apply_exponential(raw, alpha=alpha)
        letter_params[letter] = exp_params
    end
end

# إنشاء Multivector22 لكل حرف
letter_multivectors = Dict{Char, Multivector22}()
for letter in test_letters
    params = letter_params[letter]
    
    # تحويل 27D إلى 22D
    vec_22 = zeros(22)
    vec_22[1:min(22, length(params))] .= params[1:min(22, length(params))]
    
    # إنشاء bivector بسيط
    bivector = zeros(231)
    if length(params) > 22
        extra = params[23:min(27, length(params))]
        for i in 1:min(5, length(extra))
            bivector[i] = extra[i]
        end
    end
    
    mv = Multivector22(0.0, vec_22, bivector, 0.0)
    letter_multivectors[letter] = mv
    
    println("  $letter: ||mv||=$(round(norm(mv), digits=2))")
end

# ─────────────────────────────────────────────────────────
# 2. اختبار الجداء الهندسي بين حرفين
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  2. اختبار الجداء الهندسي بين حرفين                   │")
println("└─────────────────────────────────────────────────────────┘")

test_pairs = [('أ', 'ب'), ('ح', 'خ'), ('ب', 'ت'), ('ج', 'ح'), ('ص', 'ض'), ('ع', 'غ')]
for (l1, l2) in test_pairs
    if haskey(letter_multivectors, l1) && haskey(letter_multivectors, l2)
        mv1 = letter_multivectors[l1]
        mv2 = letter_multivectors[l2]
        product = geometric_product(mv1, mv2)
        println("    $l1 × $l2: ||product||=$(round(norm(product), digits=2))")
    end
end

# ─────────────────────────────────────────────────────────
# 3. اختبار تجميع الكلمات
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  3. اختبار تجميع الكلمات بجاء كليفورد                │")
println("└─────────────────────────────────────────────────────────┘")

test_words = ["سلام", "كتاب", "قلم", "بيت", "نور", "علم", "حب", "كره", "عين", "مصر"]

word_multivectors = Dict{String, Multivector22}()
for word in test_words
    letters = collect(filter(c -> !isspace(c), word))
    
    if isempty(letters)
        continue
    end
    
    # التحقق من وجود جميع الحروف
    all_exist = all(haskey(letter_multivectors, l) for l in letters)
    if !all_exist
        println("  '$word': ⚠️  بعض الحروف غير موجودة")
        continue
    end
    
    # الجداء الهندسي لجميع الحروف
    result = letter_multivectors[letters[1]]
    for i in 2:length(letters)
        if haskey(letter_multivectors, letters[i])
            result = geometric_product(result, letter_multivectors[letters[i]])
        end
    end
    
    word_multivectors[word] = result
    println("  '$word': ||mv||=$(round(norm(result), digits=2))")
end

# ─────────────────────────────────────────────────────────
# 4. اختبار المسافة بين الكلمات
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  4. اختبار المسافة بين الكلمات                        │")
println("└─────────────────────────────────────────────────────────┘")

# مسافة الكلمات
word_dist_pairs = Tuple{Float64,String,String}[]
words_list = collect(keys(word_multivectors))
for (i, w1) in enumerate(words_list)
    for (j, w2) in enumerate(words_list)
        if i < j
            mv1 = word_multivectors[w1]
            mv2 = word_multivectors[w2]
            dist = clifford_distance(mv1, mv2)
            push!(word_dist_pairs, (dist, w1, w2))
        end
    end
end
sort!(word_dist_pairs, by=x->x[1], rev=true)

println("\n  أعلى 5 مسافات بين كلمات مختلفة:")
for (dist, w1, w2) in word_dist_pairs[1:5]
    println("    $w1 ↔ $w2: $(round(dist, digits=2))")
end

println("\n  أقل 5 مسافات بين كلمات مختلفة:")
for (dist, w1, w2) in word_dist_pairs[end-4:end]
    println("    $w1 ↔ $w2: $(round(dist, digits=2))")
end

word_dists = [p[1] for p in word_dist_pairs]
println("\n  إحصائيات مسافة الكلمات:")
println("    متوسط المسافة: $(round(mean(word_dists), digits=2))")
println("    أدنى مسافة:    $(round(minimum(word_dists), digits=2))")
println("    أعلى مسافة:    $(round(maximum(word_dists), digits=2))")

# ─────────────────────────────────────────────────────────
# 5. اختبار كلمات متعارضة
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  5. اختبار كلمات متعارضة                               │")
println("└─────────────────────────────────────────────────────────┘")

opposite_pairs = [("حب", "كره"), ("نور", "ظل"), ("سرور", "حزن"), ("سلام", "حرب")]
for (w1, w2) in opposite_pairs
    if haskey(word_multivectors, w1) && haskey(word_multivectors, w2)
        mv1 = word_multivectors[w1]
        mv2 = word_multivectors[w2]
        dist = clifford_distance(mv1, mv2)
        sim = clifford_similarity(mv1, mv2)
        println("    $w1 ↔ $w2: مسافة=$(round(dist, digits=2)), تشابه=$(round(sim, digits=4))")
    else
        println("    $w1 ↔ $w2: (كلمة غير موجودة)")
    end
end

# ─────────────────────────────────────────────────────────
# 6. اختبار الترتيب (الجداء لا ي commutative)
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  6. اختبار الترتيب (عدم التبادلية)                    │")
println("└─────────────────────────────────────────────────────────┘")

if haskey(letter_multivectors, 'أ') && haskey(letter_multivectors, 'ب')
    mv_a = letter_multivectors['أ']
    mv_b = letter_multivectors['ب']
    
    product_ab = geometric_product(mv_a, mv_b)
    product_ba = geometric_product(mv_b, mv_a)
    
    println("  أ × ب: ||product||=$(round(norm(product_ab), digits=2))")
    println("  ب × أ: ||product||=$(round(norm(product_ba), digits=2))")
    println("  فرق الترتيب: $(round(norm(product_ab - product_ba), digits=2))")
    
    if norm(product_ab - product_ba) > 1e-6
        println("  ✅ الجداء غير تبادلي (كما هو متوقع في كليفورد)")
    else
        println("  ⚠️  الجداء تبادلي (غير متوقع)")
    end
end

# ─────────────────────────────────────────────────────────
# ملخص
# ─────────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("  ملخص النتائج")
println("=" ^ 70)
println("  بعد Multivector22: 454 (1 + 22 + 231)")
println("  عدد الحروف المختبرة: $(length(test_letters))")
println("  عدد الكلمات المختبرة: $(length(test_words))")
println("  متوسط مسافة الكلمات: $(round(mean(word_dists), digits=2))")

if mean(word_dists) > 1000.0
    println("  ✅ النموذج المتكامل جاهز!")
elseif mean(word_dists) > 100.0
    println("  ✅ النموذج جيد جداً")
else
    println("  ⚠️  يحتاج تحسين")
end

println("\n" * "=" ^ 70)
