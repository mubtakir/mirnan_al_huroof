"""
test_exponential_v2.jl - اختبار العامل الأسي بدون تطبيع مسبق
"""

include("../src/MirnanNew.jl")
using .MirnanNew
using .MirnanNew.Physics.WordPhysics
const WordPhysics = MirnanNew.Physics.WordPhysics
using LinearAlgebra
using Statistics

const compute_letter_signal = WordPhysics.compute_letter_signal
const get_letter_params = WordPhysics.get_letter_params

println("=" ^ 70)
println("  اختبار العامل الأسي (بدون تطبيع) - Exponential Factor v2")
println("=" ^ 70)

# ─────────────────────────────────────────────────────────
# 1. حساب المعاملات الخام
# ─────────────────────────────────────────────────────────
test_letters = ['أ', 'ب', 'ت', 'ج', 'ح', 'خ', 'د', 'ر', 'س', 'ص', 'ض', 'ع', 'غ', 'ق', 'ل', 'م', 'ن', 'و', 'ي']

raw_params = Dict{Char, Vector{Float64}}()
for letter in test_letters
    params = get_letter_params(letter)
    if params !== nothing
        vec = zeros(27)
        for (i, p) in enumerate(params)
            for (j, val) in enumerate(p)
                vec[(i-1)*9 + j] = val
            end
        end
        raw_params[letter] = vec
    end
end

# ─────────────────────────────────────────────────────────
# 2. تطبيق العامل الأسي مباشرة (بدون تطبيع)
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  2. تطبيق العامل الأسي مباشرة                          │")
println("└─────────────────────────────────────────────────────────┘")

# تجربة قيم مختلفة لـ alpha
for alpha in [0.1, 0.3, 0.5, 1.0]
    println("\n  === alpha = $alpha ===")
    
    exp_params = Dict{Char, Vector{Float64}}()
    for letter in test_letters
        v = raw_params[letter]
        # تطبيق العامل الأسي مباشرة بدون تطبيع
        exp_params[letter] = exp.(alpha .* v)
    end
    
    # حساب المسافات
    n = length(test_letters)
    dist_matrix = zeros(n, n)
    for i in 1:n
        for j in 1:n
            v1 = exp_params[test_letters[i]]
            v2 = exp_params[test_letters[j]]
            dist_matrix[i, j] = norm(v1 - v2)
        end
    end
    
    upper_tri_dist = [dist_matrix[i, j] for i in 1:n for j in i+1:n]
    
    println("    متوسط المسافة: $(round(mean(upper_tri_dist), digits=2))")
    println("    أدنى مسافة:    $(round(minimum(upper_tri_dist), digits=2))")
    println("    أعلى مسافة:    $(round(maximum(upper_tri_dist), digits=2))")
    
    # أفضل وأسوأ حالات
    dist_pairs = Tuple{Float64,Char,Char}[]
    for i in 1:n
        for j in i+1:n
            push!(dist_pairs, (dist_matrix[i, j], test_letters[i], test_letters[j]))
        end
    end
    sort!(dist_pairs, by=x->x[1], rev=true)
    
    println("    أفضل تباين: $(dist_pairs[1][2]) ↔ $(dist_pairs[1][3]): $(round(dist_pairs[1][1], digits=2))")
    println("    أسوأ تباين: $(dist_pairs[end][2]) ↔ $(dist_pairs[end][3]): $(round(dist_pairs[end][1], digits=2))")
end

# ─────────────────────────────────────────────────────────
# 3. أفضل alpha واختبار كلمات
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  3. اختبار كلمات بأفضل alpha (1.0)                     │")
println("└─────────────────────────────────────────────────────────┘")

alpha_best = 1.0
exp_params = Dict{Char, Vector{Float64}}()
for letter in test_letters
    v = raw_params[letter]
    exp_params[letter] = exp.(alpha_best .* v)
end

test_words = ["سلام", "كتاب", "قلم", "بيت", "نور", "علم", "حب", "كره", "عين", "اذن"]

word_exp_params = Dict{String, Vector{Float64}}()
for word in test_words
    letters = collect(filter(c -> !isspace(c), word))
    if isempty(letters)
        continue
    end
    
    n_letters = length(letters)
    weights = Float64[1.0 / (0.5 + 0.5 * i / n_letters) for i in 1:n_letters]
    weights ./= sum(weights)
    
    vec = zeros(27)
    for (i, letter) in enumerate(letters)
        if haskey(exp_params, letter)
            vec .+= weights[i] .* exp_params[letter]
        end
    end
    word_exp_params[word] = vec
end

# مسافة الكلمات
word_dist_pairs = Tuple{Float64,String,String}[]
words_list = collect(keys(word_exp_params))
for (i, w1) in enumerate(words_list)
    for (j, w2) in enumerate(words_list)
        if i < j
            v1 = word_exp_params[w1]
            v2 = word_exp_params[w2]
            dist = norm(v1 - v2)
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
# 4. اختبار كلمات متعارضة
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  4. اختبار كلمات متعارضة                               │")
println("└─────────────────────────────────────────────────────────┘")

opposite_pairs = [("حب", "كره"), ("نور", "ظل"), ("سرور", "حزن")]
for (w1, w2) in opposite_pairs
    letters1 = collect(filter(c -> !isspace(c), w1))
    letters2 = collect(filter(c -> !isspace(c), w2))
    
    if !isempty(letters1) && !isempty(letters2)
        vec1 = zeros(27)
        vec2 = zeros(27)
        
        for (i, letter) in enumerate(letters1)
            if haskey(exp_params, letter)
                vec1 .+= exp_params[letter] ./ length(letters1)
            end
        end
        
        for (i, letter) in enumerate(letters2)
            if haskey(exp_params, letter)
                vec2 .+= exp_params[letter] ./ length(letters2)
            end
        end
        
        dist = norm(vec1 - vec2)
        println("    $w1 ↔ $w2: مسافة=$(round(dist, digits=2))")
    end
end

# ─────────────────────────────────────────────────────────
# 5. مقارنة قبل وبعد
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  5. مقارنة شاملة                                       │")
println("└─────────────────────────────────────────────────────────┘")

# حساب المسافات قبل العامل الأسي
n = length(test_letters)
dist_before = zeros(n, n)
for i in 1:n
    for j in 1:n
        v1 = raw_params[test_letters[i]]
        v2 = raw_params[test_letters[j]]
        dist_before[i, j] = norm(v1 - v2)
    end
end
upper_tri_before = [dist_before[i, j] for i in 1:n for j in i+1:n]

# حساب المسافات بعد العامل الأسي (alpha=1.0)
dist_after = zeros(n, n)
for i in 1:n
    for j in 1:n
        v1 = exp_params[test_letters[i]]
        v2 = exp_params[test_letters[j]]
        dist_after[i, j] = norm(v1 - v2)
    end
end
upper_tri_after = [dist_after[i, j] for i in 1:n for j in i+1:n]

println("\n  قبل العامل الأسي:")
println("    متوسط المسافة: $(round(mean(upper_tri_before), digits=2))")
println("    أدنى مسافة:    $(round(minimum(upper_tri_before), digits=2))")
println("    أعلى مسافة:    $(round(maximum(upper_tri_before), digits=2))")

println("\n  بعد العامل الأسي (α=$alpha_best):")
println("    متوسط المسافة: $(round(mean(upper_tri_after), digits=2))")
println("    أدنى مسافة:    $(round(minimum(upper_tri_after), digits=2))")
println("    أعلى مسافة:    $(round(maximum(upper_tri_after), digits=2))")

println("\n  نسبة التحسن:")
println("    متوسط المسافة: $(round(mean(upper_tri_after) / mean(upper_tri_before), digits=2))x")
println("    أدنى مسافة:    $(round(minimum(upper_tri_after) / minimum(upper_tri_before), digits=2))x")
println("    أعلى مسافة:    $(round(maximum(upper_tri_after) / maximum(upper_tri_before), digits=2))x")

# ─────────────────────────────────────────────────────────
# ملخص
# ─────────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("  ملخص النتائج النهائية")
println("=" ^ 70)
println("  بعد المتجه: 27")
println("  معامل الأسي: α=$alpha_best")
println("  متوسط مسافة الحروف (قبل): $(round(mean(upper_tri_before), digits=2))")
println("  متوسط مسافة الحروف (بعد): $(round(mean(upper_tri_after), digits=2))")
println("  نسبة التحسن: $(round(mean(upper_tri_after) / mean(upper_tri_before), digits=2))x")

if mean(upper_tri_after) > 100.0
    println("  ✅ التباين ممتاز جداً! العامل الأسي طوّر النتائج بشكل كبير")
elseif mean(upper_tri_after) > 50.0
    println("  ✅ التباين جيد جداً")
else
    println("  ⚠️  يحتاج مراجعة")
end

println("\n" * "=" ^ 70)
