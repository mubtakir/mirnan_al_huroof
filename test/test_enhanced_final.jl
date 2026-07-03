"""
test_enhanced_final.jl - الاختبار النهائي للنموذج المعزز مع العامل الأسي
"""

include("../src/MirnanNew.jl")
using .MirnanNew
using .MirnanNew.Physics.WordPhysics
const WordPhysics = MirnanNew.Physics.WordPhysics
const compute_raw_param_vector = WordPhysics.compute_raw_param_vector
using LinearAlgebra
using Statistics

println("=" ^ 70)
println("  الاختبار النهائي - النموذج المعزز مع العامل الأسي")
println("=" ^ 70)

# ─────────────────────────────────────────────────────────
# 1. اختبار حساب المتجهات
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  1. اختبار حساب المتجهات                               │")
println("└─────────────────────────────────────────────────────────┘")

test_letters = ['أ', 'ب', 'ت', 'ج', 'ح', 'خ', 'د', 'ر', 'س', 'ص', 'ض', 'ع', 'غ', 'ق', 'ل', 'م', 'ن', 'و', 'ي']

for letter in test_letters
    v = compute_enhanced_vector(letter)
    println("  $letter: ||vec||=$(round(norm(v), digits=2))")
end

# ─────────────────────────────────────────────────────────
# 2. اختبار التباعد بين الحروف
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  2. اختبار التباعد بين الحروف                          │")
println("└─────────────────────────────────────────────────────────┘")

n = length(test_letters)
dist_matrix = zeros(n, n)
for i in 1:n
    for j in 1:n
        v1 = compute_enhanced_vector(test_letters[i])
        v2 = compute_enhanced_vector(test_letters[j])
        dist_matrix[i, j] = norm(v1 - v2)
    end
end

upper_tri_dist = [dist_matrix[i, j] for i in 1:n for j in i+1:n]

println("\n  إحصائيات المسافة بين الحروف:")
println("    متوسط المسافة: $(round(mean(upper_tri_dist), digits=2))")
println("    أدنى مسافة:    $(round(minimum(upper_tri_dist), digits=2))")
println("    أعلى مسافة:    $(round(maximum(upper_tri_dist), digits=2))")
println("    الانحراف المعياري: $(round(std(upper_tri_dist), digits=2))")

# أفضل وأسوأ حالات
dist_pairs = Tuple{Float64,Char,Char}[]
for i in 1:n
    for j in i+1:n
        push!(dist_pairs, (dist_matrix[i, j], test_letters[i], test_letters[j]))
    end
end
sort!(dist_pairs, by=x->x[1], rev=true)

println("\n  أعلى 5 مسافات (أكثر تبايناً):")
for (dist, l1, l2) in dist_pairs[1:5]
    println("    $l1 ↔ $l2: $(round(dist, digits=2))")
end

println("\n  أقل 5 مسافات (أكثر تشابهاً):")
for (dist, l1, l2) in dist_pairs[end-4:end]
    println("    $l1 ↔ $l2: $(round(dist, digits=2))")
end

# ─────────────────────────────────────────────────────────
# 3. اختبار كلمات
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  3. اختبار كلمات                                       │")
println("└─────────────────────────────────────────────────────────┘")

test_words = ["سلام", "كتاب", "قلم", "بيت", "نور", "علم", "حب", "كره", "عين", "اذن"]

for word in test_words
    v = compute_word_enhanced_vector(word)
    println("  '$word': ||vec||=$(round(norm(v), digits=2))")
end

# مسافة الكلمات
println("\n  مسافة الكلمات:")
word_dist_pairs = Tuple{Float64,String,String}[]
for (i, w1) in enumerate(test_words)
    for (j, w2) in enumerate(test_words)
        if i < j
            v1 = compute_word_enhanced_vector(w1)
            v2 = compute_word_enhanced_vector(w2)
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
    v1 = compute_word_enhanced_vector(w1)
    v2 = compute_word_enhanced_vector(w2)
    dist = norm(v1 - v2)
    println("    $w1 ↔ $w2: مسافة=$(round(dist, digits=2))")
end

# ─────────────────────────────────────────────────────────
# 5. اختبار القوة الجاذبية
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  5. اختبار القوة الجاذبية                             │")
println("└─────────────────────────────────────────────────────────┘")

test_pairs = [('أ', 'ب'), ('ح', 'خ'), ('ب', 'ت'), ('ج', 'ح'), ('ر', 'ز'), ('س', 'ش'), ('ص', 'ض'), ('ع', 'غ')]
for (l1, l2) in test_pairs
    force = gravitational_force_enhanced(l1, l2)
    println("    $l1 ↔ $l2: قوة=$(round(force, digits=2))")
end

# ─────────────────────────────────────────────────────────
# 6. اختبار معامل alpha المختلف
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  6. اختبار معامل alpha المختلف                        │")
println("└─────────────────────────────────────────────────────────┘")

for alpha in [0.5, 1.0, 1.5, 2.0]
    dists = Float64[]
    for i in 1:n
        for j in i+1:n
            v1 = compute_raw_param_vector(test_letters[i], alpha=alpha)
            v2 = compute_raw_param_vector(test_letters[j], alpha=alpha)
            push!(dists, norm(v1 - v2))
        end
    end
    println("  alpha=$alpha: متوسط=$(round(mean(dists), digits=2)), أدنى=$(round(minimum(dists), digits=2)), أعلى=$(round(maximum(dists), digits=2))")
end

# ─────────────────────────────────────────────────────────
# ملخص
# ─────────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("  ملخص النتائج النهائية")
println("=" ^ 70)
println("  بعد المتجه: $(ENHANCED_DIM)")
println("  متوسط مسافة الحروف: $(round(mean(upper_tri_dist), digits=2))")
println("  أدنى مسافة حروف: $(round(minimum(upper_tri_dist), digits=2))")
println("  متوسط مسافة الكلمات: $(round(mean(word_dists), digits=2))")

if mean(upper_tri_dist) > 100.0
    println("  ✅ النموذج جاهز للخطوة التالية!")
elseif mean(upper_tri_dist) > 50.0
    println("  ✅ النموذج جيد جداً")
else
    println("  ⚠️  يحتاج تحسين")
end

println("\n" * "=" ^ 70)
