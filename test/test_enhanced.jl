"""
test_enhanced.jl - اختبار المتجه الطوري المحسّن
فحص التباين بين الحروف والكلمات
"""

include("../src/MirnanNew.jl")
using .MirnanNew
using .MirnanNew.Physics.WordPhysics
const WordPhysics = MirnanNew.Physics.WordPhysics
using LinearAlgebra
using Statistics

# دمج الدوال من WordPhysics إلى Main
const compute_letter_signal = WordPhysics.compute_letter_signal
const compute_enhanced_vector = WordPhysics.compute_enhanced_vector
const compute_word_enhanced_vector = WordPhysics.compute_word_enhanced_vector
const phase_similarity_enhanced = WordPhysics.phase_similarity_enhanced
const letter_mass = WordPhysics.letter_mass
const gravitational_force_enhanced = WordPhysics.gravitational_force_enhanced
const ENHANCED_DIM = WordPhysics.ENHANCED_DIM

println("=" ^ 70)
println("  اختبار المتجه الطوري المحسّن - Enhanced Phase Vector Test")
println("  بعد المتجه: $ENHANCED_DIM")
println("=" ^ 70)

# ─────────────────────────────────────────────────────────
# 1. اختبار أساسيات الحروف
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  1. اختبار حساب الإشارة للحروف                          │")
println("└─────────────────────────────────────────────────────────┘")

test_letters = ['أ', 'آ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']

for letter in test_letters
    y = compute_letter_signal(letter, n_samples=100)
    mass = letter_mass(letter)
    vec = compute_enhanced_vector(letter)
    println("  حرف '$letter': أقصى=$(round(maximum(y), digits=1)), كتلة=$(round(mass, digits=0)), بعد=$(length(vec))")
end

# ─────────────────────────────────────────────────────────
# 2. اختبار التباين بين الحروف
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  2. اختبار التباين بين الحروف (مصفوفة التشابه)         │")
println("└─────────────────────────────────────────────────────────┘")

sample_letters = ['أ', 'ب', 'ت', 'ج', 'ح', 'خ', 'د', 'ر', 'س', 'ص', 'ض', 'ع', 'غ', 'ق', 'ل', 'م', 'ن', 'و', 'ي']

n = length(sample_letters)

# حساب المتجهات
vectors = Dict{Char, Vector{Float64}}()
for letter in sample_letters
    vectors[letter] = compute_enhanced_vector(letter)
end

# حساب مصفوفة التشابه
sim_matrix = zeros(n, n)
for i in 1:n
    for j in 1:n
        sim_matrix[i, j] = phase_similarity_enhanced(vectors[sample_letters[i]], vectors[sample_letters[j]])
    end
end

# عرض المصفوفة (جزئياً)
println("\n  أعلى 5 تشابهات بين حروف مختلفة:")
pairs_list = Tuple{Float64,Char,Char}[]
for i in 1:n
    for j in i+1:n
        push!(pairs_list, (sim_matrix[i, j], sample_letters[i], sample_letters[j]))
    end
end
sort!(pairs_list, by=x->x[1], rev=true)
for (sim, l1, l2) in pairs_list[1:5]
    println("    $l1 ↔ $l2: $(round(sim, digits=4))")
end

println("\n  أقل 5 تشابهات بين حروف مختلفة:")
for (sim, l1, l2) in pairs_list[end-4:end]
    println("    $l1 ↔ $l2: $(round(sim, digits=4))")
end

upper_tri = [sim_matrix[i, j] for i in 1:n for j in i+1:n]
println("\n  إحصائيات التشابه بين الحروف المختلفة:")
println("    متوسط التشابه: $(round(mean(upper_tri), digits=4))")
println("    أدنى تشابه:    $(round(minimum(upper_tri), digits=4))")
println("    أعلى تشابه:    $(round(maximum(upper_tri), digits=4))")
println("    الانحراف المعياري: $(round(std(upper_tri), digits=4))")

# ─────────────────────────────────────────────────────────
# 3. اختبار كلمات مختلفة
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  3. اختبار التباين بين الكلمات                          │")
println("└─────────────────────────────────────────────────────────┘")

test_words = ["سلام", "كتاب", "قلم", "بيت", "نور", "علم", "حب", "كره", "عين", "اذن"]

println("  حساب متجهات الكلمات...")
word_vectors = Dict{String, Vector{Float64}}()
for word in test_words
    word_vectors[word] = compute_word_enhanced_vector(word)
    nrm = norm(word_vectors[word])
    println("    كلمة '$word': بعد=$(length(word_vectors[word])), ||v||=$(round(nrm, digits=4))")
end

# مصفوفة تشابه الكلمات
println("\n  أعلى 5 تشابهات بين كلمات مختلفة:")
word_pairs = Tuple{Float64,String,String}[]
for (i, w1) in enumerate(test_words)
    for (j, w2) in enumerate(test_words)
        if i < j
            sim = phase_similarity_enhanced(word_vectors[w1], word_vectors[w2])
            push!(word_pairs, (sim, w1, w2))
        end
    end
end
sort!(word_pairs, by=x->x[1], rev=true)
for (sim, w1, w2) in word_pairs[1:5]
    println("    $w1 ↔ $w2: $(round(sim, digits=4))")
end

println("\n  أقل 5 تشابهات بين كلمات مختلفة:")
for (sim, w1, w2) in word_pairs[end-4:end]
    println("    $w1 ↔ $w2: $(round(sim, digits=4))")
end

word_sims = [p[1] for p in word_pairs]
println("\n  إحصائيات تشابه الكلمات المختلفة:")
println("    متوسط التشابه: $(round(mean(word_sims), digits=4))")
println("    أدنى تشابه:    $(round(minimum(word_sims), digits=4))")
println("    أعلى تشابه:    $(round(maximum(word_sims), digits=4))")
println("    الانحراف المعياري: $(round(std(word_sims), digits=4))")

# ─────────────────────────────────────────────────────────
# 4. اختبار الجاذبية بين الحروف
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  4. اختبار الجاذبية الطورية بين الحروف                  │")
println("└─────────────────────────────────────────────────────────┘")

gravity_pairs = [
    ('ب', 'ت'),   # حروف قريبة صوتياً
    ('ب', 'ض'),   # حروف مختلفة جداً
    ('ق', 'ك'),   # حروف شبه متشابهة
    ('ع', 'غ'),   # حروف قريبة
    ('س', 'ش'),   # حروف قريبة
    ('ل', 'ر'),   # حروف مختلفة
    ('أ', 'و'),   # حروف مفتوحة
]

# حساب المتجهات لجميع الحروف في gravity_pairs
for (l1, l2) in gravity_pairs
    if !haskey(vectors, l1)
        vectors[l1] = compute_enhanced_vector(l1)
    end
    if !haskey(vectors, l2)
        vectors[l2] = compute_enhanced_vector(l2)
    end
end

for (l1, l2) in gravity_pairs
    g = gravitational_force_enhanced(l1, l2)
    sim = phase_similarity_enhanced(vectors[l1], vectors[l2])
    println("    $l1 ↔ $l2: جاذبية=$(round(g, digits=2)), تشابه=$(round(sim, digits=4))")
end

# ─────────────────────────────────────────────────────────
# 5. اختبار تمييز الجذور
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  5. اختبار كلمات بنفس الجذر (تمييز بالحروف)             │")
println("└─────────────────────────────────────────────────────────┘")

root_groups = [
    ["كتب", "كاتب", "مكتوب"],
    ["علم", "عالم", "معلم"],
    ["حب", "حبيب", "محبة"],
]

for group in root_groups
    println("\n  جذر $(group[1]):")
    group_vecs = [compute_word_enhanced_vector(w) for w in group]
    for (i, w) in enumerate(group)
        for (j, w2) in enumerate(group)
            if i < j
                sim = phase_similarity_enhanced(group_vecs[i], group_vecs[j])
                println("    $w ↔ $w2: تشابه=$(round(sim, digits=4))")
            end
        end
    end
end

# ─────────────────────────────────────────────────────────
# 6. اختبار كلمات متعارضة
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  6. اختبار كلمات متعارضة (حب/كره، نور/ظلمة)           │")
println("└─────────────────────────────────────────────────────────┘")

opposite_pairs = [
    ("حب", "كره"),
    ("نور", "ظلم"),
    ("快乐", "حزن"),  # قد لا يكون معرضاً
]

for (w1, w2) in opposite_pairs
    try
        v1 = compute_word_enhanced_vector(w1)
        v2 = compute_word_enhanced_vector(w2)
        sim = phase_similarity_enhanced(v1, v2)
        println("    $w1 ↔ $w2: تشابه=$(round(sim, digits=4))")
    catch e
        println("    $w1 ↔ $w2: خطأ - $e")
    end
end

# ─────────────────────────────────────────────────────────
# ملخص النتائج
# ─────────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("  ملخص النتائج")
println("=" ^ 70)
println("  بعد المتجه المحسّن: $ENHANCED_DIM")
println("  عدد الحروف المعرّفة: $(length(test_letters))")
println("  متوسط تشابه الحروف المختلفة: $(round(mean(upper_tri), digits=4))")
println("  متوسط تشابه الكلمات المختلفة: $(round(mean(word_sims), digits=4))")
println("  نطاق تشابه الحروف: [$(round(minimum(upper_tri), digits=4)), $(round(maximum(upper_tri), digits=4))]")
println("  نطاق تشابه الكلمات: [$(round(minimum(word_sims), digits=4)), $(round(maximum(word_sims), digits=4))]")

if mean(upper_tri) < 0.5
    println("  ✅ التباين بين الحروف جيد جداً")
elseif mean(upper_tri) < 0.8
    println("  ⚠️  التباين بين الحروف مقبول")
else
    println("  ❌ التباين بين الحروف ضعيف - يحتاج تحسين")
end

if mean(word_sims) < 0.5
    println("  ✅ التباين بين الكلمات جيد جداً")
elseif mean(word_sims) < 0.8
    println("  ⚠️  التباين بين الكلمات مقبول")
else
    println("  ❌ التباين بين الكلمات ضعيف - يحتاج تحسين")
end

println("\n" * "=" ^ 70)
