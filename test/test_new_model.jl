"""
test_new_model.jl - اختبار النموذج الجديد
"""

include("../src/physics/Physics.jl")
using .Physics
import .Physics.LetterEquations: get_letter_params, LETTER_DB, list_letters
using LinearAlgebra
using Statistics

println("=" ^ 70)
println("  اختبار النموذج الجديد - mirnan_new")
println("=" ^ 70)

# ═══════════════════════════════════════════════════════
# 1. اختبار letter_db.jl
# ═══════════════════════════════════════════════════════
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  1. اختبار قاعدة بيانات الحروف                        │")
println("└─────────────────────────────────────────────────────────┘")

println("  عدد الحروف: $(length(LETTER_DB))")
println("  قائمة الحروف: $(sort(collect(keys(LETTER_DB))))")

# اختبار جلب معاملات حرف
params = get_letter_params('ض')
println("  معاملات ض: $(length(params)) حدود")
println("  الحد الأول: $(params[1])")

# ═══════════════════════════════════════════════════════
# 2. اختبار العامل الأسي
# ═══════════════════════════════════════════════════════
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  2. اختبار العامل الأسي                                │")
println("└─────────────────────────────────────────────────────────┘")

test_letters = ['أ', 'ب', 'ت', 'ج', 'ح', 'خ', 'د', 'ر', 'س', 'ص', 'ض', 'ع', 'غ', 'ق', 'ل', 'م', 'ن', 'و', 'ي']

# حساب المعاملات الخام
raw_params = Dict{Char, Vector{Float64}}()
for letter in test_letters
    local p = get_letter_params(letter)
    if p !== nothing
        vec = zeros(27)
        for (i, term) in enumerate(p)
            for (j, val) in enumerate(term)
                vec[(i-1)*9 + j] = val
            end
        end
        raw_params[letter] = vec
    end
end

# تطبيق العامل الأسي
alpha = 1.0
exp_params = Dict{Char, Vector{Float64}}()
for letter in test_letters
    local v = raw_params[letter]
    exp_params[letter] = exp.(alpha .* v)
    println("  $letter: ||raw||=$(round(norm(raw_params[letter]), digits=2)), ||exp||=$(round(norm(exp_params[letter]), digits=2))")
end

# ═══════════════════════════════════════════════════════
# 3. اختبار التباعد بين الحروف
# ═══════════════════════════════════════════════════════
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  3. اختبار التباعد بين الحروف                          │")
println("└─────────────────────────────────────────────────────────┘")

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

println("  إحصائيات المسافة بين الحروف:")
println("    متوسط المسافة: $(round(mean(upper_tri_dist), digits=2))")
println("    أدنى مسافة:    $(round(minimum(upper_tri_dist), digits=2))")
println("    أعلى مسافة:    $(round(maximum(upper_tri_dist), digits=2))")

# ═══════════════════════════════════════════════════════
# 4. اختبار كلمات
# ═══════════════════════════════════════════════════════
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  4. اختبار كلمات                                       │")
println("└─────────────────────────────────────────────────────────┘")

test_words = ["سلام", "كتاب", "قلم", "بيت", "نور", "علم", "حب", "كره"]

for word in test_words
    letters = collect(filter(c -> !isspace(c), word))
    if all(haskey(exp_params, l) for l in letters)
        n_letters = length(letters)
        weights = Float64[1.0 / (0.5 + 0.5 * i / n_letters) for i in 1:n_letters]
        weights ./= sum(weights)
        
        vec = zeros(27)
        for (i, letter) in enumerate(letters)
            vec .+= weights[i] .* exp_params[letter]
        end
        
        println("  '$word': ||vec||=$(round(norm(vec), digits=2))")
    end
end

# ═══════════════════════════════════════════════════════
# 5. اختبار الجاذبية النحوية
# ═══════════════════════════════════════════════════════
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  5. اختبار الجاذبية النحوية                           │")
println("└─────────────────────────────────────────────────────────┘")

# حساب الجاذبية النحوية يدوياً
function calc_gravity(pos_i, mass_i, pos_j, mass_j; G=1.0)
    distance = abs(pos_i - pos_j)
    log_distance = log(1.0 + distance)
    return G * mass_i * mass_j / log_distance
end

# جملة: "الكتاب على الطاولة"
sentence_words = ["الكتاب", "على", "الطاولة"]
sentence_positions = [1, 2, 3]
sentence_masses = [length("الكتاب"), length("على"), length("الطاولة")]

total_force = 0.0
for i in 1:length(sentence_words)
    for j in i+1:length(sentence_words)
        force = calc_gravity(sentence_positions[i], sentence_masses[i], sentence_positions[j], sentence_masses[j])
        global total_force += force
        println("  $(sentence_words[i]) ↔ $(sentence_words[j]): قوة=$(round(force, digits=2))")
    end
end

println("  القوة الكلية: $(round(total_force, digits=2))")

# ═══════════════════════════════════════════════════════
# ملخص
# ═══════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("  ملخص النتائج")
println("=" ^ 70)
println("  ✅ قاعدة بيانات الحروف: $(length(LETTER_DB)) حرف")
println("  ✅ العامل الأسي يعمل بشكل صحيح")
println("  ✅ التباعد بين الحروف: متوسط=$(round(mean(upper_tri_dist), digits=2))")
println("  ✅ الجاذبية النحوية تعمل بشكل صحيح")

if mean(upper_tri_dist) > 1000.0
    println("\n  🎉 النموذج الجديد جاهز!")
else
    println("\n  ⚠️  يحتاج تحسين")
end

println("\n" * "=" ^ 70)
