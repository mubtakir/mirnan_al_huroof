"""
test_raw_params.jl - اختبار المعاملات الخام فقط
بدون إشارات، بدون مشتقات، بدون تطبيع
"""

include("../src/MirnanNew.jl")
using .MirnanNew
using .MirnanNew.Physics.WordPhysics
const WordPhysics = MirnanNew.Physics.WordPhysics
using LinearAlgebra
using Statistics

# دمج الدوال
const compute_letter_signal = WordPhysics.compute_letter_signal
const compute_enhanced_vector = WordPhysics.compute_enhanced_vector
const compute_word_enhanced_vector = WordPhysics.compute_word_enhanced_vector
const phase_similarity_enhanced = WordPhysics.phase_similarity_enhanced
const letter_mass = WordPhysics.letter_mass
const gravitational_force_enhanced = WordPhysics.gravitational_force_enhanced
const ENHANCED_DIM = WordPhysics.ENHANCED_DIM
const get_letter_params = WordPhysics.get_letter_params

println("=" ^ 70)
println("  اختبار المعاملات الخام فقط - Raw Parameters Test")
println("=" ^ 70)

# ─────────────────────────────────────────────────────────
# 1. استخراج المعاملات الخام
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  1. المعاملات الخام للحروف                              │")
println("└─────────────────────────────────────────────────────────┘")

test_letters = ['أ', 'ب', 'ت', 'ج', 'ح', 'خ', 'د', 'ر', 'س', 'ص', 'ض', 'ع', 'غ', 'ق', 'ل', 'م', 'ن', 'و', 'ي']

# استخراج المعاملات الخام
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
        println("  $letter: ||raw||=$(round(norm(vec), digits=2)), min=$(round(minimum(vec), digits=2)), max=$(round(maximum(vec), digits=2))")
    end
end

# ─────────────────────────────────────────────────────────
# 2. مصفوفة التشابه بالمعاملات الخام فقط
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  2. مصفوفة التشابه بالمعاملات الخام (27 بُعد)           │")
println("└─────────────────────────────────────────────────────────┘")

n = length(test_letters)
sim_matrix = zeros(n, n)

for i in 1:n
    for j in 1:n
        v1 = raw_params[test_letters[i]]
        v2 = raw_params[test_letters[j]]
        n1 = norm(v1)
        n2 = norm(v2)
        if n1 > 1e-10 && n2 > 1e-10
            sim_matrix[i, j] = dot(v1, v2) / (n1 * n2)
        end
    end
end

# إحصائيات
upper_tri = [sim_matrix[i, j] for i in 1:n for j in i+1:n]
println("\n  إحصائيات التشابه بالمعاملات الخام:")
println("    متوسط التشابه: $(round(mean(upper_tri), digits=4))")
println("    أدنى تشابه:    $(round(minimum(upper_tri), digits=4))")
println("    أعلى تشابه:    $(round(maximum(upper_tri), digits=4))")
println("    الانحراف المعياري: $(round(std(upper_tri), digits=4))")

# أعلى وأدنى تشابه
println("\n  أعلى 5 تشابهات:")
pairs_list = Tuple{Float64,Char,Char}[]
for i in 1:n
    for j in i+1:n
        push!(pairs_list, (sim_matrix[i, j], test_letters[i], test_letters[j]))
    end
end
sort!(pairs_list, by=x->x[1], rev=true)
for (sim, l1, l2) in pairs_list[1:5]
    println("    $l1 ↔ $l2: $(round(sim, digits=4))")
end

println("\n  أقل 5 تشابهات:")
for (sim, l1, l2) in pairs_list[end-4:end]
    println("    $l1 ↔ $l2: $(round(sim, digits=4))")
end

# ─────────────────────────────────────────────────────────
# 3. اختبار كلمات بالمعاملات الخام
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  3. اختبار كلمات بالمعاملات الخام                       │")
println("└─────────────────────────────────────────────────────────┘")

test_words = ["سلام", "كتاب", "قلم", "بيت", "نور", "علم", "حب", "كره", "عين", "اذن"]

# حساب متجهات الكلمات (متوسط مرجّح للمعاملات الخام)
word_raw = Dict{String, Vector{Float64}}()
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
        if haskey(raw_params, letter)
            vec .+= weights[i] .* raw_params[letter]
        end
    end
    word_raw[word] = vec
    println("  '$word': ||raw||=$(round(norm(vec), digits=2))")
end

# مصفوفة تشابه الكلمات
println("\n  أعلى 5 تشابهات بين كلمات مختلفة:")
word_pairs = Tuple{Float64,String,String}[]
words_list = collect(keys(word_raw))
for (i, w1) in enumerate(words_list)
    for (j, w2) in enumerate(words_list)
        if i < j
            v1 = word_raw[w1]
            v2 = word_raw[w2]
            n1 = norm(v1)
            n2 = norm(v2)
            sim = 0.0
            if n1 > 1e-10 && n2 > 1e-10
                sim = dot(v1, v2) / (n1 * n2)
            end
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
println("\n  إحصائيات تشابه الكلمات:")
println("    متوسط التشابه: $(round(mean(word_sims), digits=4))")
println("    أدنى تشابه:    $(round(minimum(word_sims), digits=4))")
println("    أعلى تشابه:    $(round(maximum(word_sims), digits=4))")
println("    الانحراف المعياري: $(round(std(word_sims), digits=4))")

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
            if haskey(raw_params, letter)
                vec1 .+= raw_params[letter] ./ length(letters1)
            end
        end
        
        for (i, letter) in enumerate(letters2)
            if haskey(raw_params, letter)
                vec2 .+= raw_params[letter] ./ length(letters2)
            end
        end
        
        n1 = norm(vec1)
        n2 = norm(vec2)
        sim = 0.0
        if n1 > 1e-10 && n2 > 1e-10
            sim = dot(vec1, vec2) / (n1 * n2)
        end
        println("    $w1 ↔ $w2: تشابه=$(round(sim, digits=4))")
    end
end

# ─────────────────────────────────────────────────────────
# ملخص
# ─────────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("  ملخص النتائج")
println("=" ^ 70)
println("  بعد المتجه (معاملات خام): 27")
println("  متوسط تشابه الحروف: $(round(mean(upper_tri), digits=4))")
println("  متوسط تشابه الكلمات: $(round(mean(word_sims), digits=4))")

if mean(upper_tri) < 0.5
    println("  ✅ التباين بين الحروف جيد جداً")
elseif mean(upper_tri) < 0.8
    println("  ⚠️  التباين بين الحروف مقبول")
else
    println("  ❌ التباين بين الحروف ضعيف جداً")
end

println("\n" * "=" ^ 70)
