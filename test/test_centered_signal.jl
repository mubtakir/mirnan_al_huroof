"""
test_centered_signal.jl - اختبار الإشارة المركّزة كمتجه
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
println("  اختبار الإشارة المركّزة كمتجه - Centered Signal Test")
println("=" ^ 70)

# ─────────────────────────────────────────────────────────
# 1. حساب الإشارات المركّزة
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  1. حساب الإشارات المركّزة للحروف                       │")
println("└─────────────────────────────────────────────────────────┘")

test_letters = ['أ', 'ب', 'ت', 'ج', 'ح', 'خ', 'د', 'ر', 'س', 'ص', 'ض', 'ع', 'غ', 'ق', 'ل', 'م', 'ن', 'و', 'ي']

# حساب الإشارات المركّزة
centered_signals = Dict{Char, Vector{Float64}}()
for letter in test_letters
    y = compute_letter_signal(letter, n_samples=200)
    y_centered = y .- mean(y)
    centered_signals[letter] = y_centered
    println("  $letter: ||y||=$(round(norm(y_centered), digits=2)), min=$(round(minimum(y_centered), digits=2)), max=$(round(maximum(y_centered), digits=2))")
end

# ─────────────────────────────────────────────────────────
# 2. مصفوفة التشابه بالإشارات المركّزة
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  2. مصفوفة التشابه بالإشارات المركّزة (200 بُعد)       │")
println("└─────────────────────────────────────────────────────────┘")

n = length(test_letters)
sim_matrix = zeros(n, n)

for i in 1:n
    for j in 1:n
        v1 = centered_signals[test_letters[i]]
        v2 = centered_signals[test_letters[j]]
        n1 = norm(v1)
        n2 = norm(v2)
        if n1 > 1e-10 && n2 > 1e-10
            sim_matrix[i, j] = dot(v1, v2) / (n1 * n2)
        end
    end
end

# إحصائيات
upper_tri = [sim_matrix[i, j] for i in 1:n for j in i+1:n]
println("\n  إحصائيات التشابه بالإشارات المركّزة:")
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
# 3. اختبار كلمات
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  3. اختبار كلمات (متوسط مرجّح للإشارات المركّزة)        │")
println("└─────────────────────────────────────────────────────────┘")

test_words = ["سلام", "كتاب", "قلم", "بيت", "نور", "علم", "حب", "كره", "عين", "اذن"]

word_signals = Dict{String, Vector{Float64}}()
for word in test_words
    letters = collect(filter(c -> !isspace(c), word))
    if isempty(letters)
        continue
    end
    
    n_letters = length(letters)
    weights = Float64[1.0 / (0.5 + 0.5 * i / n_letters) for i in 1:n_letters]
    weights ./= sum(weights)
    
    vec = zeros(200)
    for (i, letter) in enumerate(letters)
        if haskey(centered_signals, letter)
            vec .+= weights[i] .* centered_signals[letter]
        end
    end
    word_signals[word] = vec
    println("  '$word': ||vec||=$(round(norm(vec), digits=2))")
end

# مصفوفة تشابه الكلمات
println("\n  أعلى 5 تشابهات بين كلمات مختلفة:")
word_pairs = Tuple{Float64,String,String}[]
words_list = collect(keys(word_signals))
for (i, w1) in enumerate(words_list)
    for (j, w2) in enumerate(words_list)
        if i < j
            v1 = word_signals[w1]
            v2 = word_signals[w2]
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
        vec1 = zeros(200)
        vec2 = zeros(200)
        
        for (i, letter) in enumerate(letters1)
            if haskey(centered_signals, letter)
                vec1 .+= centered_signals[letter] ./ length(letters1)
            end
        end
        
        for (i, letter) in enumerate(letters2)
            if haskey(centered_signals, letter)
                vec2 .+= centered_signals[letter] ./ length(letters2)
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
println("  بعد المتجه (إشارة مركّزة): 200")
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
