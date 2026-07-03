"""
test_euclidean.jl - اختبار باستخدام Euclidean distance بدلاً من cosine similarity
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
println("  اختبار Euclidean distance - Euclidean Distance Test")
println("=" ^ 70)

# ─────────────────────────────────────────────────────────
# 1. حساب المعاملات الخام
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  1. المعاملات الخام للحروف                              │")
println("└─────────────────────────────────────────────────────────┘")

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
        println("  $letter: ||raw||=$(round(norm(vec), digits=2))")
    end
end

# ─────────────────────────────────────────────────────────
# 2. مصفوفة المسافة بالإucledian
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  2. مصفوفة المسافة بالإucledian (27 بُعد)              │")
println("└─────────────────────────────────────────────────────────┘")

n = length(test_letters)
dist_matrix = zeros(n, n)

for i in 1:n
    for j in 1:n
        v1 = raw_params[test_letters[i]]
        v2 = raw_params[test_letters[j]]
        dist_matrix[i, j] = norm(v1 - v2)
    end
end

# إحصائيات
upper_tri_dist = [dist_matrix[i, j] for i in 1:n for j in i+1:n]
println("\n  إحصائيات المسافة بالإucledian:")
println("    متوسط المسافة: $(round(mean(upper_tri_dist), digits=2))")
println("    أدنى مسافة:    $(round(minimum(upper_tri_dist), digits=2))")
println("    أعلى مسافة:    $(round(maximum(upper_tri_dist), digits=2))")
println("    الانحراف المعياري: $(round(std(upper_tri_dist), digits=2))")

# أعلى وأدنى مسافة
println("\n  أعلى 5 مسافات (أكثر تبايناً):")
dist_pairs = Tuple{Float64,Char,Char}[]
for i in 1:n
    for j in i+1:n
        push!(dist_pairs, (dist_matrix[i, j], test_letters[i], test_letters[j]))
    end
end
sort!(dist_pairs, by=x->x[1], rev=true)
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
println("│  3. اختبار كلمات (مسافة إقليدية)                       │")
println("└─────────────────────────────────────────────────────────┘")

test_words = ["سلام", "كتاب", "قلم", "بيت", "نور", "علم", "حب", "كره", "عين", "اذن"]

word_params = Dict{String, Vector{Float64}}()
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
    word_params[word] = vec
    println("  '$word': ||vec||=$(round(norm(vec), digits=2))")
end

# مصفوفة مسافة الكلمات
println("\n  أعلى 5 مسافات بين كلمات مختلفة:")
word_dist_pairs = Tuple{Float64,String,String}[]
words_list = collect(keys(word_params))
for (i, w1) in enumerate(words_list)
    for (j, w2) in enumerate(words_list)
        if i < j
            v1 = word_params[w1]
            v2 = word_params[w2]
            dist = norm(v1 - v2)
            push!(word_dist_pairs, (dist, w1, w2))
        end
    end
end
sort!(word_dist_pairs, by=x->x[1], rev=true)
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
println("    الانحراف المعياري: $(round(std(word_dists), digits=2))")

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
        
        dist = norm(vec1 - vec2)
        println("    $w1 ↔ $w2: مسافة=$(round(dist, digits=2))")
    end
end

# ─────────────────────────────────────────────────────────
# ملخص
# ─────────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("  ملخص النتائج")
println("=" ^ 70)
println("  بعد المتجه (معاملات خام): 27")
println("  متوسط مسافة الحروف: $(round(mean(upper_tri_dist), digits=2))")
println("  متوسط مسافة الكلمات: $(round(mean(word_dists), digits=2))")

if mean(upper_tri_dist) > 10.0
    println("  ✅ التباين بين الحروف جيد جداً")
elseif mean(upper_tri_dist) > 5.0
    println("  ⚠️  التباين بين الحروف مقبول")
else
    println("  ❌ التباين بين الحروف ضعيف جداً")
end

println("\n" * "=" ^ 70)
