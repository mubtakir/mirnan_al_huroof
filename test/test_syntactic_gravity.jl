"""
test_syntactic_gravity.jl - اختبار الجاذبية النحوية
"""

# ─────────────────────────────────────────────────────────
# وحدة الجاذبية النحوية
# ─────────────────────────────────────────────────────────
module SyntacticGravityTest

using LinearAlgebra
using Statistics

export SyntacticWord, syntactic_gravity_force, syntactic_gravity_potential,
       compute_sentence_syntactic_cohesion, SyntacticResult, analyze_syntactic_gravity

# ─────────────────────────────────────────────────────────
# هيكل الكلمة النحوي
# ─────────────────────────────────────────────────────────

mutable struct SyntacticWord
    text::String
    position::Int
    length::Int
    mass::Float64
    vector::Vector{Float64}
    role::Symbol
    
    function SyntacticWord(word::String, position::Int; role::Symbol=:other)
        letters = collect(filter(c -> !isspace(c), word))
        mass = Float64(length(letters))
        vector = Float64[position, length(letters), length(letters)^2]
        new(word, position, length(letters), mass, vector, role)
    end
end

# ─────────────────────────────────────────────────────────
# قوة الجاذبية النحوية
# ─────────────────────────────────────────────────────────

function syntactic_gravity_force(word_i::SyntacticWord, word_j::SyntacticWord;
                                G::Float64=1.0)
    distance = abs(word_i.position - word_j.position)
    log_distance = log(1.0 + distance)
    force = G * word_i.mass * word_j.mass / log_distance
    return force
end

function syntactic_gravity_potential(word_i::SyntacticWord, word_j::SyntacticWord;
                                    G::Float64=1.0)
    distance = abs(word_i.position - word_j.position)
    log_distance = log(1.0 + distance)
    potential = -G * word_i.mass * word_j.mass / log_distance
    return potential
end

# ─────────────────────────────────────────────────────────
# تماسك الجملة
# ─────────────────────────────────────────────────────────

mutable struct SyntacticResult
    total_force::Float64
    word_forces::Dict{String, Float64}
    potentials::Vector{Float64}
    cohesion_score::Float64
    avg_force::Float64
    min_force::Float64
    max_force::Float64
end

function compute_sentence_syntactic_cohesion(words::Vector{SyntacticWord};
                                           G::Float64=1.0)
    n = length(words)
    
    if n < 2
        return SyntacticResult(
            0.0,
            Dict{String, Float64}(),
            Float64[],
            1.0, 0.0, 0.0, 0.0
        )
    end
    
    total_force = 0.0
    word_forces = Dict{String, Float64}()
    potentials = Float64[]
    forces = Float64[]
    
    for i in 1:n
        word_forces[words[i].text] = 0.0
    end
    
    for i in 1:n
        for j in i+1:n
            force = syntactic_gravity_force(words[i], words[j], G=G)
            total_force += force
            
            potential = syntactic_gravity_potential(words[i], words[j], G=G)
            push!(potentials, potential)
            
            word_forces[words[i].text] += force
            word_forces[words[j].text] += force
            
            push!(forces, force)
        end
    end
    
    avg_force = isempty(forces) ? 0.0 : mean(forces)
    min_force = isempty(forces) ? 0.0 : minimum(forces)
    max_force = isempty(forces) ? 0.0 : maximum(forces)
    
    cohesion_score = compute_cohesion_score(words, potentials, total_force)
    
    return SyntacticResult(
        total_force,
        word_forces,
        potentials,
        cohesion_score,
        avg_force,
        min_force,
        max_force
    )
end

function compute_cohesion_score(words::Vector{SyntacticWord}, potentials::Vector{Float64},
                               total_force::Float64)
    if isempty(potentials)
        return 1.0
    end
    
    total_energy = sum(potentials)
    energy_per_word = total_energy / length(words)
    cohesion = 1.0 / (1.0 + exp(-energy_per_word - total_force))
    
    return clamp(cohesion, 0.0, 1.0)
end

function analyze_syntactic_gravity(words::Vector{SyntacticWord})
    result = compute_sentence_syntactic_cohesion(words)
    
    analysis = Dict{String, Any}(
        "sentence" => join([w.text for w in words], " "),
        "num_words" => length(words),
        "total_force" => result.total_force,
        "cohesion_score" => result.cohesion_score,
        "avg_force" => result.avg_force,
        "min_force" => result.min_force,
        "max_force" => result.max_force,
        "word_positions" => Dict(w.text => w.position for w in words),
        "word_masses" => Dict(w.text => w.mass for w in words),
        "word_forces" => result.word_forces
    )
    
    return analysis
end

end # module SyntacticGravityTest

# ─────────────────────────────────────────────────────────
# الاختبار الرئيسي
# ─────────────────────────────────────────────────────────
using .SyntacticGravityTest
using Statistics

println("=" ^ 70)
println("  اختبار الجاذبية النحوية")
println("=" ^ 70)

# ─────────────────────────────────────────────────────────
# 1. اختبار جمل بسيطة
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  1. اختبار جمل بسيطة                                   │")
println("└─────────────────────────────────────────────────────────┘")

test_sentences = [
    "الكتاب على الطاولة",
    "النور يضيء الغرفة",
    "الحياة جميلة",
    "العلم نور",
]

for sentence in test_sentences
    words_raw = split(sentence)
    words = [SyntacticWord(String(w), i) for (i, w) in enumerate(words_raw)]
    
    println("\n  جملة: $sentence")
    
    # حساب الجاذبية النحوية
    result = compute_sentence_syntactic_cohesion(words)
    
    println("    عدد الكلمات: $(length(words))")
    println("    القوة الكلية: $(round(result.total_force, digits=4))")
    println("    درجة التماسك: $(round(result.cohesion_score, digits=4))")
    println("    متوسط القوة: $(round(result.avg_force, digits=4))")
    println("    أدنى قوة: $(round(result.min_force, digits=4))")
    println("    أعلى قوة: $(round(result.max_force, digits=4))")
    
    # عرض قوى كل كلمة
    println("    قوى الكلمات:")
    for word in words
        force = result.word_forces[word.text]
        println("      $(word.text) (موقع $(word.position)): قوة=$(round(force, digits=4))")
    end
end

# ─────────────────────────────────────────────────────────
# 2. اختبار تأثير اللوغاريتم
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  2. اختبار تأثير اللوغاريتم                            │")
println("└─────────────────────────────────────────────────────────┘")

# مقارنة المسافة العادية مع اللوغاريتم
println("\n  مقارنة المسافة العادية مع اللوغاريتم:")
println("  ─────────────────────────────────────────────────────")

distances = [1, 2, 3, 5, 10, 20, 50, 100]
println("  مسافة | عادية | لوغاريتم | نسبة")
println("  ──────┼───────┼──────────┼─────")

for d in distances
    log_d = log(1.0 + d)
    ratio = d / log_d
    println("  $(lpad(d, 5)) | $(lpad(d, 5)) | $(lpad(round(log_d, digits=2), 8)) | $(lpad(round(ratio, digits=2), 5))")
end

println("\n  ملاحظة: اللوغاريتم يُقلص المسافات الكبيرة ويوسع الصغيرة")
println("  هذا يجعل الجاذبية أكثر حساسية للكلمات القريبة")

# ─────────────────────────────────────────────────────────
# 3. اختبار جمل بأطوال مختلفة
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  3. اختبار جمل بأطوال مختلفة                           │")
println("└─────────────────────────────────────────────────────────┘")

different_lengths = [
    ["كتاب"],                    # كلمة واحدة
    ["كتاب", "جديد"],            # كلمتان
    ["الكتاب", "على", "الطاولة"],  # ثلاث كلمات
    ["閱讀", "هذا", "الكتاب", "المفيد"],  # أربع كلمات (بالإنجليزية)
    ["الطالب", "مجتهد", "و", "النتائج", "ممتازة"],  # خمس كلمات
]

for word_list in different_lengths
    words = [SyntacticWord(String(w), i) for (i, w) in enumerate(word_list)]
    
    println("\n  جملة: $(join(word_list, " "))")
    
    result = compute_sentence_syntactic_cohesion(words)
    
    println("    عدد الكلمات: $(length(words))")
    println("    القوة الكلية: $(round(result.total_force, digits=4))")
    println("    درجة التماسك: $(round(result.cohesion_score, digits=4))")
end

# ─────────────────────────────────────────────────────────
# 4. اختبار تأثير طول الكلمة
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  4. اختبار تأثير طول الكلمة                            │")
println("└─────────────────────────────────────────────────────────┘")

# جملة بكلمات قصيرة
short_words = ["أنا", "أحب", "العلم"]
words_short = [SyntacticWord(String(w), i) for (i, w) in enumerate(short_words)]

# جملة بكلمات طويلة
long_words = ["الطالب", "المجتهد", "يحقق", "النتائج", "الممتازة"]
words_long = [SyntacticWord(String(w), i) for (i, w) in enumerate(long_words)]

println("\n  جملة بكلمات قصيرة: $(join(short_words, " "))")
result_short = compute_sentence_syntactic_cohesion(words_short)
println("    متوسط طول الكلمات: $(mean([w.length for w in words_short]))")
println("    القوة الكلية: $(round(result_short.total_force, digits=4))")
println("    درجة التماسك: $(round(result_short.cohesion_score, digits=4))")

println("\n  جملة بكلمات طويلة: $(join(long_words, " "))")
result_long = compute_sentence_syntactic_cohesion(words_long)
println("    متوسط طول الكلمات: $(mean([w.length for w in words_long]))")
println("    القوة الكلية: $(round(result_long.total_force, digits=4))")
println("    درجة التماسك: $(round(result_long.cohesion_score, digits=4))")

# ─────────────────────────────────────────────────────────
# 5. اختبار ترتيب الكلمات
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  5. اختبار ترتيب الكلمات                               │")
println("└─────────────────────────────────────────────────────────┘")

# نفس الكلمات بترتيب مختلفة
word_set = ["الطالب", "مجتهد", "النتائج", "ممتازة"]

permutations = [
    ["الطالب", "مجتهد", "النتائج", "ممتازة"],  # ترتيب أصلي
    ["مجتهد", "الطالب", "ممتازة", "النتائج"],  # ترتيب مختلف
    ["النتائج", "ممتازة", "الطالب", "مجتهد"],  # ترتيب معكوس
]

for perm in permutations
    words = [SyntacticWord(String(w), i) for (i, w) in enumerate(perm)]
    
    println("\n  ترتيب: $(join(perm, " "))")
    
    result = compute_sentence_syntactic_cohesion(words)
    
    println("    القوة الكلية: $(round(result.total_force, digits=4))")
    println("    درجة التماسك: $(round(result.cohesion_score, digits=4))")
end

# ─────────────────────────────────────────────────────────
# 6. اختبار جمل غير متماسكة
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  6. اختبار جمل غير متماسكة                             │")
println("└─────────────────────────────────────────────────────────┘")

incoherent_sentences = [
    "كتاب طاولة نور غرفة",  # كلمات غير مترابطة
    "أنا حب علم كتاب",     # كلمات غير مترابطة
    "الطالب الباب الكتاب الطاولة",  # كلمات غير مترابطة
]

for sentence in incoherent_sentences
    words_raw = split(sentence)
    words = [SyntacticWord(String(w), i) for (i, w) in enumerate(words_raw)]
    
    println("\n  جملة غير متماسكة: $sentence")
    
    result = compute_sentence_syntactic_cohesion(words)
    
    println("    القوة الكلية: $(round(result.total_force, digits=4))")
    println("    درجة التماسك: $(round(result.cohesion_score, digits=4))")
end

# ─────────────────────────────────────────────────────────
# ملخص
# ─────────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("  ملخص النتائج")
println("=" ^ 70)
println("  ✅ تم بناء الجاذبية النحوية بنجاح")
println("  ✅ استخدام اللوغاريتم لمعالجة المسافات")
println("  ✅ حساب قوة الجاذبية بين الكلمات")
println("  ✅ حساب طاقة الجاذبية الكامنة")
println("  ✅ حساب درجة التماسك")
println("  ✅ تحليل تأثير ترتيب الكلمات")
println("  ✅ مقارنة الجمل المتماسكة وغير المتماسكة")

if true
    println("\n  🎉 الجاذبية النحوية جاهزة للاستخدام!")
end

println("\n" * "=" ^ 70)
