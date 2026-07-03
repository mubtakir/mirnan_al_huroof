"""
syntactic_gravity.jl - الجاذبية النحوية

وظيفتها: تماسك كلمات الجملة والفقرة الواحدة

الفكرة:
- الكلمات في الجملة لها "وزن نحوي" (的位置 في الجملة)
- المسافة بين الكلمات تُعالج باللوغاريتم ل_compression الكبائر والتوسيع الصغائر
- الجاذبية النحوية تجذب الكلمات المتجاورة وتحافظ على تماسك الجملة

المعادلة:
F_syntax = G_s × m_i × m_j / log(1 + |position_i - position_j|)

ملاحظة: اللوغاريتم يعمل على:
- compression المسافات الكبيرة (الكلمات البعيدة)
- توسيع المسافات الصغيرة (الكلمات القريبة)
- هذا يجعل الجاذبية أكثر حساسية للكلمات المتجاورة
"""

module SyntacticGravity

using LinearAlgebra
using Statistics

export SyntacticWord, syntactic_gravity_force, syntactic_gravity_potential,
       compute_sentence_syntactic_cohesion, SyntacticResult,
       gravitational_force_syntactic

# ─────────────────────────────────────────────────────────
# هيكل الكلمة النحوي
# ─────────────────────────────────────────────────────────

"""
    SyntacticWord

هيكل يمثل الكلمة في الفضاء النحوي
"""
mutable struct SyntacticWord
    text::String                    # نص الكلمة
    position::Int                   # موقع الكلمة في الجملة (1, 2, 3, ...)
    length::Int                     # طول الكلمة
    mass::Float64                   # الكتلة النحوية
    vector::Vector{Float64}         # المتجه النحوي
    role::Symbol                    # الدور النحوي (:subject, :verb, :object, :adjective, :other)
    
    function SyntacticWord(word::String, position::Int; role::Symbol=:other)
        letters = collect(filter(c -> !isspace(c), word))
        
        # الكتلة النحوية = عدد الحروف (كلمات أطول = أثقل)
        mass = Float64(length(letters))
        
        # المتجه النحوي (مبسط: based on position and length)
        vector = Float64[position, length(letters), length(letters)^2]
        
        new(word, position, length(letters), mass, vector, role)
    end
end

# ─────────────────────────────────────────────────────────
# قوة الجاذبية النحوية
# ─────────────────────────────────────────────────────────

"""
    syntactic_gravity_force(word_i::SyntacticWord, word_j::SyntacticWord;
                           G::Float64=1.0) -> Float64
حساب قوة الجاذبية النحوية بين كلمتين

F_syntax = G × m_i × m_j / log(1 + |pos_i - pos_j|)

اللوغاريتم يعمل على:
- compression المسافات الكبيرة
- توسيع المسافات الصغيرة
"""
function syntactic_gravity_force(word_i::SyntacticWord, word_j::SyntacticWord;
                                G::Float64=1.0)
    # حساب المسافة المطلقة بين المواقع
    distance = abs(word_i.position - word_j.position)
    
    # تطبيق اللوغاريتم على المسافة
    # log(1 + distance) لضمان عدم الت.infinity عندما تكون المسافة 0
    log_distance = log(1.0 + distance)
    
    # حساب القوة
    # F = G × m_i × m_j / log_distance
    force = G * word_i.mass * word_j.mass / log_distance
    
    return force
end

"""
    syntactic_gravity_potential(word_i::SyntacticWord, word_j::SyntacticWord;
                               G::Float64=1.0) -> Float64
حساب طاقة الجاذبية النحوية

U_syntax = -G × m_i × m_j / log(1 + |pos_i - pos_j|)
"""
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

"""
    SyntacticResult

هيكل يحتوي على نتائج حساب الجاذبية النحوية
"""
mutable struct SyntacticResult
    total_force::Float64           # القوة الكلية
    word_forces::Dict{String, Float64}  # قوة كل كلمة
    potentials::Vector{Float64}    # الطاقات الكامنة
    cohesion_score::Float64        # درجة التماسك
    avg_force::Float64            # متوسط القوة
    min_force::Float64            # أدنى قوة
    max_force::Float64            # أعلى قوة
end

"""
    compute_sentence_syntactic_cohesion(words::Vector{SyntacticWord};
                                      G::Float64=1.0) -> SyntacticResult
حساب تماسك الجملة باستخدام الجاذبية النحوية

الجملة المتماسكة = كلمات متقاربة في الموقع وأثقل
"""
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
    
    # حساب قوى الجاذبية بين جميع الأزواج
    total_force = 0.0
    word_forces = Dict{String, Float64}()
    potentials = Float64[]
    forces = Float64[]
    
    for i in 1:n
        word_forces[words[i].text] = 0.0
    end
    
    for i in 1:n
        for j in i+1:n
            # حساب القوة
            force = syntactic_gravity_force(words[i], words[j], G=G)
            total_force += force
            
            # حساب الطاقة الكامنة
            potential = syntactic_gravity_potential(words[i], words[j], G=G)
            push!(potentials, potential)
            
            # تحديث قوى الكلمات
            word_forces[words[i].text] += force
            word_forces[words[j].text] += force
            
            push!(forces, force)
        end
    end
    
    # حساب الإحصائيات
    avg_force = isempty(forces) ? 0.0 : mean(forces)
    min_force = isempty(forces) ? 0.0 : minimum(forces)
    max_force = isempty(forces) ? 0.0 : maximum(forces)
    
    # حساب درجة التماسك
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

"""
    compute_cohesion_score(words::Vector{SyntacticWord}, potentials::Vector{Float64},
                          total_force::Float64) -> Float64
حساب درجة تماسك الجملة

التماسك = -متوسط الطاقات الكامنة × متوسط القوة
(الطاقة السلبية + القوة الإيجابية = تماسك أكبر)
"""
function compute_cohesion_score(words::Vector{SyntacticWord}, potentials::Vector{Float64},
                               total_force::Float64)
    if isempty(potentials)
        return 1.0
    end
    
    # الطاقة الكلية
    total_energy = sum(potentials)
    
    # الطاقة لكل كلمة
    energy_per_word = total_energy / length(words)
    
    # تحويل إلى درجة تماسك (0-1)
    # الطاقة السلبية = تماسك أكبر
    # القوة الإيجابية = تماسك أكبر
    cohesion = 1.0 / (1.0 + exp(-energy_per_word - total_force))
    
    return clamp(cohesion, 0.0, 1.0)
end

# ─────────────────────────────────────────────────────────
# تحليل الجاذبية النحوية
# ─────────────────────────────────────────────────────────

"""
    analyze_syntactic_gravity(words::Vector{SyntacticWord}) -> Dict{String, Any}
تحليل شامل للجاذبية النحوية
"""
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

# ─────────────────────────────────────────────────────────
# مقارنة الجاذبية النحوية مع الدلالية
# ─────────────────────────────────────────────────────────

"""
    compare_gravities(syntactic_result::SyntacticResult,
                     semantic_forces::Dict{String, Float64}) -> Dict{String, Any}
مقارنة الجاذبية النحوية مع الدلالية
"""
function compare_gravities(syntactic_result::SyntacticResult,
                          semantic_forces::Dict{String, Float64})
    comparison = Dict{String, Any}(
        "syntactic_total" => syntactic_result.total_force,
        "semantic_total" => sum(values(semantic_forces)),
        "syntactic_cohesion" => syntactic_result.cohesion_score,
        "balance" => syntactic_result.total_force / (sum(values(semantic_forces)) + 1e-10)
    )
    
    return comparison
end

"""
    gravitational_force_syntactic(pos_i::Int, pv_i::AbstractVector, 
                                   pos_j::Int, pv_j::AbstractVector,
                                   distance::Float64; G::Float64=1.0) -> Float64
قوة الجاذبية النحوية بين كلمتين بناءً على مواقعهما
F = G × |pos_i - pos_j| × (pv_i · pv_j) / log(1 + distance)
"""
function gravitational_force_syntactic(pos_i::Int, pv_i::AbstractVector,
                                       pos_j::Int, pv_j::AbstractVector,
                                       distance::Float64; G::Float64=1.0)
    # حساب الكتلة من طول المتجه
    mass_i = Float64(length(pv_i))
    mass_j = Float64(length(pv_j))
    
    # حساب المسافة الموضعية
    pos_distance = abs(pos_i - pos_j)
    
    # تطبيق اللوغاريتم على المسافة الموضعية
    log_distance = log(1.0 + Float64(pos_distance))
    
    # حساب التشابه الطوري
    n_i = norm(pv_i)
    n_j = norm(pv_j)
    similarity = (n_i > 1e-10 && n_j > 1e-10) ? dot(pv_i, pv_j) / (n_i * n_j) : 0.0
    
    # حساب القوة
    # F = G × mass_product / log_positional_distance × similarity
    force = G * mass_i * mass_j * similarity / (log_distance + distance)
    
    return force
end

end # module SyntacticGravity
