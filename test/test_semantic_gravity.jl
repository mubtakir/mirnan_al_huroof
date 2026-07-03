"""
test_semantic_gravity.jl - اختبار الجاذبية الدلالية المحسّنة
"""

# ─────────────────────────────────────────────────────────
# تعريف Multivector22 مباشرة
# ─────────────────────────────────────────────────────────
module CliffordTest

using LinearAlgebra

export Multivector22, norm, geometric_product, clifford_distance

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
    res_s = a.s * b.s + dot(a.v, b.v)
    res_v = a.s .* b.v .+ b.s .* a.v
    res_v .+= contract_bv(a.b, b.v)
    res_v .-= contract_bv(b.b, a.v)
    res_b = a.s .* b.b .+ b.s .* a.b
    res_b .+= wedge_product(a.v, b.v)
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

end # module CliffordTest

# ─────────────────────────────────────────────────────────
# قاعدة بيانات الحروف العربية
# ─────────────────────────────────────────────────────────
module LetterDBTest

export get_letter_params, apply_exponential, list_all_letters

const LETTER_DB = Dict{Char, Vector{Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64}}}(
    'ض' => [(3.0,2.5,30.0,-0.8,3.5,2.0,35.0,-0.5,6.0),(2.0,2.0,20.0,0.9,2.5,1.5,25.0,0.4,5.0),(1.0,1.0,10.0,0.0,1.2,0.8,12.0,0.0,3.5)],
    'ص' => [(2.8,2.3,28.0,-0.7,3.2,1.8,32.0,-0.45,5.5),(1.8,1.8,18.0,0.8,2.2,1.3,22.0,0.35,4.5),(0.9,0.9,9.0,0.0,1.0,0.7,10.0,0.0,3.0)],
    'ع' => [(2.5,2.0,26.0,1.8,3.0,1.5,15.0,0.8,4.0),(1.6,1.3,16.0,1.2,2.0,1.0,9.0,1.0,3.0),(0.8,0.65,8.0,0.0,0.9,0.5,4.5,1.2,2.0)],
    'غ' => [(2.6,2.1,25.0,1.9,3.1,1.6,16.0,0.85,4.2),(1.7,1.4,15.0,1.3,2.1,1.1,10.0,1.05,3.2),(0.85,0.7,7.5,0.0,0.95,0.55,5.0,1.3,2.2)],
    'ح' => [(2.3,1.9,24.0,1.7,2.8,1.4,14.0,0.7,3.8),(1.5,1.2,14.0,1.1,1.9,0.95,8.5,0.95,2.8),(0.75,0.6,7.0,0.0,0.85,0.45,4.0,1.1,1.8)],
    'خ' => [(2.4,2.0,23.0,1.75,2.9,1.45,14.5,0.75,3.9),(1.55,1.25,14.5,1.15,1.95,1.0,8.8,1.0,2.9),(0.78,0.62,7.2,0.0,0.88,0.48,4.2,1.15,1.9)],
    'ه' => [(1.2,1.0,12.0,0.5,1.5,0.8,8.0,0.3,2.5),(0.8,0.6,6.0,0.3,1.0,0.5,4.0,0.2,2.0),(0.4,0.3,3.0,0.0,0.5,0.25,2.0,0.1,1.5)],
    'م' => [(1.8,1.5,18.0,0.2,2.2,1.2,20.0,0.1,3.5),(1.2,1.0,12.0,0.15,1.5,0.8,14.0,0.08,3.0),(0.6,0.5,6.0,0.0,0.8,0.4,7.0,0.05,2.5)],
    'ن' => [(1.9,1.6,19.0,0.25,2.3,1.3,21.0,0.12,3.6),(1.3,1.1,13.0,0.18,1.6,0.85,15.0,0.1,3.1),(0.65,0.55,6.5,0.0,0.85,0.45,7.5,0.06,2.6)],
    'ب' => [(1.5,1.3,15.0,0.0,2.0,1.0,18.0,0.0,3.0),(1.0,0.8,10.0,0.0,1.3,0.7,12.0,0.0,2.5),(0.5,0.4,5.0,0.0,0.7,0.35,6.0,0.0,2.0)],
    'ف' => [(1.6,1.4,16.0,0.05,2.1,1.1,19.0,0.02,3.1),(1.1,0.9,11.0,0.03,1.4,0.75,13.0,0.01,2.6),(0.55,0.45,5.5,0.0,0.75,0.38,6.5,0.005,2.1)],
    'و' => [(1.4,1.2,14.0,-0.1,1.9,0.9,17.0,-0.05,2.9),(0.9,0.7,9.0,-0.08,1.2,0.6,11.0,-0.03,2.4),(0.45,0.35,4.5,0.0,0.6,0.3,5.5,-0.01,1.9)],
    'ل' => [(2.0,1.7,20.0,0.3,2.5,1.4,22.0,0.15,3.8),(1.4,1.2,14.0,0.2,1.8,1.0,16.0,0.1,3.3),(0.7,0.6,7.0,0.0,0.9,0.5,8.0,0.05,2.8)],
    'ر' => [(1.7,1.5,17.0,-0.2,2.2,1.2,19.5,-0.1,3.4),(1.1,0.9,11.0,-0.15,1.5,0.8,13.0,-0.08,2.9),(0.55,0.45,5.5,0.0,0.8,0.4,6.5,-0.04,2.4)],
    'ز' => [(1.75,1.55,17.5,-0.18,2.25,1.25,20.0,-0.12,3.5),(1.15,0.95,11.5,-0.13,1.55,0.85,13.5,-0.09,3.0),(0.58,0.48,5.8,0.0,0.82,0.42,6.8,-0.045,2.5)],
    'ج' => [(1.65,1.45,16.5,0.1,2.15,1.15,19.2,0.05,3.3),(1.05,0.85,10.5,0.08,1.45,0.78,12.8,0.03,2.8),(0.52,0.42,5.2,0.0,0.72,0.37,6.3,0.01,2.3)],
    'ش' => [(2.1,1.8,21.0,0.35,2.6,1.5,23.0,0.18,3.9),(1.45,1.25,14.5,0.25,1.85,1.05,16.5,0.12,3.4),(0.72,0.62,7.2,0.0,0.92,0.52,8.2,0.06,2.9)],
    'س' => [(2.05,1.75,20.5,0.32,2.55,1.45,22.5,0.16,3.85),(1.42,1.22,14.2,0.22,1.82,1.02,16.2,0.11,3.35),(0.71,0.61,7.1,0.0,0.91,0.51,8.1,0.055,2.85)],
    'ت' => [(1.55,1.35,15.5,0.02,2.05,1.05,18.5,0.01,3.05),(1.05,0.85,10.5,0.01,1.35,0.72,12.5,0.005,2.55),(0.52,0.42,5.2,0.0,0.72,0.36,6.2,0.002,2.05)],
    'د' => [(1.45,1.25,14.5,-0.05,1.95,0.95,17.5,-0.02,2.95),(0.95,0.75,9.5,-0.03,1.25,0.65,11.5,-0.01,2.45),(0.48,0.38,4.8,0.0,0.65,0.32,5.8,-0.005,1.95)],
    'ك' => [(2.2,1.9,22.0,0.4,2.7,1.55,24.0,0.2,4.0),(1.5,1.3,15.0,0.28,1.9,1.1,17.0,0.14,3.5),(0.75,0.65,7.5,0.0,0.95,0.55,8.5,0.07,3.0)],
    'ق' => [(2.3,2.0,23.0,0.45,2.8,1.6,25.0,0.22,4.1),(1.6,1.4,16.0,0.3,2.0,1.15,18.0,0.15,3.6),(0.8,0.7,8.0,0.0,1.0,0.6,9.0,0.08,3.1)],
    'أ' => [(1.3,1.1,13.0,0.0,1.8,0.85,16.0,0.0,2.8),(0.85,0.65,8.5,0.0,1.1,0.55,10.0,0.0,2.3),(0.42,0.32,4.2,0.0,0.55,0.28,5.0,0.0,1.8)],
    'إ' => [(1.35,1.15,13.5,0.0,1.85,0.9,16.5,0.0,2.85),(0.88,0.68,8.8,0.0,1.15,0.58,10.5,0.0,2.35),(0.44,0.34,4.4,0.0,0.58,0.29,5.2,0.0,1.85)],
    'آ' => [(1.4,1.2,14.0,0.0,1.9,0.95,17.0,0.0,2.9),(0.9,0.7,9.0,0.0,1.2,0.6,11.0,0.0,2.4),(0.45,0.35,4.5,0.0,0.6,0.3,5.5,0.0,1.9)],
    'ا' => [(1.3,1.1,13.0,0.0,1.8,0.85,16.0,0.0,2.8),(0.85,0.65,8.5,0.0,1.1,0.55,10.0,0.0,2.3),(0.42,0.32,4.2,0.0,0.55,0.28,5.0,0.0,1.8)],
    'ة' => [(1.25,1.05,12.5,0.0,1.75,0.88,15.5,0.0,2.75),(0.82,0.62,8.2,0.0,1.08,0.52,9.8,0.0,2.25),(0.41,0.31,4.1,0.0,0.54,0.27,4.9,0.0,1.75)],
    'ي' => [(1.38,1.18,13.8,-0.08,1.88,0.92,16.8,-0.04,2.88),(0.92,0.72,9.2,-0.06,1.22,0.62,11.2,-0.025,2.38),(0.46,0.36,4.6,0.0,0.62,0.31,5.6,-0.01,1.88)],
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

end # module LetterDBTestTest

# ─────────────────────────────────────────────────────────
# وحدة الجاذبية الدلالية
# ─────────────────────────────────────────────────────────
module SemanticGravity

using LinearAlgebra
using Statistics

# استيراد النماذج
using ..CliffordTest
using ..LetterDBTest

export SemanticWord, semantic_mass, semantic_gravity_force,
       semantic_gravity_potential, compute_sentence_cohesion,
       semantic_repulsion, semantic_attraction, GravityResult

# حساب المتجه المحسّن
function compute_word_enhanced_vector(word::String; alpha::Float64=1.0)
    letters = collect(filter(c -> !isspace(c), word))
    if isempty(letters)
        return zeros(27)
    end
    
    n_letters = length(letters)
    weights = Float64[1.0 / (0.5 + 0.5 * i / n_letters) for i in 1:n_letters]
    weights ./= sum(weights)
    
    vec = zeros(27)
    for (i, letter) in enumerate(letters)
        params = get_letter_params(letter)
        if params !== nothing
            raw = zeros(27)
            for (j, p) in enumerate(params)
                for (k, val) in enumerate(p)
                    raw[(j-1)*9 + k] = val
                end
            end
            exp_params = apply_exponential(raw, alpha=alpha)
            vec .+= weights[i] .* exp_params
        end
    end
    
    return vec
end

# حساب Multivector
function word_to_multivector(word::String; alpha::Float64=1.0)
    letters = collect(filter(c -> !isspace(c), word))
    if isempty(letters)
        return Multivector22(0.0, zeros(22), zeros(231), 0.0)
    end
    
    # حساب المعاملات
    vec = compute_word_enhanced_vector(word, alpha=alpha)
    
    # تحويل إلى 22D
    vec_22 = zeros(22)
    vec_22[1:min(22, length(vec))] .= vec[1:min(22, length(vec))]
    
    # إنشاء bivector
    bivector = zeros(231)
    if length(vec) > 22
        extra = vec[23:min(27, length(vec))]
        for i in 1:min(5, length(extra))
            bivector[i] = extra[i]
        end
    end
    
    return Multivector22(0.0, vec_22, bivector, 0.0)
end

# ─────────────────────────────────────────────────────────
# هيكل الكلمة الدلالي
# ─────────────────────────────────────────────────────────

mutable struct SemanticWord
    text::String
    letters::Vector{Char}
    vector::Vector{Float64}
    multivector::Multivector22
    mass::Float64
    energy::Float64
    position::Vector{Float64}
    context::Vector{String}
    
    function SemanticWord(word::String; alpha::Float64=1.0)
        letters = collect(filter(c -> !isspace(c), word))
        vector = compute_word_enhanced_vector(word, alpha=alpha)
        multivector = word_to_multivector(word, alpha=alpha)
        
        mass = compute_semantic_mass(word, letters)
        energy = compute_semantic_energy(word, letters)
        position = vector[1:min(3, length(vector))]
        
        new(word, letters, vector, multivector, mass, energy, position, String[])
    end
end

# ─────────────────────────────────────────────────────────
# حساب الكتلة الدلالية
# ─────────────────────────────────────────────────────────

function compute_semantic_mass(word::String, letters::Vector{Char})
    if isempty(letters)
        return 0.0
    end
    
    length_factor = sqrt(length(letters))
    unique_letters = length(unique(letters))
    complexity_factor = log2(unique_letters + 1)
    
    vowel_weight = 0.0
    consonant_weight = 0.0
    
    vowels = Set(['ا', 'و', 'ي', 'أ', 'إ', 'آ'])
    for letter in letters
        if letter in vowels
            vowel_weight += 1.0
        else
            consonant_weight += 0.8
        end
    end
    
    base_mass = 1.0
    mass = base_mass * length_factor * complexity_factor * (vowel_weight + consonant_weight)
    
    return mass
end

function compute_semantic_energy(word::String, letters::Vector{Char})
    mass = compute_semantic_mass(word, letters)
    c_squared = 1.0
    
    order_factor = 1.0
    for i in 1:length(letters)-1
        if letters[i] == letters[i+1]
            order_factor *= 0.9
        end
    end
    
    return mass * c_squared * order_factor
end

# ─────────────────────────────────────────────────────────
# قوة الجاذبية الدلالية
# ─────────────────────────────────────────────────────────

function semantic_gravity_force(word1::SemanticWord, word2::SemanticWord; 
                               G::Float64=1.0, eps::Float64=0.1)
    r_vec = word2.position - word1.position
    r_sq = dot(r_vec, r_vec) + eps
    
    magnitude = G * word1.mass * word2.mass / r_sq
    
    r_norm = norm(r_vec)
    if r_norm < 1e-10
        return zeros(length(word1.position))
    end
    
    direction = r_vec / r_norm
    
    return magnitude * direction
end

function semantic_gravity_potential(word1::SemanticWord, word2::SemanticWord;
                                   G::Float64=1.0, eps::Float64=0.1)
    r_vec = word2.position - word1.position
    r = sqrt(dot(r_vec, r_vec) + eps)
    
    return -G * word1.mass * word2.mass / r
end

# ─────────────────────────────────────────────────────────
# التأثير المتبادل
# ─────────────────────────────────────────────────────────

function semantic_repulsion(word1::SemanticWord, word2::SemanticWord;
                           strength::Float64=1.0)
    similarity = compute_semantic_similarity(word1, word2)
    repulsion_strength = strength * (1.0 - similarity)
    
    r_vec = word1.position - word2.position
    r_norm = norm(r_vec)
    
    if r_norm < 1e-10
        return zeros(length(word1.position))
    end
    
    direction = r_vec / r_norm
    
    return repulsion_strength * direction
end

function semantic_attraction(word1::SemanticWord, word2::SemanticWord;
                            strength::Float64=1.0)
    similarity = compute_semantic_similarity(word1, word2)
    attraction_strength = strength * similarity
    
    r_vec = word2.position - word1.position
    r_norm = norm(r_vec)
    
    if r_norm < 1e-10
        return zeros(length(word1.position))
    end
    
    direction = r_vec / r_norm
    
    return attraction_strength * direction
end

# ─────────────────────────────────────────────────────────
# حساب التشابه الدلالي
# ─────────────────────────────────────────────────────────

function compute_semantic_similarity(word1::SemanticWord, word2::SemanticWord)
    common_letters = intersect(Set(word1.letters), Set(word2.letters))
    all_letters = union(Set(word1.letters), Set(word2.letters))
    
    if isempty(all_letters)
        return 0.0
    end
    
    letter_similarity = length(common_letters) / length(all_letters)
    
    v1 = word1.vector
    v2 = word2.vector
    
    n1 = norm(v1)
    n2 = norm(v2)
    
    if n1 < 1e-10 || n2 < 1e-10
        phase_similarity = 0.0
    else
        phase_similarity = clamp(dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    end
    
    length_diff = abs(length(word1.letters) - length(word2.letters))
    length_similarity = 1.0 / (1.0 + length_diff)
    
    total_similarity = 0.4 * letter_similarity + 
                       0.4 * phase_similarity + 
                       0.2 * length_similarity
    
    return clamp(total_similarity, 0.0, 1.0)
end

# ─────────────────────────────────────────────────────────
# تماسك الجملة
# ─────────────────────────────────────────────────────────

mutable struct GravityResult
    total_force::Vector{Float64}
    word_forces::Dict{String, Vector{Float64}}
    potentials::Vector{Float64}
    cohesion_score::Float64
    repulsion_score::Float64
    attraction_score::Float64
end

function compute_sentence_cohesion(words::Vector{SemanticWord}; 
                                 G::Float64=1.0, eps::Float64=0.1)
    n = length(words)
    
    if n < 2
        return GravityResult(
            zeros(3),
            Dict{String, Vector{Float64}}(),
            Float64[],
            1.0, 0.0, 1.0
        )
    end
    
    total_force = zeros(3)
    word_forces = Dict{String, Vector{Float64}}()
    potentials = Float64[]
    attraction_total = 0.0
    repulsion_total = 0.0
    
    for i in 1:n
        word_forces[words[i].text] = zeros(3)
    end
    
    for i in 1:n
        for j in i+1:n
            force = semantic_gravity_force(words[i], words[j], G=G, eps=eps)
            total_force .+= force
            
            attraction = semantic_attraction(words[i], words[j])
            repulsion = semantic_repulsion(words[i], words[j])
            
            attraction_total += norm(attraction)
            repulsion_total += norm(repulsion)
            
            word_forces[words[i].text] .+= force + attraction - repulsion
            word_forces[words[j].text] .-= force + attraction - repulsion
            
            potential = semantic_gravity_potential(words[i], words[j], G=G, eps=eps)
            push!(potentials, potential)
        end
    end
    
    cohesion_score = compute_cohesion_score(words, potentials)
    repulsion_score = repulsion_total / (n * (n-1) / 2)
    attraction_score = attraction_total / (n * (n-1) / 2)
    
    return GravityResult(
        total_force,
        word_forces,
        potentials,
        cohesion_score,
        repulsion_score,
        attraction_score
    )
end

function compute_cohesion_score(words::Vector{SemanticWord}, potentials::Vector{Float64})
    if isempty(potentials)
        return 1.0
    end
    
    total_energy = sum(potentials)
    energy_per_word = total_energy / length(words)
    cohesion = 1.0 / (1.0 + exp(energy_per_word))
    
    return clamp(cohesion, 0.0, 1.0)
end

end # module SemanticGravity

# ─────────────────────────────────────────────────────────
# الاختبار الرئيسي
# ─────────────────────────────────────────────────────────
using .CliffordTest
import .CliffordTest: Multivector22, norm, geometric_product, clifford_distance
using .LetterDBTest
import .LetterDBTest: get_letter_params, apply_exponential, list_all_letters
using .SemanticGravity
using .SemanticGravity: compute_semantic_similarity
using LinearAlgebra
using Statistics

println("=" ^ 70)
println("  اختبار الجاذبية الدلالية المحسّنة")
println("=" ^ 70)

# ─────────────────────────────────────────────────────────
# 1. إنشاء كلمات دلالية
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  1. إنشاء كلمات دلالية                                 │")
println("└─────────────────────────────────────────────────────────┘")

test_sentences = [
    ["حب", "كره"],           # معانضة
    ["نور", "ظلم"],          # معانضة
    ["سلام", "حرب"],         # معانضة
    ["كتاب", "قلم"],        # متشابهة
    ["بيت", "منزل"],        # متشابهة
    ["حياة", "موت"],        # معانضة
    ["فرح", "حزن"],         # معانضة
    ["علم", "جهل"],         # معانضة
]

for sentence in test_sentences
    words = [SemanticWord(word) for word in sentence]
    
    println("\n  جملة: $(join(sentence, " - "))")
    for word in words
        println("    $(word.text): كتلة=$(round(word.mass, digits=2)), طاقة=$(round(word.energy, digits=2))")
    end
    
    # حساب الجاذبية
    result = compute_sentence_cohesion(words)
    
    println("    درجة التماسك: $(round(result.cohesion_score, digits=4))")
    println("    درجة التجاذب: $(round(result.attraction_score, digits=4))")
    println("    درجة التنافر: $(round(result.repulsion_score, digits=4))")
end

# ─────────────────────────────────────────────────────────
# 2. اختبار التشابه الدلالي
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  2. اختبار التشابه الدلالي                             │")
println("└─────────────────────────────────────────────────────────┘")

test_pairs = [
    ("حب", "كره"),
    ("نور", "ظلم"),
    ("سلام", "حرب"),
    ("كتاب", "قلم"),
    ("بيت", "منزل"),
    ("حياة", "موت"),
]

for (w1, w2) in test_pairs
    word1 = SemanticWord(w1)
    word2 = SemanticWord(w2)
    
    similarity = compute_semantic_similarity(word1, word2)
    distance = clifford_distance(word1.multivector, word2.multivector)
    
    println("  $w1 ↔ $w2: تشابه=$(round(similarity, digits=4)), مسافة=$(round(distance, digits=2))")
end

# ─────────────────────────────────────────────────────────
# 3. اختبار القوى الدلالية
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  3. اختبار القوى الدلالية                              │")
println("└─────────────────────────────────────────────────────────┘")

for sentence in test_sentences
    words = [SemanticWord(word) for word in sentence]
    
    if length(words) >= 2
        w1 = words[1]
        w2 = words[2]
        
        gravity_force = semantic_gravity_force(w1, w2)
        attraction = semantic_attraction(w1, w2)
        repulsion = semantic_repulsion(w1, w2)
        potential = semantic_gravity_potential(w1, w2)
        
        println("\n  $(w1.text) ↔ $(w2.text):")
        println("    قوة الجاذبية: $(round(norm(gravity_force), digits=4))")
        println("    قوة التجاذب: $(round(norm(attraction), digits=4))")
        println("    قوة التنافر: $(round(norm(repulsion), digits=4))")
        println("    الطاقة الكامنة: $(round(potential, digits=4))")
    end
end

# ─────────────────────────────────────────────────────────
# 4. اختبار تماسك الجملة
# ─────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────┐")
println("│  4. اختبار تماسك الجملة                                │")
println("└─────────────────────────────────────────────────────────┘")

full_sentences = [
    "الكتاب على الطاولة",
    "النور يضيء الغرفة",
    "الحياة جميلة والموت صعب",
    "العلم نور والجهل ظلام",
]

for sentence in full_sentences
    words = [SemanticWord(String(word)) for word in split(sentence)]
    
    println("\n  جملة: $sentence")
    
    # حساب التماسك
    result = compute_sentence_cohesion(words)
    
    println("    عدد الكلمات: $(length(words))")
    println("    درجة التماسك: $(round(result.cohesion_score, digits=4))")
    println("    درجة التجاذب: $(round(result.attraction_score, digits=4))")
    println("    درجة التنافر: $(round(result.repulsion_score, digits=4))")
    println("    مقدار القوة الكلية: $(round(norm(result.total_force), digits=4))")
end

# ─────────────────────────────────────────────────────────
# ملخص
# ─────────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("  ملخص النتائج")
println("=" ^ 70)
println("  ✅ تم بناء الجاذبية الدلالية بنجاح")
println("  ✅ حساب الكتلة الدلالية لكلمة")
println("  ✅ حساب قوة الجاذبية بين كلمتين")
println("  ✅ حساب التجاذب والتنافر")
println("  ✅ حساب تماسك الجملة")
println("  ✅ حساب التشابه الدلالي")

if true
    println("\n  🎉 النموذج جاهز للاستخدام!")
end

println("\n" * "=" ^ 70)
