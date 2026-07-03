"""
semantic_gravity.jl - الجاذبية الدلالية المحسّنة

قوة الجاذبية بين كلمتين:
F = G × m1 × m2 / (r² + ε)

حيث:
- m1, m2: كتلة الكلمتين (من حروفها)
- r: المسافة الطورية بين المتجهين
- G: ثابت الجاذبية
- ε: ثابت صغير لمنع القسمة على صفر
"""

module SemanticGravity

using LinearAlgebra, Statistics

using ..Constants: ENHANCED_DIM
using ..WordPhysics: compute_word_enhanced_vector, compute_enhanced_vector,
       phase_similarity_enhanced, letter_mass

export gravitational_force_enhanced, letter_mass, phase_similarity_enhanced

"""
    letter_mass(letter::Char) -> Float64
كتلة الحرف = طاقة إشارته = ∫y²dx
"""
function gravitational_force_enhanced(letter1::String, letter2::String; G::Float64=1.0, eps::Float64=0.01)
    # تحويل الحروف إلى متجهات
    v1 = compute_enhanced_vector(letter1[1])
    v2 = compute_enhanced_vector(letter2[1])
    
    # حساب الكتلة
    m1 = letter_mass(letter1[1])
    m2 = letter_mass(letter2[1])
    
    # حساب المسافة الطورية
    r = 1.0 - phase_similarity_enhanced(v1, v2)
    
    # قوة الجاذبية
    return (G * m1 * m2) / (r^2 + eps)
end

"""
    gravitational_force_word(word1::String, word2::String; G::Float64=1.0, eps::Float64=0.01) -> Float64
قوة الجاذبية بين كلمتين
"""
function gravitational_force_word(word1::String, word2::String; G::Float64=1.0, eps::Float64=0.01)
    v1 = compute_word_enhanced_vector(word1)
    v2 = compute_word_enhanced_vector(word2)
    
    m1 = length(filter(!isspace, word1))
    m2 = length(filter(!isspace, word2))
    
    r = 1.0 - phase_similarity_enhanced(v1, v2)
    
    return (G * m1 * m2) / (r^2 + eps)
end

end # module SemanticGravity

module GravityEngine

using LinearAlgebra
using ..Constants: GRAVITY_G

export gravitational_force

function gravitational_force(m_i::Real, v_i::AbstractVector,
                             m_j::Real, v_j::AbstractVector,
                             distance::Real=1.0; eps::Float64=0.01)
    d = min(length(v_i), length(v_j))
    direction = Float64.(v_j[1:d] .- v_i[1:d])
    dir_norm = norm(direction)
    dir_norm < 1e-10 && return zeros(Float64, d)
    r2 = max(Float64(distance)^2, eps)
    magnitude = GRAVITY_G * Float64(m_i) * Float64(m_j) / r2
    return magnitude .* direction ./ dir_norm
end

end # module GravityEngine
