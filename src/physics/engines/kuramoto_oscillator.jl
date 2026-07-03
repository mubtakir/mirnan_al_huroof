"""
مذبذبات كوراموتو — Kuramoto Oscillator Engine (محسّن).

يحكي تزامن الأطوار عبر معادلات كوراموتو + قوى الجاذبية بين الكلمات.
يستخدم تكامل RK4 بخطوة زمنية dt.
يستخدم العامل الأسي لحساب المتجهات الطورية.
"""

module KuramotoOscillator

using LinearAlgebra, Random

using ..Constants: ENHANCED_DIM
using ..GravityEngine: gravitational_force
using ..WordPhysics: compute_word_enhanced_vector, phase_similarity_enhanced

export OscillatorEngine, simulate, _derivative

"""
    OscillatorEngine

محرك مذبذبات كوراموتو. يحاكي تزامن الكلمات كأنظمة غير خطية مقترنة.

الحقول:
- `coupling`: مصفوفة الاقتران K (اختيارية)
- `dim`: أبعاد فضاء الطور
- `damping_coeff`: معامل التخميد لمنع التباعد الانفجاري للأطوار
"""
mutable struct OscillatorEngine
    coupling::Union{AbstractMatrix{Float64},Nothing}
    dim::Int
    damping_coeff::Float64
end

OscillatorEngine() = OscillatorEngine(nothing, ENHANCED_DIM, 0.1)

function _derivative(engine::OscillatorEngine,
                     omega::AbstractVector, phases::AbstractMatrix,
                     masses::AbstractVector, pvs::Vector{<:AbstractVector},
                     coupling_sub::AbstractMatrix; temperature::Float64=0.0)
    n = length(omega)
    dphi = zeros(Float64, (n, engine.dim))
    for i in 1:n
        kuramoto = zeros(Float64, engine.dim)
        grav = zeros(Float64, engine.dim)
        for j in 1:n
            j == i && continue
            d = phases[j, :] .- phases[i, :]
            kuramoto .+= coupling_sub[i, j] .* sin.(d)
            
            # حساب قوة الجاذبية الاتجاهية بناءً على الأطوار والكتل الفعلية والمسافة الطورية
            dist = norm(d)
            f = gravitational_force(masses[i], phases[i, :], masses[j], phases[j, :], dist)
            grav .+= f
        end
        noise = randn(engine.dim) .* (temperature * 0.1)
        val = omega[i] .+ kuramoto .+ grav .+ noise
        
        # تطبيق التخميد (Damping)
        val .*= (1.0 - engine.damping_coeff)
        
        # قص المطال (Amplitude Clipping) لمنع الانفجار الرقمي
        val_norm = norm(val)
        max_amp = 50.0
        if val_norm > max_amp
            val .*= (max_amp / val_norm)
        end
        
        dphi[i, :] .= val
    end
    return dphi
end

"""
    simulate(engine::OscillatorEngine, omega, phases, masses, pvs,
             coupling_sub; dt=0.01, steps=100, temperature=0.0) -> Tuple{Matrix, Array}

محاكاة RK4 لتطور الأطوار. تُرجع (phases_final, history).
"""
function simulate(engine::OscillatorEngine,
                  omega::AbstractVector, phases::AbstractMatrix,
                  masses::AbstractVector, pvs::Vector{<:AbstractVector},
                  coupling_sub::AbstractMatrix;
                  dt::Float64=0.01, steps::Int=100, temperature::Float64=0.0)
    n = length(omega)
    history = Array{Float64}(undef, steps + 1, n, engine.dim)
    history[1, :, :] .= phases

    for step in 1:steps
        k1 = _derivative(engine, omega, phases, masses, pvs, coupling_sub; temperature=temperature)
        k2 = _derivative(engine, omega, phases .+ 0.5*dt .* k1, masses, pvs, coupling_sub; temperature=temperature)
        k3 = _derivative(engine, omega, phases .+ 0.5*dt .* k2, masses, pvs, coupling_sub; temperature=temperature)
        k4 = _derivative(engine, omega, phases .+ dt .* k3, masses, pvs, coupling_sub; temperature=temperature)
        phases .= phases .+ (dt / 6.0) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        history[step+1, :, :] .= phases
    end
    return phases, history
end

end # module KuramotoOscillator
