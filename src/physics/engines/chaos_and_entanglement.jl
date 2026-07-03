"""Chaos, Phase Coupling, and Molecular Physics — ThermalChaos, PhaseCoupling, MolecularBinder."""
module ChaosEntanglement
using LinearAlgebra, Random

export ThermalChaosEngine, PhaseCouplingEngine, MolecularBinder,
       WaveOscillator, split_wave, merge_waves, couple_waves, project_coupled_wave

mutable struct ThermalChaosEngine; base_noise::Float64; end
ThermalChaosEngine(; base_noise=0.15) = ThermalChaosEngine(base_noise)

function perturb(eng::ThermalChaosEngine, pv::AbstractVector, temperature::Float64)
    temperature <= 0.1 && return Float64.(pv)
    noise_level = eng.base_noise * (temperature - 0.1)
    noise_level <= 0 && return Float64.(pv)
    noise = randn(length(pv)) .* noise_level
    pv_copy = Float64.(pv) .+ noise
    nrm = norm(pv_copy)
    return nrm > 1e-10 ? pv_copy ./ nrm : pv_copy
end

mutable struct PhaseCouplingEngine; threshold::Float64; super_gravity::Float64; end
PhaseCouplingEngine(; threshold=0.85, super_gravity=3.0) = PhaseCouplingEngine(threshold, super_gravity)

function compute_bonus(eng::PhaseCouplingEngine, cand_pv, ctx_pvs, ctx_masses)
    bonus = 0.0; d=min(9958, length(cand_pv))
    nc = norm(view(cand_pv,1:d)); nc < 1e-10 && return 0.0
    for (ctx_pv, mass) in zip(ctx_pvs, ctx_masses)
        d2 = min(9958, length(ctx_pv))
        nx = norm(view(ctx_pv,1:d2)); nx < 1e-10 && continue
        sim = dot(view(cand_pv,1:min(d,d2)), view(ctx_pv,1:min(d,d2))) / (nc * nx)
        if sim >= eng.threshold; bonus += mass * sim * eng.super_gravity; end
    end
    return bonus
end

mutable struct MolecularBinder; threshold::Float64; end
MolecularBinder(; threshold=0.90) = MolecularBinder(threshold)

function bind(eng::MolecularBinder, words, pvs, masses)
    length(words) < 2 && return words, pvs, masses
    new_w, new_p, new_m = String[], Vector{Float64}[], Float64[]
    i = 1
    while i <= length(words)
        if i < length(words)
            d = min(9958, min(length(pvs[i]), length(pvs[i+1])))
            sim = dot(view(pvs[i],1:d), view(pvs[i+1],1:d)) / (norm(view(pvs[i],1:d))*norm(view(pvs[i+1],1:d)) + 1e-10)
            if sim >= eng.threshold
                bound = words[i] * "_" * words[i+1]
                bound_pv = Float64.(pvs[i][1:d]) .+ Float64.(pvs[i+1][1:d])
                nrm = norm(bound_pv); if nrm>1e-10; bound_pv./=nrm; end
                push!(new_w,bound); push!(new_p,bound_pv); push!(new_m,masses[i]+masses[i+1]); i+=2; continue
            end
        end
        push!(new_w,words[i]); push!(new_p,pvs[i]); push!(new_m,masses[i]); i+=1
    end
    return new_w, new_p, new_m
end

# ═══ المذبذب الموجي التناظري والاقتران الطوري المستمر ═══

mutable struct WaveOscillator
    psi::Vector{ComplexF64} # المتجه الطوري الكلي (N)
end

"""
    split_wave(v::AbstractVector{ComplexF64}, theta::Float64, phi_shift::Float64) -> Tuple{Vector{ComplexF64}, Vector{ComplexF64}, Vector{ComplexF64}}

انشطار موجة إلى طورين ونظير معاكس لها:
- الموجة الأولى (المعكوسة / Reflected): v_1 = cos(theta) * v
- الموجة الثانية (النافذة / Transmitted): v_2 = sin(theta) * v * exp(i * phi_shift)
- النظير المعاكس لها (الهدام / Anti-wave): v_anti = -v_1 = v_1 * exp(i * pi)
"""
function split_wave(v::AbstractVector{ComplexF64}, theta::Real=pi/4, phi_shift::Real=pi/2)
    N = length(v)
    v1 = cos(theta) .* v
    v2 = sin(theta) .* v .* exp(im * phi_shift)
    v_anti = -v1
    return v1, v2, v_anti
end

function split_wave(qubit::WaveOscillator, theta::Real=pi/4, phi_shift::Real=pi/2)
    return split_wave(qubit.psi, theta, phi_shift)
end

"""
    merge_waves(v1::AbstractVector{ComplexF64}, v2::AbstractVector{ComplexF64}; phase_shift::Real=0.0) -> Vector{ComplexF64}

دمج موجتين بتداخل طوري معين:
- تداخل بناء تام عندما تكون phase_shift = 0
- تداخل هدام/متضاد تام عندما تكون phase_shift = pi
"""
function merge_waves(v1::AbstractVector{ComplexF64}, v2::AbstractVector{ComplexF64}; phase_shift::Real=0.0)
    N = min(length(v1), length(v2))
    res = v1[1:N] .+ v2[1:N] .* exp(im * phase_shift)
    nrm = norm(res)
    return nrm > 1e-10 ? res ./ nrm : res
end

"""
    couple_waves(v_a::AbstractVector{ComplexF64}, v_b::AbstractVector{ComplexF64}) -> Matrix{ComplexF64}

توليد مصفوفة اقتران طوري هولوغرافي (مصفوفة الكثافة المشتركة) بين موجتين:
E_ab = v_a * v_b' (الجداء الخارجي)
"""
function couple_waves(v_a::AbstractVector{ComplexF64}, v_b::AbstractVector{ComplexF64})
    N_a = length(v_a)
    N_b = length(v_b)
    # تطبيع
    na = norm(v_a); v_an = na > 1e-10 ? v_a ./ na : v_a
    nb = norm(v_b); v_bn = nb > 1e-10 ? v_b ./ nb : v_b
    return v_an * v_bn'
end

"""
    project_coupled_wave(entangled_matrix::Matrix{ComplexF64}, test_v::AbstractVector{ComplexF64}) -> Vector{ComplexF64}

قياس أو إسقاط الموجة المقترنة عبر موجة اختبار معينة (تحليل الاقتران الطوري):
W_out = C_ab * test_v
"""
function project_coupled_wave(entangled_matrix::Matrix{ComplexF64}, test_v::AbstractVector{ComplexF64})
    N_mat = size(entangled_matrix, 2)
    vt = test_v[1:min(length(test_v), N_mat)]
    if length(vt) < N_mat
        vt = vcat(vt, zeros(ComplexF64, N_mat - length(vt)))
    end
    res = entangled_matrix * vt
    nrm = norm(res)
    return nrm > 1e-10 ? res ./ nrm : res
end

end
