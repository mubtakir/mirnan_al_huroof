"""
بوابة الإنتروبيا — EntropyGate (محسّنة).

يراقب إنتروبيا النظام S(t) ويقرر متى تنهار الدالة الموجية.
عندما S(t) < S_crit تنهار. يقوم بالتبريد والتسخين الذاتي
عبر ضبط k_B و β ديناميكياً.
"""

module EntropyGateModule

using LinearAlgebra

using ..Constants: BOLTZMANN_KB
using ..WordPhysics: phase_similarity_enhanced, phase_distance_enhanced

export EntropyGate, compute_S, evaluate, correct!, reset!

"""
    EntropyGate

بوابة إنتروبيا — كم محرك انهيار الدالة الموجية.

الحقول:
- `S_crit`: العتبة الحرجة للإنتروبيا
- `k_B_orig`: ثابت بولتزمان الأصلي
- `k_B`: ثابت بولتزمان الحالي
- `beta_0`: ثابت بولتزمان العكسي الابتدائي
- `corrections_applied`: عدد التصحيحات المطبقة
"""
mutable struct EntropyGate
    S_crit::Float64
    k_B::Float64
    beta_0::Float64
    k_B_orig::Float64
    corrections_applied::Int

    function EntropyGate(; S_crit::Float64=2.5, k_B::Float64=BOLTZMANN_KB, beta_0::Float64=2.0)
        return new(S_crit, k_B, beta_0, k_B, 0)
    end
end

"""
    compute_S(gate::EntropyGate, pv_list, target_pv) -> Float64

حساب إنتروبيا شانون للتوزيع الطوري حول الهدف.
"""
function compute_S(gate::EntropyGate, pv_list::Vector{<:AbstractVector},
                   target_pv::AbstractVector)
    if isempty(pv_list)
        return 0.0
    end
    sims = Float64[phase_similarity_enhanced(pv, target_pv) for pv in pv_list]
    sims .= sims .- minimum(sims)
    s_sum = sum(sims)
    if s_sum < 1e-10
        return 0.0
    end
    p = sims ./ s_sum
    S = -sum(p .* log.(p .+ 1e-10))
    return S
end

"""
    correct!(gate::EntropyGate, S, k_B_cur, beta_cur) -> Tuple{Float64, Float64, Float64}

تصحيح حراري عند تجاوز العتبة. يُخفض k_B ويُرفع β.
"""
function correct!(gate::EntropyGate, S::Float64, k_B_cur::Float64, beta_cur::Float64)
    if S < gate.S_crit
        return k_B_cur, beta_cur, 1.0
    end
    k_B_new = k_B_cur / 2.0
    beta_new = min(beta_cur * 1.5, 6.0)
    gate.corrections_applied += 1
    return k_B_new, beta_new, 1.5
end

"""
    evaluate(gate::EntropyGate, pv_list, target_pv, k_B_cur, beta_cur) -> Tuple

تقييم إنتروبي كامل: حساب ثم تصحيح.
"""
function evaluate(gate::EntropyGate, pv_list::Vector{<:AbstractVector},
                  target_pv::AbstractVector, k_B_cur::Float64, beta_cur::Float64)
    S = compute_S(gate, pv_list, target_pv)
    return correct!(gate, S, k_B_cur, beta_cur)
end

"""
    reset!(gate::EntropyGate)

إعادة تعيين عداد التصحيحات.
"""
function reset!(gate::EntropyGate)
    gate.corrections_applied = 0
end

end # module EntropyGateModule
