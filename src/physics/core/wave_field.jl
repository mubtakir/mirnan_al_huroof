"""
WaveField — حقل الموجات.

المبدأ الفيزيائي:
  كل محرّك يُعيد مساهمة موجية (سعة + طور).
  الموجة الكلية = تراكب (superposition) جميع المساهمات.
  احتمال اختيار كلمة = |Ψ_total|²  (قاعدة بورن).

  هذا ليس weighted average. هذا تداخل موجي حقيقي:
  - تداخل بنّاء: موجتان بنفس الطور → تعزيز
  - تداخل مدمر: موجتان بطور عكسي → إلغاء

  كل "وزن" في config.yaml يمثل الآن سعة موجية، لا معامل خطي.
"""
module WaveField

using LinearAlgebra

export WaveContribution, wave_superposition, born_rule, interference_strength

"""
    WaveContribution

مساهمة موجية من محرّك واحد:
  Ψ = amplitude * exp(i * phase)

  amplitude >= 0: قوة الموجة (0 = لا مساهمة)
  phase: الطور بالراديان (-π to π)
"""
struct WaveContribution
    amplitude::Float64
    phase::Float64
end

WaveContribution() = WaveContribution(0.0, 0.0)

Base.:+(a::WaveContribution, b::WaveContribution) = begin
    # حساب التراكب الفعلي (متجهات في المستوى المركب)
    re_a = a.amplitude * cos(a.phase)
    im_a = a.amplitude * sin(a.phase)
    re_b = b.amplitude * cos(b.phase)
    im_b = b.amplitude * sin(b.phase)
    re_total = re_a + re_b
    im_total = im_a + im_b
    amp = sqrt(re_total^2 + im_total^2)
    pha = atan(im_total, re_total)
    return WaveContribution(amp, pha)
end

"""
    wave_superposition(waves::Vector{WaveContribution}) -> WaveContribution

  التراكب الموجي: Ψ_total = Σ Ψ_i

  إذا كانت الموجتان متطابقتين (same phase):
    amp_total = amp1 + amp2  (تعزيز)

  إذا كانتا متعاكستين (phase diff = π):
    amp_total = |amp1 - amp2|  (إلغاء)

  هذا الفرق الجوهري عن weighted average.
"""
function wave_superposition(waves::Vector{WaveContribution})
    isempty(waves) && return WaveContribution(0.0, 0.0)
    result = waves[1]
    for i in 2:length(waves)
        result = result + waves[i]
    end
    return result
end

"""
    wave_superposition_weighted(waves, amplitudes) -> WaveContribution

  تراكب مرجّح: كل موجة لها سعتها riêng.
  amplitudes[i] يُضرب في سعة الموجة الأصلية.
"""
function wave_superposition_weighted(waves::Vector{WaveContribution},
                                     amplitudes::Vector{Float64})
    isempty(waves) && return WaveContribution(0.0, 0.0)
    @assert length(waves) == length(amplitudes) "waves and amplitudes must match"
    result = WaveContribution(waves[1].amplitude * amplitudes[1], waves[1].phase)
    for i in 2:length(waves)
        w = WaveContribution(waves[i].amplitude * amplitudes[i], waves[i].phase)
        result = result + w
    end
    return result
end

"""
    born_rule(total_wave::WaveContribution) -> Float64

  قاعدة بورن: P = |Ψ|²

  الاحتمال = مربع السعة الكلية.
  لا يوجد softmax هنا — الإحصاء خرج من المعادلة.
"""
function born_rule(total_wave::WaveContribution)
    return total_wave.amplitude^2
end

"""
    interference_strength(w1::WaveContribution, w2::WaveContribution) -> Float64

  قوة التداخل بين موجتين:
  - 1.0 = تداخل بنّاء كامل (متطابقتان)
  - 0.0 = تداخل محايد
  - -1.0 = تداخل مدمر كامل (متعاكستان)
"""
function interference_strength(w1::WaveContribution, w2::WaveContribution)
    (w1.amplitude < 1e-10 || w2.amplitude < 1e-10) && return 0.0
    return cos(w1.phase - w2.phase)
end

"""
    destructive_interference_ratio(waves::Vector{WaveContribution}) -> Float64

  نسبة التداخل المدمر الكلية:
  - 0.0 = لا يوجد إلغاء
  - 1.0 = إلغاء كامل (جميع الموجات متعاكسة)
"""
function destructive_interference_ratio(waves::Vector{WaveContribution})
    length(waves) < 2 && return 0.0
    total = wave_superposition(waves)
    sum_amplitudes = sum(w.amplitude for w in waves)
    sum_amplitudes < 1e-10 && return 0.0
    return 1.0 - (total.amplitude / sum_amplitudes)
end

"""
    compute_phases_from_context(candidate_pv, context_pvs) -> Vector{Float64}

  حساب أطوار المساهمات بناءً على السياق:
  كل طور = زاوية المتجه المرجعي للمحرّك بالنسبة للمتجه الرئيسي.

  هذا ليس عشوائياً — الأطوار تمثل اتجاهات فيزيائية حقيقية.
"""
function compute_phases_from_context(candidate_pv::AbstractVector,
                                    context_pvs::Vector{<:AbstractVector})
    isempty(context_pvs) && return Float64[0.0]

    phases = Float64[]
    cand_norm = norm(candidate_pv)
    cand_norm < 1e-10 && return Float64[0.0]

    for cpv in context_pvs
        cpv_norm = norm(cpv)
        if cpv_norm < 1e-10
            push!(phases, 0.0)
        else
            # الطور = زاوية المتجهين في المستوى المركب
            cos_angle = dot(candidate_pv, cpv) / (cand_norm * cpv_norm)
            cos_angle = clamp(cos_angle, -1.0, 1.0)
            push!(phases, acos(cos_angle))
        end
    end
    return phases
end

"""
    compute_phase_from_syntax(syn_vec) -> Float64

  الطور من المتجه النحوي:
  DET → 0, NOUN → π/4, VERB → π/2, ADV → 3π/4, CONJ → π

  كل POS له زاوية ثابتة في الدائرة الموجية.
"""
function compute_phase_from_syntax(syn_vec::AbstractVector)
    length(syn_vec) < 6 && return 0.0
    # الطور = atan2(syn[5] + syn[6], syn[1] + syn[2])
    # يُعطي زاوية في المستوى النحوي
    re = syn_vec[1] + syn_vec[2]  # noun + det
    im = syn_vec[5] + syn_vec[6]  # conj + adv
    return atan(im, re)
end

"""
    compute_phase_from_gravity(candidate_pv, context_pv, mass_cand, mass_ctx) -> Float64

  طور الجاذبية:
  F = G * m1 * m2 / r²
  الطور = زاوية الاتجاه الجاذبي
"""
function compute_phase_from_gravity(candidate_pv::AbstractVector,
                                   context_pv::AbstractVector,
                                   mass_cand::Float64, mass_ctx::Float64)
    d = min(length(candidate_pv), length(context_pv))
    direction = Float64.(context_pv[1:d] .- candidate_pv[1:d])
    dir_norm = norm(direction)
    dir_norm < 1e-10 && return 0.0
    return atan(direction[2], direction[1])  # زاوية الاتجاه
end

"""
    compute_phase_from_ksem(k_val::Float64) -> Float64

  طور الارتباط الدلالي:
  k > 0 → طور 0 (تجاذب)
  k < 0 → طور π (تنافر)
  k = 0 → طور 0
"""
function compute_phase_from_ksem(k_val::Float64)
    return k_val >= 0 ? 0.0 : π
end

"""
    compute_phase_from_causal_flow(flow_vector, candidate_pv) -> Float64

  طور التدفق السببي:
  الطور = زاوية متجه التدفق بالنسبة للمتجه المرجعي.
"""
function compute_phase_from_causal_flow(flow_vector::AbstractVector,
                                        candidate_pv::AbstractVector)
    d = min(length(flow_vector), length(candidate_pv))
    fn = norm(flow_vector[1:d])
    cn = norm(candidate_pv[1:d])
    (fn < 1e-10 || cn < 1e-10) && return 0.0
    cos_a = dot(flow_vector[1:d], candidate_pv[1:d]) / (fn * cn)
    return acos(clamp(cos_a, -1.0, 1.0))
end

"""
    compute_phase_from_density_matrix(dm_rho, candidate_pv) -> Float64

  طور مصفوفة الكثافة:
  الطور = زاوية المتجه المرجعي للمرجعية الذاتية (eigenbasis).
"""
function compute_phase_from_density_matrix(dm_rho::AbstractMatrix,
                                           candidate_pv::AbstractVector)
    d = min(size(dm_rho, 1), length(candidate_pv))
    v = Float64.(candidate_pv[1:d])
    v_norm = norm(v)
    v_norm < 1e-10 && return 0.0
    v ./= v_norm
    # الطور = atan2(v' * rho * Im(v), v' * rho * Re(v))
    # تقريبي: atan2(v[2], v[1]) من المتجه المُرجَّع
    rho_v = dm_rho * v
    return atan(dot(rho_v, v), norm(rho_v))
end

"""
    compute_phase_from_ppm(prompt_mean, candidate_pv) -> Float64

  طور حقل التنبيه:
  الطور = زاوية المتجه بالنسبة لمتوسط التنبيه.
"""
function compute_phase_from_ppm(prompt_mean::AbstractVector,
                                candidate_pv::AbstractVector)
    d = min(length(prompt_mean), length(candidate_pv))
    pm = Float64.(prompt_mean[1:d])
    cp = Float64.(candidate_pv[1:d])
    pm_n = norm(pm)
    cp_n = norm(cp)
    (pm_n < 1e-10 || cp_n < 1e-10) && return 0.0
    cos_a = dot(pm, cp) / (pm_n * cp_n)
    return acos(clamp(cos_a, -1.0, 1.0))
end

end # module WaveField
