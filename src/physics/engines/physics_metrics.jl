"""
PhysicsMetrics — مقاييس جودة المحرك الفيزيائي (Phase 2).

Three core metrics:
1. K Density — كثافة مصفوفة الارتباط (كمية المعلومات المشتركة)
2. Transition Entropy — إنتروبيا الانتقال (تنويع التنبؤ)
3. Density Matrix Purity — نقاء مصفوفة الكثافة (وضوح الحالة)
"""
module PhysicsMetrics

using LinearAlgebra, Statistics, SparseArrays, Printf

using ..DensityMatrix: PhaseDensityMatrix, get_purity, get_spectral_entropy, get_trace

export PhysicsQualityReport, compute_quality_report,
       k_density, transition_entropy, transition_entropy_per_row,
       effective_connections, sparsity_ratio,
       density_purity, density_spectral_entropy,
       overall_quality_report

struct PhysicsQualityReport
    k_density::Float64
    k_sparsity::Float64
    k_effective_connections::Int
    transition_entropy::Float64
    transition_entropy_std::Float64
    density_purity::Float64
    density_trace::Float64
    density_spectral_entropy::Float64
    overall_score::Float64
end

"""
K Density — كثافة مصفوفة الارتباط الدلالي.
H = (عدد العناصر غير الصفرية) / (حجم المصفوفة الكلي)
قيمة عالية = معلومات مشتركة أكثر بين الكلمات.
قيمة مثالية: 0.1 - 0.5 (sparse but informative)
"""
function k_density(K::AbstractMatrix)
    V = size(K, 1)
    V == 0 && return 0.0
    total = V * V
    if K isa AbstractSparseMatrix
        nnz_count = nnz(K)
    else
        nnz_count = count(!iszero, K)
    end
    return nnz_count / total
end

"""
Sparsity Ratio — نسبة الخواء.
1.0 = مصفوفة فارغة تماماً، 0.0 = مصفوفة ممتلئة.
"""
sparsity_ratio(K::AbstractMatrix) = 1.0 - k_density(K)

"""
Effective Connections — عدد الارتباطات الفعّالة (أقوى من عتبة معينة).
العتبة الافتراضية: 0.01 (ارتباط طفيف لكنه موجود).
"""
function effective_connections(K::AbstractMatrix; threshold::Float64=0.01)
    if K isa SparseMatrixCSC
        return count(x -> abs(x) > threshold, K.nzval)
    else
        return count(x -> abs(x) > threshold, K)
    end
end

"""
Transition Entropy — إنتروبيا الانتقال لكل صف.
H_i = -Σ_j p(j|i) log(p(j|i))
حيث p(j|i) = |K_ji| / Σ_j |K_ji|

entropy عالية = تنويع كبير في الانتقالات (less predictable)
entropy منخفضة = انتقالات حتمية (more predictable)
"""
function transition_entropy_per_row(K::AbstractMatrix)
    V = size(K, 1)
    V == 0 && return Float64[]
    entropies = Float64[]
    if K isa SparseMatrixCSC
        colptr = K.colptr
        rowval = K.rowval
        nzval = K.nzval
        for i in 1:V
            s = 0.0
            for p in colptr[i]:(colptr[i+1]-1)
                s += abs(nzval[p])
            end
            s < 1e-10 && continue
            H = 0.0
            for p in colptr[i]:(colptr[i+1]-1)
                pv = abs(nzval[p]) / s
                pv > 1e-10 && (H -= pv * log(pv))
            end
            push!(entropies, H)
        end
    else
        for i in 1:V
            vals = abs.(view(K, i, :))
            s = sum(vals)
            s < 1e-10 && continue
            p = vals ./ s
            p = p[p .> 1e-10]
            H = -sum(p .* log.(p))
            push!(entropies, H)
        end
    end
    return entropies
end

"""
Transition Entropy — المتوسط الحسابي لإنتروبيا الانتقال.
قيمة عالية = محرك غير حتمي (more creative/diverse).
قيمة منخفضة = محرك حتمي (more focused/deterministic).
"""
function transition_entropy(K::AbstractMatrix)
    ents = transition_entropy_per_row(K)
    isempty(ents) && return (mean=0.0, std=0.0)
    return (mean=mean(ents), std=std(ents))
end

"""
Density Purity — نقاء مصفوفة الكثافة.
Tr(ρ²) = 1.0: حالة نقية (pure state) — كل الكلمات متزامنة.
Tr(ρ²) = 1/d: حالة خليط عشوائي (maximally mixed).
قيمة أعلى = تماسك أقوى بين كلمات السياق.
"""
density_purity(dm::PhaseDensityMatrix) = get_purity(dm)

"""
Density Trace — أثر مصفوفة الكثافة.
يجب أن يكون ≈ 1.0 (مصفوفة الكثافة المقننة).
"""
density_trace(dm::PhaseDensityMatrix) = get_trace(dm)

"""
Density Spectral Entropy — الإنتروبيا الطيفية لمصفوفة الكثافة.
قيمة عالية = حالة خليط (less coherent).
قيمة منخفضة = حالة نقية (more coherent).
"""
density_spectral_entropy(dm::PhaseDensityMatrix) = get_spectral_entropy(dm)

"""
Overall Quality Score — النتيجة الشاملة لمقياس الجودة الفيزيائية.
(0.0 = ضعيف جداً، 1.0 = ممتاز)
"""
function overall_quality_report(k_dens::Float64, trans_ent::Float64,
                                dm_purity::Float64)
    # K Density: مثالية حول 0.2
    k_score = 1.0 - abs(k_dens - 0.2) / 0.8
    k_score = clamp(k_score, 0.0, 1.0)

    # Transition Entropy: مثالية حول 3.0-5.0 (توسط الكلمات ≈ 1201)
    # log(1201) ≈ 7.09 — entropy أقصى = 7.09
    # entropy جيدة ≈ 3.0-5.0
    ent_norm = clamp(trans_ent / 5.0, 0.0, 1.0)

    # Purity: أعلى = أفضل (تماسك)
    pur_score = clamp(dm_purity, 0.0, 1.0)

    # الوزن: K=0.3, Entropy=0.3, Purity=0.4
    return clamp(0.3 * k_score + 0.3 * ent_norm + 0.4 * pur_score, 0.0, 1.0)
end

"""
Compute full quality report for a generator's K_sem and density matrix.
"""
function compute_quality_report(K::Union{AbstractMatrix,Nothing},
                                dm::Union{PhaseDensityMatrix,Nothing})
    k_d = K !== nothing ? k_density(K) : 0.0
    k_s = K !== nothing ? sparsity_ratio(K) : 1.0
    k_ec = K !== nothing ? effective_connections(K) : 0
    te = K !== nothing ? transition_entropy(K) : (mean=0.0, std=0.0)
    dp = dm !== nothing ? density_purity(dm) : 0.0
    dt = dm !== nothing ? density_trace(dm) : 0.0
    dse = dm !== nothing ? density_spectral_entropy(dm) : 0.0
    os = overall_quality_report(k_d, te.mean, dp)
    return PhysicsQualityReport(k_d, k_s, k_ec, te.mean, te.std, dp, dt, dse, os)
end

function Base.show(io::IO, r::PhysicsQualityReport)
    println(io, "╔══════════════════════════════════════════╗")
    println(io, "║   Physics Quality Report — تقرير الجودة   ║")
    println(io, "╠══════════════════════════════════════════╣")
    println(io, @sprintf("║ K Density:          %.4f (%.1f%%)      ║", r.k_density, r.k_density * 100))
    println(io, @sprintf("║ K Sparsity:         %.4f                ║", r.k_sparsity))
    println(io, @sprintf("║ Effective Links:    %d                   ║", r.k_effective_connections))
    println(io, "╠══════════════════════════════════════════╣")
    println(io, @sprintf("║ Trans. Entropy:     %.3f ± %.3f         ║", r.transition_entropy, r.transition_entropy_std))
    println(io, "╠══════════════════════════════════════════╣")
    println(io, @sprintf("║ Density Purity:     %.4f                ║", r.density_purity))
    println(io, @sprintf("║ Density Trace:      %.4f                ║", r.density_trace))
    println(io, @sprintf("║ Spectral Entropy:   %.4f                ║", r.density_spectral_entropy))
    println(io, "╠══════════════════════════════════════════╣")
    println(io, @sprintf("║ ★ Overall Score:    %.3f / 1.000        ║", r.overall_score))
    println(io, "╚══════════════════════════════════════════╝")
end

end # module PhysicsMetrics
