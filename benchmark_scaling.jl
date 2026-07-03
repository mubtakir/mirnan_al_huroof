#!/usr/bin/env julia
"""
benchmark_scaling.jl - Physics Scaling Benchmark

Measures how physical quantities behave as vocabulary grows.
This is the core diagnostic tool for Phase 1: Scaling Physics.

Usage:
    julia --project=. benchmark_scaling.jl
    julia --project=. benchmark_scaling.jl --vocab-sizes 100 500 1000 5000
"""

using Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "src", "MirnanNew.jl"))
using .MirnanNew

include(joinpath(@__DIR__, "train.jl"))
using .Main: load_all_corpus, build_vocab, _strip_diacritics, _is_meaningful_word, _strip_punct_boundary

using LinearAlgebra, Statistics, SparseArrays, Random

const WORD_PHYSICS = MirnanNew.Physics.WordPhysics

# ═══════════════════════════════════════════════════════
# Sampling utilities
# ═══════════════════════════════════════════════════════

function sample_vocab(vocab::Dict{String,Int}, n::Int)
    words = collect(keys(vocab))
    if length(words) <= n
        return words
    end
    Random.seed!(42)
    return sort(Random.shuffle(words)[1:n])
end

# ═══════════════════════════════════════════════════════
# Physics quality metrics
# ═══════════════════════════════════════════════════════

function measure_mass_distribution(words)
    masses = Float64[]
    for w in words
        try
            m = WORD_PHYSICS.compute_word_mass(w)
            m > 0.0 && push!(masses, m)
        catch
        end
    end
    isempty(masses) && return (mean=0.0, std=0.0, entropy=0.0, skew=0.0)
    mu = mean(masses)
    sig = std(masses)
    bins = min(50, max(3, div(length(masses), 10)))
    counts = zeros(Int, bins)
    rmin, rmax = extrema(masses)
    span = rmax - rmin
    span < 1e-10 && return (mean=mu, std=sig, entropy=0.0, skew=0.0)
    for m in masses
        idx = clamp(ceil(Int, (m - rmin) / span * bins), 1, bins)
        counts[idx] += 1
    end
    total = sum(counts)
    H = 0.0
    for c in counts
        if c > 0
            p = c / total
            H -= p * log(p)
        end
    end
    skew = sig > 1e-10 ? mean((masses .- mu).^3) / (sig^3) : 0.0
    return (mean=mu, std=sig, entropy=H, skew=skew)
end

function measure_resonance_spectrum(words; sample_limit=300)
    n = length(words)
    n < 2 && return (mean=0.0, std=0.0, entropy=0.0, max_sim=0.0)
    pvs = Vector{Float64}[]
    for w in words
        try
            push!(pvs, Float64.(WORD_PHYSICS.compute_extended_phase_vector(w)))
        catch
        end
    end
    length(pvs) < 2 && return (mean=0.0, std=0.0, entropy=0.0, max_sim=0.0)
    sims = Float64[]
    if length(pvs) <= 80
        for i in 1:length(pvs), j in i+1:length(pvs)
            push!(sims, WORD_PHYSICS.phase_similarity(pvs[i], pvs[j]))
        end
    else
        for _ in 1:min(1000, div(sample_limit * (sample_limit - 1), 2))
            a = rand(1:length(pvs))
            b = rand(1:length(pvs))
            a != b && push!(sims, WORD_PHYSICS.phase_similarity(pvs[a], pvs[b]))
        end
    end
    isempty(sims) && return (mean=0.0, std=0.0, entropy=0.0, max_sim=0.0)
    mu_s = mean(sims)
    sig_s = std(sims)
    max_s = maximum(sims)
    bins = 30
    counts = zeros(Int, bins)
    for s in sims
        idx = clamp(ceil(Int, (s + 1.0) / 2.0 * bins), 1, bins)
        counts[idx] += 1
    end
    total = sum(counts)
    H = 0.0
    for c in counts
        if c > 0
            p = c / total
            H -= p * log(p)
        end
    end
    return (mean=mu_s, std=sig_s, entropy=H, max_sim=max_s)
end

function measure_collision_rate(words; sample_limit=200)
    n = length(words)
    n < 2 && return (collision_rate=0.0, unique_ratio=1.0)
    pvs = Vector{Float64}[]
    for w in words
        try
            push!(pvs, Float64.(WORD_PHYSICS.compute_extended_phase_vector(w)))
        catch
        end
    end
    length(pvs) < 2 && return (collision_rate=0.0, unique_ratio=1.0)
    sn = min(sample_limit, length(pvs))
    idxs = Random.randperm(length(pvs))[1:sn]
    colls = 0
    tot = 0
    thresholds = [0.99, 0.95, 0.90]
    at_99 = 0
    at_95 = 0
    at_90 = 0
    for k in 1:length(idxs)
        for l in k+1:length(idxs)
            sim = WORD_PHYSICS.phase_similarity(pvs[idxs[k]], pvs[idxs[l]])
            tot += 1
            sim > 0.99 && (at_99 += 1)
            sim > 0.95 && (at_95 += 1)
            sim > 0.90 && (at_90 += 1)
        end
    end
    rate = tot > 0 ? at_99 / tot : 0.0
    return (
        collision_rate=rate,
        unique_ratio=1.0 - rate,
        collisions_95=tot > 0 ? at_95 / tot : 0.0,
        collisions_90=tot > 0 ? at_90 / tot : 0.0
    )
end

function measure_gravity_strength(words; sample_limit=50)
    n = length(words)
    n < 2 && return (mean=0.0, std=0.0, density=0.0, max_force=0.0)
    sn = min(sample_limit, n)
    idxs = sort(Random.randperm(n)[1:sn])
    sw = words[idxs]
    strengths = Float64[]
    conns = 0
    total_p = 0
    for i in 1:length(sw)
        try
            mi = WORD_PHYSICS.compute_word_mass(sw[i])
            pvi = WORD_PHYSICS.compute_word_phase_vector(sw[i])
            for j in i+1:length(sw)
                mj = WORD_PHYSICS.compute_word_mass(sw[j])
                pvj = WORD_PHYSICS.compute_word_phase_vector(sw[j])
                sim = WORD_PHYSICS.phase_similarity(pvi, pvj)
                r = 1.0 - sim
                F = MirnanNew.Physics.Constants.GRAVITY_G * mi * mj / (r^2 + 0.01)
                push!(strengths, F)
                total_p += 1
                F > 1.0 && (conns += 1)
            end
        catch
        end
    end
    density = total_p > 0 ? conns / total_p : 0.0
    avg_s = isempty(strengths) ? 0.0 : mean(strengths)
    std_s = isempty(strengths) ? 0.0 : std(strengths)
    max_f = isempty(strengths) ? 0.0 : maximum(strengths)
    return (mean=avg_s, std=std_s, density=density, max_force=max_f)
end

function measure_root_coverage(words)
    n = length(words)
    n == 0 && return 0.0
    roots = Set{String}()
    for w in words
        try
            rc = WORD_PHYSICS._extract_root_light(w)
            !isempty(rc) && push!(roots, String(rc))
        catch
        end
    end
    return length(roots) / n
end

# ═══════════════════════════════════════════════════════
# K matrix scaling
# ═══════════════════════════════════════════════════════

function measure_K_quality(K)
    K === nothing && return (density=0.0, avg_weight=0.0, max_weight=0.0, nnz=0)
    nnz = length(K.nzval)
    V = size(K, 1)
    max_possible = V * V
    avg_w = nnz > 0 ? mean(K.nzval) : 0.0
    max_w = nnz > 0 ? maximum(K.nzval) : 0.0
    return (density=nnz / max_possible, avg_weight=avg_w, max_weight=max_w, nnz=nnz)
end

# ═══════════════════════════════════════════════════════
# Quality score
# ═══════════════════════════════════════════════════════

function physics_quality_score(mass, res, coll, grav, root_cov)
    score = 0.0
    max_score = 0.0

    mass_diversity = mass.std / max(mass.mean, 1e-10)
    s1 = clamp(mass_diversity, 0.0, 3.0) / 3.0
    score += s1 * 2.0; max_score += 2.0

    s2 = clamp(mass.entropy, 0.0, 4.0) / 4.0
    score += s2 * 1.5; max_score += 1.5

    s3 = 1.0 - coll.collision_rate
    score += s3 * 3.0; max_score += 3.0

    ideal_d = 0.3
    s4 = 1.0 - min(abs(grav.density - ideal_d) / ideal_d, 1.0)
    score += s4 * 2.0; max_score += 2.0

    s5 = clamp(root_cov, 0.0, 0.5) / 0.5
    score += s5 * 1.5; max_score += 1.5

    return max_score > 0 ? score / max_score : 0.0
end

# ═══════════════════════════════════════════════════════
# Main benchmark
# ═══════════════════════════════════════════════════════

function run_benchmark(; vocab_sizes=[100, 500, 1000, 3000, 10000])
    println("=" ^ 60)
    println("  Physics Scaling Benchmark - Mirnan")
    println("=" ^ 60)
    println()

    println("Loading corpus...")
    texts = load_all_corpus(; granularity=:document)
    if isempty(texts)
        println("No corpus found. Using built-in texts.")
        texts = [
            "العلم نور والجهل ظلام والسماء صافية والأرض خضراء والحياة جميلة والعالم كبير",
            "الله خالق كل شيء والكتاب مفيد والعلم نور والماء سر الحياة والأرض خضراء",
            "السلام عليكم ورحمة الله وبركاته القلب الكبير يعرف الحب والطريق الحق",
        ]
    end
    println("  Corpus: $(length(texts)) documents")
    println()

    println("Building full vocabulary...")
    full_vocab = build_vocab(texts; min_count=1)
    all_words = collect(keys(full_vocab))
    println("  Full vocab: $(length(full_vocab)) words")
    println()

    results = []

    for vs in vocab_sizes
        vs_actual = min(vs, length(all_words))
        if vs_actual < 10
            println("Skipping vocab_size=$vs (only $(length(all_words)) words available)")
            continue
        end

        println("-" ^ 50)
        println("Testing vocab_size = $vs_actual")
        println("-" ^ 50)

        words = sample_vocab(full_vocab, vs_actual)
        t_start = time()

        mass = measure_mass_distribution(words)
        t_mass = time() - t_start

        t_res_start = time()
        res = measure_resonance_spectrum(words)
        t_res = time() - t_res_start

        t_coll_start = time()
        coll = measure_collision_rate(words)
        t_coll = time() - t_coll_start

        t_grav_start = time()
        grav = measure_gravity_strength(words)
        t_grav = time() - t_grav_start

        root_cov = measure_root_coverage(words)
        score = physics_quality_score(mass, res, coll, grav, root_cov)

        t_total = time() - t_start

        push!(results, (
            vocab_size=vs_actual,
            mass=mass,
            resonance=res,
            collision=coll,
            gravity=grav,
            root_coverage=root_cov,
            quality_score=score,
            time_total=t_total
        ))

        println()
        println("  Mass:     mean=$(round(mass.mean, digits=6)) std=$(round(mass.std, digits=6)) H=$(round(mass.entropy, digits=3)) skew=$(round(mass.skew, digits=3))")
        println("  Resonance: mean=$(round(res.mean, digits=4)) std=$(round(res.std, digits=4)) H=$(round(res.entropy, digits=3)) max=$(round(res.max_sim, digits=4))")
        println("  Collision: 99%=$(round(coll.collision_rate*100, digits=2))% 95%=$(round(coll.collisions_95*100, digits=2))% 90%=$(round(coll.collisions_90*100, digits=2))%")
        println("  Gravity:   mean=$(round(grav.mean, digits=4)) density=$(round(grav.density*100, digits=1))% max=$(round(grav.max_force, digits=2))")
        println("  Root coverage: $(round(root_cov*100, digits=1))%")
        println("  Quality score: $(round(score, digits=3)) / 1.000")
        println("  Time: $(round(t_total, digits=1))s")
        println()
    end

    println("=" ^ 60)
    println("  SCALING SUMMARY")
    println("=" ^ 60)
    println()
    println("  Vocab | Mass-μ     | Mass-σ    | Reson-μ  | Coll-99% | Grav-ρ  | Root% | Score | Time")
    println("  ------|------------|-----------|----------|----------|---------|-------|-------|------")
    for r in results
        println("  $(lpad(string(r.vocab_size), 5)) | " *
                "$(lpad(string(round(r.mass.mean, digits=4)), 10)) | " *
                "$(lpad(string(round(r.mass.std, digits=4)), 9)) | " *
                "$(lpad(string(round(r.resonance.mean, digits=4)), 8)) | " *
                "$(lpad(string(round(r.collision.collision_rate * 100, digits=1)), 8))% | " *
                "$(lpad(string(round(r.gravity.density * 100, digits=1)), 7))% | " *
                "$(lpad(string(round(r.root_coverage * 100, digits=1)), 5))% | " *
                "$(lpad(string(round(r.quality_score, digits=3)), 5)) | " *
                "$(lpad(string(round(r.time_total, digits=1)), 4))s")
    end
    println()

    # Scaling trends
    if length(results) >= 2
        println("  SCALING TRENDS:")
        first_r = results[1]
        last_r = results[end]
        println("    Mass diversity: $(round(first_r.mass.std/first_r.mass.mean, digits=3)) -> $(round(last_r.mass.std/last_r.mass.mean, digits=3))")
        println("    Collision rate: $(round(first_r.collision.collision_rate*100, digits=1))% -> $(round(last_r.collision.collision_rate*100, digits=1))%")
        println("    Gravity density: $(round(first_r.gravity.density*100, digits=1))% -> $(round(last_r.gravity.density*100, digits=1))%")
        println("    Quality score: $(round(first_r.quality_score, digits=3)) -> $(round(last_r.quality_score, digits=3))")
    end
    println()
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
