"""
ScalingAnalysis.jl - Physics Scaling Analysis

Measures how physical quantities (mass, resonance, gravity, entropy)
behave as vocabulary grows. Essential for understanding whether
physics scales with data.
"""

module ScalingAnalysis

using LinearAlgebra, Statistics, Random

using ..Constants: GRAVITY_G
using ..WordPhysics: compute_word_phase_vector, compute_extended_phase_vector,
                     compute_word_mass, compute_word_frequency, phase_similarity

export ScalingMetrics, compute_scaling_metrics, compare_scaling_regimes,
       measure_mass_distribution, measure_resonance_spectrum,
       measure_gravity_network, measure_phase_collision_rate,
       measure_vocab_density, compute_physics_quality_score

struct ScalingMetrics
    vocab_size::Int
    mass_mean::Float64
    mass_std::Float64
    mass_entropy::Float64
    resonance_mean::Float64
    resonance_std::Float64
    resonance_entropy::Float64
    gravity_strength_mean::Float64
    gravity_network_density::Float64
    phase_collision_rate::Float64
    unique_phase_ratio::Float64
    root_coverage::Float64
    avg_word_length::Float64
end

function measure_mass_distribution(words::Vector{String})
    masses = Float64[]
    for w in words
        try
            m = compute_word_mass(w)
            m > 0.0 && push!(masses, m)
        catch e
            @debug "Scaling analysis: mass computation failed for '$w': $e"
        end
    end
    isempty(masses) && return (mean=0.0, std=0.0, entropy=0.0)
    mu = mean(masses)
    sig = std(masses)
    bins = min(50, max(3, div(length(masses), 10)))
    counts = zeros(Int, bins)
    rmin, rmax = extrema(masses)
    span = rmax - rmin
    span < 1e-10 && return (mean=mu, std=sig, entropy=0.0)
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
    return (mean=mu, std=sig, entropy=H)
end

function measure_resonance_spectrum(words::Vector{String})
    n = length(words)
    n < 2 && return (mean=0.0, std=0.0, entropy=0.0)
    pvs = Vector{Float64}[]
    for w in words
        try
            push!(pvs, Float64.(compute_extended_phase_vector(w)))
        catch e
            @debug "Scaling analysis: resonance PV failed for '$w': $e"
        end
    end
    length(pvs) < 2 && return (mean=0.0, std=0.0, entropy=0.0)
    sims = Float64[]
    if length(pvs) <= 80
        for i in 1:length(pvs), j in i+1:length(pvs)
            push!(sims, phase_similarity(pvs[i], pvs[j]))
        end
    else
        for _ in 1:min(500, div(length(pvs) * (length(pvs) - 1), 2))
            a = rand(1:length(pvs))
            b = rand(1:length(pvs))
            a != b && push!(sims, phase_similarity(pvs[a], pvs[b]))
        end
    end
    isempty(sims) && return (mean=0.0, std=0.0, entropy=0.0)
    mu_s = mean(sims)
    sig_s = std(sims)
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
    return (mean=mu_s, std=sig_s, entropy=H)
end

function measure_gravity_network(words::Vector{String})
    n = length(words)
    n < 2 && return (strength_mean=0.0, density=0.0)
    sn = min(50, n)
    idxs = sort(Random.randperm(n)[1:sn])
    sw = words[idxs]
    strengths = Float64[]
    conns = 0
    total_p = 0
    for i in 1:length(sw)
        try
            mi = compute_word_mass(sw[i])
            pvi = compute_word_phase_vector(sw[i])
            for j in i+1:length(sw)
                mj = compute_word_mass(sw[j])
                pvj = compute_word_phase_vector(sw[j])
                sim = phase_similarity(pvi, pvj)
                r = 1.0 - sim
                F = GRAVITY_G * mi * mj / (r^2 + 0.01)
                push!(strengths, F)
                total_p += 1
                F > 1.0 && (conns += 1)
            end
        catch e
            @debug "Scaling analysis: gravity network failed: $e"
        end
    end
    density = total_p > 0 ? conns / total_p : 0.0
    avg_s = isempty(strengths) ? 0.0 : mean(strengths)
    return (strength_mean=avg_s, density=density)
end

function measure_phase_collision_rate(words::Vector{String})
    n = length(words)
    n < 2 && return (collision_rate=0.0, unique_ratio=1.0)
    pvs = Vector{Float64}[]
    for w in words
        try
            push!(pvs, Float64.(compute_extended_phase_vector(w)))
        catch e
            @debug "Scaling analysis: phase collision PV failed for '$w': $e"
        end
    end
    length(pvs) < 2 && return (collision_rate=0.0, unique_ratio=1.0)
    sn = min(200, length(pvs))
    idxs = Random.randperm(length(pvs))[1:sn]
    colls = 0
    tot = 0
    for k in 1:length(idxs)
        for l in k+1:length(idxs)
            sim = phase_similarity(pvs[idxs[k]], pvs[idxs[l]])
            tot += 1
            sim > 0.99 && (colls += 1)
        end
    end
    rate = tot > 0 ? colls / tot : 0.0
    return (collision_rate=rate, unique_ratio=1.0 - rate)
end

function measure_vocab_density(words::Vector{String})
    n = length(words)
    n == 0 && return (avg_length=0.0, root_coverage=0.0)
    avg_len = mean(Float64[length(strip(w)) for w in words])
    roots = Set{String}()
    for w in words
        try
            rc = MirnanNew.Physics.WordPhysics._extract_root_light(w)
            !isempty(rc) && push!(roots, String(rc))
        catch e
            @debug "Scaling analysis: root extraction failed for '$w': $e"
        end
    end
    coverage = n > 0 ? length(roots) / n : 0.0
    return (avg_length=avg_len, root_coverage=coverage)
end

function compute_physics_quality_score(metrics::ScalingMetrics)
    score = 0.0
    md = metrics.mass_std / max(metrics.mass_mean, 1e-10)
    score += clamp(md, 0.0, 2.0) * 2.0
    score += clamp(metrics.mass_entropy, 0.0, 5.0) * 1.0
    score += metrics.unique_phase_ratio * 3.0
    ideal_d = 0.3
    ds = 1.0 - abs(metrics.gravity_network_density - ideal_d) / ideal_d
    score += clamp(ds, 0.0, 1.0) * 2.0
    score += clamp(metrics.root_coverage, 0.0, 1.0) * 2.0
    return clamp(score / 10.0, 0.0, 1.0)
end

function compute_scaling_metrics(words::Vector{String})
    isempty(words) && return nothing
    md = measure_mass_distribution(words)
    res = measure_resonance_spectrum(words)
    grav = measure_gravity_network(words)
    coll = measure_phase_collision_rate(words)
    dens = measure_vocab_density(words)
    return ScalingMetrics(
        length(words),
        md.mean, md.std, md.entropy,
        res.mean, res.std, res.entropy,
        grav.strength_mean, grav.density,
        coll.collision_rate, coll.unique_ratio,
        dens.root_coverage, dens.avg_length
    )
end

function compare_scaling_regimes(small::Vector{String}, med::Vector{String}, large::Vector{String})
    return (
        small=compute_scaling_metrics(small),
        medium=compute_scaling_metrics(med),
        large=compute_scaling_metrics(large)
    )
end

end # module ScalingAnalysis
