"""TrainingBalanceModule — soft corpus-frequency balancing for K builders."""
module TrainingBalanceModule

export WordBalanceConfig, build_word_balance_weights, balance_summary, pair_balance_weight

Base.@kwdef struct WordBalanceConfig
    target_quantile::Float64 = 0.60
    power::Float64 = 0.35
    min_weight::Float64 = 0.35
    max_weight::Float64 = 3.0
    rare_count_floor::Float64 = 2.0
    rare_max_weight::Float64 = 1.25
end

function _positive_quantile(values::Vector{Float64}, q::Float64)
    isempty(values) && return 1.0
    sort!(values)
    qq = clamp(q, 0.0, 1.0)
    idx = clamp(Int(ceil(qq * length(values))), 1, length(values))
    return max(values[idx], 1.0)
end

function _safe_config(config::WordBalanceConfig)
    min_w = max(config.min_weight, 1e-6)
    max_w = max(config.max_weight, min_w)
    rare_max = clamp(config.rare_max_weight, min_w, max_w)
    return WordBalanceConfig(;
        target_quantile=clamp(config.target_quantile, 0.05, 0.95),
        power=clamp(config.power, 0.0, 1.0),
        min_weight=min_w,
        max_weight=max_w,
        rare_count_floor=max(config.rare_count_floor, 0.0),
        rare_max_weight=rare_max,
    )
end

"""
    build_word_balance_weights(counts; config=WordBalanceConfig())

Builds hidden balancing weights from raw token counts.

The weights are intentionally soft: common words are damped, underrepresented
words are lifted, and extremely rare words are not allowed to receive an
unbounded boost. These weights are for training K matrices only; they are not
treated as corpus text.
"""
function build_word_balance_weights(counts::AbstractVector{<:Real};
                                    config::WordBalanceConfig=WordBalanceConfig())
    cfg = _safe_config(config)
    n = length(counts)
    positive = Float64[Float64(c) for c in counts if Float64(c) > 0.0 && isfinite(Float64(c))]
    target = _positive_quantile(positive, cfg.target_quantile)
    weights = ones(Float64, n)

    for i in 1:n
        c = Float64(counts[i])
        if !(isfinite(c)) || c <= 0.0
            weights[i] = 1.0
            continue
        end
        raw = (target / max(c, 1.0)) ^ cfg.power
        if c <= cfg.rare_count_floor
            raw = min(raw, cfg.rare_max_weight)
        end
        weights[i] = clamp(raw, cfg.min_weight, cfg.max_weight)
    end

    meta = Dict{String,Any}(
        "version" => 1,
        "target_count" => target,
        "target_quantile" => cfg.target_quantile,
        "power" => cfg.power,
        "min_weight" => cfg.min_weight,
        "max_weight" => cfg.max_weight,
        "rare_count_floor" => cfg.rare_count_floor,
        "rare_max_weight" => cfg.rare_max_weight,
    )
    merge!(meta, balance_summary(weights))
    return weights, meta
end

function balance_summary(weights::AbstractVector{<:Real}; neutral_band::Float64=0.05)
    n = length(weights)
    n == 0 && return Dict{String,Any}(
        "n_weights" => 0,
        "boosted" => 0,
        "reduced" => 0,
        "neutral" => 0,
        "min_observed_weight" => 1.0,
        "max_observed_weight" => 1.0,
        "mean_weight" => 1.0,
    )

    low = 1.0 - abs(neutral_band)
    high = 1.0 + abs(neutral_band)
    vals = Float64[isfinite(Float64(w)) ? Float64(w) : 1.0 for w in weights]
    boosted = count(w -> w > high, vals)
    reduced = count(w -> w < low, vals)
    return Dict{String,Any}(
        "n_weights" => n,
        "boosted" => boosted,
        "reduced" => reduced,
        "neutral" => n - boosted - reduced,
        "min_observed_weight" => minimum(vals),
        "max_observed_weight" => maximum(vals),
        "mean_weight" => sum(vals) / n,
    )
end

function pair_balance_weight(weights::Union{Nothing,AbstractVector{<:Real}}, i::Integer, j::Integer)
    weights === nothing && return 1.0
    if i < 1 || j < 1 || i > length(weights) || j > length(weights)
        return 1.0
    end
    wi = Float64(weights[i])
    wj = Float64(weights[j])
    if !(isfinite(wi) && isfinite(wj)) || wi <= 0.0 || wj <= 0.0
        return 1.0
    end
    return sqrt(wi * wj)
end

end # module TrainingBalanceModule
