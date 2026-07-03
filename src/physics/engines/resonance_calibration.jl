"""ResonanceCalibrator — معايرة رنينية ذاتية لأوزان التسجيل."""
module ResonanceCalibration
using Logging, Statistics
export ResonanceCalibrator

mutable struct ResonanceCalibrator
    lr::Float64; min_samples::Int; max_iter::Int; convergence::Float64; history::Vector{Dict}
end
ResonanceCalibrator(; lr=0.02, min_samples=50, max_iter=10, convergence=0.001) =
    ResonanceCalibrator(lr, min_samples, max_iter, convergence, Dict[])

function calibrate!(rc::ResonanceCalibrator, weights::Dict{String,Float64};
                     reference_scores::Dict{String,Vector{Float64}}=Dict())
    isempty(reference_scores) && return weights
    for iter in 1:rc.max_iter
        new_W = copy(weights)
        changed = false
        for (name, refs) in reference_scores
            if isempty(refs); continue; end
            avg_ref = mean(refs)
            current = get(weights, name, 1.0)
            delta = rc.lr * (avg_ref - 0.5)
            new_W[name] = max(0.1, current + delta)
            if abs(delta) > rc.convergence; changed = true; end
        end
        weights = new_W
        !changed && break
    end
    return weights
end
end
