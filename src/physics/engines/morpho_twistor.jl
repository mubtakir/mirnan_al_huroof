"""
MorphoTwistor — عامل الالتواء الطوري.
Learns morphological relationships between words via vector operations.
Simplified version using 27D phase vectors (no Clifford algebra).
"""
module MorphoTwistor

using LinearAlgebra, Statistics

export TwistorOperator, TwistorEngine, MorphPattern,
       compute_twistor, apply_twistor, twistor_similarity,
       learn_patterns!, predict_derivation, find_morphological_pairs,
       score_twistor_candidate

struct TwistorOperator
    vec::Vector{Float64}
    magnitude::Float64
    rotation::Float64
    scale::Float64
end

function TwistorOperator(vec::Vector{Float64})
    nrm = norm(vec)
    return TwistorOperator(vec, nrm, 0.0, abs(vec[1]))
end

TwistorOperator() = TwistorOperator(ones(27) ./ sqrt(27))

function Base.show(io::IO, t::TwistorOperator)
    print(io, "Twistor(|T|=", round(t.magnitude; digits=3),
          ", rot=", round(t.rotation; digits=3),
          ", scale=", round(t.scale; digits=3), ")")
end

mutable struct MorphPattern
    center::TwistorOperator
    label::String
    count::Int
    examples::Vector{Tuple{String,String}}
end

MorphPattern(center::TwistorOperator; label::String="") =
    MorphPattern(center, label, 0, Tuple{String,String}[])

mutable struct TwistorEngine
    patterns::Vector{MorphPattern}
    pair_cache::Dict{Tuple{String,String},TwistorOperator}
    active::Bool
end

TwistorEngine() = TwistorEngine(MorphPattern[], Dict{Tuple{String,String},TwistorOperator}(), true)

function compute_twistor(w1::String, w2::String, pv_fn::Function)
    pv1 = Float64.(pv_fn(w1))
    pv2 = Float64.(pv_fn(w2))
    d = min(length(pv1), length(pv2))
    v1 = pv1[1:d]
    v2 = pv2[1:d]
    n1 = norm(v1); n2 = norm(v2)
    if n1 > 1e-10; v1 ./= n1; end
    if n2 > 1e-10; v2 ./= n2; end
    T_vec = v2 .- v1
    return TwistorOperator(T_vec)
end

function apply_twistor(T::TwistorOperator, word::String, pv_fn::Function)
    pv = Float64.(pv_fn(word))
    d = min(length(pv), length(T.vec))
    result = pv[1:d] .+ T.vec[1:d]
    nrm = norm(result)
    if nrm > 1e-10; result ./= nrm; end
    return result
end

function apply_twistor_pv(T::TwistorOperator, pv::AbstractVector)
    d = min(length(pv), length(T.vec))
    result = Float64.(pv[1:d]) .+ T.vec[1:d]
    nrm = norm(result)
    if nrm > 1e-10; result ./= nrm; end
    return result
end

function twistor_similarity(T1::TwistorOperator, T2::TwistorOperator)
    d = min(length(T1.vec), length(T2.vec))
    if d == 0; return 0.0; end
    v1 = T1.vec[1:d]
    v2 = T2.vec[1:d]
    n1 = norm(v1); n2 = norm(v2)
    if n1 < 1e-10 || n2 < 1e-10; return 0.0; end
    return max(0.0, min(1.0, dot(v1, v2) / (n1 * n2)))
end

function find_morphological_pairs(words::Vector{String})
    pairs = Tuple{String,String}[]
    seen = Set{Tuple{String,String}}()
    n = length(words)
    for i in 1:n
        for j in max(1, i-5):min(n, i+5)
            j == i && continue
            w1, w2 = words[i], words[j]
            (length(w1) < 3 || length(w2) < 3) && continue
            w1 == w2 && continue
            w1_chars = collect(w1)
            w2_chars = collect(w2)
            l1 = length(w1_chars); l2 = length(w2_chars)
            min_len = min(l1, l2, 3)
            shared = 0
            for k in 1:min_len
                if w1_chars[k] == w2_chars[k]
                    shared += 1
                end
            end
            if shared >= 2 && shared >= min_len - 1
                key = (w1, w2)
                key in seen && continue
                push!(seen, key)
                push!(pairs, (w1, w2))
            end
        end
    end
    return pairs
end

function _blend_twistors(a::TwistorOperator, b::TwistorOperator, alpha::Float64)
    beta = 1.0 - alpha
    d = min(length(a.vec), length(b.vec))
    blended = beta .* a.vec[1:d] .+ alpha .* b.vec[1:d]
    return TwistorOperator(blended)
end

function learn_patterns!(engine::TwistorEngine, pairs::Vector{Tuple{String,String}}, pv_fn::Function;
                          sim_threshold::Float64=0.7, min_cluster_size::Int=2)
    isempty(pairs) && return engine.patterns
    engine.active || return engine.patterns

    computed = Tuple{Float64,TwistorOperator,Tuple{String,String}}[]
    for (w1, w2) in pairs
        T = get!(engine.pair_cache, (w1, w2)) do
            compute_twistor(w1, w2, pv_fn)
        end
        push!(computed, (T.magnitude, T, (w1, w2)))
    end

    valid = filter(x -> x[1] > 0.01, computed)
    isempty(valid) && return engine.patterns

    clusters = Tuple{TwistorOperator, Vector{Tuple{String,String}}}[]
    assigned = falses(length(valid))

    for i in 1:length(valid)
        assigned[i] && continue
        Ti = valid[i][2]
        cluster_center = Ti
        cluster_members = [valid[i][3]]
        assigned[i] = true

        for j in (i+1):length(valid)
            assigned[j] && continue
            Tj = valid[j][2]
            sim = twistor_similarity(Ti, Tj)
            if sim >= sim_threshold
                alpha = 1.0 / (length(cluster_members) + 1)
                cluster_center = _blend_twistors(cluster_center, Tj, alpha)
                push!(cluster_members, valid[j][3])
                assigned[j] = true
            end
        end

        if length(cluster_members) >= min_cluster_size
            push!(clusters, (cluster_center, cluster_members))
        end
    end

    new_patterns = MorphPattern[]
    for (center, members) in clusters
        pat = MorphPattern(center; label="pattern_$(length(new_patterns)+1)")
        pat.count = length(members)
        pat.examples = members[1:min(5, end)]
        push!(new_patterns, pat)
    end

    for np in new_patterns
        merged = false
        for ep in engine.patterns
            if twistor_similarity(np.center, ep.center) >= 0.85
                ep.count += np.count
                append!(ep.examples, np.examples)
                ep.center = _blend_twistors(ep.center, np.center, 0.3)
                merged = true
                break
            end
        end
        merged || push!(engine.patterns, np)
    end

    return engine.patterns
end

function predict_derivation(engine::TwistorEngine, w1::String, w2::String,
                             query::String, pv_fn::Function;
                             vocab::Dict{String,Int}=Dict(), top_k::Int=10)
    T_example = get!(engine.pair_cache, (w1, w2)) do
        compute_twistor(w1, w2, pv_fn)
    end

    best_T = T_example
    best_sim = 0.0
    for pat in engine.patterns
        sim = twistor_similarity(T_example, pat.center)
        if sim > best_sim && sim > 0.5
            best_sim = sim
            best_T = pat.center
        end
    end

    pred_v = apply_twistor_pv(best_T, pv_fn(query))

    if isempty(vocab)
        return Tuple{String,Float64}[]
    end

    results = Tuple{String,Float64}[]
    for (word, _) in vocab
        word == query && continue
        word_pv = Float64.(pv_fn(word))
        d = min(length(word_pv), length(pred_v))
        wv = word_pv[1:d]
        pv = pred_v[1:d]
        nw = norm(wv); np = norm(pv)
        if nw > 1e-10 && np > 1e-10
            sim = max(0.0, dot(wv, pv) / (nw * np))
            push!(results, (word, sim))
        end
    end

    sort!(results; by=x -> -x[2])
    return results[1:min(top_k, end)]
end

function score_twistor_candidate(engine::TwistorEngine, candidate::String,
                                  query::String, T::TwistorOperator, pv_fn::Function)
    pred_v = apply_twistor_pv(T, pv_fn(query))
    cand_pv = Float64.(pv_fn(candidate))
    d = min(length(cand_pv), length(pred_v))
    cv = cand_pv[1:d]
    pv = pred_v[1:d]
    nw = norm(cv); np = norm(pv)
    if nw < 1e-10 || np < 1e-10
        return 0.0
    end
    return max(0.0, dot(cv, pv) / (nw * np))
end

end # module MorphoTwistor
