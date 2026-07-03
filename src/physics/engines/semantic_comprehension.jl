"""
SemanticComprehension — إدراك دلالي: حرف ← كلمة ← جملة ← موضوع.
Builds on phase vectors to create sentence-level semantic vectors
and topic density matrices for context awareness.
"""
module SemanticComprehension

using LinearAlgebra, Statistics

export SentenceSemanticVector, TopicDensityMatrix,
       compute_sentence_vector, topic_resonance, build_topic_density!,
       update_topic_with_word!, topic_alignment_score

mutable struct SentenceSemanticVector
    vec::Vector{Float64}
    n_words::Int
    coherence::Float64
end

SentenceSemanticVector() = SentenceSemanticVector(zeros(Float64, 27), 0, 0.0)

function compute_sentence_vector(words::Vector{String}, pv_fn::Function;
                                  weights::Union{Vector{Float64},Nothing}=nothing)
    n = length(words)
    n == 0 && return SentenceSemanticVector()

    pvs = Vector{Float64}[]
    for w in words
        pv = try pv_fn(w) catch; nothing end
        if pv !== nothing && length(pv) >= 27
            push!(pvs, Float64.(pv[1:27]))
        end
    end

    isempty(pvs) && return SentenceSemanticVector()

    if weights === nothing
        weights = Float64[1.0 / sqrt(i) for i in 1:length(pvs)]
        weights ./= sum(weights)
    end

    result = zeros(Float64, 27)
    for (i, pv) in enumerate(pvs)
        w = i <= length(weights) ? weights[i] : 1.0
        result .+= pv .* w
    end

    nrm = norm(result)
    if nrm > 1e-10
        result ./= nrm
    end

    coherence = length(pvs) > 1 ? mean([dot(pvs[i], pvs[j]) / (norm(pvs[i]) * norm(pvs[j]) + 1e-10)
                                         for i in 1:length(pvs) for j in (i+1):length(pvs)]) : 1.0

    return SentenceSemanticVector(result, n, coherence)
end

mutable struct TopicDensityMatrix
    dim::Int
    decay::Float64
    rho::Matrix{Float64}
    n_contributions::Int
    active::Bool
end

function TopicDensityMatrix(; dim::Int=27, decay::Float64=0.85)
    return TopicDensityMatrix(dim, decay, zeros(Float64, dim, dim), 0, true)
end

function build_topic_density!(topic::TopicDensityMatrix,
                               sentences::Vector{Vector{String}},
                               pv_fn::Function)
    n = length(sentences)
    n == 0 && return topic

    sent_vecs = Vector{Float64}[]
    for words in sentences
        sv = compute_sentence_vector(words, pv_fn)
        if sv.n_words > 0
            push!(sent_vecs, sv.vec[1:min(topic.dim, end)])
        end
    end

    isempty(sent_vecs) && return topic

    topic.rho .= 0.0
    for (i, sv) in enumerate(sent_vecs)
        w = exp(-topic.decay * (n - i))
        topic.rho .+= w .* (sv * sv')
        topic.n_contributions += 1
    end

    tr_val = tr(topic.rho)
    tr_val > 1e-10 && (topic.rho ./= tr_val)

    return topic
end

function update_topic_with_word!(topic::TopicDensityMatrix, word::String, pv_fn::Function)
    pv = try pv_fn(word) catch; nothing end
    if pv === nothing || length(pv) < topic.dim
        return topic
    end

    word_vec = Float64.(pv[1:topic.dim])
    nrm = norm(word_vec)
    nrm > 1e-10 && (word_vec ./= nrm)

    alpha = 0.15
    topic.rho .= (1.0 - alpha) .* topic.rho .+ alpha .* (word_vec * word_vec')
    topic.n_contributions += 1

    return topic
end

function topic_resonance(topic::TopicDensityMatrix, word::String, pv_fn::Function)
    !topic.active && return 0.5
    topic.n_contributions < 2 && return 0.5

    pv = try pv_fn(word) catch; nothing end
    if pv === nothing || length(pv) < topic.dim
        return 0.5
    end

    word_vec = Float64.(pv[1:topic.dim])
    nrm = norm(word_vec) + 1e-10
    word_vec ./= nrm

    R = dot(word_vec, topic.rho * word_vec)
    return clamp(R, 0.0, 1.0)
end

function topic_alignment_score(topic::TopicDensityMatrix, word::String,
                                pv_fn::Function, excitation_score::Float64)
    tr = topic_resonance(topic, word, pv_fn)
    return 0.4 * tr + 0.6 * excitation_score
end

end # module SemanticComprehension
