"""
IntentDetector — كاشف القصد عبر متجهات طورية + كلمات مفتاحية.

الأغراض: GREETING, QUESTION, COMMAND, REQUEST, FAREWELL, STATEMENT, OPINION, SUGGESTION, COMPLAINT, PROMISE, THANK.
"""
module IntentDetectorModule
using LinearAlgebra, Statistics
using JSON
using ..WordPhysics: compute_extended_phase_vector

export IntentDetector

const _INTENT_PATTERNS_FILE = joinpath(@__DIR__, "..", "..", "data", "intent_patterns.json")

function _load_intent_patterns()
    keywords = Dict{String,Vector{String}}()
    canonical = Dict{String,Vector{String}}()
    
    # Inline fallback
    fallback_keywords = Dict{String,Vector{String}}(
        "GREETING" => ["سلام", "مرحبا", "اهلا", "صباح", "مساء", "hello", "hi", "hey"],
        "QUESTION" => ["هل", "ما", "كيف", "لماذا", "متى", "أين", "من", "كم", "what", "how", "why", "where", "when", "who"],
        "COMMAND" => ["افعل", "اكتب", "اذهب", "قل", "اعمل", "do", "go", "make", "write", "say"],
        "REQUEST" => ["من فضلك", "لو سمحت", "أرجو", "please", "could", "would"],
        "FAREWELL" => ["وداعا", "مع السلامة", "bye", "goodbye", "see you"],
        "THANK" => ["شكرا", "مشكور", "thanks", "thank"],
    )
    for (k, v) in fallback_keywords
        keywords[k] = v
        canonical[k] = [k]
    end
    
    if isfile(_INTENT_PATTERNS_FILE)
        try
            data = JSON.parsefile(_INTENT_PATTERNS_FILE)
            if haskey(data, "intents")
                intents = data["intents"]
                for (name, info) in intents
                    kws = get(info, "keywords", String[])
                    can = get(info, "canonical", String[])
                    keywords[String(name)] = String[String(x) for x in kws]
                    canonical[String(name)] = String[String(x) for x in can]
                end
            end
        catch e
            @warn "Failed to load intent patterns from $_INTENT_PATTERNS_FILE: $e"
        end
    end
    return keywords, canonical
end

const INTENT_KEYWORDS, INTENT_CANONICAL = _load_intent_patterns()

function _phrase_pv(phrase::String)
    words = split(phrase)
    if isempty(words)
        return nothing
    end
    pvs = Vector{Float64}[]
    for w in words
        try
            pv = compute_extended_phase_vector(String(w))
            push!(pvs, Float64.(pv))
        catch e
            @debug "Intent detector: failed to compute PV for '$w': $e"
        end
    end
    if isempty(pvs)
        return nothing
    end
    avg = vec(mean(reduce(hcat, pvs)'; dims=1))
    nrm = norm(avg)
    if nrm > 1e-10
        avg ./= nrm
    end
    return avg
end

function _build_intent_attractors()
    intent_pvs = Dict{String,Vector{Float64}}()
    for (intent_name, phrases) in INTENT_CANONICAL
        pvs = Vector{Float64}[]
        for phrase in phrases
            pv = _phrase_pv(phrase)
            if pv !== nothing
                push!(pvs, pv)
            end
        end
        if !isempty(pvs)
            attractor = vec(mean(reduce(hcat, pvs)'; dims=1))
            nrm = norm(attractor)
            if nrm > 1e-10
                attractor ./= nrm
            end
            intent_pvs[intent_name] = attractor
        end
    end
    return intent_pvs
end

mutable struct IntentDetector
    intent_vectors::Dict{String,Vector{Float64}}
    pv_fn::Any
end

function IntentDetector(; pv_fn=nothing)
    intent_pvs = _build_intent_attractors()
    return IntentDetector(intent_pvs, pv_fn)
end

function build!(det::IntentDetector, canonical_texts::Dict{String,Vector{String}})
    # Kept for compatibility, attractors are now built at construction
end

function detect(det::IntentDetector, text::String)
    if isempty(text)
        return Dict("intent" => "STATEMENT", "confidence" => 0.0, "scores" => Dict{String,Float64}())
    end

    input_pv = _phrase_pv(text)
    if input_pv === nothing
        return Dict("intent" => "STATEMENT", "confidence" => 0.0, "scores" => Dict{String,Float64}())
    end

    distances = Dict{String,Float64}()
    for (name, attractor) in det.intent_vectors
        cos_sim = mean(cos.(input_pv .- attractor))
        distances[name] = cos_sim
    end

    # Keyword boost
    text_lower = lowercase(text)
    words_set = Set{String}(String(w) for w in split(text_lower))
    keyword_boost = Dict{String,Int}()
    for (intent_name, kws) in INTENT_KEYWORDS
        sorted_kws = sort(kws; by=length, rev=true)
        for kw in sorted_kws
            multi = split(kw)
            if length(multi) > 1
                if occursin(lowercase(kw), text_lower)
                    keyword_boost[intent_name] = get(keyword_boost, intent_name, 0) + 1
                    break
                end
            else
                if kw in words_set
                    keyword_boost[intent_name] = get(keyword_boost, intent_name, 0) + 1
                    break
                end
            end
        end
    end

    for (intent_name, boost) in keyword_boost
        if haskey(distances, intent_name)
            distances[intent_name] += boost * 0.15
        end
    end

    if isempty(distances)
        return Dict("intent" => "STATEMENT", "confidence" => 0.0, "scores" => Dict{String,Float64}())
    end

    best = argmax(distances)
    raw_confidence = distances[best]
    min_dist = minimum(values(distances))
    confidence = clamp(raw_confidence - min_dist + 0.3, 0.0, 1.0)

    return Dict(
        "intent" => best,
        "confidence" => confidence,
        "scores" => distances
    )
end

end

