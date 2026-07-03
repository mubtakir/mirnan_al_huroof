"""
Holographic Knowledge Base — الذاكرة الهولوغرافية المعرفية.
Resonant Filter Bank for knowledge storage and retrieval via quantum superposition.
- Each fact = (subject_pv, relation, object_pv)
- Query = weighted superposition of all facts at once
- Answer = reconstructed vector via weighted sum
"""
module HolographicKB

using LinearAlgebra

export HolographicKnowledgeBase, RELATION_TYPES, store_fact!, query, query_by_word, reconstruct_vector, clear!

const RELATION_TYPES = Dict(
    "IS_A" => 0,
    "HAS_PROPERTY" => 1,
    "LOCATED_IN" => 2,
    "CAPABLE_OF" => 3,
    "PART_OF" => 4,
    "SYNONYM" => 5,
    "CURATED" => 6,
    "CAPITAL_OF" => 7,
    "MADE_OF" => 8,
    "USED_FOR" => 9,
    "BECOMES_WHEN" => 10,
    "PRODUCES" => 11,
)

mutable struct HolographicKnowledgeBase
    banks::Dict{String,Dict{String,Vector}}
    fact_count::Int
    built::Bool

    function HolographicKnowledgeBase()
        return new(Dict{String,Dict{String,Vector}}(), 0, false)
    end
end

function store_fact!(kb::HolographicKnowledgeBase, subj_pv::AbstractVector,
                     obj_pv::AbstractVector, rel_type::String;
                     subj_word::String="", obj_word::String="")
    if !haskey(RELATION_TYPES, rel_type)
        return false
    end

    if !haskey(kb.banks, rel_type)
        kb.banks[rel_type] = Dict(
            "keys" => Vector{Float64}[],
            "values" => Vector{Float64}[],
            "subj_words" => String[],
            "obj_words" => String[],
        )
    end

    bank = kb.banks[rel_type]

    for i in 1:length(bank["subj_words"])
        if bank["subj_words"][i] == subj_word && bank["obj_words"][i] == obj_word
            return false
        end
    end

    push!(bank["keys"], Float64.(subj_pv))
    push!(bank["values"], Float64.(obj_pv))
    push!(bank["subj_words"], subj_word)
    push!(bank["obj_words"], obj_word)
    kb.fact_count += 1
    kb.built = true
    return true
end

function query(kb::HolographicKnowledgeBase, query_pv::AbstractVector;
               rel_type::Union{String,Nothing}=nothing, top_k::Int=10,
               sharpening::Float64=3.0, cutoff::Float64=0.0)
    if !kb.built
        return Tuple{Float64,String,String}[]
    end

    query_norm = norm(query_pv)
    if query_norm < 1e-10
        return Tuple{Float64,String,String}[]
    end
    q = Float64.(query_pv) ./ query_norm

    rel_types = rel_type === nothing ? collect(keys(kb.banks)) : [rel_type]
    all_results = Tuple{Float64,String,String}[]

    for rt in rel_types
        if !haskey(kb.banks, rt)
            continue
        end
        bank = kb.banks[rt]
        keys = bank["keys"]
        values = bank["values"]
        obj_words = bank["obj_words"]

        if isempty(keys)
            continue
        end

        keys_matrix = reduce(hcat, keys)'
        keys_norms = [norm(k) for k in keys]
        keys_normed = copy(keys_matrix)
        for i in 1:size(keys_normed, 1)
            if keys_norms[i] > 1e-10
                keys_normed[i, :] ./= keys_norms[i]
            end
        end

        resonances = keys_normed * q
        sharpened = exp.(sharpening .* resonances)
        total = sum(sharpened)
        if total < 1e-10
            continue
        end
        sharpened ./= total

        dynamic_cutoff = max(cutoff, 2.0 / max(length(keys), 1))
        above = sharpened .> dynamic_cutoff

        for i in findall(above)
            obj_word = i <= length(obj_words) ? obj_words[i] : ""
            if length(obj_word) >= 2
                push!(all_results, (sharpened[i], obj_word, rt))
            end
        end
    end

    sort!(all_results; by=x -> -x[1])
    return all_results[1:min(top_k, end)]
end

function query_by_word(kb::HolographicKnowledgeBase, query_word::String;
                       rel_type::Union{String,Nothing}=nothing, top_k::Int=10)
    if !kb.built
        return Tuple{Float64,String,String}[]
    end

    rel_types = rel_type === nothing ? collect(keys(kb.banks)) : [rel_type]
    all_results = Tuple{Float64,String,String}[]

    for rt in rel_types
        if !haskey(kb.banks, rt)
            continue
        end
        bank = kb.banks[rt]
        for i in 1:length(bank["subj_words"])
            if bank["subj_words"][i] == query_word
                obj_word = i <= length(bank["obj_words"]) ? bank["obj_words"][i] : ""
                push!(all_results, (1.0, obj_word, rt))
            end
        end
    end

    sort!(all_results; by=x -> -x[1])
    return all_results[1:min(top_k, end)]
end

function reconstruct_vector(kb::HolographicKnowledgeBase, query_pv::AbstractVector;
                            rel_type::Union{String,Nothing}=nothing)
    if !kb.built
        return nothing
    end

    q = Float64.(query_pv) / max(norm(query_pv), 1e-10)
    rel_types = rel_type === nothing ? collect(keys(kb.banks)) : [rel_type]
    reconstructed = zeros(Float64, length(q))
    total_weight = 0.0

    for rt in rel_types
        if !haskey(kb.banks, rt)
            continue
        end
        bank = kb.banks[rt]
        keys = bank["keys"]
        values = bank["values"]

        if isempty(keys)
            continue
        end

        resonances = Float64[clamp(dot(k, q) / (norm(k) * norm(q) + 1e-10), -1.0, 1.0)
                             for k in keys]
        weights = exp.(3.0 .* resonances)
        weights ./= (sum(weights) + 1e-10)

        for i in 1:length(weights)
            if weights[i] > 0.01
                reconstructed .+= weights[i] .* values[i]
                total_weight += weights[i]
            end
        end
    end

    if total_weight > 1e-10
        reconstructed ./= total_weight
        nrm = norm(reconstructed)
        if nrm > 1e-10
            reconstructed ./= nrm
        end
        return reconstructed
    end
    return nothing
end

function clear!(kb::HolographicKnowledgeBase)
    empty!(kb.banks)
    kb.fact_count = 0
    kb.built = false
end

end # module HolographicKB
