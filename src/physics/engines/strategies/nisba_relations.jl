function _nisba_endpoint_overlap(prompt_tokens::AbstractVector{<:AbstractString}, concepts::AbstractVector{<:AbstractString})
    length(concepts) >= 2 || return false
    length(prompt_tokens) >= 2 || return false
    
    first_ok = false
    last_ok = false
    first_keys = _identity_generation_keys(first(prompt_tokens))
    last_keys = _identity_generation_keys(last(prompt_tokens))
    
    for c in concepts
        c_keys = _identity_generation_keys(c)
        if !first_ok && !isempty(intersect(c_keys, first_keys))
            first_ok = true
        end
        if !last_ok && !isempty(intersect(c_keys, last_keys))
            last_ok = true
        end
    end
    return first_ok && last_ok
end

function _nisba_ordered_endpoint_overlap(prompt_tokens::AbstractVector{<:AbstractString}, concepts::AbstractVector{<:AbstractString})
    length(concepts) >= 2 || return false
    length(prompt_tokens) >= 2 || return false
    
    matching_indices = Int[]
    first_matched = false
    last_matched = false
    first_keys = _identity_generation_keys(first(prompt_tokens))
    last_keys = _identity_generation_keys(last(prompt_tokens))
    
    for c in concepts
        c_keys = _identity_generation_keys(c)
        for (i, tok) in enumerate(prompt_tokens)
            tok_keys = _identity_generation_keys(tok)
            if !isempty(intersect(tok_keys, c_keys))
                push!(matching_indices, i)
                if i == 1
                    first_matched = true
                elseif i == length(prompt_tokens)
                    last_matched = true
                end
                break
            end
        end
    end
    
    first_matched && last_matched || return false
    length(matching_indices) >= 2 || return false
    return issorted(matching_indices)
end

function _nisba_relation_marker_overlap(rec, relation_words::Vector{String})
    isempty(relation_words) && return true
    relation_keys = Set{String}()
    for w in relation_words
        union!(relation_keys, _light_verb_keys(w))
    end
    isempty(relation_keys) && return true
    marker_keys = Set{String}()
    for marker in rec.markers
        union!(marker_keys, _light_verb_keys(marker))
    end
    for ex in rec.evidences
        for tok in split(String(ex))
            union!(marker_keys, _light_verb_keys(tok))
        end
    end
    !isempty(intersect(relation_keys, marker_keys)) && return true
    if _prompt_relation_has_action(relation_words, _positive_relation_action_keys()) &&
       _record_has_action_keys(rec, _positive_relation_action_keys())
        return true
    end
    if _prompt_relation_has_action(relation_words, _negative_relation_action_keys()) &&
       _record_has_action_keys(rec, _negative_relation_action_keys())
        return true
    end
    return false
end

function _select_endpoint_nisba_relation(mem::NisbaMemory,
                                         relation_prompt::AbstractString,
                                         positive_words::Vector{String};
                                         active_paras=nothing,
                                         min_score::Float64=0.32,
                                         ordered_endpoints::Bool=false,
                                         relation_words::Vector{String}=String[])
    rec = select_nisba_relation(mem, relation_prompt; min_score=min_score, active_paras=active_paras)
    endpoint_ok(r) = ordered_endpoints ? _nisba_ordered_endpoint_overlap(positive_words, r.concepts) :
                                        _nisba_endpoint_overlap(positive_words, r.concepts)
    if rec !== nothing && _nisba_prompt_overlap(positive_words, rec.concepts) >= 2 &&
       endpoint_ok(rec) && _nisba_relation_marker_overlap(rec, relation_words)
        return rec
    end
    best = nothing
    best_score = 0.0
    prompt_keys = _token_keyset(positive_words)
    for candidate in values(mem.relations)
        _nisba_prompt_overlap(positive_words, candidate.concepts) >= 2 || continue
        endpoint_ok(candidate) || continue
        _nisba_relation_marker_overlap(candidate, relation_words) || continue
        concept_hits = 0
        marker_hits = 0
        for concept in candidate.concepts
            !isempty(intersect(_generation_keys(concept), prompt_keys)) && (concept_hits += 1)
        end
        for marker in candidate.markers
            !isempty(intersect(_generation_keys(marker), prompt_keys)) && (marker_hits += 1)
        end
        score = concept_hits + 0.35 * marker_hits + 0.15 * candidate.count + candidate.intensity
        if score > best_score
            best = candidate
            best_score = score
        end
    end
    best_score >= min_score && return best
    return nothing
end
