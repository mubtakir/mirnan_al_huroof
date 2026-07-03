"""PathIntegralReasoner — مسبب تكامل مساري (جاذبية مزدوجة + KB-first)."""
module PathIntegralReasonerModule
using LinearAlgebra
export PathIntegralReasoner

mutable struct PathIntegralReasoner
    beam_width::Int; max_depth::Int; min_depth::Int; convergence::Float64; goal_weight::Float64
end
PathIntegralReasoner(; beam_width=10, max_depth=8, min_depth=2, convergence=0.75, goal_weight=0.3) =
    PathIntegralReasoner(beam_width, max_depth, min_depth, convergence, goal_weight)

function reason!(pir::PathIntegralReasoner, start_pv::AbstractVector, goal_pv::AbstractVector,
                 pv_fn, vocab::Dict, K_sem=nothing; kb=nothing)
    chain = [(start_pv, "START")]; current_pv = start_pv
    for depth in 1:pir.max_depth
        # Get candidates from K_sem or KB
        candidates = Tuple{Float64,String,Vector{Float64}}[]
        if kb !== nothing && kb.built
            kb_results = kb_query(kb, current_pv; top_k=5)
            for result in kb_results
                conf, word, rel = result
                if haskey(vocab, word)
                    push!(candidates, (conf, word, Float64.(pv_fn(word))))
                end
            end
        elseif K_sem !== nothing
            # Fallback: use K_sem to find candidates
            cid = get(vocab, last(chain)[2], nothing)
            if cid !== nothing && cid <= size(K_sem, 1)
                row = K_sem isa AbstractSparseMatrix ? Vector(K_sem[cid, :]) : Vector(K_sem[cid, :])
                sorted = sortperm(row; rev=true)[1:min(pir.beam_width, end)]
                id2word_local = Dict(v => k for (k, v) in vocab)
                for tid in sorted
                    row[tid] > 1e-6 || break
                    w = get(id2word_local, tid, nothing)
                    w !== nothing && push!(candidates, (row[tid], w, Float64.(pv_fn(w))))
                end
            end
        end

        if isempty(candidates)
            sim = dot(current_pv, goal_pv) / (norm(current_pv)*norm(goal_pv)+1e-10)
            if sim > pir.convergence || depth >= pir.min_depth
                break
            end
            continue
        end

        # Sort by goal alignment + KB confidence
        scored = [(pir.goal_weight*max(0.0,dot(pv,goal_pv)/(norm(pv)*norm(goal_pv)+1e-10)) + conf, word, pv) for (conf,word,pv) in candidates]
        sort!(scored; by=x->-x[1])
        best = scored[1]
        push!(chain, (best[3], best[2]))
        current_pv = best[3]
    end
    return chain
end

function kb_query(kb, pv; top_k=5)
    try
        return kb_query_internal(kb, pv; top_k=top_k)
    catch e
        @warn "Path integral: KB query failed: $e"
        return Tuple{Float64,String,String}[]
    end
end

function kb_query_internal(kb, pv; top_k=5)
    results = Tuple{Float64,String,String}[]
    for (rel_type, bank) in kb.banks
        keys = bank["keys"]
        isempty(keys) && continue
        for i in 1:length(keys)
            k = keys[i]
            nrm = norm(k) * norm(pv)
            nrm < 1e-10 && continue
            sim = dot(k, pv) / nrm
            if sim > 0.3
                word = i <= length(bank["obj_words"]) ? bank["obj_words"][i] : ""
                push!(results, (Float64(sim), word, rel_type))
            end
        end
    end
    sort!(results; by=x->-x[1])
    return results[1:min(top_k, end)]
end

end # module