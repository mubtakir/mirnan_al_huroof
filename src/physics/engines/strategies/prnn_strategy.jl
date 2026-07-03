# prnn_strategy.jl
# Part of Phase 9 — PRNN 2.0 Stuart-Landau Generation Strategy

struct PRNNStrategy <: GenerationStrategy end

function try_generate(::PRNNStrategy, gen::MirnanGenerator, prompt::String, prompt_tokens::Vector{String}, mode::String, cereb_obs, cereb_policy, response_plan, active_paras, max_words::Int)
    effective_mode = mode
    if mode == "auto"
        effective_mode = cereb_policy.mode
    end

    if effective_mode != "prnn"
        return nothing
    end

    # PRNN dynamic generation requires K_sem
    K_sem = gen.K_sem
    if K_sem === nothing
        @warn "PRNNStrategy: K_sem is null. Falling back."
        return nothing
    end

    vocab = gen.vocab
    id2word = gen.id2word
    seed_words = prompt_tokens
    
    # 1. Identify active vocabulary neighbors of prompt tokens in K_sem
    seed_ids = Int32[get(vocab, w, Int32(-1)) for w in seed_words]
    filter!(x -> x > 0, seed_ids)
    
    active_ids = Set{Int32}(seed_ids)
    
    for cid in seed_ids
        if cid > 0 && cid <= size(K_sem, 1)
            row = K_sem[cid, :]
            for (idx, val) in zip(row.nzind, row.nzval)
                if val > 1e-4
                    push!(active_ids, idx)
                end
            end
        end
    end
    
    # Filter active vocabulary to valid Arabic/English words
    active_vocab = String[]
    for id in active_ids
        w = get(id2word, Int(id), nothing)
        if w !== nothing && (any(ch -> '\u0621' <= ch <= '\u064A', w) || any(ch -> 'a' <= lowercase(ch) <= 'z', w))
            push!(active_vocab, w)
        end
    end
    
    if isempty(active_vocab)
        return nothing
    end
    
    # 2. Build dense phase vectors (Geometric Phase Expansion)
    N = Constants.TOTAL_DIM # 10000
    base_vectors = Dict{String, Vector{ComplexF64}}()
    for w in active_vocab
        base_vectors[w] = PRNNLearner.build_dense_phase_vector(w, N)
    end
    
    # 3. Build transitions dynamically from K_sem weights
    transitions = Tuple{Vector{ComplexF64}, Vector{ComplexF64}, Float64}[]
    beta = get(gen.scoring_weights, "prnn_coupling_beta", 3.0)
    
    for w_curr in active_vocab
        id_curr = vocab[w_curr]
        if id_curr > 0 && id_curr <= size(K_sem, 1)
            row = K_sem[id_curr, :]
            for (idx, val) in zip(row.nzind, row.nzval)
                if val > 1e-4
                    w_next = get(id2word, Int(idx), nothing)
                    if w_next !== nothing && w_next in active_vocab
                        v_curr = base_vectors[w_curr]
                        v_next = base_vectors[w_next]
                        v_next_adapted = PRNNCore.bind_phase(v_next, v_curr)
                        push!(transitions, (v_curr, v_next_adapted, Float64(val)))
                    end
                end
            end
        end
    end
    
    if isempty(transitions)
        return nothing
    end
    
    # 4. Simulate Stuart-Landau dynamics to update phase/amplitude state
    state = PRNNCore.PRNNState(N)
    model = PRNNCore.LowRankPRNN(transitions, beta)
    
    current_word = prompt_tokens[end]
    if haskey(base_vectors, current_word)
        if length(prompt_tokens) >= 2 && haskey(base_vectors, prompt_tokens[end-1])
            state.z .= PRNNCore.bind_phase(base_vectors[current_word], base_vectors[prompt_tokens[end-1]])
        else
            state.z .= base_vectors[current_word]
        end
    else
        found = false
        for tok in reverse(prompt_tokens)
            if haskey(base_vectors, tok)
                state.z .= base_vectors[tok]
                current_word = tok
                found = true
                break
            end
        end
        if !found
            state.z .= randn(ComplexF64, N) ./ sqrt(N)
        end
    end
    
    output = String[]
    used = Set{String}(prompt_tokens)
    
    mu = get(gen.scoring_weights, "prnn_mu", 1.0)
    g_inh = get(gen.scoring_weights, "prnn_g_inh", 0.5)
    gamma = get(gen.scoring_weights, "prnn_gamma", 2.0)
    tau_a = get(gen.scoring_weights, "prnn_tau_a", 1.5)
    steps = Int(get(gen.scoring_weights, "prnn_steps", 40.0))
    dt = get(gen.scoring_weights, "prnn_dt", 0.02)
    overlap_threshold = get(gen.scoring_weights, "prnn_overlap_threshold", 0.10)
    
    noise_amp = get(gen.scoring_weights, "prnn_noise_amp", 0.20)
    anneal_rate = get(gen.scoring_weights, "prnn_anneal_rate", 0.05)
    
    for _ in 1:max_words
        PRNNCore.simulate_stuart_landau!(state, model;
            mu=mu, g_inh=g_inh, gamma=gamma, tau_a=tau_a, steps=steps, dt=dt,
            noise_amp=noise_amp, anneal_rate=anneal_rate)
            
        if !haskey(base_vectors, current_word)
            break
        end
        
        z_unbind = PRNNCore.unbind_phase(state.z, base_vectors[current_word])
        
        best_w, best_ov = "", -Inf
        for w in active_vocab
            ov = real(dot(z_unbind, base_vectors[w])) / N
            if ov > best_ov
                best_ov = ov
                best_w = w
            end
        end
        
        if isempty(best_w) || best_w in used || best_ov < overlap_threshold
            break
        end
        
        push!(output, best_w)
        push!(used, best_w)
        
        state.z .= PRNNCore.bind_phase(base_vectors[best_w], base_vectors[current_word])
        current_word = best_w
    end
    
    if isempty(output)
        return nothing
    end
    
    generated_text = join(output, " ")
    
    return _finish_generation!(gen, prompt, prompt_tokens, generated_text,
                               cereb_obs, cereb_policy;
                               sanitize_output=true,
                               apply_templates=true)
end
