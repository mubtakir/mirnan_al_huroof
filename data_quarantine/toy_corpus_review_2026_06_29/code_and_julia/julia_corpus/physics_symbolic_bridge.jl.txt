"""SymbolicBridge — جسر رمزي-طوري (نفي، عطف، جر، استفهام، شرط)."""
module SymbolicBridgeModule
using LinearAlgebra, Statistics
using ..WordPhysics: compute_extended_phase_vector, phase_similarity

export SymbolicBridge, evaluate_rules

mutable struct SymbolicBridge
    vocab::Any; K::Any; rules::Vector{Function}
    pv_cache::Dict{String,Vector{Float64}}
end
SymbolicBridge(vocab=nothing, K=nothing) = SymbolicBridge(vocab, K, Function[], Dict())

import YAML

const _DEFAULT_YAML_PATH = joinpath(@__DIR__, "..", "..", "data", "rules", "symbolic_rules.yaml")

function load_yaml_rules!(sb::SymbolicBridge, yaml_path::String)
    if !isfile(yaml_path)
        @warn "YAML rules file not found: $yaml_path"
        return sb
    end
    try
        data = YAML.load_file(yaml_path)
        if data === nothing || !haskey(data, "rules")
            return sb
        end
        for rule_def in data["rules"]
            name = get(rule_def, "name", "unnamed_rule")
            lookback = get(rule_def, "lookback", 1)
            
            raw_markers = get(rule_def, "markers", String[])
            markers = Set{String}()
            if raw_markers isa Vector
                for m in raw_markers
                    push!(markers, String(m))
                end
            end
            
            raw_targets = get(rule_def, "targets", String[])
            targets = Set{String}()
            if raw_targets isa Vector
                for t in raw_targets
                    push!(targets, String(t))
                end
            end
            
            score = Float64(get(rule_def, "score", 0.0))
            
            raw_anti_targets = get(rule_def, "anti_targets", String[])
            anti_targets = Set{String}()
            if raw_anti_targets isa Vector
                for at in raw_anti_targets
                    push!(anti_targets, String(at))
                end
            end
            
            anti_score = Float64(get(rule_def, "anti_score", 0.0))
            
            rule_fn = (word, prev_words, context_ids, vocab, K, pv_cache) -> begin
                if isempty(prev_words)
                    return 0.0
                end
                
                lb = min(lookback, length(prev_words))
                has_marker = false
                for i in (length(prev_words) - lb + 1):length(prev_words)
                    if prev_words[i] in markers
                        has_marker = true
                        break
                    end
                end
                
                if has_marker
                    if !isempty(targets)
                        if word in targets
                            return score
                        end
                    else
                        return score
                    end
                    
                    if !isempty(anti_targets) && word in anti_targets
                        return anti_score
                    end
                end
                return 0.0
            end
            
            push!(sb.rules, rule_fn)
        end
    catch e
        @warn "Failed to load YAML rules from $yaml_path: $e"
    end
    return sb
end

function load_builtin_rules!(sb::SymbolicBridge)
    # 1. Load dynamic rules from YAML
    load_yaml_rules!(sb, _DEFAULT_YAML_PATH)
    
    # 2. Add complex programmatic rules
    # Filament resonance rule
    push!(sb.rules, (word, prev_words, ctx_ids, vocab, K, pv_cache) -> begin
        isempty(prev_words) && return 0.0
        try
            w_pv = get!(pv_cache, word) do; compute_extended_phase_vector(word); end
            cat_scores = Float64[]
            for pw in prev_words[max(1, end-2):end]
                pw_pv = get!(pv_cache, pw) do; compute_extended_phase_vector(pw); end
                push!(cat_scores, mean(cos.(w_pv .- pw_pv)))
            end
            avg = mean(cat_scores)
            avg > 0.85 && return 0.12
            avg > 0.75 && return 0.05
        catch e
            @debug "SymbolicBridge: structural pattern check failed: $e"
        end
        return 0.0
    end)
    
    # Complex Preposition rule (using K)
    push!(sb.rules, (word, prev_words, ctx_ids, vocab, K, pv_cache) -> begin
        if isempty(prev_words) || vocab === nothing || K === nothing
            return 0.0
        end
        prepositions = Set(["في", "من", "على", "إلى", "عن", "مع", "بين", "تحت", "فوق", "ب", "ل", "ك", "خلال", "عند", "لدى", "حول", "ضد", "نحو", "تجاه"])
        pw = prev_words[end]
        if pw in prepositions
            if haskey(vocab, word) && haskey(vocab, pw)
                i = vocab[pw]
                j = vocab[word]
                sz = size(K)
                if i <= sz[1] && j <= sz[2]
                    k_val = Float64(K[i, j])
                    if k_val > 0
                        return min(k_val * 0.01, 0.15)
                    end
                end
            end
        end
        return 0.0
    end)
    
    # Complex Gravitational Pull rule
    push!(sb.rules, (word, prev_words, ctx_ids, vocab, K, pv_cache) -> begin
        if isempty(prev_words)
            return 0.0
        end
        try
            pw = prev_words[end]
            prev_pv = get!(pv_cache, pw) do
                compute_extended_phase_vector(pw)
            end
            n = length(prev_pv)
            energy = Float64(prev_pv[n - 3])
            op_var = Float64(prev_pv[n - 1])
            mass = energy * (1.0 + op_var)
            if mass > 0.6 && length(pw) >= 4
                verb_like = count(c -> c in ('ي', 'ت', 'س', 'ن', 'أ', 'ف'), word)
                if verb_like >= 2 && length(word) >= 4
                    return min(mass * 0.15, 0.12)
                end
                if endswith(word, "ة") && length(word) >= 4
                    return -0.06
                end
            end
        catch e
            @debug "SymbolicBridge: pattern match failed: $e"
        end
        return 0.0
    end)
    
    # Complex Number Sequence rule
    push!(sb.rules, (word, prev_words, ctx_ids, vocab, K, pv_cache) -> begin
        is_number = occursin(r"^[\+\-]?\d+\.?\d*$", word)
        if is_number
            return 0.15
        end
        units = Set(["سنتيمتر", "متر", "كيلومتر", "غرام", "كيلو", "ليتر", "ثانية", "دقيقة", "ساعة"])
        if word in units && !isempty(prev_words)
            lb = min(2, length(prev_words))
            for i in (length(prev_words) - lb + 1):length(prev_words)
                if occursin(r"^[\+\-]?\d+\.?\d*$", prev_words[i])
                    return 0.2
                end
            end
        end
        return 0.0
    end)
    
    # Complex Arithmetic Operator rule
    push!(sb.rules, (word, prev_words, ctx_ids, vocab, K, pv_cache) -> begin
        arith_ops = Set(["+", "-", "*", "/", "**", "//", "%", "^"])
        if !isempty(prev_words)
            if prev_words[end] in arith_ops
                return 0.1
            end
        end
        for c in word
            if string(c) in arith_ops
                return 0.08
            end
        end
        return 0.0
    end)
    
    return sb
end

function evaluate_rules(sb::SymbolicBridge, word::String, prev_words::Vector{String}, context_ids=nothing)
    score = 0.0
    for rule in sb.rules
        try
            score += rule(word, prev_words, context_ids, sb.vocab, sb.K, sb.pv_cache)
        catch e
            @debug "SymbolicBridge: rule evaluation failed: $e"
        end
    end
    return min(score, 0.6)
end
end

