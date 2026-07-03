"""
SelfReviewModule -- lightweight internal feedback for generated text.

This is not a language model. It is a compact critic that checks a candidate
answer for language shape, repetition, phase coherence, prompt alignment, and
simple logic errors before the result is used as a learning signal.
"""
module SelfReviewModule

using LinearAlgebra, Statistics

using ..BayanLogicKernelModule: BayanLogicKernel, audit_bayan_logic,
    bayan_logic_summary

export SelfReviewEngine, GenerationReview, ReviewMemoryPrediction,
       ReviewTreatmentPrediction, review_generation!, review_summary,
       predict_review_repair, review_memory_summary,
       learn_review_treatment!, predict_review_treatment,
       treatment_memory_summary, self_review_state_dict,
       restore_self_review_state!, reset_self_review!

struct GenerationReview
    prompt::String
    output::String
    accepted::Bool
    score::Float64
    action::String
    language_score::Float64
    logic_score::Float64
    coherence_score::Float64
    repetition_score::Float64
    script_score::Float64
    alignment_score::Float64
    issues::Vector{String}
    issue_scores::Dict{String,Float64}
    primary_issue::String
    repair_target::String
    repair_hint::String
end

struct ReviewMemoryPrediction
    signature::String
    primary_issue::String
    repair_target::String
    confidence::Float64
    issue_scores::Dict{String,Float64}
    observations::Int
    reason::String
end

struct ReviewTreatmentPrediction
    signature::String
    repair_target::String
    expected_delta::Float64
    confidence::Float64
    observations::Int
    success_rate::Float64
    reason::String
end

mutable struct SelfReviewEngine
    enabled::Bool
    min_accept_score::Float64
    retry_threshold::Float64
    max_history::Int
    history::Vector{Dict{String,Any}}
    last_review::Union{GenerationReview,Nothing}
    logic_kernel::BayanLogicKernel
    diagnostic_memory::Dict{String,Dict{String,Any}}
    treatment_memory::Dict{String,Dict{String,Any}}
end

function SelfReviewEngine(; enabled::Bool=true,
                          min_accept_score::Float64=0.45,
                          retry_threshold::Float64=0.25,
                          max_history::Int=200)
    return SelfReviewEngine(enabled, min_accept_score, retry_threshold,
                            max_history, Dict{String,Any}[], nothing,
                            BayanLogicKernel(), Dict{String,Dict{String,Any}}(),
                            Dict{String,Dict{String,Any}}())
end

const STOPWORDS = Set([
    "ما", "ماذا", "هل", "كيف", "لماذا", "من", "في", "على", "عن", "الى", "إلى",
    "و", "او", "أو", "هو", "هي", "هذا", "هذه", "ذلك", "تلك", "the", "a", "an",
    "is", "are", "of", "to", "in", "on", "and", "or", "what", "how", "why",
])

const NEGATIVE_CUES = Set([
    "لا", "ليس", "ليست", "لن", "لم", "غير", "مستحيل", "not", "never", "no",
])

const AFFIRMATIVE_CUES = Set([
    "نعم", "صحيح", "ممكن", "يمكن", "بلى", "yes", "true", "can",
])

const QUESTION_CUES = Set([
    "ما", "ماذا", "هل", "كيف", "لماذا", "من", "متى", "اين", "أين",
    "what", "how", "why", "who", "when", "where",
])

const MEANING_CUES = Set([
    "معنى", "يعني", "تعني", "فسر", "اشرح", "meaning", "explain",
])

const CODE_CUES = Set(["code", "function", "python", "julia", "كود", "برمج", "دالة"])
const MATH_CUES = Set(["calculate", "math", "sum", "احسب", "ناتج", "+", "-", "*", "/"])

_empty_pv(::AbstractString) = Float64[]

function _clean_token(token::AbstractString)
    s = lowercase(strip(String(token)))
    return strip(s, [' ', '\t', '\n', '\r', '.', ',', '،', '؟', '?', '!', ':', ';', '"'])
end

function _tokenize(text::AbstractString)
    out = String[]
    for raw in split(String(text))
        tok = _clean_token(raw)
        isempty(tok) || push!(out, tok)
    end
    return out
end

const SEMANTIC_METHOD_CUES = Set(["by", "through", "via", "using", "with", "practice", "review", "step", "steps"])
const SEMANTIC_DEFINITION_CUES = Set(["is", "are", "means", "meaning", "defined", "definition"])
const SEMANTIC_REASON_CUES = Set(["because", "reason", "therefore", "since", "so"])
const SEMANTIC_TIME_CUES = Set(["when", "during", "after", "before"])
const SEMANTIC_PLACE_CUES = Set(["where", "in", "at", "inside"])

function _semantic_terms(guidance)
    guidance isa AbstractDict || return String[]
    raw = get(guidance, "target_terms", String[])
    return String[_clean_token(String(t)) for t in raw if !isempty(_clean_token(String(t)))]
end

function _movement_cues(movement::String)
    movement == "method" && return SEMANTIC_METHOD_CUES
    movement == "definition" && return SEMANTIC_DEFINITION_CUES
    (movement == "reason" || movement == "cause_result") && return SEMANTIC_REASON_CUES
    movement == "judgment" && return union(AFFIRMATIVE_CUES, NEGATIVE_CUES)
    movement == "time" && return SEMANTIC_TIME_CUES
    movement == "place" && return SEMANTIC_PLACE_CUES
    return Set{String}()
end

function _semantic_guidance_score(guidance, output_tokens::Vector{String}, issues::Vector{String})
    guidance isa AbstractDict || return 1.0
    Bool(get(guidance, "active", false)) || return 1.0
    confidence = Float64(get(guidance, "confidence", 0.0))
    confidence < 0.55 && return 1.0
    movement = String(get(guidance, "movement", "none"))
    movement in ("none", "continuation", "semantic_shift") && return 0.85

    output_set = Set(output_tokens)
    terms = Set(_semantic_terms(guidance))
    term_hit = !isempty(intersect(output_set, terms))
    cue_hit = any(t -> t in _movement_cues(movement), output_tokens)
    score = term_hit && cue_hit ? 1.0 :
            term_hit ? 0.82 :
            cue_hit ? 0.70 : 0.20
    score < 0.45 && push!(issues, "semantic_movement_mismatch")
    return score
end

function _script_family(tokens::Vector{String})
    text = join(tokens, " ")
    arabic = count(c -> '\u0600' <= c <= '\u06FF', text)
    latin = count(c -> ('a' <= c <= 'z') || ('A' <= c <= 'Z'), text)
    arabic > 0 && latin > 0 && return "mixed"
    arabic > 0 && return "arabic"
    latin > 0 && return "latin"
    return "other"
end

_length_bucket(n::Int) = n <= 3 ? "short" : (n <= 8 ? "medium" : "long")
_has_any(tokens::Vector{String}, cues::Set{String}) = any(t -> t in cues, tokens)

function _prompt_signature(tokens::Vector{String}, prompt::AbstractString="")
    clean = String[_clean_token(t) for t in tokens if !isempty(_clean_token(t))]
    isempty(clean) && (clean = _tokenize(prompt))
    features = String[
        string("len:", _length_bucket(length(clean))),
        string("script:", _script_family(clean)),
    ]
    if _has_any(clean, QUESTION_CUES)
        push!(features, "question")
        first_q = findfirst(t -> t in QUESTION_CUES, clean)
        first_q === nothing || push!(features, string("q:", clean[first_q]))
    end
    _has_any(clean, MEANING_CUES) && push!(features, "meaning")
    _has_any(clean, CODE_CUES) && push!(features, "code")
    (_has_any(clean, MATH_CUES) || occursin(r"\d+\s*[\+\-\*/]\s*\d+", String(prompt))) && push!(features, "math")
    (_has_any(clean, NEGATIVE_CUES) || occursin(" not ", string(" ", lowercase(String(prompt)), " "))) && push!(features, "negation")
    return join(sort(unique(features)), "|")
end

function _preview(text::String; limit::Int=240)
    length(text) <= limit && return text
    io = IOBuffer()
    for (i, c) in enumerate(text)
        i > limit && break
        print(io, c)
    end
    return String(take!(io)) * "..."
end

function _script_counts(text::String)
    arabic = count(c -> '\u0600' <= c <= '\u06FF', text)
    latin = count(c -> ('a' <= c <= 'z') || ('A' <= c <= 'Z'), text)
    digits = count(isdigit, text)
    return arabic, latin, digits
end

function _script_score(text::String)
    arabic, latin, _ = _script_counts(text)
    letters = arabic + latin
    letters == 0 && return 0.75
    minor_ratio = min(arabic, latin) / letters
    return clamp(1.0 - min(0.65, 1.8 * minor_ratio), 0.0, 1.0)
end

function _language_score(text::String, tokens::Vector{String}, issues::Vector{String})
    n = length(tokens)
    if n == 0
        push!(issues, "empty_output")
        return 0.0
    end

    usable = count(t -> any(c -> isletter(c) || isdigit(c), t), tokens) / n
    length_score = n == 1 ? 0.45 : clamp(n / 8.0, 0.0, 1.0)
    punctuation_score = any(p -> endswith(strip(text), p), [".", "؟", "?", "!", "؛", ":"]) ? 1.0 : 0.82
    score = 0.45 * length_score + 0.30 * usable +
            0.15 * punctuation_score + 0.10 * _script_score(text)

    if n >= 4 && !any(p -> occursin(p, text), [".", "،", ",", "؛", ";", ":", "؟", "?", "!", "\n"])
        push!(issues, "list_like_output")
        score *= 0.55
    end
    if n >= 4 && usable < 0.60
        push!(issues, "many_unreadable_tokens")
    end
    return clamp(score, 0.0, 1.0)
end

function _repetition_score(tokens::Vector{String}, issues::Vector{String})
    n = length(tokens)
    n == 0 && return 0.0
    counts = Dict{String,Int}()
    for t in tokens
        counts[t] = get(counts, t, 0) + 1
    end
    unique_ratio = length(keys(counts)) / n
    max_repeat = maximum(values(counts)) / n

    longest_run = 1
    run = 1
    for i in 2:n
        if tokens[i] == tokens[i - 1]
            run += 1
            longest_run = max(longest_run, run)
        else
            run = 1
        end
    end
    run_ratio = longest_run / n

    n >= 4 && max_repeat >= 0.45 && push!(issues, "dominant_repetition")
    n >= 5 && unique_ratio < 0.50 && push!(issues, "low_lexical_variety")
    return clamp(0.55 * unique_ratio + 0.25 * (1.0 - max_repeat) +
                 0.20 * (1.0 - run_ratio), 0.0, 1.0)
end

function _safe_pv(pv_fn::Function, token::String)
    try
        pv = pv_fn(token)
        pv isa AbstractVector || return nothing
        length(pv) == 0 && return nothing
        return Float64.(collect(pv))
    catch
        return nothing
    end
end

function _phase_similarity(a::AbstractVector, b::AbstractVector)
    d = min(length(a), length(b))
    d == 0 && return 0.5
    av = view(a, 1:d)
    bv = view(b, 1:d)
    na = norm(av)
    nb = norm(bv)
    (na < 1e-10 || nb < 1e-10) && return 0.5
    return clamp(max(0.0, dot(av, bv) / (na * nb)), 0.0, 1.0)
end

function _coherence_score(tokens::Vector{String}, pv_fn::Function,
                          issues::Vector{String})
    length(tokens) < 2 && return 0.55
    pvs = Vector{Vector{Float64}}()
    for t in tokens
        pv = _safe_pv(pv_fn, t)
        pv === nothing || push!(pvs, pv)
    end
    length(pvs) < 2 && return 0.55

    sims = Float64[]
    for i in 1:(length(pvs) - 1)
        push!(sims, _phase_similarity(pvs[i], pvs[i + 1]))
    end
    score = isempty(sims) ? 0.55 : mean(sims)
    length(tokens) >= 4 && score < 0.18 && push!(issues, "weak_phase_coherence")
    return clamp(score, 0.0, 1.0)
end

function _mean_vector(tokens::Vector{String}, pv_fn::Function)
    pvs = Vector{Vector{Float64}}()
    for t in tokens
        pv = _safe_pv(pv_fn, t)
        pv === nothing || push!(pvs, pv)
    end
    isempty(pvs) && return nothing
    d = minimum(length.(pvs))
    d == 0 && return nothing
    acc = zeros(Float64, d)
    for pv in pvs
        acc .+= view(pv, 1:d)
    end
    acc ./= length(pvs)
    norm(acc) < 1e-10 && return nothing
    return acc
end

function _content_tokens(tokens::Vector{String})
    return String[t for t in tokens if !(t in STOPWORDS) && length(t) >= 2]
end

function _alignment_score(prompt_tokens::Vector{String}, output_tokens::Vector{String},
                          pv_fn::Function, issues::Vector{String})
    isempty(prompt_tokens) && return 0.5
    isempty(output_tokens) && return 0.0

    pv_prompt = _mean_vector(prompt_tokens, pv_fn)
    pv_output = _mean_vector(output_tokens, pv_fn)
    vector_score = (pv_prompt === nothing || pv_output === nothing) ?
                   0.5 : _phase_similarity(pv_prompt, pv_output)

    prompt_content = _content_tokens(prompt_tokens)
    if isempty(prompt_content)
        overlap_score = 0.5
    else
        output_set = Set(output_tokens)
        overlap_count = count(t -> t in output_set, prompt_content)
        overlap_score = overlap_count / length(prompt_content)
        if length(output_tokens) >= 3 && overlap_count == 0
            push!(issues, "missing_prompt_anchor")
        elseif length(output_tokens) >= 4 && overlap_count < min(2, length(prompt_content))
            push!(issues, "weak_prompt_anchor")
        end
    end

    score = clamp(0.65 * vector_score + 0.35 * overlap_score, 0.0, 1.0)
    length(output_tokens) >= 3 && score < 0.15 && push!(issues, "weak_prompt_alignment")
    return score
end

function _logic_score(prompt::String, output::String,
                      tokens::Vector{String}, issues::Vector{String})
    score = 1.0

    for m in eachmatch(r"(-?\d+(?:\.\d+)?)\s*([\+\-\*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)", output)
        a = parse(Float64, m.captures[1])
        op = m.captures[2]
        b = parse(Float64, m.captures[3])
        c = parse(Float64, m.captures[4])
        expected = op == "+" ? a + b :
                   op == "-" ? a - b :
                   op == "*" ? a * b :
                   abs(b) < 1e-12 ? c : a / b
        if abs(expected - c) > 1e-6
            score -= 0.45
            push!(issues, "numeric_inconsistency")
            break
        end
    end

    neg = any(t -> t in NEGATIVE_CUES, tokens)
    affirm = any(t -> t in AFFIRMATIVE_CUES, tokens)
    if neg && affirm && length(tokens) <= 4
        score -= 0.25
        push!(issues, "short_affirmation_negation_conflict")
    end

    return clamp(score, 0.0, 1.0)
end

function _looks_like_code_output(text::AbstractString)
    s = lowercase(String(text))
    return occursin("def ", s) || occursin("function ", s) ||
           occursin("return ", s) || occursin("class ", s) ||
           occursin("import ", s) || occursin("println", s)
end

function _looks_like_verified_math_output(text::AbstractString)
    s = lowercase(String(text))
    return occursin(r"\d+\s*[\+\-\*/]\s*\d+\s*(=>|=)\s*-?\d+", s) ||
           occursin("verified: true", s)
end

function _remove_alignment_anchor_issues!(issues::Vector{String})
    filter!(issue -> !(issue in ("missing_prompt_anchor", "weak_prompt_anchor",
                                 "weak_prompt_alignment")), issues)
    return issues
end

function _issue_scores(language::Float64, logic::Float64, coherence::Float64,
                       repetition::Float64, script::Float64, alignment::Float64,
                       issues::Vector{String})
    scores = Dict{String,Float64}(
        "language" => 1.0 - language,
        "logic" => 1.0 - logic,
        "coherence" => 1.0 - coherence,
        "repetition" => 1.0 - repetition,
        "script" => 1.0 - script,
        "alignment" => 1.0 - alignment,
    )
    for issue in issues
        if issue in ("empty_output", "many_unreadable_tokens")
            scores["language"] = max(scores["language"], 1.0)
        elseif issue == "list_like_output"
            scores["language"] = max(scores["language"], 0.65)
        elseif issue in ("numeric_inconsistency", "short_affirmation_negation_conflict",
                         "bayan_logic_contradiction")
            scores["logic"] = max(scores["logic"], 1.0)
        elseif issue == "dominant_repetition" || issue == "low_lexical_variety"
            scores["repetition"] = max(scores["repetition"], 1.0)
        elseif issue == "weak_phase_coherence"
            scores["coherence"] = max(scores["coherence"], 0.85)
        elseif issue in ("weak_prompt_alignment", "missing_prompt_anchor", "weak_prompt_anchor")
            scores["alignment"] = max(scores["alignment"], 0.85)
        elseif issue == "semantic_movement_mismatch"
            scores["semantic"] = max(get(scores, "semantic", 0.0), 1.0)
        end
    end
    return Dict(k => clamp(v, 0.0, 1.0) for (k, v) in scores)
end

function _primary_issue(issue_scores::Dict{String,Float64})
    isempty(issue_scores) && return "none"
    ranked = sort(collect(issue_scores); by=x -> -x[2])
    ranked[1][2] < 0.22 && return "none"
    return ranked[1][1]
end

function _repair_target(primary_issue::String, issues::Vector{String})
    "empty_output" in issues && return "fallback"
    primary_issue == "logic" && return "logic"
    primary_issue == "repetition" && return "diversity"
    primary_issue == "alignment" && return "prompt_alignment"
    primary_issue == "coherence" && return "coherence"
    primary_issue == "language" && return "syntax"
    primary_issue == "script" && return "language"
    primary_issue == "semantic" && return "none"
    return "none"
end

function _repair_hint(target::String)
    target == "logic" && return "raise causal/AQL checks and avoid unsupported contradictions"
    target == "diversity" && return "increase repulsion and diversity, then regenerate"
    target == "prompt_alignment" && return "raise prompt alignment and planning weights"
    target == "coherence" && return "raise resonant chain and density coherence"
    target == "syntax" && return "raise syntax and surface affinity"
    target == "language" && return "prefer one script and cleaner language shape"
    target == "fallback" && return "use fallback generation or a simpler response path"
    target == "semantic_guidance" && return "follow the semantic movement expected by al_hisban"
    return "no repair needed"
end

function _count_map_increment!(d::Dict{String,Any}, key::String)
    counts = get!(d, "counts") do
        Dict{String,Int}()
    end
    counts = Dict{String,Int}(string(k) => Int(v) for (k, v) in counts)
    counts[key] = get(counts, key, 0) + 1
    d["counts"] = counts
    return counts
end

function _learn_review_memory!(engine::SelfReviewEngine,
                               prompt_tokens::Vector{String},
                               review::GenerationReview)
    signature = _prompt_signature(prompt_tokens, review.prompt)
    entry = get!(engine.diagnostic_memory, signature) do
        Dict{String,Any}(
            "count" => 0,
            "avg_score" => 0.0,
            "issue_scores" => Dict{String,Float64}(),
            "repair_counts" => Dict{String,Int}(),
            "primary_counts" => Dict{String,Int}(),
            "last_repair_hint" => "",
        )
    end

    count = Int(get(entry, "count", 0)) + 1
    old_avg = Float64(get(entry, "avg_score", 0.0))
    entry["count"] = count
    entry["avg_score"] = old_avg + (review.score - old_avg) / count

    memory_scores = Dict{String,Float64}(
        string(k) => Float64(v) for (k, v) in get(entry, "issue_scores", Dict())
    )
    for (issue, value) in review.issue_scores
        old = get(memory_scores, issue, 0.0)
        memory_scores[issue] = old + (Float64(value) - old) / count
    end
    entry["issue_scores"] = memory_scores

    repair_counts = Dict{String,Int}(
        string(k) => Int(v) for (k, v) in get(entry, "repair_counts", Dict())
    )
    repair_counts[review.repair_target] = get(repair_counts, review.repair_target, 0) + 1
    entry["repair_counts"] = repair_counts

    primary_counts = Dict{String,Int}(
        string(k) => Int(v) for (k, v) in get(entry, "primary_counts", Dict())
    )
    primary_counts[review.primary_issue] = get(primary_counts, review.primary_issue, 0) + 1
    entry["primary_counts"] = primary_counts
    entry["last_repair_hint"] = review.repair_hint
    return signature
end

function _top_count(counts::Dict{String,Int})
    isempty(counts) && return "none", 0
    ranked = sort(collect(counts); by=x -> (-last(x), first(x)))
    return first(ranked[1]), last(ranked[1])
end

function predict_review_repair(engine::SelfReviewEngine,
                               prompt_tokens::AbstractVector{<:AbstractString};
                               prompt::AbstractString="")
    clean = String[_clean_token(t) for t in prompt_tokens if !isempty(_clean_token(t))]
    signature = _prompt_signature(clean, prompt)
    entry = get(engine.diagnostic_memory, signature, nothing)
    entry === nothing && return ReviewMemoryPrediction(
        signature, "none", "none", 0.0, Dict{String,Float64}(), 0, "no_memory")

    issue_scores = Dict{String,Float64}(
        string(k) => Float64(v) for (k, v) in get(entry, "issue_scores", Dict())
    )
    observations = Int(get(entry, "count", 0))
    primary = _primary_issue(issue_scores)
    learned_target = _repair_target(primary, String[])
    repair_counts = Dict{String,Int}(
        string(k) => Int(v) for (k, v) in get(entry, "repair_counts", Dict())
    )
    counted_target, counted_n = _top_count(filter(p -> first(p) != "none", repair_counts))
    target = learned_target == "none" ? counted_target : learned_target
    severity = isempty(issue_scores) ? 0.0 : maximum(values(issue_scores))
    support = observations <= 0 ? 0.0 : observations / (observations + 3.0)
    count_bias = observations <= 0 ? 0.0 : counted_n / observations
    confidence = clamp((0.75 * severity + 0.25 * count_bias) * support, 0.0, 0.95)
    confidence < 0.18 && (target = "none")
    target == "none" && (primary = "none")
    return ReviewMemoryPrediction(signature, primary, target, confidence,
                                  issue_scores, observations,
                                  target == "none" ? "memory_below_threshold" : "learned_review_memory")
end

function review_memory_summary(engine::SelfReviewEngine)
    items = Vector{Dict{String,Any}}()
    for (signature, entry) in engine.diagnostic_memory
        issue_scores = Dict{String,Float64}(
            string(k) => Float64(v) for (k, v) in get(entry, "issue_scores", Dict())
        )
        primary = _primary_issue(issue_scores)
        push!(items, Dict{String,Any}(
            "signature" => signature,
            "count" => Int(get(entry, "count", 0)),
            "avg_score" => round(Float64(get(entry, "avg_score", 0.0)), digits=4),
            "primary_issue" => primary,
            "repair_target" => _repair_target(primary, String[]),
        ))
    end
    sort!(items; by=item -> -Int(get(item, "count", 0)))
    return Dict{String,Any}(
        "signature_count" => length(engine.diagnostic_memory),
        "top" => items[1:min(5, length(items))],
    )
end

function learn_review_treatment!(engine::SelfReviewEngine,
                                 prompt_tokens::AbstractVector{<:AbstractString};
                                 prompt::AbstractString="",
                                 repair_target::String,
                                 before_score::Float64,
                                 after_score::Float64,
                                 chosen::Bool=false,
                                 source::String="directed")
    target = isempty(strip(repair_target)) ? "none" : repair_target
    target == "none" && return false
    clean = String[_clean_token(t) for t in prompt_tokens if !isempty(_clean_token(t))]
    signature = _prompt_signature(clean, prompt)
    entry = get!(engine.treatment_memory, signature) do
        Dict{String,Any}("treatments" => Dict{String,Any}())
    end
    treatments = Dict{String,Any}(string(k) => v for (k, v) in get(entry, "treatments", Dict()))
    rec = get!(treatments, target) do
        Dict{String,Any}(
            "attempts" => 0,
            "successes" => 0,
            "chosen_count" => 0,
            "avg_delta" => 0.0,
            "avg_after_score" => 0.0,
            "last_source" => "",
        )
    end

    attempts = Int(get(rec, "attempts", 0)) + 1
    delta = after_score - before_score
    success = delta > 0.03 || (chosen && after_score >= before_score)
    old_delta = Float64(get(rec, "avg_delta", 0.0))
    old_after = Float64(get(rec, "avg_after_score", 0.0))

    rec["attempts"] = attempts
    rec["successes"] = Int(get(rec, "successes", 0)) + (success ? 1 : 0)
    rec["chosen_count"] = Int(get(rec, "chosen_count", 0)) + (chosen ? 1 : 0)
    rec["avg_delta"] = old_delta + (delta - old_delta) / attempts
    rec["avg_after_score"] = old_after + (after_score - old_after) / attempts
    rec["last_source"] = source
    treatments[target] = rec
    entry["treatments"] = treatments
    engine.treatment_memory[signature] = entry
    return true
end

function _treatment_utility(record::Dict{String,Any})
    attempts = Int(get(record, "attempts", 0))
    attempts <= 0 && return -Inf
    successes = Int(get(record, "successes", 0))
    chosen = Int(get(record, "chosen_count", 0))
    avg_delta = Float64(get(record, "avg_delta", 0.0))
    success_rate = successes / attempts
    chosen_rate = chosen / attempts
    support = attempts / (attempts + 3.0)
    return support * (avg_delta + 0.18 * success_rate + 0.06 * chosen_rate)
end

function predict_review_treatment(engine::SelfReviewEngine,
                                  prompt_tokens::AbstractVector{<:AbstractString};
                                  prompt::AbstractString="",
                                  default_target::String="none")
    clean = String[_clean_token(t) for t in prompt_tokens if !isempty(_clean_token(t))]
    signature = _prompt_signature(clean, prompt)
    entry = get(engine.treatment_memory, signature, nothing)
    if entry === nothing
        return ReviewTreatmentPrediction(signature, default_target, 0.0, 0.0, 0, 0.0,
                                         "no_treatment_memory")
    end
    treatments = Dict{String,Any}(string(k) => v for (k, v) in get(entry, "treatments", Dict()))
    isempty(treatments) && return ReviewTreatmentPrediction(
        signature, default_target, 0.0, 0.0, 0, 0.0, "empty_treatment_memory")

    ranked = sort(collect(treatments); by=p -> -_treatment_utility(last(p)))
    target = first(ranked[1])
    record = last(ranked[1])
    attempts = Int(get(record, "attempts", 0))
    successes = Int(get(record, "successes", 0))
    avg_delta = Float64(get(record, "avg_delta", 0.0))
    success_rate = attempts <= 0 ? 0.0 : successes / attempts
    support = attempts / (attempts + 3.0)
    confidence = clamp(support * (max(avg_delta, 0.0) * 2.0 + 0.60 * success_rate), 0.0, 0.95)
    if confidence < 0.12 || avg_delta <= -0.03
        return ReviewTreatmentPrediction(signature, default_target, avg_delta,
                                         confidence, attempts, success_rate,
                                         "treatment_below_threshold")
    end
    return ReviewTreatmentPrediction(signature, target, avg_delta, confidence,
                                     attempts, success_rate, "learned_treatment_memory")
end

function treatment_memory_summary(engine::SelfReviewEngine)
    items = Vector{Dict{String,Any}}()
    for (signature, entry) in engine.treatment_memory
        treatments = Dict{String,Any}(string(k) => v for (k, v) in get(entry, "treatments", Dict()))
        isempty(treatments) && continue
        ranked = sort(collect(treatments); by=p -> -_treatment_utility(last(p)))
        target = first(ranked[1])
        record = last(ranked[1])
        attempts = Int(get(record, "attempts", 0))
        successes = Int(get(record, "successes", 0))
        push!(items, Dict{String,Any}(
            "signature" => signature,
            "best_repair_target" => target,
            "attempts" => attempts,
            "success_rate" => attempts <= 0 ? 0.0 : round(successes / attempts, digits=4),
            "avg_delta" => round(Float64(get(record, "avg_delta", 0.0)), digits=4),
        ))
    end
    sort!(items; by=item -> -Int(get(item, "attempts", 0)))
    return Dict{String,Any}(
        "signature_count" => length(engine.treatment_memory),
        "top" => items[1:min(5, length(items))],
    )
end

function _review_action(engine::SelfReviewEngine, text::String, score::Float64)
    isempty(strip(text)) && return false, "reject_empty"
    score < engine.retry_threshold && return false, "retry"
    score < engine.min_accept_score && return false, "revise"
    return true, "accept"
end

function _review_dict(review::GenerationReview)
    return Dict{String,Any}(
        "prompt" => _preview(review.prompt),
        "output" => _preview(review.output),
        "accepted" => review.accepted,
        "score" => review.score,
        "action" => review.action,
        "language_score" => review.language_score,
        "logic_score" => review.logic_score,
        "coherence_score" => review.coherence_score,
        "repetition_score" => review.repetition_score,
        "script_score" => review.script_score,
        "alignment_score" => review.alignment_score,
        "issues" => copy(review.issues),
        "issue_scores" => copy(review.issue_scores),
        "primary_issue" => review.primary_issue,
        "repair_target" => review.repair_target,
        "repair_hint" => review.repair_hint,
    )
end

function review_generation!(engine::SelfReviewEngine,
                            prompt::AbstractString,
                            output::AbstractString;
                            prompt_tokens::AbstractVector{<:AbstractString}=String[],
                            pv_fn::Function=_empty_pv,
                            semantic_guidance=nothing)
    p = String(prompt)
    text = String(output)
    if !engine.enabled
        review = GenerationReview(p, text, true, 0.5, "disabled",
                                  0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                  String[], Dict{String,Float64}(),
                                  "none", "none", "disabled")
        engine.last_review = review
        return review
    end

    p_tokens = isempty(prompt_tokens) ? _tokenize(p) :
               String[_clean_token(t) for t in prompt_tokens if !isempty(_clean_token(t))]
    o_tokens = _tokenize(text)
    issues = String[]

    language = _language_score(text, o_tokens, issues)
    logic = _logic_score(p, text, o_tokens, issues)
    bayan_audit = audit_bayan_logic(engine.logic_kernel, p, text)
    append!(issues, bayan_audit.issues)
    logic = min(logic, bayan_audit.score)
    coherence = _coherence_score(o_tokens, pv_fn, issues)
    repetition = _repetition_score(o_tokens, issues)
    script = _script_score(text)
    alignment = _alignment_score(p_tokens, o_tokens, pv_fn, issues)
    semantic = _semantic_guidance_score(semantic_guidance, o_tokens, issues)
    prompt_is_code = _has_any(p_tokens, CODE_CUES)
    prompt_is_math = _has_any(p_tokens, MATH_CUES) || occursin(r"\d+\s*[\+\-\*/]\s*\d+", p)

    if prompt_is_code && _looks_like_code_output(text)
        _remove_alignment_anchor_issues!(issues)
        alignment = max(alignment, 0.75)
        language = max(language, 0.80)
        coherence = max(coherence, 0.70)
    elseif prompt_is_math && _looks_like_verified_math_output(text)
        _remove_alignment_anchor_issues!(issues)
        alignment = max(alignment, 0.80)
        logic = max(logic, 0.95)
    end

    score = clamp(0.22 * language + 0.18 * logic + 0.22 * coherence +
                  0.18 * repetition + 0.10 * script + 0.10 * alignment,
                  0.0, 1.0)

    if isempty(strip(text))
        score = 0.0
    elseif "numeric_inconsistency" in issues || "bayan_logic_contradiction" in issues
        score = min(score, max(0.0, engine.min_accept_score - 0.01))
    elseif "dominant_repetition" in issues && repetition < 0.25
        score = min(score, max(0.0, engine.min_accept_score - 0.01))
    elseif "list_like_output" in issues || "missing_prompt_anchor" in issues
        score = min(score, max(0.0, engine.min_accept_score - 0.01))

    end
    accepted, action = _review_action(engine, text, score)
    issue_scores = _issue_scores(language, logic, coherence, repetition,
                                 script, alignment, issues)
    primary = _primary_issue(issue_scores)
    target = _repair_target(primary, issues)
    hint = _repair_hint(target)

    review = GenerationReview(p, text, accepted, score, action, language, logic,
                              coherence, repetition, script, alignment, issues,
                              issue_scores, primary, target, hint)
    engine.last_review = review
    push!(engine.history, _review_dict(review))
    while length(engine.history) > engine.max_history
        popfirst!(engine.history)
    end
    _learn_review_memory!(engine, p_tokens, review)
    return review
end

function review_summary(engine::SelfReviewEngine)
    scores = Float64[Float64(get(item, "score", 0.0)) for item in engine.history]
    accepted = count(item -> Bool(get(item, "accepted", false)), engine.history)
    last = engine.last_review
    return Dict{String,Any}(
        "enabled" => engine.enabled,
        "history_count" => length(engine.history),
        "avg_score" => isempty(scores) ? 0.0 : round(mean(scores), digits=4),
        "accepted_count" => accepted,
        "rejected_count" => length(engine.history) - accepted,
        "last_review" => last === nothing ? nothing : Dict(
            "accepted" => last.accepted,
            "score" => round(last.score, digits=4),
            "action" => last.action,
            "issues" => copy(last.issues),
            "primary_issue" => last.primary_issue,
            "repair_target" => last.repair_target,
            "repair_hint" => last.repair_hint,
            "issue_scores" => copy(last.issue_scores),
        ),
        "bayan_logic" => bayan_logic_summary(engine.logic_kernel),
        "review_memory" => review_memory_summary(engine),
        "treatment_memory" => treatment_memory_summary(engine),
    )
end

function self_review_state_dict(engine::SelfReviewEngine)
    return Dict{String,Any}(
        "version" => 3,
        "enabled" => engine.enabled,
        "min_accept_score" => engine.min_accept_score,
        "retry_threshold" => engine.retry_threshold,
        "max_history" => engine.max_history,
        "history" => engine.history,
        "diagnostic_memory" => engine.diagnostic_memory,
        "treatment_memory" => engine.treatment_memory,
    )
end

function restore_self_review_state!(engine::SelfReviewEngine, data)
    data === nothing && return false
    data isa AbstractDict || return false
    engine.enabled = Bool(get(data, "enabled", engine.enabled))
    engine.min_accept_score = Float64(get(data, "min_accept_score", engine.min_accept_score))
    engine.retry_threshold = Float64(get(data, "retry_threshold", engine.retry_threshold))
    engine.max_history = Int(get(data, "max_history", engine.max_history))
    empty!(engine.history)
    for item in get(data, "history", Any[])
        item isa AbstractDict || continue
        push!(engine.history, Dict{String,Any}(string(k) => v for (k, v) in item))
    end
    while length(engine.history) > engine.max_history
        popfirst!(engine.history)
    end
    empty!(engine.diagnostic_memory)
    for (signature, item) in get(data, "diagnostic_memory", Dict())
        item isa AbstractDict || continue
        entry = Dict{String,Any}()
        entry["count"] = Int(get(item, "count", 0))
        entry["avg_score"] = Float64(get(item, "avg_score", 0.0))
        entry["last_repair_hint"] = string(get(item, "last_repair_hint", ""))
        entry["issue_scores"] = Dict{String,Float64}(
            string(k) => Float64(v) for (k, v) in get(item, "issue_scores", Dict())
        )
        entry["repair_counts"] = Dict{String,Int}(
            string(k) => Int(v) for (k, v) in get(item, "repair_counts", Dict())
        )
        entry["primary_counts"] = Dict{String,Int}(
            string(k) => Int(v) for (k, v) in get(item, "primary_counts", Dict())
        )
        engine.diagnostic_memory[string(signature)] = entry
    end
    empty!(engine.treatment_memory)
    for (signature, item) in get(data, "treatment_memory", Dict())
        item isa AbstractDict || continue
        entry = Dict{String,Any}()
        treatments = Dict{String,Any}()
        for (target, rec) in get(item, "treatments", Dict())
            rec isa AbstractDict || continue
            treatments[string(target)] = Dict{String,Any}(
                "attempts" => Int(get(rec, "attempts", 0)),
                "successes" => Int(get(rec, "successes", 0)),
                "chosen_count" => Int(get(rec, "chosen_count", 0)),
                "avg_delta" => Float64(get(rec, "avg_delta", 0.0)),
                "avg_after_score" => Float64(get(rec, "avg_after_score", 0.0)),
                "last_source" => string(get(rec, "last_source", "")),
            )
        end
        entry["treatments"] = treatments
        engine.treatment_memory[string(signature)] = entry
    end
    engine.last_review = nothing
    return true
end

function reset_self_review!(engine::SelfReviewEngine)
    empty!(engine.history)
    empty!(engine.logic_kernel.history)
    empty!(engine.diagnostic_memory)
    empty!(engine.treatment_memory)
    engine.last_review = nothing
    return engine
end

end # module SelfReviewModule
