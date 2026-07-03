"""
MirnanCerebellum -- a small symbolic/meta-control layer.

It is not a language model. It observes a prompt, chooses a generation policy,
nudges engine weights for that single generation, then records a lightweight
reward signal. Think of it as a cerebellum: routing, balance, and correction.
"""
module MirnanCerebellumModule

using Statistics

export MirnanCerebellum, CerebellumObservation, CerebellumPolicy, PIDController,
       observe_prompt, choose_policy!, apply_cerebellum_policy!,
       learn_from_outcome!, policy_summary,
       cerebellum_state_dict, restore_cerebellum_state!, reset_cerebellum!,
       correct!, correct_weight!, record_error!, update_integral!, reset_pid!

const AMBIGUOUS_ARABIC_WORDS = Set([
    "عين", "قلب", "لسان", "جذر", "شحن", "حق", "نور", "روح",
    "رأس", "راس", "يد", "وجه", "موجة", "موجه", "تيار", "حقل",
])

const MEANING_CUES = Set([
    "معنى", "يعني", "تعني", "فسر", "اشرح", "دلالة", "مقصود",
    "يقصد",
])

const CREATIVE_CUES = Set([
    "قصة", "قصه", "قصيدة", "شعر", "خيال", "مجاز", "صورة", "رمز",
])

const CODE_CUES = Set([
    "كود", "برمج", "دالة", "برنامج", "function", "code", "python", "julia",
])

const MATH_CUES = Set([
    "احسب", "ناتج", "جمع", "طرح", "ضرب", "قسمة", "معادلة", "math", "+", "-", "*", "/",
])

struct CerebellumObservation
    prompt::String
    tokens::Vector{String}
    tags::Set{String}
    signals::Dict{String,Float64}
end

struct CerebellumPolicy
    mode::String
    sense_mode::String
    weight_multipliers::Dict{String,Float64}
    confidence::Float64
    reason::String
end

mutable struct PIDController
    Kp::Float64
    Ki::Float64
    Kd::Float64
    integral::Float64
    prev_error::Float64
    prev_output::Float64
    window_size::Int
    error_history::Vector{Float64}
    setpoint::Float64
end

PIDController(; Kp=0.6, Ki=0.0, Kd=0.0, setpoint=0.5, window_size=10) =
    PIDController(Kp, Ki, Kd, 0.0, 0.0, 0.0, window_size, Float64[], setpoint)

function record_error!(pid::PIDController, err::Float64)
    push!(pid.error_history, err)
    if length(pid.error_history) > pid.window_size
        popfirst!(pid.error_history)
    end
end

function update_integral!(pid::PIDController)
    if !isempty(pid.error_history)
        pid.integral += pid.error_history[end]
        pid.integral = clamp(pid.integral, -10.0, 10.0)
    end
end

function reset_pid!(pid::PIDController)
    pid.integral = 0.0
    pid.prev_error = 0.0
    pid.prev_output = 0.0
    empty!(pid.error_history)
end

function correct!(pid::PIDController, weights::Dict{String,Float64}, signal::Dict{String,Float64})
    target = get(signal, "target", pid.setpoint)
    current = get(signal, "current", 0.0)
    err = target - current
    
    record_error!(pid, err)
    
    p_term = pid.Kp * err
    i_term = pid.Ki * pid.integral
    
    d_term = 0.0
    if length(pid.error_history) >= 2
        d_term = pid.Kd * (err - pid.prev_error)
    end
    
    output = p_term + i_term + d_term
    pid.prev_error = err
    pid.prev_output = output
    
    alpha = 0.1
    factor = 1.0 + alpha * output
    key = get(signal, "key", "all")
    if key == "all"
        for (k, v) in weights
            weights[k] = clamp(v * factor, 0.0, 50.0)
        end
    else
        if haskey(weights, key)
            weights[key] = clamp(weights[key] * factor, 0.0, 50.0)
        end
    end
    return output
end

function correct_weight!(pid::PIDController, weights::Dict{String,Float64}, key::String, err::Float64)
    p_term = pid.Kp * err
    factor = 1.0 + 0.1 * p_term
    if haskey(weights, key)
        weights[key] = clamp(weights[key] * factor, 0.0, 50.0)
    end
    return p_term
end

mutable struct MirnanCerebellum
    max_history::Int
    history::Vector{Dict{String,Any}}
    policy_counts::Dict{String,Int}
    policy_rewards::Dict{String,Float64}
    last_observation::Union{CerebellumObservation,Nothing}
    last_policy::Union{CerebellumPolicy,Nothing}
    pid::PIDController
    pid_enabled::Bool
    integration_log::Vector{Dict{String,Any}}
end

MirnanCerebellum(; max_history::Int=200, pid_enabled::Bool=true) =
    MirnanCerebellum(max_history, Dict{String,Any}[],
                     Dict{String,Int}(), Dict{String,Float64}(),
                     nothing, nothing, PIDController(), pid_enabled,
                     Dict{String,Any}[])

function _clean_token(token::AbstractString)
    s = lowercase(strip(String(token)))
    return strip(s, [' ', '\t', '\n', '\r', '.', ',', '،', '؟', '?', '!', ':', ';', '"', '\''])
end

function _tokenize(prompt::AbstractString)
    return String[_clean_token(t) for t in split(String(prompt)) if !isempty(_clean_token(t))]
end

_has_any(tokens::Vector{String}, cues::Set{String}) = any(t -> t in cues, tokens)

function _has_math_shape(prompt::String)
    occursin(r"\d+\s*[\+\-\*/]\s*\d+", prompt) && return true
    return false
end

function _question_score(tokens::Vector{String})
    isempty(tokens) && return 0.0
    q = Set(["ما", "ماذا", "هل", "كيف", "لماذا", "اين", "أين", "متى", "من", "كم", "why", "how", "what"])
    return any(t -> t in q, tokens) ? 1.0 : 0.0
end

function _ambiguity_score(tokens::Vector{String})
    isempty(tokens) && return 0.0
    hits = count(t -> t in AMBIGUOUS_ARABIC_WORDS, tokens)
    meaning = _has_any(tokens, MEANING_CUES) ? 1 : 0
    return clamp((hits + meaning) / max(length(tokens), 1), 0.0, 1.0)
end

function observe_prompt(cb::MirnanCerebellum, prompt::AbstractString;
                        vocab_size::Int=0, k_density::Float64=0.0)
    return observe_prompt(cb, _tokenize(prompt);
                          prompt=String(prompt), vocab_size=vocab_size,
                          k_density=k_density)
end

function observe_prompt(::MirnanCerebellum, tokens::AbstractVector{<:AbstractString};
                        prompt::AbstractString="",
                        vocab_size::Int=0, k_density::Float64=0.0)
    clean = String[_clean_token(t) for t in tokens if !isempty(_clean_token(t))]
    p = isempty(prompt) ? join(clean, " ") : String(prompt)
    tags = Set{String}()

    _question_score(clean) > 0 && push!(tags, "question")
    _ambiguity_score(clean) > 0 && push!(tags, "ambiguity")
    _has_any(clean, MEANING_CUES) && push!(tags, "meaning_query")
    _has_any(clean, CREATIVE_CUES) && push!(tags, "creative")
    (_has_any(clean, CODE_CUES) || occursin("```", p)) && push!(tags, "code")
    (_has_any(clean, MATH_CUES) || _has_math_shape(p)) && push!(tags, "math")
    length(clean) <= 3 && push!(tags, "short_prompt")
    vocab_size > 50000 && push!(tags, "large_vocab")
    k_density > 0.001 && push!(tags, "trained_resonance")

    signals = Dict{String,Float64}(
        "token_count" => Float64(length(clean)),
        "vocab_size" => Float64(vocab_size),
        "k_density" => k_density,
        "question" => _question_score(clean),
        "ambiguity" => _ambiguity_score(clean),
        "meaning_query" => ("meaning_query" in tags ? 1.0 : 0.0),
        "creative" => ("creative" in tags ? 1.0 : 0.0),
    )

    return CerebellumObservation(p, clean, tags, signals)
end

function _policy_key(policy::CerebellumPolicy)
    return string(policy.mode, "|", policy.sense_mode, "|", policy.reason)
end

function _reward_bias(cb::MirnanCerebellum, mode::String, reason::String)
    key_prefix = string(mode, "|")
    vals = Float64[]
    for (key, reward) in cb.policy_rewards
        startswith(key, key_prefix) && occursin(reason, key) && push!(vals, reward)
    end
    isempty(vals) && return 0.0
    return clamp(mean(vals) - 0.5, -0.15, 0.15)
end

function choose_policy!(cb::MirnanCerebellum, obs::CerebellumObservation;
                        requested_mode::String="auto")
    mode = requested_mode == "auto" ? "standard" : requested_mode
    sense_mode = "off"
    confidence = 0.55
    reason = "default"
    multipliers = Dict{String,Float64}()

    if "math" in obs.tags
        mode = requested_mode == "auto" ? "math" : mode
        reason = "math_route"
        confidence = 0.9
    elseif "code" in obs.tags
        mode = requested_mode == "auto" ? "code" : mode
        reason = "code_route"
        confidence = 0.9
    elseif "ambiguity" in obs.tags || "meaning_query" in obs.tags
        sense_mode = "measure"
        reason = "sense_measurement"
        confidence = 0.72
        # Prefer full wave scoring for small/medium vocabularies. For huge
        # vocabularies, keep resonant mode to avoid an expensive full scan.
        if requested_mode == "auto"
            mode = get(obs.signals, "vocab_size", 0.0) <= 20000 ? "standard" :
                   (get(obs.signals, "k_density", 0.0) > 0.001 ? "resonant" : "standard")
        end
        multipliers["prompt_align"] = 1.20
        multipliers["context_tension"] = 1.25
        multipliers["root_affinity"] = 1.15
        multipliers["density_resonance"] = 1.12
        multipliers["k_sem"] = 1.10
        multipliers["syntax"] = 1.10
        multipliers["aql_guidance"] = 1.15
        multipliers["polarity"] = 0.6      # accuracy λ=0.3 → less polarity
        multipliers["anatomical"] = 0.8
        multipliers["objectivity"] = 1.2   # accuracy prefers objectivity
    elseif "creative" in obs.tags
        reason = "creative_exploration"
        confidence = 0.68
        if requested_mode == "auto"
            mode = get(obs.signals, "k_density", 0.0) > 0.001 ? "prnn" : "standard"
        end
        multipliers["diversity"] = 1.30
        multipliers["poetic"] = 1.10
        multipliers["k_sem"] = 0.92
        multipliers["polarity"] = 3.0      # emotional λ=2.0 → ×3 polarity
        multipliers["anatomical"] = 2.0     # anatomy enriches poetic imagery
        multipliers["objectivity"] = 0.8
    elseif "question" in obs.tags
        reason = "question_answering"
        confidence = 0.66
        if requested_mode == "auto"
            mode = get(obs.signals, "k_density", 0.0) > 0.001 ? "resonant" : "standard"
        end
        multipliers["prompt_align"] = 1.15
        multipliers["causal_flow_align"] = 1.15
        multipliers["syntax"] = 1.12
        multipliers["diversity"] = 0.92
        multipliers["polarity"] = 0.8      # factual q → reduce polarity
        multipliers["anatomical"] = 0.9
        multipliers["objectivity"] = 1.1
    else
        if requested_mode == "auto"
            mode = get(obs.signals, "k_density", 0.0) > 0.001 ? "resonant" : "standard"
        end
        if "short_prompt" in obs.tags
            reason = "short_prompt_grounding"
            multipliers["root_affinity"] = 1.12
            multipliers["prompt_align"] = 1.10
            multipliers["syntax"] = 1.08
        end
    end

    confidence = clamp(confidence + _reward_bias(cb, mode, reason), 0.05, 0.98)
    policy = CerebellumPolicy(mode, sense_mode, multipliers, confidence, reason)
    cb.last_observation = obs
    cb.last_policy = policy
    return policy
end

function apply_cerebellum_policy!(weights::Dict{String,Float64},
                                  policy::CerebellumPolicy;
                                  min_weight::Float64=0.05,
                                  max_weight::Float64=20.0)
    for (name, mult) in policy.weight_multipliers
        haskey(weights, name) || continue
        weights[name] <= 0.0 && continue
        weights[name] = clamp(weights[name] * mult, min_weight, max_weight)
    end
    return weights
end

function _script_mix_penalty(text::AbstractString)
    arabic = count(c -> ('\u0600' <= c <= '\u06FF'), text)
    latin = count(c -> ('a' <= lowercase(c) <= 'z'), text)
    total = arabic + latin
    total == 0 && return 0.0
    minor = min(arabic, latin) / total
    return clamp(minor * 1.5, 0.0, 0.5)
end

function _outcome_reward(output::AbstractString)
    text = String(strip(String(output)))
    isempty(text) && return 0.0
    words = split(text)
    n = length(words)
    n == 0 && return 0.0
    unique_ratio = length(Set(words)) / n
    counts = Dict{String,Int}()
    for w in words
        counts[String(w)] = get(counts, String(w), 0) + 1
    end
    max_repeat = maximum(values(counts)) / n
    length_score = clamp(n / 10.0, 0.0, 1.0)
    repetition_score = 1.0 - max_repeat
    script_score = 1.0 - _script_mix_penalty(text)
    return clamp(0.25 * length_score + 0.30 * unique_ratio +
                 0.25 * repetition_score + 0.20 * script_score, 0.0, 1.0)
end

function learn_from_outcome!(cb::MirnanCerebellum,
                             obs::CerebellumObservation,
                             policy::CerebellumPolicy,
                             output::AbstractString;
                             reward::Union{Nothing,Float64}=nothing)
    r = reward === nothing ? _outcome_reward(output) : clamp(Float64(reward), 0.0, 1.0)
    
    if cb.pid_enabled
        err = cb.pid.setpoint - r
        record_error!(cb.pid, err)
    end
    
    key = _policy_key(policy)
    count = get(cb.policy_counts, key, 0) + 1
    old = get(cb.policy_rewards, key, 0.5)
    cb.policy_counts[key] = count
    cb.policy_rewards[key] = old + (r - old) / count

    push!(cb.history, Dict{String,Any}(
        "prompt" => obs.prompt,
        "tags" => collect(obs.tags),
        "mode" => policy.mode,
        "sense_mode" => policy.sense_mode,
        "reason" => policy.reason,
        "confidence" => policy.confidence,
        "reward" => r,
    ))
    while length(cb.history) > cb.max_history
        popfirst!(cb.history)
    end
    return r
end

function policy_summary(cb::MirnanCerebellum)
    return Dict{String,Any}(
        "history_count" => length(cb.history),
        "last_policy" => cb.last_policy === nothing ? nothing : Dict(
            "mode" => cb.last_policy.mode,
            "sense_mode" => cb.last_policy.sense_mode,
            "reason" => cb.last_policy.reason,
            "confidence" => cb.last_policy.confidence,
        ),
        "policy_counts" => copy(cb.policy_counts),
        "policy_rewards" => copy(cb.policy_rewards),
    )
end

function cerebellum_state_dict(cb::MirnanCerebellum)
    return Dict{String,Any}(
        "version" => 1,
        "max_history" => cb.max_history,
        "history" => cb.history,
        "policy_counts" => cb.policy_counts,
        "policy_rewards" => cb.policy_rewards,
    )
end

function restore_cerebellum_state!(cb::MirnanCerebellum, data)
    data === nothing && return false
    data isa Dict || return false
    cb.max_history = Int(get(data, "max_history", cb.max_history))
    empty!(cb.history)
    for item in get(data, "history", Any[])
        item isa Dict && push!(cb.history, Dict{String,Any}(string(k) => v for (k, v) in item))
    end
    empty!(cb.policy_counts)
    for (k, v) in get(data, "policy_counts", Dict())
        cb.policy_counts[string(k)] = Int(v)
    end
    empty!(cb.policy_rewards)
    for (k, v) in get(data, "policy_rewards", Dict())
        cb.policy_rewards[string(k)] = Float64(v)
    end
    return true
end

function reset_cerebellum!(cb::MirnanCerebellum)
    empty!(cb.history)
    empty!(cb.policy_counts)
    empty!(cb.policy_rewards)
    empty!(cb.integration_log)
    cb.last_observation = nothing
    cb.last_policy = nothing
    reset_pid!(cb.pid)
    return cb
end

end # module MirnanCerebellumModule
