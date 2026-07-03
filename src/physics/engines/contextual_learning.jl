"""
ContextualLearning — advanced interaction learning for compounds, question intent,
metaphor, named entities, feedback, and action effects.
"""
module ContextualLearning

using LinearAlgebra

using ..WordPhysics: compute_extended_phase_vector
using ..PhaseReinforcement

export CompoundExpression, CompoundExpressionMemory, QuestionIntent,
       QuestionIntentLearner, MetaphorEntry, MetaphorField, EffectMemory,
       ContextualLearningState, detect_question_intent, detect_compounds,
       learn_compound!, learn_metaphor!, learn_effect!, contextual_answer,
       learn_from_feedback!, compound_vector

struct CompoundExpression
    text::String
    tokens::Vector{String}
    kind::String
    vector::Vector{Float64}
    strength::Float64
    observations::Int
end

mutable struct CompoundExpressionMemory
    entries::Dict{String,CompoundExpression}
    max_entries::Int
end

CompoundExpressionMemory(; max_entries::Int=1000) =
    CompoundExpressionMemory(Dict{String,CompoundExpression}(), max_entries)

struct QuestionIntent
    kind::String
    required_slots::Vector{String}
    subject_tokens::Vector{String}
    confidence::Float64
end

mutable struct QuestionIntentLearner
    patterns::Dict{String,QuestionIntent}
end

function QuestionIntentLearner()
    return QuestionIntentLearner(Dict(
        "ماذا فعل" => QuestionIntent("action_query", ["verb", "subject", "effect"], String[], 1.0),
        "ماذا يفعل" => QuestionIntent("action_query", ["verb", "subject", "effect"], String[], 1.0),
        "لماذا" => QuestionIntent("reason_query", ["reason"], String[], 0.9),
        "كيف" => QuestionIntent("mechanism_query", ["mechanism"], String[], 0.9),
        "أين" => QuestionIntent("place_query", ["place"], String[], 0.9),
        "متى" => QuestionIntent("time_query", ["time"], String[], 0.9),
    ))
end

struct MetaphorEntry
    compound::String
    interpretation::String
    actions::Vector{String}
    effects::Vector{String}
    strength::Float64
end

mutable struct MetaphorField
    entries::Dict{String,MetaphorEntry}
    cues::Set{String}
end

function MetaphorField()
    return MetaphorField(
        Dict(
            "جيش الليل" => MetaphorEntry(
                "جيش الليل", "الظلام كجيش",
                ["زحف", "غطى", "حاصر", "انتشر"],
                ["غطى المكان بالظلام والسكون", "أطفأ ضجيج النهار", "ترك المدينة في سكونها"],
                0.8,
            ),
            "البحر يبتلع" => MetaphorEntry(
                "البحر يبتلع", "البحر ككائن يبتلع",
                ["ابتلع", "غمر"],
                ["غمر الشاطئ بالموج", "أخفى الأثر تحت الماء"],
                0.6,
            ),
            "الوقت يركض" => MetaphorEntry(
                "الوقت يركض", "الزمن ككائن متحرك",
                ["ركض", "تسارع"],
                ["تركنا خلفه", "اختصر المسافة بين اللحظات"],
                0.6,
            ),
        ),
        Set(["كأن", "مثل", "كـ", "صورة", "مجاز", "شعر", "رمز"]),
    )
end

mutable struct EffectMemory
    effects::Dict{String,Dict{String,Float64}}
end

function EffectMemory()
    return EffectMemory(Dict(
        "زحف" => Dict("غطى المكان بالظلام والسكون" => 0.8, "حاصر المكان" => 0.5),
        "غطى" => Dict("ساد السكون" => 0.7, "اختفى الضوء" => 0.7),
        "خيم" => Dict("ساد السكون" => 0.8),
        "هطل" => Dict("ابتلت الأرض" => 0.8),
    ))
end

mutable struct ContextualLearningState
    compounds::CompoundExpressionMemory
    question_intents::QuestionIntentLearner
    metaphors::MetaphorField
    effects::EffectMemory
    entity_kinds::Dict{String,String}
    feedback_log::Vector{Dict{String,Any}}
end

function ContextualLearningState()
    state = ContextualLearningState(
        CompoundExpressionMemory(),
        QuestionIntentLearner(),
        MetaphorField(),
        EffectMemory(),
        Dict{String,String}(),
        Dict{String,Any}[],
    )
    learn_compound!(state.compounds, ["جيش", "الليل"]; kind="metaphor")
    learn_compound!(state.compounds, ["باب", "الشمس"]; kind="metaphor")
    learn_compound!(state.compounds, ["سيف", "الزمن"]; kind="metaphor")
    learn_compound!(state.compounds, ["قلب", "المدينة"]; kind="metaphor")
    return state
end

_clean_token(w::AbstractString) = String(strip(lowercase(String(w)), [' ', '\t', '\n', '\r', '.', ',', '،', '؟', '?', '!', ':', ';', '"', '\'']))

function _tokenize(text::String)
    return String[_clean_token(w) for w in split(text) if !isempty(_clean_token(w))]
end

function compound_vector(tokens::AbstractVector{<:AbstractString}, pv_fn::Function=compute_extended_phase_vector)
    isempty(tokens) && return zeros(Float64, 0)
    vecs = [Float64.(pv_fn(w)) for w in tokens]
    d = minimum(length.(vecs))
    out = zeros(Float64, d)
    for (i, v) in enumerate(vecs)
        weight = 1.0 / sqrt(i)
        out .+= weight .* view(v, 1:d)
    end
    nrm = norm(out)
    nrm > 1e-10 && (out ./= nrm)
    return out
end

function learn_compound!(mem::CompoundExpressionMemory, tokens::AbstractVector{<:AbstractString};
                         kind::String="ordinary", pv_fn::Function=compute_extended_phase_vector,
                         strength::Float64=0.5)
    clean = [_clean_token(t) for t in tokens if !isempty(_clean_token(t))]
    length(clean) < 2 && return nothing
    key = join(clean, " ")
    vec = compound_vector(clean, pv_fn)
    old = get(mem.entries, key, nothing)
    if old !== nothing
        new_strength = clamp(old.strength + 0.1 * strength, 0.0, 1.0)
        mem.entries[key] = CompoundExpression(key, clean, kind, vec, new_strength, old.observations + 1)
    else
        if length(mem.entries) >= mem.max_entries
            weakest = first(sort(collect(keys(mem.entries)); by=k -> mem.entries[k].strength))
            delete!(mem.entries, weakest)
        end
        mem.entries[key] = CompoundExpression(key, clean, kind, vec, clamp(strength, 0.0, 1.0), 1)
    end
    return mem.entries[key]
end

function detect_compounds(mem::CompoundExpressionMemory, tokens::AbstractVector{<:AbstractString})
    clean = [_clean_token(t) for t in tokens]
    found = CompoundExpression[]
    for n in min(4, length(clean)):-1:2
        for i in 1:(length(clean)-n+1)
            key = join(clean[i:i+n-1], " ")
            if haskey(mem.entries, key)
                push!(found, mem.entries[key])
            end
        end
    end
    return found
end

function detect_question_intent(qil::QuestionIntentLearner, tokens::AbstractVector{<:AbstractString})
    clean = [_clean_token(t) for t in tokens]
    length(clean) >= 2 || return QuestionIntent("statement", String[], String[], 0.0)

    prefix2 = join(clean[1:min(2, end)], " ")
    if haskey(qil.patterns, prefix2)
        subject = clean[min(3, length(clean)+1):end]
        return QuestionIntent(qil.patterns[prefix2].kind, qil.patterns[prefix2].required_slots, subject, 1.0)
    end

    first_token = clean[1]
    if haskey(qil.patterns, first_token)
        subject = clean[min(2, length(clean)+1):end]
        return QuestionIntent(qil.patterns[first_token].kind, qil.patterns[first_token].required_slots, subject, 0.9)
    end

    return QuestionIntent("statement", String[], String[], 0.0)
end

function learn_metaphor!(field::MetaphorField, compound::String, interpretation::String;
                         actions::Vector{String}=String[], effects::Vector{String}=String[],
                         strength::Float64=0.5)
    key = join(_tokenize(compound), " ")
    isempty(key) && return nothing
    old = get(field.entries, key, nothing)
    if old !== nothing
        merged_actions = unique(vcat(old.actions, actions))
        merged_effects = unique(vcat(old.effects, effects))
        field.entries[key] = MetaphorEntry(key, interpretation, merged_actions, merged_effects,
                                           clamp(old.strength + 0.1 * strength, 0.0, 1.0))
    else
        field.entries[key] = MetaphorEntry(key, interpretation, actions, effects, clamp(strength, 0.0, 1.0))
    end
    return field.entries[key]
end

function learn_effect!(mem::EffectMemory, action::String, effect::String; strength::Float64=0.5)
    a = _clean_token(action)
    e = strip(effect)
    (isempty(a) || isempty(e)) && return false
    bucket = get!(mem.effects, a, Dict{String,Float64}())
    bucket[e] = clamp(get(bucket, e, 0.0) + strength, 0.0, 1.0)
    return true
end

function _best_effect(mem::EffectMemory, action::String, fallback_effects::Vector{String}=String[])
    bucket = get(mem.effects, _clean_token(action), Dict{String,Float64}())
    if !isempty(bucket)
        return first(sort(collect(bucket); by=x -> -x[2]))[1]
    end
    isempty(fallback_effects) ? "" : fallback_effects[1]
end

function _metaphor_for(field::MetaphorField, compounds::Vector{CompoundExpression})
    for c in compounds
        if haskey(field.entries, c.text)
            return field.entries[c.text]
        end
    end
    return nothing
end

function contextual_answer(state::ContextualLearningState, prompt::String; style::String="auto")
    tokens = _tokenize(prompt)
    isempty(tokens) && return ""
    intent = detect_question_intent(state.question_intents, tokens)
    intent.kind == "action_query" || return ""

    compounds = detect_compounds(state.compounds, tokens)
    subject = isempty(compounds) ? join(intent.subject_tokens, " ") : compounds[1].text
    isempty(subject) && return ""

    entity_kind = get(state.entity_kinds, subject, "")
    if !isempty(entity_kind)
        return "إذا كان \"$(subject)\" اسما لكيان في هذا السياق، فأحتاج حدثه أو سياق القصة لأجيب بدقة."
    end

    metaphor = _metaphor_for(state.metaphors, compounds)
    if metaphor !== nothing
        action = isempty(metaphor.actions) ? "تحرك" : metaphor.actions[1]
        effect = _best_effect(state.effects, action, metaphor.effects)
        if style == "cautious"
            return "إن كنت تقصدها كصورة شعرية، فقد $(action) $(subject) و$(effect). وإن كان \"$(subject)\" اسما لكيان في قصة، فأحتاج سياق القصة لأجيب بدقة."
        elseif style == "poetic"
            return "$(action) $(subject) من حواف الأفق، ف$(effect)."
        else
            return "$(action) $(subject)، ف$(effect)."
        end
    end

    action = "تحرك"
    effect = _best_effect(state.effects, action)
    isempty(effect) && (effect = "ترك أثرا في المكان")
    return "$(action) $(subject)، ف$(effect)."
end

function _learn_entity_note!(state::ContextualLearningState, note::String)
    tokens = _tokenize(note)
    isempty(tokens) && return false
    joined = join(tokens, " ")
    markers = ["اسم", "كيان", "قبيلة", "فرقة", "شخصية", "مدينة"]
    any(m -> occursin(m, joined), markers) || return false
    # Learn the two words before the first marker as a named entity when possible.
    for (i, t) in enumerate(tokens)
        if t in markers && i > 2
            name = join(tokens[i-2:i-1], " ")
            state.entity_kinds[name] = t
            learn_compound!(state.compounds, tokens[i-2:i-1]; kind="named_entity", strength=0.8)
            return true
        end
    end
    return false
end

function learn_from_feedback!(state::ContextualLearningState, prompt::String, response::String;
                              rating::Float64=1.0, note::String="")
    prompt_tokens = _tokenize(prompt)
    response_tokens = _tokenize(response)
    rating = clamp(rating, -1.0, 1.0)

    compounds = detect_compounds(state.compounds, prompt_tokens)
    for c in compounds
        learn_compound!(state.compounds, c.tokens; kind=c.kind, strength=max(0.1, abs(rating)))
    end

    if !isempty(note)
        _learn_entity_note!(state, note)
        if any(cue -> occursin(cue, note), state.metaphors.cues)
            subj = isempty(compounds) ? join(prompt_tokens[max(1, end-1):end], " ") : compounds[1].text
            learn_metaphor!(state.metaphors, subj, note; actions=response_tokens[1:min(2, end)],
                            effects=[response], strength=max(0.2, rating))
        end
    end

    if rating > 0 && length(response_tokens) >= 2
        for i in 1:length(response_tokens)-1
            learn_effect!(state.effects, response_tokens[i], join(response_tokens[i+1:end], " "); strength=0.1 * rating)
        end
    end

    push!(state.feedback_log, Dict{String,Any}(
        "prompt" => prompt,
        "response" => response,
        "rating" => rating,
        "note" => note,
    ))
    length(state.feedback_log) > 500 && popfirst!(state.feedback_log)
    return state
end

function learn_from_feedback!(gen, prompt::String, response::String; rating::Float64=1.0, note::String="")
    if hasproperty(gen, :contextual_learning)
        learn_from_feedback!(getfield(gen, :contextual_learning), prompt, response; rating=rating, note=note)
    end

    words = _tokenize(response)
    if hasproperty(gen, :reinforcement) && hasproperty(gen, :pv_cache)
        for w in words
            pv = get!(getfield(gen, :pv_cache), w) do
                Float64.(compute_extended_phase_vector(w))
            end
            if rating >= 0 && hasproperty(gen, :reinforcement)
                try
                    PhaseReinforcement.reinforce!(getfield(gen, :reinforcement), w, pv; reward=rating)
                catch e
                    @warn "Contextual learning: reinforce failed for '$w': $e"
                end
            end
        end
    end

    if rating > 0 && hasproperty(gen, :K_sem) && getfield(gen, :K_sem) !== nothing && hasproperty(gen, :vocab)
        K = getfield(gen, :K_sem)
        vocab = getfield(gen, :vocab)
        for i in 1:length(words)-1
            a = get(vocab, words[i], 0)
            b = get(vocab, words[i+1], 0)
            if a > 0 && b > 0 && a <= size(K, 1) && b <= size(K, 2)
                K[a, b] = min(K[a, b] + 0.05 * rating, 100.0)
            end
        end
    end
    return gen
end

end # module ContextualLearning
