"""
SenseSuperposition -- word-sense waves and contextual collapse.

A word can carry several possible sense states. Context acts like a measurement
field: it changes the probability of each sense and may collapse the word to a
single sense when confidence and margin are high enough.
"""
module SenseSuperpositionModule

using LinearAlgebra, Statistics

using ..WordPhysics: compute_extended_phase_vector, phase_similarity

export SenseCandidate, SenseSuperposition, SenseMeasurement,
       has_sense_inventory, sense_inventory, build_superposition,
       measure_senses, explain_measurement, top_sense

struct SenseCandidate
    word::String
    id::String
    label::String
    gloss::String
    anchors::Vector{String}
    prior::Float64
    vector::Vector{Float64}
    phase::Float64
end

struct SenseSuperposition
    word::String
    candidates::Vector{SenseCandidate}
    amplitudes::Dict{String,Float64}
    entropy::Float64
end

struct SenseMeasurement
    word::String
    context_words::Vector{String}
    probabilities::Dict{String,Float64}
    collapsed::Bool
    selected_id::String
    selected_label::String
    confidence::Float64
    margin::Float64
    entropy::Float64
    reason::String
end

const _SENSE_LEXICON = Dict{String,Vector{Dict{String,Any}}}(
    "عين" => [
        Dict("id" => "sight", "label" => "عضو البصر", "gloss" => "العين التي يرى بها الإنسان أو الكائن",
             "anchors" => ["بصر", "رؤية", "نظر", "وجه", "رمش", "دمع", "حدقة"], "prior" => 0.28),
        Dict("id" => "spring", "label" => "نبع ماء", "gloss" => "عين الماء أو الينبوع",
             "anchors" => ["ماء", "نبع", "ينبوع", "جبل", "وادي", "شرب", "جدول"], "prior" => 0.28),
        Dict("id" => "spy", "label" => "جاسوس أو مراقب", "gloss" => "عين القوم أو الجاسوس",
             "anchors" => ["جاسوس", "مراقبة", "عدو", "سر", "مخابرات", "حارس", "تجسس"], "prior" => 0.24),
        Dict("id" => "essence", "label" => "الذات أو الجوهر", "gloss" => "عين الشيء: ذاته أو حقيقته",
             "anchors" => ["ذات", "جوهر", "حقيقة", "نفس", "عَيْنُه", "عينه"], "prior" => 0.20),
    ],
    "قلب" => [
        Dict("id" => "organ", "label" => "عضو القلب", "gloss" => "القلب العضوي أو موضع النبض",
             "anchors" => ["نبض", "دم", "صدر", "شريان", "مرض", "حياة"], "prior" => 0.30),
        Dict("id" => "emotion", "label" => "العاطفة والوجدان", "gloss" => "القلب بمعنى الشعور والحب والخوف",
             "anchors" => ["حب", "خوف", "حزن", "فرح", "شعور", "وجدان"], "prior" => 0.30),
        Dict("id" => "center", "label" => "المركز أو اللب", "gloss" => "قلب المدينة أو قلب الفكرة",
             "anchors" => ["مركز", "مدينة", "لب", "وسط", "محور", "داخل"], "prior" => 0.25),
        Dict("id" => "invert", "label" => "التحويل أو العكس", "gloss" => "قلب الشيء أي عكسه أو بدله",
             "anchors" => ["عكس", "بدل", "حوّل", "قلب", "تغيير", "اتجاه"], "prior" => 0.15),
    ],
    "لسان" => [
        Dict("id" => "organ", "label" => "عضو اللسان", "gloss" => "اللسان الجسدي",
             "anchors" => ["فم", "ذوق", "كلام", "نطق", "أسنان"], "prior" => 0.32),
        Dict("id" => "language", "label" => "لغة أو كلام", "gloss" => "لسان بمعنى لغة أو طريقة كلام",
             "anchors" => ["لغة", "عربية", "كلام", "بيان", "نحو", "ترجمة"], "prior" => 0.36),
        Dict("id" => "projection", "label" => "طرف ممتد", "gloss" => "لسان البحر أو النار، امتداد ضيق",
             "anchors" => ["بحر", "نار", "لهب", "امتداد", "طرف", "شاطئ"], "prior" => 0.32),
    ],
    "جذر" => [
        Dict("id" => "plant", "label" => "جذر نبات", "gloss" => "أصل النبات في الأرض",
             "anchors" => ["نبات", "أرض", "شجرة", "تربة", "غصن", "ورقة"], "prior" => 0.26),
        Dict("id" => "linguistic", "label" => "جذر لغوي", "gloss" => "أصل صرفي للكلمة",
             "anchors" => ["كلمة", "صرف", "وزن", "حروف", "لغة", "اشتقاق"], "prior" => 0.30),
        Dict("id" => "math", "label" => "جذر رياضي", "gloss" => "الجذر في الحساب أو المعادلة",
             "anchors" => ["رياضيات", "معادلة", "تربيعي", "عدد", "حل", "حساب"], "prior" => 0.30),
        Dict("id" => "origin", "label" => "أصل أو سبب", "gloss" => "جذر المشكلة أو أصلها",
             "anchors" => ["أصل", "سبب", "مشكلة", "منبع", "أساس", "بداية"], "prior" => 0.14),
    ],
    "شحن" => [
        Dict("id" => "electric", "label" => "شحن كهربائي", "gloss" => "ملء بطارية أو تراكم شحنة",
             "anchors" => ["كهرباء", "بطارية", "تيار", "طاقة", "هاتف", "سلك"], "prior" => 0.35),
        Dict("id" => "shipping", "label" => "نقل وإرسال", "gloss" => "شحن البضائع أو الطرود",
             "anchors" => ["بضاعة", "طرد", "ميناء", "سفينة", "نقل", "إرسال"], "prior" => 0.35),
        Dict("id" => "emotional", "label" => "تعبئة نفسية", "gloss" => "شحن عاطفي أو تحريض",
             "anchors" => ["غضب", "حماس", "تحريض", "خطاب", "توتر", "مشاعر"], "prior" => 0.30),
    ],
)

_clean_token(w::AbstractString) =
    strip(lowercase(String(w)), [' ', '\t', '\n', '\r', '.', ',', '،', '؟', '?', '!', ':', ';', '"', '\''])

_norm_key(w::AbstractString) = replace(_clean_token(w), 'أ' => 'ا', 'إ' => 'ا', 'آ' => 'ا', 'ى' => 'ي', 'ة' => 'ه')

function _norm_anchor(w::AbstractString)
    key = _norm_key(w)
    if startswith(key, "ال") && length(key) > 2
        return key[nextind(key, firstindex(key), 2):end]
    end
    return key
end

function has_sense_inventory(word::AbstractString)
    return haskey(_SENSE_LEXICON, _norm_key(word))
end

function sense_inventory(word::AbstractString)
    return get(_SENSE_LEXICON, _norm_key(word), Dict{String,Any}[])
end

function _safe_pv(word::AbstractString, pv_fn::Function)
    try
        return Float64.(pv_fn(String(word)))
    catch
        return Float64[]
    end
end

function _context_vector(context_words::Vector{String}, pv_fn::Function)
    vec = Float64[]
    total = 0.0
    for (age, word) in enumerate(reverse(context_words))
        pv = _safe_pv(word, pv_fn)
        isempty(pv) && continue
        if isempty(vec)
            vec = zeros(Float64, length(pv))
        end
        d = min(length(vec), length(pv))
        w = 1.0 / sqrt(age)
        vec[1:d] .+= w .* pv[1:d]
        total += w
    end
    total > 1e-10 && (vec ./= total)
    nrm = norm(vec)
    nrm > 1e-10 && (vec ./= nrm)
    return vec
end

function _sense_vector(word::String, anchors::Vector{String}, pv_fn::Function)
    base = _safe_pv(word, pv_fn)
    vec = isempty(base) ? Float64[] : 0.25 .* base
    total = isempty(base) ? 0.0 : 0.25
    for anchor in anchors
        pv = _safe_pv(anchor, pv_fn)
        isempty(pv) && continue
        if isempty(vec)
            vec = zeros(Float64, length(pv))
        end
        d = min(length(vec), length(pv))
        vec[1:d] .+= pv[1:d]
        total += 1.0
    end
    total > 1e-10 && (vec ./= total)
    nrm = norm(vec)
    nrm > 1e-10 && (vec ./= nrm)
    return vec
end

function _phase_of(v::Vector{Float64})
    length(v) >= 2 || return 0.0
    return atan(v[2], v[1])
end

function build_superposition(word::AbstractString;
                             inventory=nothing,
                             pv_fn::Function=compute_extended_phase_vector)
    key = _norm_key(word)
    raw = inventory === nothing ? sense_inventory(key) : inventory
    candidates = SenseCandidate[]
    for item in raw
        anchors = String[string(x) for x in get(item, "anchors", String[])]
        prior = clamp(Float64(get(item, "prior", 1.0)), 0.0, 1.0)
        vec = _sense_vector(key, anchors, pv_fn)
        push!(candidates, SenseCandidate(
            key,
            string(get(item, "id", get(item, "label", key))),
            string(get(item, "label", get(item, "id", key))),
            string(get(item, "gloss", "")),
            anchors,
            prior,
            vec,
            _phase_of(vec),
        ))
    end
    if isempty(candidates)
        return SenseSuperposition(key, SenseCandidate[], Dict{String,Float64}(), 0.0)
    end
    total_prior = sum(c.prior for c in candidates)
    total_prior <= 1e-10 && (total_prior = length(candidates))
    amplitudes = Dict(c.id => sqrt(max(c.prior / total_prior, 0.0)) for c in candidates)
    probs = [amp^2 for amp in values(amplitudes)]
    entropy = -sum(p > 1e-12 ? p * log(p) : 0.0 for p in probs)
    return SenseSuperposition(key, candidates, amplitudes, entropy)
end

function _anchor_overlap(candidate::SenseCandidate, context_words::Vector{String})
    isempty(context_words) && return 0.0
    ctx = Set(_norm_anchor(w) for w in context_words)
    anchors = Set(_norm_anchor(w) for w in candidate.anchors)
    isempty(anchors) && return 0.0
    hits = length(intersect(ctx, anchors))
    hits == 0 && return 0.0
    anchor_ratio = hits / length(anchors)
    context_ratio = hits / max(length(ctx), 1)
    return clamp(0.35 * anchor_ratio + 0.65 * context_ratio, 0.0, 1.0)
end

function _sense_score(candidate::SenseCandidate, context_vec::Vector{Float64},
                      context_words::Vector{String})
    anchor = _anchor_overlap(candidate, context_words)
    if isempty(context_vec) || isempty(candidate.vector)
        return max(candidate.prior, 1e-6) * (1.0 + 2.0 * anchor)
    end
    d = min(length(context_vec), length(candidate.vector))
    sim = phase_similarity(view(candidate.vector, 1:d), view(context_vec, 1:d))
    resonance = max(0.0, sim)^2
    return max(candidate.prior, 1e-6) * (0.15 + 0.75 * resonance + 4.5 * anchor)
end

function measure_senses(word::AbstractString, context_words::AbstractVector{<:AbstractString};
                        inventory=nothing,
                        collapse_threshold::Float64=0.55,
                        margin_threshold::Float64=0.14,
                        pv_fn::Function=compute_extended_phase_vector)
    super = build_superposition(word; inventory=inventory, pv_fn=pv_fn)
    if isempty(super.candidates)
        return SenseMeasurement(_norm_key(word), context_words, Dict{String,Float64}(),
            false, "", "", 0.0, 0.0, 0.0, "no_inventory")
    end

    clean_context = String[_clean_token(w) for w in context_words if !isempty(_clean_token(w))]
    context_vec = _context_vector(clean_context, pv_fn)
    raw = Dict{String,Float64}()
    for candidate in super.candidates
        raw[candidate.id] = _sense_score(candidate, context_vec, clean_context)
    end
    total = sum(values(raw))
    if total <= 1e-12
        p = 1.0 / length(super.candidates)
        probs = Dict(c.id => p for c in super.candidates)
    else
        probs = Dict(k => v / total for (k, v) in raw)
    end
    ranked = sort(collect(probs); by=x -> -x[2])
    best_id, best_p = ranked[1]
    second_p = length(ranked) >= 2 ? ranked[2][2] : 0.0
    margin = best_p - second_p
    selected = first(c for c in super.candidates if c.id == best_id)
    entropy = -sum(p > 1e-12 ? p * log(p) : 0.0 for p in values(probs))
    collapsed = best_p >= collapse_threshold && margin >= margin_threshold
    reason = collapsed ? "contextual_collapse" : "mixed_state"
    return SenseMeasurement(super.word, clean_context, probs, collapsed,
        selected.id, selected.label, best_p, margin, entropy, reason)
end

function top_sense(measurement::SenseMeasurement)
    return measurement.selected_id, measurement.selected_label, measurement.confidence
end

function explain_measurement(measurement::SenseMeasurement)
    isempty(measurement.probabilities) && return ""
    ranked = sort(collect(measurement.probabilities); by=x -> -x[2])
    parts = String[]
    for (id, p) in ranked[1:min(3, end)]
        push!(parts, "$(id)=$(round(p * 100; digits=1))%")
    end
    if measurement.collapsed
        return "قياس معنى [$(measurement.word)]: انهارت موجة المعنى إلى [$(measurement.selected_label)] بثقة $(round(measurement.confidence * 100; digits=1))%. الاحتمالات: $(join(parts, "، "))."
    else
        return "قياس معنى [$(measurement.word)]: بقيت الكلمة في حالة معانٍ مختلطة؛ أقرب معنى هو [$(measurement.selected_label)] بثقة $(round(measurement.confidence * 100; digits=1))%. الاحتمالات: $(join(parts, "، "))."
    end
end

end # module SenseSuperpositionModule
