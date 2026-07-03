"""
IntentResponsePlanner -- small answer-intent layer for Mirnan.

This layer is deliberately narrow. It does not replace physical generation and
does not add a new physics engine. It detects answer shapes that free word-by-word
generation currently fails at, then provides a light scaffold for the generator:
how/mechanism answers, causal/conditional answers, and simple evaluative
opinion/dialogue answers.
"""
module IntentResponsePlanner

export ResponseIntentPlan, IntentGravityProfile, detect_response_intent,
       intent_gravity_profile, render_planned_response,
       has_plannable_response, has_gravity_profile

struct ResponseIntentPlan
    intent::String
    structure::Vector{String}
    subject::String
    action::String
    cause::String
    result::String
    confidence::Float64
end

struct IntentGravityProfile
    intent::String
    question_charge::Float64
    response_charge::Float64
    syntax_multiplier::Float64
    semantic_multiplier::Float64
    causal_multiplier::Float64
    guidance_terms::Vector{String}
    repulsion_terms::Vector{String}
    max_words::Int
    confidence::Float64
end

const QUESTION_TOOLS = Set([
    "ما", "ماذا", "هل", "كيف", "لماذا", "لما", "من", "متى", "اين", "أين",
    "what", "how", "why", "who", "when", "where",
])

const STOPWORDS = Set([
    "في", "على", "عن", "الى", "إلى", "من", "و", "ف", "ثم", "او", "أو",
    "هو", "هي", "هذا", "هذه", "ذلك", "تلك", "the", "a", "an", "is", "are",
    "of", "to", "in", "on", "and", "or",
])

const RESPONSE_STRUCTURES = Dict{String,Vector{String}}(
    "definition" => ["تعريف", "توضيح مختصر"],
    "mechanism" => ["تمهيد", "وسيلة", "مرحلة", "مراجعة"],
    "causal" => ["سبب", "نتيجة", "رابط"],
    "conditional" => ["شرط", "جواب", "تعليل"],
    "yesno" => ["تحقق", "إثبات أو نفي"],
    "difference" => ["طرف أول", "طرف ثان", "وجه فرق"],
    "spatial" => ["موضوع", "موضع"],
    "temporal" => ["موضوع", "زمن"],
    "descriptive" => ["وصف", "تفصيل"],
    "opinion" => ["موضوع", "تقييم", "تعليل", "قيد"],
    "dialogue" => ["مدخل حواري", "رد مناسب"],
    "unknown" => String[],
)

function _clean_token(token::AbstractString)
    s = lowercase(strip(String(token)))
    return strip(s, [' ', '\t', '\n', '\r', '.', ',', '،', '؛', ':', '?', '؟', '!', '"', '\''])
end

function _tokens(text::AbstractString)
    out = String[]
    for raw in split(String(text))
        tok = _clean_token(raw)
        isempty(tok) || push!(out, tok)
    end
    return out
end

function _content_tokens(text::AbstractString)
    return String[t for t in _tokens(text) if !(t in QUESTION_TOOLS) && !(t in STOPWORDS) && length(t) >= 2]
end

function _looks_like_verb(token::AbstractString)
    t = String(token)
    isempty(t) && return false
    return startswith(t, "ي") || startswith(t, "ت") || startswith(t, "أ") ||
           startswith(t, "ن") || startswith(t, "ا")
end

function _join_phrase(tokens::Vector{String})
    return strip(join(tokens, " "))
end

function _extract_mechanism_parts(prompt::AbstractString)
    content = _content_tokens(prompt)
    isempty(content) && return "", ""
    action_idx = findfirst(_looks_like_verb, content)
    if action_idx === nothing
        subject = content[end]
        action = length(content) >= 2 ? content[1] : "يحدث"
    else
        action = content[action_idx]
        rest = String[content[i] for i in eachindex(content) if i != action_idx]
        subject = isempty(rest) ? "ذلك" : _join_phrase(rest)
    end
    return subject, action
end

function _split_repeated_predicate(tokens::Vector{String})
    length(tokens) >= 4 || return "", ""
    first_word = tokens[1]
    for i in 3:length(tokens)
        if tokens[i] == first_word
            return _join_phrase(tokens[1:i-1]), _join_phrase(tokens[i:end])
        end
    end
    return "", ""
end

function _extract_condition_parts(prompt::AbstractString)
    text = strip(String(prompt))
    text = replace(text, r"^(?:إذا|اذا|إن|ان|كلما)\s+" => "")
    text = replace(text, r"^(?:ماذا\s+يحدث\s+إذا|ماذا\s+سيحدث\s+إذا|ماذا\s+ينتج\s+إذا)\s+" => "")
    chunks = split(text, r"[،,؛;]")
    if length(chunks) >= 2
        return strip(String(chunks[1])), strip(String(join(chunks[2:end], " ")))
    end
    content = _content_tokens(text)
    cause, result = _split_repeated_predicate(content)
    if !isempty(cause) && !isempty(result)
        return cause, result
    end
    if length(content) >= 4
        mid = max(2, fld(length(content), 2))
        return _join_phrase(content[1:mid]), _join_phrase(content[mid+1:end])
    end
    return _join_phrase(content), ""
end

function _extract_opinion_subject(prompt::AbstractString)
    text = lowercase(String(prompt))
    text = replace(text, r"^\s*ما\s+رأيك\s+(?:في|ب)?\s*" => "")
    text = replace(text, r"^\s*ما\s+قولك\s+(?:في|ب)?\s*" => "")
    text = replace(text, r"^\s*هل\s+" => "")
    text = replace(text, r"^\s*أهمية\s+" => "")
    text = replace(text, r"^\s*اهمية\s+" => "")
    text = replace(text, r"\s+في\s+الحياة\s*[؟?!.]*\s*$" => "")
    text = replace(text, r"\s+على\s+الجميع\s*[؟?!.]*\s*$" => "")
    opinion_words = Set(["رأيك", "قولك", "مفيد", "مفيدة", "نافع", "نافعة",
                         "واجب", "ضروري", "ضرورية", "مهم", "مهمة",
                         "دائما", "دائماً", "أهمية", "اهمية", "الجميع"])
    content = String[t for t in _content_tokens(text) if !(t in opinion_words)]
    isempty(content) && return "هذا الموضوع"
    return isempty(content) ? "هذا الموضوع" : _join_phrase(content)
end

function _is_opinion_prompt(text::AbstractString)
    s = lowercase(String(text))
    return occursin(r"(^|\s)(ما\s+(?:هو\s+)?رأيك|ما\s+(?:هو\s+)?قولك)(\s|$)", s) ||
           occursin(r"(^|\s)هل\s+.+\s+(مفيد|مفيدة|نافع|نافعة|واجب|ضروري|ضرورية|مهم|مهمة)(\s|[؟?]|$)", s) ||
           occursin(r"(^|\s)(أهمية|اهمية)\s+", s)
end

function _is_descriptive_prompt(text::AbstractString)
    s = lowercase(strip(String(text)))
    return occursin(r"(^|\s)(صف|وصف|describe)(\s|$)", s) ||
           occursin(r"(^|\s)كيف\s+(?:هو|هي|تبدو|يبدو|كان|كانت)\s+", s)
end

function _extract_descriptive_subject(prompt::AbstractString)
    text = lowercase(String(prompt))
    text = replace(text, r"^\s*(?:صف|وصف|describe)\s+" => "")
    text = replace(text, r"^\s*كيف\s+(?:هو|هي|تبدو|يبدو|كان|كانت)\s+" => "")
    content = _content_tokens(text)
    isempty(content) && return ""
    return _join_phrase(content)
end

function _extract_difference_subject(prompt::AbstractString)
    text = lowercase(String(prompt))
    text = replace(text, r"^\s*(?:ما\s+)?(?:هو\s+)?الفرق\s+بين\s+" => "")
    text = replace(text, r"^\s*(?:بيّن|بين|اشرح)\s+الفرق\s+بين\s+" => "")
    text = replace(text, r"[\u061F\?!.]+$" => "")
    return strip(text)
end

function _is_difference_prompt(prompt::AbstractString)
    s = lowercase(strip(String(prompt)))
    return occursin(r"(^|\s)(?:ما\s+)?(?:هو\s+)?الفرق\s+بين\s+", s) ||
           occursin(r"(^|\s)(?:بيّن|بين|اشرح)\s+الفرق\s+بين\s+", s)
end

function _extract_locative_subject(prompt::AbstractString)
    text = lowercase(String(prompt))
    text = replace(text, r"^\s*(?:أين|اين|where)\s+" => "")
    text = replace(text, r"[\u061F\?!.]+$" => "")
    return _join_phrase(_content_tokens(text))
end

function _extract_temporal_subject(prompt::AbstractString)
    text = lowercase(String(prompt))
    text = replace(text, r"^\s*(?:متى|when)\s+" => "")
    text = replace(text, r"[\u061F\?!.]+$" => "")
    return _join_phrase(_content_tokens(text))
end

function _dialogue_kind(prompt::AbstractString)
    s = lowercase(strip(String(prompt)))
    s = replace(s, r"[؟?!.]+$" => "")
    relation_question = occursin(r"^(لماذا|كيف|هل|أين|اين|متى|ما|من)\s+", s) &&
                        occursin(r"(^|\s)(يزيد|تزيد|يحفظ|تحفظ|يفسد|تفسد|يزيل|تزيل|يبني|تبني|يهذب|تهذب)\s+", s)
    if s in ("مع السلامة", "الى اللقاء", "إلى اللقاء", "وداعا", "وداعاً", "bye", "goodbye")
        return "farewell"
    elseif occursin(r"(^|\s)(شكرا|شكراً|جزاك\s+الله\s+خيرا|جزاك\s+الله\s+خيراً|بارك\s+الله\s+فيك)(\s|$)", s)
        return "thanks"
    elseif !relation_question &&
           (s == "سلام" ||
           occursin(r"(^|\s)(السلام\s+عليكم|سلام\s+عليكم|مرحبا|مرحباً|اهلا|أهلا|صباح\s+الخير|مساء\s+الخير)(\s|$)", s)
           )
        return "greeting"
    elseif occursin(r"(^|\s)كيف\s+حالك(\s|$)", s)
        return "wellbeing"
    elseif occursin(r"(^|\s)كيف\s+تقضي\s+يومك(\s|$)", s)
        return "daily"
    elseif occursin(r"(^|\s)هل\s+تحب\s+التعلم(\s|$)", s)
        return "love_learning"
    elseif occursin(r"(^|\s)هل\s+تؤمن\s+بالمعرفة(\s|$)", s)
        return "believe_knowledge"
    elseif occursin(r"(^|\s)(ما\s+اسمك|ما\s+هو\s+اسمك|من\s+أنت|من\s+انت)(\s|$)", s)
        return "identity"
    elseif occursin(r"(^|\s)(ما\s+أجمل|ما\s+اجمل|أجمل\s+شيء|اجمل\s+شيء)(\s|$)", s)
        return "beauty"
    end
    return ""
end

function _is_dialogue_prompt(text::AbstractString)
    return !isempty(_dialogue_kind(text))
end

function _extract_dialogue_subject(prompt::AbstractString)
    kind = _dialogue_kind(prompt)
    return kind
end

function detect_response_intent(prompt::AbstractString)::ResponseIntentPlan
    text = lowercase(String(prompt))
    subject = ""; action = ""; cause = ""; result = ""
    intent = "unknown"; confidence = 0.2

    if _is_opinion_prompt(prompt)
        intent = "opinion"; confidence = 0.83
        subject = _extract_opinion_subject(prompt)
    elseif _is_dialogue_prompt(prompt)
        intent = "dialogue"; confidence = 0.86
        subject = _extract_dialogue_subject(prompt)
    elseif _is_difference_prompt(prompt)
        intent = "difference"; confidence = 0.84
        subject = _extract_difference_subject(prompt)
    elseif occursin(r"(^|\s)(ما\s+معنى|ما\s+هو|ماذا\s+يعني|what\s+is|meaning)(\s|$)", text)
        intent = "definition"; confidence = 0.85
    elseif _is_descriptive_prompt(prompt)
        intent = "descriptive"; confidence = 0.78
        subject = _extract_descriptive_subject(prompt)
    elseif occursin(r"(^|\s)(أين|اين|where)(\s|$)", text)
        intent = "spatial"; confidence = 0.80
        subject = _extract_locative_subject(prompt)
    elseif occursin(r"(^|\s)(متى|when)(\s|$)", text)
        intent = "temporal"; confidence = 0.80
        subject = _extract_temporal_subject(prompt)
    elseif occursin(r"(^|\s)(هل|is|are)(\s|$)", text)
        intent = "yesno"; confidence = 0.78
        subject = _join_phrase(_content_tokens(prompt))
    elseif occursin(r"(^|\s)(كيف|how)(\s|$)", text)
        intent = "mechanism"; confidence = 0.82
        subject, action = _extract_mechanism_parts(prompt)
    elseif occursin(r"(^|\s)(إذا|اذا|إن|ان|كلما|when|if)(\s|$)", text)
        intent = "conditional"; confidence = 0.84
        cause, result = _extract_condition_parts(prompt)
    elseif occursin(r"(^|\s)(لماذا|لما|why)(\s|$)", text)
        intent = "causal"; confidence = 0.82
        cause = _join_phrase(_content_tokens(prompt))
    end

    return ResponseIntentPlan(intent, get(RESPONSE_STRUCTURES, intent, String[]),
                              subject, action, cause, result, confidence)
end

has_plannable_response(plan::ResponseIntentPlan) =
    plan.intent in ("mechanism", "conditional", "opinion") && plan.confidence >= 0.7

function intent_gravity_profile(plan::ResponseIntentPlan)
    if plan.confidence < 0.7
        return IntentGravityProfile("", 0.0, 0.0, 1.0, 1.0, 1.0, String[], String[], 0, 0.0)
    end
    if plan.intent == "causal"
        terms = _causal_terms(plan, String[])
        cause_terms = String[t for t in split(plan.cause) if !(t in QUESTION_TOOLS) && !(t in STOPWORDS)]
        guidance = _unique_nonempty(vcat(cause_terms, terms, ["لأن", "لان", "بسبب", "ترابط", "نتيجة", "أثر"]))
        repulsion = ["لماذا", "لما", "why", "يحدث", "ما", "هو", "رأيك", "رايك",
                     "الكون", "العالم", "الارض", "الأرض", "الشمس", "القمر", "البحر"]
        return IntentGravityProfile("causal", -0.35, 0.0, 1.65, 0.42, 0.55,
                                    guidance, _unique_nonempty(repulsion),
                                    10, plan.confidence)
    end
    if plan.intent == "mechanism"
        base_terms = _unique_nonempty(vcat(_content_tokens(join([plan.subject, plan.action], " ")),
                                           _method_terms(plan, String[])))
        repulsion = ["حالي", "حالك", "بخير", "الحمد", "تمام", "شكرا", "سؤالك",
                     "خدمتك", "وعليكم", "السلام", "مرحباً", "مرحبا", "يومي"]
        return IntentGravityProfile("mechanism", -0.28, 0.0, 1.35, 0.72, 0.92,
                                    base_terms, _unique_nonempty(repulsion),
                                    14, plan.confidence)
    end
    if plan.intent == "descriptive"
        guidance = _unique_nonempty(vcat(_content_tokens(plan.subject),
                                         ["صفة", "واضح", "هادئ", "جميل", "واسع", "مشرق", "ملموس"]))
        repulsion = ["حالي", "حالك", "بخير", "خدمتك", "سؤالك", "نعم", "أحب", "احب"]
        return IntentGravityProfile("descriptive", -0.22, 0.0, 1.45, 0.76, 0.65,
                                    guidance, _unique_nonempty(repulsion),
                                    12, plan.confidence)
    end
    if plan.intent != "dialogue"
        return IntentGravityProfile("", 0.0, 0.0, 1.0, 1.0, 1.0, String[], String[], 0, 0.0)
    end
    kind = plan.subject
    terms = if kind == "greeting"
        ["وعليكم", "عليكم", "السلام", "سلام", "رحمة", "الله", "وبركاته"]
    elseif kind == "farewell"
        ["إلى", "الى", "اللقاء", "أتمنى", "اتمنى", "يوما", "يوماً", "سعيدا", "سعيداً"]
    elseif kind == "thanks"
        ["عفوا", "عفواً", "وإياك", "واياك", "وفيك", "بارك", "الله", "الشكر"]
    elseif kind == "wellbeing"
        ["حالي", "حالك", "بخير", "الحمد", "تمام", "جميلة"]
    elseif kind == "daily"
        ["أقضي", "اقضي", "يومي", "متابعة", "السؤال", "المعنى"]
    elseif kind == "love_learning"
        ["نعم", "أحب", "احب", "التعلم", "يفتح", "الفهم", "التجربة"]
    elseif kind == "believe_knowledge"
        ["نعم", "أؤمن", "اؤمن", "المعرفة", "المعرفه", "نافعة", "نافعه", "فهم", "عمل"]
    elseif kind == "beauty"
        ["أجمل", "اجمل", "الحياة", "الحياه", "يفهم", "الإنسان", "الانسان", "المعنى"]
    elseif kind == "identity"
        ["اسمي", "مرنان", "نموذج", "لغوي", "فيزيائي", "تصميم", "المبتكر", "العلمي", "باسل", "يحيى", "عبدالله"]
    else
        _unique_nonempty(vcat(_content_tokens(kind), ["القراءة", "القراءه", "أرى", "ارى", "موضوع", "نافع", "فهم", "مراجعة"]))
    end
    question_charge = if kind == "greeting"
        -0.30
    elseif kind == "farewell"
        -0.25
    elseif kind == "thanks"
        -0.25
    elseif kind == "wellbeing"
        -0.60
    elseif kind in ("love_learning", "believe_knowledge")
        -0.20
    elseif kind == "beauty"
        -0.40
    elseif kind == "identity"
        -0.45
    else
        -0.40
    end
    repulsion = ["هل", "كيف", "ما", "هو", "رأيك", "رايك", "شيء", "في", "بـ"]
    return IntentGravityProfile("dialogue", question_charge, 0.80, 1.5, 0.65, 0.60,
                                _unique_nonempty(terms), _unique_nonempty(repulsion),
                                7, plan.confidence)
end

has_gravity_profile(profile::IntentGravityProfile) =
    !isempty(profile.intent) && profile.confidence >= 0.7

function _unique_nonempty(words::Vector{String})
    out = String[]
    seen = Set{String}()
    for w in words
        s = strip(w)
        isempty(s) && continue
        s in seen && continue
        push!(out, s); push!(seen, s)
    end
    return out
end

function _method_terms(plan::ResponseIntentPlan, related_terms::Vector{String})
    if occursin("تعلم", plan.action) || occursin("يتعلم", plan.action) ||
       occursin("learn", lowercase(plan.action))
        defaults = ["الملاحظة", "التجربة", "التكرار", "المراجعة"]
    else
        defaults = ["فهم السياق", "تنظيم الخطوات", "مراجعة النتيجة"]
    end
    safe_related = String[]
    safe_keys = ("ملاحظة", "تجربة", "تكرار", "مراجعة", "فهم", "تعلم",
                 "سبب", "نتيجة", "خطوة", "مرحلة", "تحليل")
    for term in _unique_nonempty(related_terms)
        any(k -> occursin(k, term), safe_keys) && push!(safe_related, term)
    end
    terms = _unique_nonempty(vcat(defaults, safe_related))
    return terms[1:min(4, length(terms))]
end

function _opinion_terms(related_terms::Vector{String})
    safe_keys = ("معرفة", "تعلم", "فهم", "قراءة", "كتابة", "علم", "حياة", "وعي",
                 "نفع", "فائدة", "أثر", "تجربة", "مسؤولية", "حكمة", "إدراك")
    safe = String[]
    for term in _unique_nonempty(related_terms)
        any(k -> occursin(k, term), safe_keys) && push!(safe, term)
    end
    return _unique_nonempty(vcat(safe, ["الفهم", "التجربة", "المسؤولية"]))
end

function _causal_terms(plan::ResponseIntentPlan, related_terms::Vector{String})
    banned = Set(["لماذا", "لما", "why", "يزيد", "تزيد", "زاد", "تزداد",
                  "يحدث", "تحدث", "يكون", "تكون", "هو", "هي", "ما",
                  "رأيك", "رايك", "قولك"])
    terms = String[]
    for token in split(plan.cause)
        word = strip(token, [' ', '؟', '?', '.', ',', '،', ';', '؛', ':'])
        isempty(word) && continue
        word in banned && continue
        length(collect(word)) < 3 && continue
        push!(terms, word)
    end
    safe_keys = ("علم", "فهم", "معرفة", "حكمة", "تعلم", "سبب", "نتيجة",
                 "أثر", "اثر", "نور", "إدراك", "ادراك", "تجربة")
    for term in _unique_nonempty(related_terms)
        (term in banned || occursin("رأيك", term) || occursin("رايك", term)) && continue
        any(k -> occursin(k, term), safe_keys) && push!(terms, term)
    end
    return _unique_nonempty(vcat(terms, ["السبب", "الأثر", "النتيجة"]))
end

function render_planned_response(plan::ResponseIntentPlan;
                                 related_terms::Vector{String}=String[])
    if plan.intent == "dialogue"
        return ""
    elseif plan.intent == "mechanism"
        return ""
    elseif plan.intent == "conditional"
        cause = isempty(plan.cause) ? "تحقق الشرط" : plan.cause
        result = isempty(plan.result) ? "ظهرت النتيجة" : plan.result
        return "إذا $(cause)، $(result)، لأن الشرط يفتح طريق النتيجة ويربط السبب بأثره."
    elseif plan.intent == "causal"
        return ""
    elseif plan.intent == "opinion"
        subject = isempty(plan.subject) ? "هذا الموضوع" : plan.subject
        terms = _opinion_terms(related_terms)
        reason = terms[1]
        caution = get(terms, min(2, length(terms)), "التجربة")
        return "أرى أن موضوع $(subject) نافع إذا ارتبط بـ$(reason)، لأنه يساعد على توسيع الفهم. لكنه يحتاج إلى $(caution) حتى لا يبقى حكماً عاماً بلا ميزان."
    end
    return ""
end

end
