"""
محرك الحوار — Dialogue Engine.

يفضّل مسارات الحوار عبر:
- عقوبة anti-completion (منع تكرار كلمات المستخدم)
- مكافأة بدء الجملة (sentence starters)
- عقوبة التكرار الموضعي
- كشف الحاجة للحوار عبر intent detection
"""
module DialogueEngineFull

using LinearAlgebra

export DialogueEngine, SENTENCE_STARTERS

const SENTENCE_STARTERS = Set{String}([
    "إن", "قد", "لقد", "سوف", "هل", "ما", "من", "هذا", "هذه",
    "ذلك", "تلك", "هناك", "هنا", "عندما", "حيث", "بينما", "ربما",
    "كان", "كانت", "ليس", "يكون", "تكون", "أصبح", "يجب", "يمكن",
    "لا", "لن", "لم", "إذا", "لو", "كل", "بعض", "نفس", "ذات",
    "في", "على", "عن", "منذ", "حتى", "ثم", "أو", "بل",
    "the", "a", "an", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they",
    "there", "here", "what", "where", "when", "why", "how",
    "وعليكم", "أهلا", "مرحبا", "بالطبع", "حسناً", "نعم", "بالتأكيد",
])

"""
    DialogueEngine

محرك حوار — anti-completion + sentence starter bonus + repetition penalty.

الحقول:
- `intent_detector`: كاشف القصد (اختياري)
- `associative_memory`: ذاكرة ترابطية (اختياري)
- `anti_completion_weight`: شدة عقوبة استكمال كلمة المستخدم
- `starter_bonus`: مكافأة بدء جملة جديدة
"""
mutable struct DialogueEngine
    intent_detector::Any
    associative_memory::Any
    anti_completion_weight::Float64
    starter_bonus::Float64
end

DialogueEngine(; intent_detector=nothing, associative_memory=nothing,
               anti_completion_weight::Float64=3.0, starter_bonus::Float64=3.0) =
    DialogueEngine(intent_detector, associative_memory, anti_completion_weight, starter_bonus)

"""
    compute_dialogue_score(engine, candidate_pv, candidate_word;
                           intent_name="STATEMENT", last_user_pv=nothing,
                           context_words=nothing) -> Float64

حساب درجة الحوار لمرشح:
- عقوبة anti-completion (إذا تشابه المرشح مع كلمة المستخدم > 0.97)
- مكافأة بدء الجملة
- عقوبة التكرار الموضعي
"""
function compute_dialogue_score(engine::DialogueEngine,
                                candidate_pv::AbstractVector,
                                candidate_word::String;
                                intent_name::String="STATEMENT",
                                last_user_pv::Union{AbstractVector,Nothing}=nothing,
                                context_words::Union{Vector{String},Nothing}=nothing)
    score = 0.0

    # عقوبة anti-completion
    if last_user_pv !== nothing
        c_norm = norm(candidate_pv)
        u_norm = norm(last_user_pv)
        if c_norm > 1e-10 && u_norm > 1e-10
            continuation_sim = dot(candidate_pv, last_user_pv) / (c_norm * u_norm)
            if continuation_sim > 0.97
                factor = (continuation_sim - 0.97) / 0.03
                score -= engine.anti_completion_weight * factor
            end
        end
    end

    # مكافأة بدء جملة
    if candidate_word in SENTENCE_STARTERS
        score += engine.starter_bonus
    end

    # عقوبة التكرار الموضعي
    if context_words !== nothing && length(context_words) >= 4
        if candidate_word in context_words[end-3:end]
            score -= 2.0
        end
    end

    return score
end

"""
    detect_need_for_dialogue(engine, user_text) -> Tuple{Bool,String,Float64}

هل يحتاج النص إلى رد حواري؟ (تحية، سؤال، أمر، طلب...)
"""
function detect_need_for_dialogue(engine::DialogueEngine, user_text::String)
    if engine.intent_detector === nothing
        return false, "STATEMENT", 0.0
    end
    result = engine.intent_detector.detect(user_text)
    intent = result["intent"]
    confidence = result["confidence"]
    is_dialogue = confidence > 0.5 && intent in (
        "GREETING", "QUESTION", "COMMAND", "REQUEST",
        "FAREWELL", "OPINION", "SUGGESTION", "COMPLAINT", "THANK")
    return is_dialogue, intent, confidence
end

"""
    should_stop_after_sentence(words) -> Bool

هل يجب التوقف بعد هذه الكلمات؟ (علامات ترقيم، أدعية...)
"""
function should_stop_after_sentence(words::Vector{String})
    if isempty(words)
        return false
    end
    if length(words) >= 2 && last(words) in (".", "!", "؟", "?", "…")
        return true
    end
    if length(words) >= 3 && words[end-1] == "الله"
        return true
    end
    return false
end

end # module DialogueEngineFull
