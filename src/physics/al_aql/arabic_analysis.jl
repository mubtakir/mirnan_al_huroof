module ArabicAnalysis

export MorphAnalysis, TemporalInfo, BinaryRelation, InferredEvent, EntityKnowledge,
       AdvancedKnowledgeBase, strip_diacritics, strip_al, strip_prep_prefix,
       has_tanwin_nasb, tokenize_arabic, analyze_arabic_word, analyze_arabic_sentence,
       segment_sentences, deep_understand, get_or_create_entity!,
       add_relation!, add_role!, add_temporal_info!, get_entity_relations,
       get_entities_with_attribute, find_relations_between

struct MorphAnalysis
    word::String
    kind::String
    normalized::String
    core::String
    has_al::Bool
    reasons::Vector{String}
end

struct TemporalInfo
    expression::String
    normalized::String
    kind::String
    precision::String
end

struct BinaryRelation
    source::String
    target::String
    relation_type::String
    evidence::String
    confidence::Float64
end

struct InferredEvent
    description::String
    actor::Union{Nothing,String}
    location::Union{Nothing,String}
    time::Union{Nothing,TemporalInfo}
    participants::Vector{String}
end

mutable struct EntityKnowledge
    name::String
    attributes::Vector{String}
    capabilities::Vector{String}
    roles::Vector{String}
    workplace::Union{Nothing,String}
    locations::Vector{String}
    time_slots::Vector{TemporalInfo}
    events::Vector{InferredEvent}
    relations_as_source::Vector{BinaryRelation}
    relations_as_target::Vector{BinaryRelation}
    inferred_relations::Vector{String}
end

EntityKnowledge(name::String) =
    EntityKnowledge(name, String[], String[], String[], nothing, String[],
                    TemporalInfo[], InferredEvent[], BinaryRelation[],
                    BinaryRelation[], String[])

mutable struct AdvancedKnowledgeBase
    entities::Dict{String,EntityKnowledge}
    events::Vector{InferredEvent}
    relations::Vector{BinaryRelation}
    temporal_expressions::Vector{TemporalInfo}
    inference_log::Vector{Dict{String,Any}}
end

AdvancedKnowledgeBase() =
    AdvancedKnowledgeBase(Dict{String,EntityKnowledge}(), InferredEvent[],
                          BinaryRelation[], TemporalInfo[], Dict{String,Any}[])

const PRONOUNS = Set(["أنا", "أنت", "أنتِ", "أنتما", "أنتم", "أنتن",
    "هو", "هي", "هما", "هم", "هن", "نحن", "إياي", "إياك", "إياكِ",
    "إياه", "إياها", "إيانا", "إياكم", "إياهم", "إياكن", "إياهن"])

const DEMONSTRATIVES = Set(["هذا", "هذه", "هذان", "هاتان", "هؤلاء",
    "ذلك", "تلك", "ذانك", "تانك", "أولئك", "هنا", "هناك", "ثمة", "ثَمَّ"])

const RELATIVES = Set(["الذي", "التي", "اللذان", "اللتان", "الذين",
    "اللواتي", "اللاتي", "اللائي", "ما", "من", "مهما", "أيّ", "أيّما"])

const CONJUNCTIONS = Set(["و", "ف", "ثم", "أو", "أم", "لكن", "بل", "حتى", "إذ", "إذا", "لما", "كلما"])
const PREPOSITIONS = Set(["في", "من", "إلى", "الى", "على", "عن", "مع", "عند", "لدى",
    "حول", "بين", "أمام", "خلف", "فوق", "تحت", "قبل", "بعد", "منذ", "خلال", "حتى", "كي"])
const PARTICLES = Set(["لم", "لن", "لا", "ما", "قد", "سوف", "سـ", "إن", "أن", "كي", "هل", "أ"])
const FUNCTION_WORDS = union(CONJUNCTIONS, PREPOSITIONS, PARTICLES)

const KNOWN_NOUNS = Set(["يد", "يوم", "يمين", "يسار", "نهر", "نور", "نار",
    "تراب", "تمر", "تفاح", "تفاحة", "أمل", "أخ", "أخت", "أرض", "أسد",
    "أسرة", "نعجة", "نعمة"])

const LEXICAL_EXCEPTIONS = Set(["وجد", "وجه", "وطن", "وقت", "ولد", "وصل",
    "ورد", "وضع", "ورق", "وسط", "وحيد", "وداع", "ودود", "وفاء", "وفاة",
    "وحشة", "وعد", "ولي", "وراء", "وثيق", "وليد", "ومضة", "وهم", "وهج",
    "وميض", "لبن", "لون", "لغة", "لحظة", "لعب", "لجأ", "لذة", "لطف",
    "لمس", "لواء", "لهيب", "لجنة", "لئيم", "لحم", "لؤلؤ", "لباس", "لسان",
    "لحاء", "لقاء", "لحد", "لتر", "لصق", "لغو", "بيت", "بحر", "بدر",
    "برق", "بئر", "بدن", "بطل", "بكاء", "بلاغ", "بنية", "بهجة", "بصر",
    "بساط", "بسمة", "براء", "باب", "بدع", "فجر", "فخر", "فكر", "فقر",
    "فهم", "فلك", "فطر", "فيض", "فضل", "فراق", "فضاء", "فداء", "فتنة",
    "كتاب", "كلام", "كرم", "كمال", "كنز", "كثير", "كيف", "كذب", "كسب"])

strip_diacritics(word::AbstractString) = String(word)

function _drop_prefix(s::AbstractString, prefix::String)
    s = String(s)
    startswith(s, prefix) || return s
    idx = firstindex(s)
    for _ in 1:length(prefix)
        idx = nextind(s, idx)
    end
    return s[idx:end]
end

function strip_al(word::AbstractString)
    s = strip_diacritics(word)
    if startswith(s, "ال") && length(s) > 2
        return _drop_prefix(s, "ال"), true
    elseif startswith(s, "آل") && length(s) > 3
        return _drop_prefix(s, "آل"), true
    end
    return s, false
end

function strip_prep_prefix(word::AbstractString; known_vocab::Union{Nothing,Set{String}}=nothing)
    s = strip_diacritics(word)
    s in LEXICAL_EXCEPTIONS && return s
    known_vocab !== nothing && s in known_vocab && return s
    if length(s) >= 4 && first(s) in ('ب', 'ل')
        remainder = s[nextind(s, firstindex(s)):end]
        if length(remainder) >= 2 && !(remainder in LEXICAL_EXCEPTIONS)
            if !(known_vocab !== nothing && !(remainder in known_vocab) && length(remainder) < 3)
                return remainder
            end
        end
    end
    return s
end

has_tanwin_nasb(word::AbstractString) =
    occursin('\u064B', String(word)) || endswith(String(word), "اً") || endswith(String(word), "ً")

tokenize_arabic(text::AbstractString) =
    [String(w) for w in split(strip(String(text))) if !isempty(w)]

function analyze_word(word::AbstractString; context_words::Union{Nothing,Vector{String}}=nothing)
    original = String(word)
    reasons = String[]
    plain = strip_diacritics(original)
    stripped_pre = strip_prep_prefix(plain)
    core, has_al = strip_al(stripped_pre)
    s = core

    if isempty(s) || length(s) < 2
        return MorphAnalysis(original, "غير معروف", plain, s, has_al, ["الكلمة قصيرة جدا أو فارغة"])
    end

    for (lst, label) in ((PRONOUNS, "ضمير منفصل"), (DEMONSTRATIVES, "اسم إشارة"), (RELATIVES, "اسم موصول"))
        if plain in lst || s in lst
            push!(reasons, "$label من القوائم المغلقة -> اسم")
            return MorphAnalysis(original, "اسم", plain, s, has_al, reasons)
        end
    end

    if plain in FUNCTION_WORDS
        push!(reasons, "حرف أو أداة وظيفية")
        return MorphAnalysis(original, "أداة", plain, s, has_al, reasons)
    end

    if occursin(r"[تط][\u064F\u064E\u0650]$", original)
        push!(reasons, "اتصلت بها تاء الفاعل المتحركة -> فعل")
        return MorphAnalysis(original, "فعل", plain, s, has_al, reasons)
    end

    verb_suffix_rules = [
        (r"نا$", "نا الفاعلين"),
        (r"نَ$|نَّ$", "نون النسوة"),
        (r"ون$|وا$", "واو الجماعة"),
        (r"ين$", "ياء المخاطبة"),
        (r"نَّ$", "نون التوكيد الثقيلة"),
        (r"نْ$", "نون التوكيد الخفيفة"),
    ]
    for (pat, label) in verb_suffix_rules
        if occursin(pat, s)
            if label == "واو الجماعة" && occursin(r"ون$", s) && has_al
                break
            end
            push!(reasons, "اتصلت بها $label -> فعل")
            return MorphAnalysis(original, "فعل", plain, s, has_al, reasons)
        end
    end

    if context_words !== nothing
        idx = findfirst(w -> strip_diacritics(w) == plain, context_words)
        if idx !== nothing && idx > 1
            prev = strip_diacritics(context_words[idx - 1])
            if prev in Set(["لم", "لن", "لا"])
                push!(reasons, "سبقتها أداة تختص بالفعل المضارع -> فعل")
                return MorphAnalysis(original, "فعل", plain, s, has_al, reasons)
            elseif prev in Set(["سوف", "سـ", "قد"])
                push!(reasons, "سبقتها أداة خاصة بالفعل -> فعل")
                return MorphAnalysis(original, "فعل", plain, s, has_al, reasons)
            end
        end
    end

    if occursin(r"^[اإ][\u0600-\u06FF]{2,3}$", s) && !has_al && !occursin("ا", s[nextind(s, firstindex(s)):end])
        push!(reasons, "تبدأ بهمزة وصل وعلى وزن افعل -> فعل أمر")
        return MorphAnalysis(original, "فعل", plain, s, has_al, reasons)
    end

    if occursin(r"^أ[\u0600-\u06FF]{3}$", s) && length(s) == 4
        push!(reasons, "على وزن أفعل يدل على لون أو عيب -> صفة")
        return MorphAnalysis(original, "صفة", plain, s, has_al, reasons)
    end

    if occursin(r"^[\u0600-\u06FF]{2,4}ان$", s) && length(s) <= 6 && !has_al
        push!(reasons, "على وزن فعلان -> صفة")
        return MorphAnalysis(original, "صفة", plain, s, has_al, reasons)
    end

    if occursin(r"^[يتنأ][\u0600-\u06FF]{2,}$", s) && !has_al &&
       4 <= length(s) <= 8 && !(plain in KNOWN_NOUNS) && !(s in KNOWN_NOUNS)
        push!(reasons, "تبدأ بحرف مضارعة وبطول مناسب -> فعل")
        return MorphAnalysis(original, "فعل", plain, s, has_al, reasons)
    end

    if occursin(r"^[\u0600-\u06FF]ا[\u0600-\u06FF]{1,2}$", s) && length(s) in (4, 5)
        push!(reasons, "على وزن فاعل -> صفة")
        return MorphAnalysis(original, "صفة", plain, s, has_al, reasons)
    end

    is_maf3ul = occursin(r"^م[\u0600-\u06FF]{1,2}و[\u0600-\u06FF]$", s) ||
                occursin(r"^م[\u0600-\u06FF]+ول$", s)
    if is_maf3ul
        if !has_al && has_tanwin_nasb(original)
            push!(reasons, "اسم مفعول نكرة منصوب -> حال")
            return MorphAnalysis(original, "حال", plain, s, has_al, reasons)
        end
        push!(reasons, "اسم مفعول -> صفة")
        return MorphAnalysis(original, "صفة", plain, s, has_al, reasons)
    end

    if occursin(r"^[\u0600-\u06FF][\u0600-\u06FF]ي[\u0600-\u06FF]$", s) && length(s) == 4
        if !has_al && has_tanwin_nasb(original)
            push!(reasons, "على وزن فعيل نكرة منصوب -> حال")
            return MorphAnalysis(original, "حال", plain, s, has_al, reasons)
        end
        push!(reasons, "على وزن فعيل -> صفة")
        return MorphAnalysis(original, "صفة", plain, s, has_al, reasons)
    end

    if !has_al && has_tanwin_nasb(original)
        push!(reasons, "نكرة منصوبة بتنوين النصب -> حال محتملة")
        return MorphAnalysis(original, "حال", plain, s, has_al, reasons)
    end

    if occursin(r"^م[\u0600-\u06FF]{4,}$", s) && length(s) >= 6
        push!(reasons, "على وزن مفاعل/مفاعيل -> اسم")
        return MorphAnalysis(original, "اسم", plain, s, has_al, reasons)
    end

    if endswith(s, "ة") && length(s) >= 3 && !occursin(r"^[\u0600-\u06FF]ا[\u0600-\u06FF]ة$", s)
        push!(reasons, "تنتهي بتاء مربوطة -> اسم")
        return MorphAnalysis(original, "اسم", plain, s, has_al, reasons)
    end

    if has_al
        push!(reasons, "معرفة بأل ولم تنطبق أوزان الصفة أو الفعل -> اسم")
        return MorphAnalysis(original, "اسم", plain, s, has_al, reasons)
    end

    push!(reasons, "الأصل اسم")
    return MorphAnalysis(original, "اسم", plain, s, has_al, reasons)
end

function analyze_sentence(text::AbstractString)
    words = tokenize_arabic(text)
    return [analyze_word(w; context_words=words) for w in words]
end

analyze_arabic_word(word::AbstractString; context_words::Union{Nothing,Vector{String}}=nothing) =
    analyze_word(word; context_words=context_words)

analyze_arabic_sentence(text::AbstractString) = analyze_sentence(text)

function _clean_name(name::AbstractString)
    s = strip(String(name))
    if startswith(s, "ال") && length(s) > 3
        return _drop_prefix(s, "ال")
    end
    return s
end

function get_or_create_entity!(kb::AdvancedKnowledgeBase, name::AbstractString)
    clean = _clean_name(name)
    if !haskey(kb.entities, clean)
        kb.entities[clean] = EntityKnowledge(String(strip(String(name))))
    end
    return kb.entities[clean]
end

function _push_unique!(xs::Vector{String}, value::AbstractString)
    text = String(value)
    isempty(text) && return xs
    text in xs || push!(xs, text)
    return xs
end

function add_temporal_info!(kb::AdvancedKnowledgeBase, entity_name::AbstractString,
                            temporal::TemporalInfo, pattern_name::AbstractString)
    entity = get_or_create_entity!(kb, entity_name)
    temporal in entity.time_slots || push!(entity.time_slots, temporal)
    temporal in kb.temporal_expressions || push!(kb.temporal_expressions, temporal)
    push!(kb.inference_log, Dict{String,Any}(
        "type" => "temporal", "entity" => _clean_name(entity_name),
        "value" => temporal.expression, "normalized" => temporal.normalized,
        "pattern" => String(pattern_name)))
    return temporal
end

function add_relation!(kb::AdvancedKnowledgeBase, source::AbstractString, target::AbstractString,
                       relation_type::AbstractString, evidence::AbstractString, confidence::Real)
    rel = BinaryRelation(_clean_name(source), _clean_name(target), String(relation_type),
                         String(evidence), Float64(confidence))
    push!(kb.relations, rel)
    push!(get_or_create_entity!(kb, source).relations_as_source, rel)
    push!(get_or_create_entity!(kb, target).relations_as_target, rel)
    push!(kb.inference_log, Dict{String,Any}(
        "type" => "relation", "source" => rel.source, "target" => rel.target,
        "relation" => rel.relation_type, "confidence" => rel.confidence,
        "evidence" => rel.evidence))
    return rel
end

function add_role!(kb::AdvancedKnowledgeBase, entity_name::AbstractString,
                   role::AbstractString, pattern_name::AbstractString)
    entity = get_or_create_entity!(kb, entity_name)
    _push_unique!(entity.roles, strip(String(role)))
    push!(kb.inference_log, Dict{String,Any}(
        "type" => "role", "entity" => _clean_name(entity_name),
        "value" => strip(String(role)), "pattern" => String(pattern_name)))
    return entity
end

get_entity_relations(kb::AdvancedKnowledgeBase, entity_name::AbstractString) =
    haskey(kb.entities, _clean_name(entity_name)) ?
    vcat(kb.entities[_clean_name(entity_name)].relations_as_source,
         kb.entities[_clean_name(entity_name)].relations_as_target) : BinaryRelation[]

get_entities_with_attribute(kb::AdvancedKnowledgeBase, attribute::AbstractString) =
    [name for (name, entity) in kb.entities if String(attribute) in entity.attributes]

function find_relations_between(kb::AdvancedKnowledgeBase, entity1::AbstractString, entity2::AbstractString)
    c1, c2 = _clean_name(entity1), _clean_name(entity2)
    return [rel for rel in kb.relations
            if (rel.source == c1 && rel.target == c2) || (rel.source == c2 && rel.target == c1)]
end

function segment_sentences(text::AbstractString)
    raw = split(String(text), r"(?<=[.!؟؛])\s+|\n+")
    return [strip(s) for s in raw if !isempty(strip(s))]
end

function _normalize_temporal(expression::AbstractString, kind::AbstractString)
    time_map = Dict(
        "الصباح" => "morning", "صباحاً" => "morning", "صباحا" => "morning",
        "المساء" => "evening", "مساءً" => "evening", "مساء" => "evening",
        "الليل" => "night", "ليلاً" => "night", "ليلا" => "night",
        "الظهر" => "noon", "ظهراً" => "noon", "العصر" => "afternoon",
        "عصراً" => "afternoon", "الفجر" => "dawn", "فجراً" => "dawn",
        "العصر العباسي" => "Abbasid Era (750-1258 CE)",
        "العصر الأموي" => "Umayyad Era (661-750 CE)",
        "العصر العثماني" => "Ottoman Era (1299-1922 CE)",
        "العصر الجاهلي" => "Pre-Islamic Era",
    )
    expr = String(expression)
    kind_text = String(kind)
    return TemporalInfo(expr, get(time_map, expr, expr), kind_text, "approximate")
end

const ROLE_WORDS = "طبيب|مهندس|قاض|معلم|تاجر|نجار|حداد|خياط|فلاح|صياد|كاتب|شاعر|فقيه|قارئ|أمير|وزير|ملك|خليفة|قائد|زعيم|شيخ|أستاذ"

function _capture(m::RegexMatch, name::Symbol)
    value = m[name]
    return value === nothing ? "" : String(value)
end

function _analyze_inference_sentence!(kb::AdvancedKnowledgeBase, sentence::AbstractString)
    sentence_text = String(sentence)
    results = Dict{String,Any}[]

    temporal_patterns = [
        (r"(?:في\s+)?(?<time>الصباح|المساء|الظهر|العصر|الليل|الفجر|الغروب|صباحاً|مساءً|ظهراً|عصراً|ليلاً|فجراً|غروباً)", "time_of_day", 0.95),
        (r"(?:في|خلال|إبان)\s+(?:ال)?(?<time>العصر\s+(?:العباسي|الأموي|العثماني|الجاهلي|الإسلامي|الحديث|القديم|الوسيط))", "historical_period", 0.85),
        (r"(?:في|خلال|سنة|عام)\s+(?<time>\d{1,4}\s*(?:هـ|ق\.م|م)?)", "date", 0.90),
    ]
    for (pat, kind, confidence) in temporal_patterns
        m = match(pat, sentence_text)
        if m !== nothing
            time_expr = _capture(m, :time)
            temporal = _normalize_temporal(time_expr, kind)
            add_temporal_info!(kb, "الراوي", temporal, kind)
            push!(results, Dict{String,Any}("type" => "temporal", "sub_type" => kind,
                                            "confidence" => confidence, "value" => time_expr))
        end
    end

    relation_patterns = [
        (r"(?<entity1>[\u0621-\u063A\u0641-\u064A]+)\s+(?:هو|كان|يكون)\s+(?:ابن|أخو|أبو|عم|خال|جد|زوج|ابن\s+عم)\s+(?<entity2>[\u0621-\u063A\u0641-\u064A]+)", "قرابة", 0.90),
        (r"(?<entity2>[\u0621-\u063A\u0641-\u064A]+)\s+(?:ابن|أخو|أبو|عم|خال|جد|زوج)\s+(?<entity1>[\u0621-\u063A\u0641-\u064A]+)", "قرابة", 0.85),
        (r"(?:تعلم|درس|تتلمذ)\s+(?<entity1>[\u0621-\u063A\u0641-\u064A]+)\s+(?:على\s+يد|عند)\s+(?<entity2>[\u0621-\u063A\u0641-\u064A]+)", "تعليم", 0.85),
        (r"(?<entity2>[\u0621-\u063A\u0641-\u064A]+)\s+(?:علم|درب|لقن)\s+(?<entity1>[\u0621-\u063A\u0641-\u064A]+)", "تعليم", 0.85),
        (r"(?<entity1>[\u0621-\u063A\u0641-\u064A]+)\s+(?:يملك|يمتلك|لديه|عنده|له)\s+(?<entity2>[\u0621-\u063A\u0641-\u064A]+(?:ة|ات)?)", "ملكية", 0.80),
        (r"(?<entity1>[\u0621-\u063A\u0641-\u064A]+)\s+(?:صديق|صاحب|رفيق|جليس)\s+(?<entity2>[\u0621-\u063A\u0641-\u064A]+)", "صداقة", 0.80),
        (r"(?<entity1>[\u0621-\u063A\u0641-\u064A]+)\s+(?:عدو|خصم|غريم)\s+(?<entity2>[\u0621-\u063A\u0641-\u064A]+)", "عداوة", 0.80),
        (r"(?<entity1>[\u0621-\u063A\u0641-\u064A]+)\s+(?:في|داخل|بـ)\s+(?<entity2>[\u0621-\u063A\u0641-\u064A]+(?:ة|ات)?(?:ه|ها)?)", "موقع", 0.75),
    ]
    for (pat, relation_type, confidence) in relation_patterns
        m = match(pat, sentence_text)
        if m !== nothing
            e1, e2 = _capture(m, :entity1), _capture(m, :entity2)
            if !isempty(e1) && !isempty(e2)
                add_relation!(kb, e1, e2, relation_type, sentence_text, confidence)
                push!(results, Dict{String,Any}("type" => "relation", "relation" => relation_type,
                                                "source" => e1, "target" => e2,
                                                "confidence" => confidence))
            end
        end
    end

    role_patterns = [
        Regex("(?<entity>[\\u0621-\\u063A\\u0641-\\u064A]+)\\s+(?:هو|كان|يكون)\\s+(?:ال)?(?<role>$ROLE_WORDS)"),
        Regex("(?:ال)?(?<role>$ROLE_WORDS)\\s+(?<entity>[\\u0621-\\u063A\\u0641-\\u064A]+)"),
    ]
    for (idx, pat) in enumerate(role_patterns)
        m = match(pat, sentence_text)
        if m !== nothing
            entity, role = _capture(m, :entity), _capture(m, :role)
            if !isempty(entity) && !isempty(role)
                add_role!(kb, entity, role, "role_$idx")
                push!(results, Dict{String,Any}("type" => "role", "entity" => entity,
                                                "role" => role, "confidence" => idx == 1 ? 0.90 : 0.85))
            end
        end
    end

    return results
end

function deep_understand(text::AbstractString)
    kb = AdvancedKnowledgeBase()
    for sentence in segment_sentences(text)
        analyses = analyze_sentence(sentence)
        for a in analyses
            if a.kind in ("اسم", "صفة")
                get_or_create_entity!(kb, a.word)
            end
        end
        _analyze_inference_sentence!(kb, sentence)
    end
    return kb
end

end # module ArabicAnalysis
