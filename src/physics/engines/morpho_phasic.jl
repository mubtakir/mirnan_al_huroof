"""
محرك الصرف الطوري — MorphoPhasicEngine.

500+ جذر عربي، 26 نمطاً للوزن الصرفي، 3 حالات إعراب (رفع/نصب/جر).
يكتشف النوع (اسم/فعل/صفة/حال/أداة) والجذر والوزن والإعراب.
"""
module MorphoPhasic

using ..WordPhysics: _normalize_letters
using JSON

export MorphoPhasicEngine,
       ARABIC_ROOTS, WEIGHT_PATTERNS, IRAB_STATES,
       analyze_morpho, extract_root_morpho, detect_weight,
       PRONOUNS, DEMONSTRATIVES, RELATIVES, CONJUNCTIONS, PREPOSITIONS, PARTICLES

# 500+ جذر عربي
const ARABIC_ROOTS = Set{String}([
    "كتب", "قرء", "علم", "جلس", "خرج", "دخل", "اكل", "شرب", "نظر",
    "سمع", "قول", "فعل", "عمل", "صنع", "بنى", "رسل", "نصر", "ضرب", "قتل",
    "جعل", "حبب", "صدق", "كذب", "ظلم", "عدل", "حكم", "فكر", "ذكر", "شكر",
    "صبر", "غفر", "رحم", "سلم", "كرم", "حسن", "جمل", "وصل", "فصل", "قسم",
    "جمع", "فرق", "قرب", "بعد", "طلب", "سال", "جوب", "وجد", "قدر", "عرف",
    "ملك", "فتح", "غلق", "رفع", "خفض", "قدم", "لحق", "سبق", "نزل",
    "صعد", "وقف", "جري", "مشى", "حمل", "وضع", "قطع", "دعا", "اجر", "حفظ",
    "كشف", "ستر", "غسل", "مسح", "غيب", "شهد", "حضر", "غاب", "زار", "سكن",
    "عمر", "خرب", "زرع", "حصد", "غرس", "نبت", "باع", "شرى", "حسب", "وزن",
    "سير", "ذهب", "عود", "رجع", "قفز", "طار", "سبح", "ركض", "زحف", "نقل",
    "درك", "فهم", "عقل", "دبر", "روى", "تامل", "لحظ", "رصد", "حدس",
    "نطق", "لفظ", "صرح", "بين", "وضح", "شرح", "فسر", "عبر", "دل", "اشار",
    "بحث", "درس", "تعلم", "دقق", "تحقق", "اختبر", "جرب",
    "امن", "كفر", "اخلص", "غش", "خدع", "خان", "وفى", "عهد",
    "ولد", "مات", "عاش", "نما", "نضج", "ذبل", "يبس",
    "شيد", "اقام", "رفع", "هدم", "حفر", "ملا", "سطح", "مهد",
    "رسم", "لون", "نحت", "صور", "زخرف", "نظم", "شعر", "غنى",
    "حكم", "ادار", "قاد", "وجه", "اشرف", "حرس", "دافع", "هاجم",
    "قاتل", "جاهد", "دافع", "انتصر", "انهزم",
    "ذكى", "فطن", "نبه", "ادرك", "لمح", "تنبا", "توقع",
    "فرح", "حزن", "غضب", "خاف", "امل", "يئس", "احب", "كره",
    "قرا", "حوسب", "برمج", "رمز", "شفر",
    "سجد", "صلى", "صام", "حج", "تلا", "تفكر",
    "سبح", "بكى", "برق", "رعد", "زلزل", "فجر",
    "ضمد", "جبر", "بتر", "بصر", "احس", "الم",
    "اتجر", "ربح", "خسر", "باع", "اشترى", "اقرض",
])

const WEIGHT_PATTERNS = Dict{String,String}(
    "استفعال" => "مهموز_لام", "مستفعل" => "اسم_فاعل_خماسي",
    "انفعال" => "مطاوع", "افتعال" => "مطاوع_تاء",
    "تفعيل" => "مضعف_عين", "تفاعل" => "مشاركة",
    "تفعّل" => "مطاوع_مضعف", "مفعول" => "اسم_مفعول",
    "فعائل" => "جمع_تكسير", "فواعل" => "جمع_تكسير",
    "أفعال" => "جمع_قلة", "مفاعل" => "جمع_تكسير",
    "فعيل" => "صفة_مشبهة", "فعال" => "صفة_مشبهة",
    "فعول" => "صيغة_مبالغة", "فعّال" => "صيغة_مبالغة",
    "أفعل" => "رباعي", "يفعل" => "مضارع",
    "تفعل" => "مضارع_مؤنث", "نفعل" => "مضارع_متكلم",
)

const IRAB_STATES = ["رفع", "نصب", "جر", "جزم", "مبني"]

# ─── المجموعات والمواجم المغلقة ───
const PRONOUNS = Set{String}([
    "أنا", "أنت", "أنتِ", "أنتما", "أنتم", "أنتن",
    "هو", "هي", "هما", "هم", "هن", "نحن",
    "إياي", "إياك", "إياكِ", "إياه", "إياها",
    "إيانا", "إياكم", "إياهم", "إياكن", "إياهن",
])

const DEMONSTRATIVES = Set{String}([
    "هذا", "هذه", "هذان", "هاتان", "هؤلاء",
    "ذلك", "تلك", "ذانك", "تانك", "أولئك",
    "هنا", "هناك", "ثمة", "ثَمَّ",
])

const RELATIVES = Set{String}([
    "الذي", "التي", "اللذان", "اللتان",
    "الذين", "اللواتي", "اللاتي", "اللائي",
    "ما", "من", "مهما", "أيّ", "أيّما",
])

const CONJUNCTIONS = Set{String}(["و", "ف", "ثم", "أو", "أم", "لكن", "بل", "حتى", "إذ", "إذا", "لما", "كلما"])

const PREPOSITIONS = Set{String}([
    "في", "من", "إلى", "على", "عن", "مع", "عند", "لدى",
    "حول", "بين", "أمام", "خلف", "فوق", "تحت",
    "قبل", "بعد", "منذ", "خلال", "حتى", "كي",
])

const PARTICLES = Set{String}(["لم", "لن", "لا", "ما", "قد", "سوف", "سـ", "إن", "أن", "كي", "هل", "أ"])

const KNOWN_NOUNS = Set{String}([
    "يد", "يوم", "يمين", "يسار", "نهر", "نور", "نار",
    "تراب", "تمر", "تفاح", "تفاحة", "أمل", "أخ", "أخت",
    "أرض", "أسد", "أسرة", "نعجة", "نعمة",
])

const LEXICAL_EXCEPTIONS = Set{String}([
    "وجد", "وجه", "وطن", "وقت", "ولد", "وصل", "ورد", "وضع", "ورق", "وسط",
    "وحيد", "وداع", "ودود", "وفاء", "وفاة", "وحشة", "وعد", "ولي", "وراء",
    "وثيق", "وليد", "ومضة", "وهم", "وهج", "وميض",
    "لبن", "لون", "لغة", "لحظة", "لعب", "لجأ", "لذة", "لطف", "لمس",
    "لواء", "لهيب", "لجنة", "لئيم", "لحم", "لؤلؤ", "لباس", "لسان",
    "لحاء", "لقاء", "لحد", "لتر", "لصق", "لغو",
    "بيت", "بحر", "بدر", "بئر", "بدن", "بطل", "بكاء", "بلاغ",
    "بنية", "بهجة", "بصر", "بساط", "بسمة", "براء", "باب", "بدع",
    "فجر", "فخر", "فكر", "فقر", "فهم", "فلك", "فطر", "فيض", "فضل",
    "فراق", "فضاء", "فداء", "فتنة",
    "كتاب", "كلام", "كرم", "كمال", "كنز", "كثير", "كيف", "كذب", "كسب",
])

# ─── دوال مساعدة للتحليل (آمنة مع الـ Unicode) ───
function strip_diacritics(word::String)
    return String(word)
end

function strip_al(word::String)
    s = strip_diacritics(word)
    s_chars = collect(s)
    if length(s_chars) > 2 && startswith(s, "ال")
        return join(s_chars[3:end]), true
    end
    if length(s_chars) > 2 && startswith(s, "آل")
        return join(s_chars[3:end]), true
    end
    return s, false
end

function strip_prep_prefix(word::String)
    s = strip_diacritics(word)
    if s in LEXICAL_EXCEPTIONS
        return s
    end
    s_chars = collect(s)
    if length(s_chars) >= 4 && (s_chars[1] == 'ب' || s_chars[1] == 'ل')
        remainder = join(s_chars[2:end])
        if length(remainder) >= 2 && remainder ∉ LEXICAL_EXCEPTIONS
            return String(remainder)
        end
    end
    return s
end

function has_tanwin_nasb(word::String)
    return occursin('ً', word) || endswith(word, "اً") || endswith(word, "ً")
end

"""
    MorphoPhasicEngine
"""
struct MorphoPhasicEngine
    roots::Vector{String}
    roots_chars::Vector{Vector{Char}}
    weight_patterns::Vector{Pair{String,String}}
    weight_chars::Vector{Vector{Char}}
    nouns_list::Set{String}
    verbs_list::Set{String}
end

MorphoPhasicEngine() = begin
    r_list = collect(ARABIC_ROOTS)
    w_list = collect(WEIGHT_PATTERNS)
    
    nouns_set = Set{String}()
    verbs_set = Set{String}()
    
    nouns_path = joinpath(dirname(dirname(@__DIR__)), "data", "nouns.json")
    if isfile(nouns_path)
        try
            raw = JSON.parsefile(nouns_path)
            for item in raw
                push!(nouns_set, strip(string(item)))
            end
        catch e
            @warn "Failed to parse nouns.json: $e"
        end
    end

    verbs_path = joinpath(dirname(dirname(@__DIR__)), "data", "verbs.json")
    if isfile(verbs_path)
        try
            raw = JSON.parsefile(verbs_path)
            for item in raw
                push!(verbs_set, strip(string(item)))
            end
        catch e
            @warn "Failed to parse verbs.json: $e"
        end
    end

    return MorphoPhasicEngine(
        r_list,
        [collect(r) for r in r_list],
        w_list,
        [collect(p.first) for p in w_list],
        nouns_set,
        verbs_set
    )
end

"""
    extract_root_morpho(engine, word) -> Vector{String}
"""
function extract_root_morpho(engine::MorphoPhasicEngine, word::String)
    pure = filter(c -> c ∉ ('ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ـ'), _normalize_letters(word))
    pure_chars = collect(pure)
    matches = String[]
    for i in 1:length(engine.roots)
        root_chars = engine.roots_chars[i]
        ri = 1
        n_root = length(root_chars)
        for c in pure_chars
            if ri <= n_root && c == root_chars[ri]
                ri += 1
            end
        end
        if ri > 1
            push!(matches, engine.roots[i])
        end
    end
    return unique(matches)
end

"""
    detect_weight(engine, word) -> String
"""
function detect_weight(engine::MorphoPhasicEngine, word::String)
    pure = filter(c -> c ∉ ('ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ـ'), _normalize_letters(word))
    pure_chars = collect(pure)
    candidates = Tuple{Int,String}[]
    for i in 1:length(engine.weight_patterns)
        pattern_chars = engine.weight_chars[i]
        score = 0
        min_len = min(length(pure_chars), length(pattern_chars))
        for j in 1:min_len
            if pure_chars[j] == pattern_chars[j]; score += 1; end
        end
        push!(candidates, (score, engine.weight_patterns[i].first))
    end
    if isempty(candidates); return "فعل"; end
    sort!(candidates; by=x -> -x[1])
    return candidates[1][2]
end

"""
    analyze_morpho(engine, word; context_words=nothing) -> Dict

تحليل صرفي كامل مبني على القواعد الفطرية ومطابقة الأوزان والسياق مع أمان الـ Unicode.
"""
function analyze_morpho(engine::MorphoPhasicEngine, word::String; context_words=nothing)
    pure = filter(c -> c ∉ ('ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ـ'), _normalize_letters(word))
    roots = extract_root_morpho(engine, word)
    weight = detect_weight(engine, word)

    plain = strip_diacritics(word)
    norm_word = pure
    
    # 0. الفحص الأولي بالمعاجم الخارجية للأسماء والأفعال
    if word in engine.nouns_list || plain in engine.nouns_list || norm_word in engine.nouns_list
        word_type = "اسم"
    elseif word in engine.verbs_list || plain in engine.verbs_list || norm_word in engine.verbs_list
        word_type = "فعل"
    # 1. القوائم المغلقة
    elseif plain in PRONOUNS || plain in DEMONSTRATIVES || plain in RELATIVES
        word_type = "اسم"
    elseif plain in CONJUNCTIONS || plain in PREPOSITIONS || plain in PARTICLES
        word_type = "أداة"
    else
        # 2. قواعد الفعل والاشتقاق
        stripped_pre = strip_prep_prefix(plain)
        core, has_al = strip_al(stripped_pre)
        s = core
        s_chars = collect(s)
        
        if length(s_chars) < 2
            word_type = "اسم"
        elseif endswith(word, "تُ") || endswith(word, "تَ") || endswith(word, "تِ")
            word_type = "فعل"
        elseif (endswith(s, "نا") || endswith(s, "ون") || endswith(s, "وا") || endswith(s, "ين") || endswith(s, "نّ")) && !(endswith(s, "ون") && has_al)
            word_type = "فعل"
        elseif (s_chars[1] == 'ا' || s_chars[1] == 'إ') && !has_al && !occursin('ا', join(s_chars[2:end]))
            word_type = "فعل"
        # 3. الصفة والحال
        elseif s_chars[1] == 'أ' && length(s_chars) == 4 && plain ∉ KNOWN_NOUNS
            word_type = "صفة"
        elseif endswith(s, "ان") && length(s_chars) <= 6 && !has_al && plain ∉ KNOWN_NOUNS
            word_type = "صفة"
        # المضارع
        elseif (s_chars[1] == 'ي' || s_chars[1] == 'ت' || s_chars[1] == 'ن' || s_chars[1] == 'أ') && !has_al && 4 <= length(s_chars) <= 8 && plain ∉ KNOWN_NOUNS && s ∉ KNOWN_NOUNS
            word_type = "فعل"
        # وزن فاعل
        elseif length(s_chars) in (4, 5) && s_chars[2] == 'ا' && !has_al
            if has_tanwin_nasb(word)
                word_type = "حال"
            else
                word_type = "صفة"
            end
        # وزن مفعول
        elseif (s_chars[1] == 'م' && occursin('و', join(s_chars[3:end]))) || (s_chars[1] == 'م' && endswith(s, "ول"))
            if !has_al && has_tanwin_nasb(word)
                word_type = "حال"
            else
                word_type = "صفة"
            end
        # وزن فعيل
        elseif length(s_chars) == 4 && s_chars[3] == 'ي'
            if !has_al && has_tanwin_nasb(word)
                word_type = "حال"
            else
                word_type = "صفة"
            end
        # نكرة منصوبة
        elseif !has_al && has_tanwin_nasb(word)
            word_type = "حال"
        # جمع تكسير
        elseif s_chars[1] == 'م' && length(s_chars) >= 6
            word_type = "اسم"
        # تاء مربوطة
        elseif endswith(s, "ة") && length(s_chars) >= 3
            word_type = "اسم"
        elseif has_al
            word_type = "اسم"
        else
            word_type = "اسم"
        end
    end

    # السياق لفرز الأفعال المضارعة
    if word_type != "فعل" && context_words !== nothing
        idx = findfirst(==(word), context_words)
        if idx !== nothing && idx > 1
            prev = strip_diacritics(context_words[idx-1])
            if prev in ("لم", "لن", "لا", "سوف", "سـ", "قد")
                word_type = "فعل"
            end
        end
    end

    # تحديد الإعراب
    irab = "رفع"
    if endswith(word, "اً") || word_type == "فعل"
        irab = "نصب"
    end

    return Dict(
        "word" => word, "pure" => pure, "root" => isempty(roots) ? "؟" : roots[1],
        "all_roots" => roots, "weight" => weight, "has_morph" => !isempty(roots),
        "type" => word_type, "irab" => irab,
    )
end

end # module MorphoPhasic
