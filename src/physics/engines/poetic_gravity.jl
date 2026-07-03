"""
محرك الجاذبية الشعرية — Poetic Gravity Engine.

البحور الـ 16: الطويل، المديد، البسيط، الوافر، الكامل، الهزج،
الرجز، الرمل، السريع، المنسرح، الخفيف، المضارع، المقتضب،
المجتث، المتقارب، المتدارك.

يدعم: تقطيع عربي، عد مقاطع إنجليزية، كشف البحر، قافية.
"""
module PoeticGravity

export PoeticGravityEngine, ARABIC_METERS, syllabify_arabic,
       count_syllables_english, detect_meter, rhyme_score

# ═══ البحور العربية الـ 16 ═══
const ARABIC_METERS = Dict{String,String}(
    "طويل"   => "فعولن مفاعيلن فعولن مفاعيلن",
    "مديد"   => "فاعلاتن فاعلن فاعلاتن",
    "بسيط"   => "مستفعلن فاعلن مستفعلن فاعلن",
    "وافر"   => "مفاعلتن مفاعلتن فعولن",
    "كامل"   => "متفاعلن متفاعلن متفاعلن",
    "هزج"    => "مفاعيلن مفاعيلن",
    "رجز"    => "مستفعلن مستفعلن مستفعلن",
    "رمل"    => "فاعلاتن فاعلاتن فاعلاتن",
    "سريع"   => "مستفعلن مستفعلن فاعلن",
    "منسرح"  => "مستفعلن مفعولات مفتعلن",
    "خفيف"   => "فاعلاتن مستفعلن فاعلاتن",
    "مضارع"  => "مفاعيلن فاعلاتن مفاعيلن",
    "مقتضب"  => "فاعلات مفتعلن",
    "مجتث"   => "مستفعلن فاعلاتن",
    "متقارب" => "فعولن فعولن فعولن فعولن",
    "متدارك" => "فاعلن فاعلن فاعلن فاعلن",
)

"""
    syllabify_arabic(word) -> Vector{Int}

تقطيع الكلمة العربية إلى مقاطع (1 = ساكن/مد، 0 = حركة).
"""
function syllabify_arabic(word::String)
    vowels = Set(['ا', 'و', 'ي', 'ى', 'آ'])
    pattern = Int[]
    for c in word
        if c in ('َ', 'ِ', 'ُ')
            push!(pattern, 0)  # حركة قصيرة
        else
            push!(pattern, 1)  # حرف / مد / سكون
        end
    end
    return pattern
end

"""
    count_syllables_english(word) -> Int

عد المقاطع في كلمة إنجليزية.
"""
function count_syllables_english(word::String)
    w = lowercase(strip(word))
    isempty(w) && return 1
    count = 0
    prev_vowel = false
    vowels = Set("aeiouy")
    for ch in w
        is_v = ch in vowels
        if is_v && !prev_vowel
            count += 1
        end
        prev_vowel = is_v
    end
    if endswith(w, "e") && count > 1 && !endswith(w, "le")
        count -= 1
    end
    if endswith(w, "le") && length(w) > 2 && !(w[end-2] in vowels)
        count += 1
    end
    return max(1, count)
end

"""
    detect_meter(words) -> String

كشف البحر الشعري من قائمة كلمات عبر مقارنة التفعيلات.
"""
function detect_meter(words::Vector{String})
    if isempty(words); return "غير_معروف"; end
    pattern = Int[]
    for w in words
        append!(pattern, syllabify_arabic(w))
    end
    best_meter = "غير_معروف"
    best_score = 0
    for (meter, tafeela) in ARABIC_METERS
        taf_vec = Int[c == ' ' ? -1 : (c in ('ْ', 'ن') ? 1 : 0) for c in tafeela]
        filter!(x -> x >= 0, taf_vec)
        score = 0
        for i in 1:min(length(pattern), length(taf_vec))
            if pattern[i] == taf_vec[i]; score += 1; end
        end
        if score > best_score
            best_score = score
            best_meter = meter
        end
    end
    return best_meter
end

"""
    rhyme_score(word1, word2) -> Float64

درجة القافية: التشابه في آخر حرفين.
"""
function rhyme_score(word1::String, word2::String)
    if length(word1) < 2 || length(word2) < 2
        return 0.0
    end
    suffix1 = word1[end-1:end]
    suffix2 = word2[end-1:end]
    return suffix1 == suffix2 ? 1.0 : 0.0
end

"""
    PoeticGravityEngine

محرك الجاذبية الشعرية.

الحقول:
- `meter`: اسم البحر
- `rhyme_char`: حرف القافية
- `meter_weight`, `rhyme_weight`: أوزان الجاذبية الشعرية
"""
struct PoeticGravityEngine
    meter::String
    rhyme_char::Char
    meter_weight::Float64
    rhyme_weight::Float64
end

PoeticGravityEngine(; meter::String="", rhyme_char::Char='\0',
                    meter_weight::Float64=0.3, rhyme_weight::Float64=0.15) =
    PoeticGravityEngine(meter, rhyme_char, meter_weight, rhyme_weight)

end # module PoeticGravity
