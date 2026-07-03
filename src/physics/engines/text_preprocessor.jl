"""
text_preprocessor.jl — معالج النصوص التدريبية

ي enforcing القواعد:
1. الجمل التامة → نقطة .
2. الجمل غير التامة → فاصلة ،
3. الجمل المكملة → فاصلة منقوطة ؛
4. جمل القول → نقطتين :
5. علامات الترقيم ملاصقة للكلمة
6. فصل الحوار بـ tab
"""

module TextPreprocessor

export preprocess_text, validate_and_fix, split_sentences, parse_dialogue

const SENTENCE_ENDERS = Set(['.', '!', '?', '؟'])
const CLAUSE_ENDERS = Set([';', '؛'])
const PAUSE_ENDERS = Set([',', '،'])
const SPEECH_ENDERS = Set([':'])
const ALL_PUNCT = union(SENTENCE_ENDERS, CLAUSE_ENDERS, PAUSE_ENDERS, SPEECH_ENDERS)

"""
    preprocess_text(text::String) -> String

معالجة النص وتقويمه حسب القواعد.
"""
function preprocess_text(text::String)
    lines = split(text, '\n')
    result = String[]

    for line in lines
        stripped = strip(line)

        if isempty(stripped)
            push!(result, "")
            continue
        end

        # Fix: Remove space before punctuation
        fixed = fix_punctuation_spacing(stripped)

        # Fix: Ensure dialogue uses tab separation
        fixed = fix_dialogue_separation(fixed)

        push!(result, fixed)
    end

    # Ensure double newline between paragraphs
    output = join(result, '\n')
    output = fix_paragraph_separation(output)

    return output
end

"""
    fix_punctuation_spacing(line::String) -> String

إزالة المسافات قبل علامات الترقيم.
"""
function fix_punctuation_spacing(line::String)
    chars = collect(line)
    result = Char[]

    for (i, ch) in enumerate(chars)
        if ch in ALL_PUNCT && i > 1 && result[end] == ' '
            # Remove the trailing space
            pop!(result)
        end
        push!(result, ch)
    end

    return String(result)
end

"""
    fix_dialogue_separation(line::String) -> String

تحويل فواصل الحوار إلى tab.
يكتشف أنماط مثل:
  "قال: كلام" → "قال:\tكلام"
"""
function fix_dialogue_separation(line::String)
    # Pattern: word followed by colon, then Arabic text
    # "قال: كلام" → "قال:\tكلام"
    m = match(r"^([^:\n]+:\s*)(.+)$", line)
    if m !== nothing
        prefix = m.captures[1]
        content = m.captures[2]

        # Only fix if it looks like a speech tag
        speech_words = ["قال", "قالت", "يقول", "تقول", "هذا", "هذاك",
                        "أجاب", "أجابت", "سأل", "سألت", "صرخ", "همس",
                        "أضاف", "أضافت", "أوضح", "أشار", "نادى", "نادت"]
        prefix_clean = strip(prefix, [':', ' '])

        if any(startswith(prefix_clean, w) for w in speech_words)
            return rstrip(prefix) * "\t" * lstrip(content)
        end
    end

    return line
end

"""
    fix_paragraph_separation(text::String) -> String

التأكد من أن بين الفقرات سطر فارغ واحد فقط.
"""
function fix_paragraph_separation(text::String)
    # Replace 3+ newlines with 2 (one empty line)
    text = replace(text, r"\n{3,}" => "\n\n")
    return text
end

"""
    split_sentences(text::String) -> Vector{String}

تقسيم النص إلى جمل بناءً على علامات الترقيم.
"""
function split_sentences(text::String)
    sentences = String[]
    current = IOBuffer()

    for ch in text
        write(current, ch)
        if ch in SENTENCE_ENDERS
            push!(sentences, String(take!(current)))
        end
    end

    remaining = String(take!(current))
    if !isempty(strip(remaining))
        push!(sentences, remaining)
    end

    return sentences
end

"""
    parse_dialogue(text::String) -> Vector{Tuple{String, String}}

تحليل الحوار وفصل القول عن الجواب.
يُرجع أزواج (القائل، الكلام).
"""
function parse_dialogue(text::String)
    dialogues = Tuple{String, String}[]
    lines = split(text, '\n')

    i = 1
    while i <= length(lines)
        line = strip(lines[i])

        if isempty(line)
            i += 1
            continue
        end

        # Check for tab-separated dialogue
        if '\t' in line
            parts = split(line, '\t', limit=2)
            if length(parts) == 2
                push!(dialogues, (strip(parts[1]), strip(parts[2])))
                i += 1
                continue
            end
        end

        # Check for "سؤال:" / "جواب:" pattern
        for prefix in ["سؤال:", "جواب:", "س:", "ج:"]
            if startswith(line, prefix)
                rest = strip(line[nextind(line, firstindex(line), length(prefix)):end])
                label = strip(prefix, [':', ' '])
                push!(dialogues, (label, rest))
                break
            end
        end

        i += 1
    end

    return dialogues
end

"""
    validate_and_fix(text::String) -> (String, Vector{String})

التحقق من النص وإصلاح الأخطاء.
يُرجع النص المُصلح وقائمة بالأخطاء التي تم إصلاحها.
"""
function validate_and_fix(text::String)
    fixes = String[]
    fixed = text

    # Fix 1: Space before punctuation
    for ch in ALL_PUNCT
        pattern = Regex(" $ch")
        if occursin(pattern, fixed)
            fixed = replace(fixed, pattern => string(ch))
            push!(fixes, "تمت إزالة مسافة قبل '$ch'")
        end
    end

    # Fix 2: Multiple newlines
    if occursin(r"\n{3,}", fixed)
        fixed = replace(fixed, r"\n{3,}" => "\n\n")
        push!(fixes, "تمت تسوية فواصل الفقرات")
    end

    return fixed, fixes
end

end # module TextPreprocessor
