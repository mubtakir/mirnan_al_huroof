"""
النحو العربي - تحليل الجمل والمرتكزات النحوية
"""

module Grammar

export Sentence, Phrase, Word, Morpheme,
       SyntacticRole, PhraseType, SentenceType,
       analyze_sentence, parse_sentence, get_syntactic_roles,
       GrammarAnalyzer

# ═══════════════════════════════════════════════════════
# الأنواع
# ═══════════════════════════════════════════════════════

@enum SyntacticRole begin
    SUBJECT       # فاعل
    OBJECT        # مفعول به
    VERB          # فعل
    ADJECTIVE     # صفة
    ADVERB        # حال
    PREPOSITION   # حرف جر
    CONJUNCTION   # حرف عطف
    POSSESSIVE    # مضاف إليه
    IDafa         # إضافة
    PREDICATE     # خبر
    TOPIC         # مبتدأ
    UNKNOWN
end

@enum PhraseType begin
    NOUN_PHRASE     # اسمية
    VERB_PHRASE     # فعلية
    PREP_PHRASE     # جار ومجرور
    ADJ_PHRASE      # صفة
    ADV_PHRASE      # حال
end

@enum SentenceType begin
    INDICATIVE     # خبرية
    INTERROGATIVE  # استفهامية
    IMPERATIVE     # طلبية
    EXCLAMATORY    # تعجبية
    CONDITIONAL    # شرطية
end

# ═══════════════════════════════════════════════════════
# البيانات
# ═══════════════════════════════════════════════════════

struct Morpheme
    text::String
    prefix::String
    root::String
    suffix::String
end

struct Word
    text::String
    morpheme::Morpheme
    role::SyntacticRole
    case_ending::Char  # ضمة، فتحة، كسرة
    is_indefinite::Bool
end

struct Phrase
    words::Vector{Word}
    phrase_type::PhraseType
    head_index::Int
end

struct Sentence
    text::String
    words::Vector{Word}
    phrases::Vector{Phrase}
    sentence_type::SentenceType
    subject::Union{Word,Nothing}
    predicate::Union{Word,Nothing}
    verb::Union{Word,Nothing}
    object::Union{Word,Nothing}
end

# ═══════════════════════════════════════════════════════
# محلل النحو
# ═══════════════════════════════════════════════════════

struct GrammarAnalyzer
    verb_patterns::Dict{String,String}
    pronoun_map::Dict{String,String}
    preposition_map::Dict{String,String}
    conjunction_map::Dict{String,String}
end

function GrammarAnalyzer()
    verb_patterns = Dict(
        "يَفْعَلُ" => "present",
        "فَعَلَ" => "past",
        "يَفْعِلُ" => "present",
        "فَعِلَ" => "past",
        "يَفْعُلُ" => "present",
        "فَعُلَ" => "past",
    )

    pronoun_map = Dict(
        "هُوَ" => "he",
        "هِيَ" => "she",
        "أَنَا" => "I",
        "نَحْنُ" => "we",
        "أَنْتَ" => "you_m",
        "أَنْتِ" => "you_f",
        "أَنْتُمْ" => "you_mp",
        "أَنْتُنَّ" => "you_fp",
        "هُمْ" => "they_m",
        "هُنَّ" => "they_f",
    )

    preposition_map = Dict(
        "فِي" => "in",
        "عَلَى" => "on",
        "مِن" => "from",
        "إِلَى" => "to",
        "بِ" => "with",
        "لِ" => "for",
        "عَن" => "about",
        "مَع" => "with",
        "بَعْد" => "after",
        "قَبْل" => "before",
    )

    conjunction_map = Dict(
        "وَ" => "and",
        "فَ" => "so",
        "ثُمَّ" => "then",
        "أَو" => "or",
        "لَكِن" => "but",
        "إِذَن" => "therefore",
    )

    GrammarAnalyzer(verb_patterns, pronoun_map, preposition_map, conjunction_map)
end

# ═══════════════════════════════════════════════════════
# تحليل الجملة
# ═══════════════════════════════════════════════════════

function analyze_sentence(analyzer::GrammarAnalyzer, sentence_text::AbstractString)::Sentence
    words = _tokenize_sentence(analyzer, sentence_text)
    sentence_type = _detect_sentence_type(sentence_text)
    phrases = _build_phrases(words)

    subject = _find_word_with_role(words, SUBJECT)
    verb = _find_word_with_role(words, VERB)
    object = _find_word_with_role(words, OBJECT)
    predicate = _find_word_with_role(words, PREDICATE)

    Sentence(
        String(sentence_text),
        words,
        phrases,
        sentence_type,
        subject,
        predicate,
        verb,
        object
    )
end

function parse_sentence(analyzer::GrammarAnalyzer, sentence_text::AbstractString)::Dict{String,Any}
    sentence = analyze_sentence(analyzer, sentence_text)

    return Dict(
        "text" => sentence.text,
        "type" => string(sentence.sentence_type),
        "words" => [
            Dict(
                "text" => w.text,
                "role" => string(w.role),
                "case" => string(w.case_ending),
                "indefinite" => w.is_indefinite
            ) for w in sentence.words
        ],
        "subject" => sentence.subject !== nothing ? sentence.subject.text : nothing,
        "verb" => sentence.verb !== nothing ? sentence.verb.text : nothing,
        "object" => sentence.object !== nothing ? sentence.object.text : nothing,
        "predicate" => sentence.predicate !== nothing ? sentence.predicate.text : nothing,
    )
end

function get_syntactic_roles(sentence::Sentence)::Dict{SyntacticRole,Vector{Word}}
    roles = Dict{SyntacticRole,Vector{Word}}()
    for word in sentence.words
        if !haskey(roles, word.role)
            roles[word.role] = Word[]
        end
        push!(roles[word.role], word)
    end
    return roles
end

# ═══════════════════════════════════════════════════════
# دوال مساعدة
# ═══════════════════════════════════════════════════════

function _tokenize_sentence(analyzer::GrammarAnalyzer, text::AbstractString)::Vector{Word}
    tokens = split(text)
    words = Word[]

    for token in tokens
        token_str = String(token)
        morpheme = _analyze_morpheme(token_str)
        role = _assign_role(analyzer, token_str)
        case_ending = _detect_case_ending(token_str)
        is_indef = _is_indefinite(token_str)

        push!(words, Word(token_str, morpheme, role, case_ending, is_indef))
    end

    return words
end

function _analyze_morpheme(word::AbstractString)::Morpheme
    prefix = ""
    root = word
    suffix = ""

    chars = collect(word)
    
    prefixes = ["ال", "و", "ف", "ب", "ك", "ل"]
    for p in prefixes
        p_chars = collect(p)
        if length(chars) > length(p_chars) && chars[1:length(p_chars)] == p_chars
            prefix = p
            root = String(chars[length(p_chars)+1:end])
            chars = chars[length(p_chars)+1:end]
            break
        end
    end

    suffixes = ["ون", "ين", "ات", "ة", "ها", "هم", "هن", "كم", "كن"]
    for s in suffixes
        s_chars = collect(s)
        if length(chars) > length(s_chars) && chars[end-length(s_chars)+1:end] == s_chars
            suffix = s
            root = String(chars[1:end-length(s_chars)])
            break
        end
    end

    Morpheme(word, prefix, root, suffix)
end

function _assign_role(analyzer::GrammarAnalyzer, word::AbstractString)::SyntacticRole
    if haskey(analyzer.verb_patterns, word)
        return VERB
    end

    if haskey(analyzer.pronoun_map, word)
        return SUBJECT
    end

    if haskey(analyzer.preposition_map, word)
        return PREPOSITION
    end

    if haskey(analyzer.conjunction_map, word)
        return CONJUNCTION
    end

    if endswith(word, "ِ") || endswith(word, "َ") || endswith(word, "ُ")
        return SUBJECT
    end

    return UNKNOWN
end

function _detect_case_ending(word::AbstractString)::Char
    if endswith(word, "ُ") || endswith(word, "و")
        return 'ُ'  # ضمة - مرفوع
    elseif endswith(word, "َ") || endswith(word, "ا")
        return 'َ'  # فتحة - منصوب
    elseif endswith(word, "ِ") || endswith(word, "ي")
        return 'ِ'  # كسرة - مجرور
    end
    return ' '
end

function _is_indefinite(word::AbstractString)::Bool
    return !startswith(word, "ال") && !endswith(word, "ِ")
end

function _detect_sentence_type(text::AbstractString)::SentenceType
    if occursin("؟", text) || occursin("?", text)
        return INTERROGATIVE
    end

    interrogative_words = ["هل", "ما", "ماذا", "من", "أين", "كيف", "لماذا", "متى"]
    for w in interrogative_words
        if startswith(text, w)
            return INTERROGATIVE
        end
    end

    imperative_words = ["افعل", "لا تفعل", "دع"]
    for w in imperative_words
        if startswith(text, w)
            return IMPERATIVE
        end
    end

    if occursin("!", text)
        return EXCLAMATORY
    end

    conditional_words = ["إذا", "لو", "إن"]
    for w in conditional_words
        if startswith(text, w)
            return CONDITIONAL
        end
    end

    return INDICATIVE
end

function _build_phrases(words::Vector{Word})::Vector{Phrase}
    phrases = Phrase[]
    current_phrase_words = Word[]
    current_type = NOUN_PHRASE

    for word in words
        if word.role in [VERB, SUBJECT, OBJECT, PREDICATE]
            if !isempty(current_phrase_words)
                push!(phrases, Phrase(current_phrase_words, current_type, 1))
                current_phrase_words = Word[]
            end
            current_type = word.role == VERB ? VERB_PHRASE : NOUN_PHRASE
            push!(current_phrase_words, word)
        else
            push!(current_phrase_words, word)
        end
    end

    if !isempty(current_phrase_words)
        push!(phrases, Phrase(current_phrase_words, current_type, 1))
    end

    return phrases
end

function _find_word_with_role(words::Vector{Word}, role::SyntacticRole)::Union{Word,Nothing}
    for word in words
        if word.role == role
            return word
        end
    end
    return nothing
end

end # module Grammar
