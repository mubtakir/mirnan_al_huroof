"""
Arabic and mixed-text preprocessing utilities.
"""

module Preprocessing

export normalize_text, extract_sentences, extract_words,
       normalize_arabic, remove_diacritics, normalize_whitespace,
       tokenize_text, preprocess_text, TextPreprocessor

const ARABIC_DIACRITICS = Dict(
    '\u064B' => "",
    '\u064C' => "",
    '\u064D' => "",
    '\u064E' => "",
    '\u064F' => "",
    '\u0650' => "",
    '\u0651' => "",
    '\u0652' => "",
    '\u0653' => "",
    '\u0654' => "",
    '\u0655' => "",
    '\u0670' => "",
)

const ARABIC_NORMALIZATION = Dict(
    '\u0622' => '\u0627',
    '\u0623' => '\u0627',
    '\u0625' => '\u0627',
    '\u0629' => '\u0647',
    '\u0649' => '\u064A',
)

const SENTENCE_TERMINATORS = ['.', '!', '?', ';', '؟', '؛', '،']

struct TextPreprocessor
    normalize_diacritics::Bool
    normalize_alef::Bool
    normalize_ta_marbuta::Bool
    remove_tatweel::Bool
    normalize_whitespace::Bool
end

function TextPreprocessor(;
    normalize_diacritics::Bool = false,
    normalize_alef::Bool = true,
    normalize_ta_marbuta::Bool = true,
    remove_tatweel::Bool = true,
    normalize_whitespace::Bool = true
)
    TextPreprocessor(
        normalize_diacritics,
        normalize_alef,
        normalize_ta_marbuta,
        remove_tatweel,
        normalize_whitespace
    )
end

function preprocess_text(text::AbstractString; preprocessor::TextPreprocessor = TextPreprocessor())::String
    result = String(text)

    if preprocessor.normalize_diacritics
        result = remove_diacritics(result)
    end

    result = normalize_arabic(result;
        normalize_alef = preprocessor.normalize_alef,
        normalize_ta_marbuta = preprocessor.normalize_ta_marbuta
    )

    if preprocessor.remove_tatweel
        result = replace(result, '\u0640' => "")
    end

    if preprocessor.normalize_whitespace
        result = normalize_whitespace(result)
    end

    return result
end

normalize_text(text::AbstractString; preprocessor::TextPreprocessor = TextPreprocessor()) =
    preprocess_text(text; preprocessor = preprocessor)

function normalize_arabic(text::AbstractString;
    normalize_alef::Bool = true,
    normalize_ta_marbuta::Bool = true
)::String
    result = collect(String(text))

    if normalize_alef
        for (from, to) in ARABIC_NORMALIZATION
            if from == '\u0629' && !normalize_ta_marbuta
                continue
            end
            result = [c == from ? to : c for c in result]
        end
    end

    return String(result)
end

function remove_diacritics(text::AbstractString)::String
    return String(text)
end

function normalize_whitespace(text::AbstractString)::String
    result = strip(String(text))
    result = replace(result, r"\s+" => " ")
    return result
end

function extract_sentences(text::AbstractString)::Vector{String}
    sentences = String[]
    current = IOBuffer()

    for c in text
        if c in SENTENCE_TERMINATORS
            sentence = strip(String(take!(current)))
            if !isempty(sentence)
                push!(sentences, sentence)
            end
            current = IOBuffer()
        else
            print(current, c)
        end
    end

    last_sentence = strip(String(take!(current)))
    if !isempty(last_sentence)
        push!(sentences, last_sentence)
    end

    return sentences
end

function extract_words(text::AbstractString)::Vector{String}
    words = String[]
    current = IOBuffer()

    for c in text
        if is_arabic(c) || is_latin(c) || isdigit(c)
            print(current, c)
        else
            word = strip(String(take!(current)))
            if !isempty(word)
                push!(words, word)
            end
            current = IOBuffer()
        end
    end

    last_word = strip(String(take!(current)))
    if !isempty(last_word)
        push!(words, last_word)
    end

    return words
end

function tokenize_text(text::AbstractString; preprocessor::TextPreprocessor = TextPreprocessor())::Vector{String}
    normalized = preprocess_text(text; preprocessor = preprocessor)
    return extract_words(normalized)
end

function is_arabic(c::Char)::Bool
    return '\u0600' <= c <= '\u06FF' || '\u0750' <= c <= '\u077F' ||
           '\uFB50' <= c <= '\uFDFF' || '\uFE70' <= c <= '\uFEFF'
end

function is_latin(c::Char)::Bool
    return ('a' <= c <= 'z') || ('A' <= c <= 'Z')
end

function count_arabic_letters(text::AbstractString)::Int
    return count(is_arabic, text)
end

function count_latin_letters(text::AbstractString)::Int
    return count(is_latin, text)
end

function is_mostly_arabic(text::AbstractString; threshold::Float64 = 0.5)::Bool
    arabic_count = count_arabic_letters(text)
    total_count = length(filter(c -> is_arabic(c) || is_latin(c), text))
    total_count == 0 && return false
    return (arabic_count / total_count) >= threshold
end

end # module Preprocessing
