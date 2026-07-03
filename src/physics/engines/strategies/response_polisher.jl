const RESPONSE_POLISHER_PUNCT = Set([".", "!", "?", "\u061F", "\u06D4"])

function _polisher_has_arabic(s::AbstractString)
    return occursin(r"[\u0600-\u06FF]", String(s))
end

function _polisher_language(text::AbstractString)
    return _polisher_has_arabic(text) ? "arabic" : "latin"
end

function _polisher_contains_word(text::AbstractString, word::AbstractString)
    return occursin(" " * String(word) * " ", " " * String(text) * " ")
end

function _polisher_should_preserve(text::AbstractString)
    t = String(text)
    lt = lowercase(t)
    return occursin("```", t) ||
           occursin("\u2500\u2500\u2500", t) ||
           occursin("sha1:", lt) ||
           occursin("\n", t) && count(==('\n'), t) >= 2
end

function _polisher_sentence_end(text::AbstractString)
    t = strip(String(text))
    isempty(t) && return t
    last_char = string(t[lastindex(t)])
    last_char in RESPONSE_POLISHER_PUNCT && return t
    return t * "."
end

function _polisher_clean_spacing(text::AbstractString)
    t = replace(String(text), r"\s+" => " ")
    t = replace(t, r"\s+([,.;:!\?])" => s"\1")
    t = replace(t, r"\s+([\u060C\u061B\u061F])" => s"\1")
    t = replace(t, r"([,.;:!\?])([^\s,.;:!\?])" => s"\1 \2")
    t = replace(t, r"([\u060C\u061B\u061F])([^\s\u060C\u061B\u061F])" => s"\1 \2")
    return strip(t)
end

function _polisher_collapse_repeated_words(text::AbstractString)
    parts = split(_polisher_clean_spacing(text))
    isempty(parts) && return ""
    out = String[]
    last_norm = ""
    repeat_count = 0
    for p in parts
        norm = lowercase(replace(String(p), r"^[\W_]+|[\W_]+$" => ""))
        if !isempty(norm) && norm == last_norm
            repeat_count += 1
            continue
        else
            repeat_count = 0
        end
        push!(out, String(p))
        last_norm = norm
    end
    return join(out, " ")
end

function _polisher_fix_light_arabic_style(text::AbstractString)
    _polisher_has_arabic(text) || return String(text)
    t = String(text)
    t = replace(t, r"\s*,\s*" => "\u060C ")
    t = replace(t, r"\s*;\s*" => "\u061B ")
    return t
end

function _polisher_run_on_words(text::AbstractString)
    words = split(strip(String(text)))
    length(words) >= 18 || return false
    return !occursin(r"[\.\!\?\u061F\u06D4]", String(text))
end

function _polisher_group_run_on_words(text::AbstractString)
    words = split(strip(String(text)))
    length(words) >= 18 || return String(text)
    chunks = String[]
    for i in 1:12:length(words)
        push!(chunks, join(words[i:min(i + 11, length(words))], " "))
    end
    sep = _polisher_has_arabic(text) ? "\u060C " : ", "
    return join(chunks, sep)
end

function _polisher_answer_kind(prompt::AbstractString, text::AbstractString)
    raw = String(text)
    t = lowercase(_polisher_clean_spacing(raw))
    padded = " " * t * " "
    stripped = strip(raw)

    if startswith(stripped, "\u0625\u0630\u0627") ||
           startswith(stripped, "\u0627\u0646 ") ||
           startswith(t, "if ")
        return "conditional"
    elseif occursin("\u064A\u0646\u062A\u0642\u0644 \u0627\u0644\u0645\u062A\u0623\u062B\u0631", raw) ||
           occursin("\u0623\u062B\u0631 \u062F\u0644\u0627\u0644\u064A", raw) ||
           occursin("\u0641\u0627\u0646\u062A\u0642\u0644", raw) ||
           occursin("\u0638\u0647\u0631\u062A \u0622\u062B\u0627\u0631", raw) ||
           occursin("\u0648\u0638\u0647\u0631\u062A \u0622\u062B\u0627\u0631", raw) ||
           occursin("\u0648\u0643\u0627\u0646\u062A \u0627\u0644\u063A\u0627\u064A\u0629", raw) ||
           occursin("semantic effect", t) ||
           (startswith(t, "when ") && occursin(" affects ", padded))
        return "scene"
    elseif occursin("\u0627\u0644\u063A\u0627\u064A\u0629", raw) ||
           occursin("\u0647\u062F\u0641", raw) ||
           _polisher_contains_word(t, "purpose")
        return "purpose"
    elseif startswith(stripped, "\u0642\u0628\u0644") ||
           startswith(stripped, "\u0628\u0639\u062F") ||
           occursin("\u0642\u0628\u0644", raw) ||
           occursin("\u0628\u0639\u062F", raw) ||
           _polisher_contains_word(t, "before") ||
           _polisher_contains_word(t, "after") ||
           _polisher_contains_word(t, "when")
        return "temporal"
    elseif startswith(stripped, "\u0641\u064A ") ||
           occursin("\u0627\u0644\u0645\u0643\u0627\u0646", raw) ||
           occursin("\u0645\u0643\u0627\u0646", raw) ||
           startswith(t, "place:") ||
           startswith(t, "the place of") ||
           _polisher_contains_word(t, "inside") ||
           _polisher_contains_word(t, "above") ||
           _polisher_contains_word(t, "under")
        return "spatial"
    elseif occursin("\u062D\u0627\u0644", raw) ||
           occursin("\u0628\u064A\u0646\u0645\u0627", raw) ||
           startswith(t, "state:") ||
           _polisher_contains_word(t, "while")
        return "state"
    elseif startswith(stripped, "\u0639\u062F\u062F ") ||
           startswith(stripped, "\u0645\u0642\u062F\u0627\u0631 ") ||
           occursin("\u0627\u0644\u0645\u0642\u0627\u0631\u0646\u0629", raw) ||
           occursin("\u0645\u0646 \u062B\u0644\u0627\u062B\u0629 \u0625\u0644\u0649 \u0639\u0634\u0631\u0629", raw) ||
           (occursin("\u064A\u062F\u0644", raw) && occursin("\u0639\u0644\u0649", raw)) ||
           _polisher_contains_word(t, "number") ||
           _polisher_contains_word(t, "quantity") ||
           _polisher_contains_word(t, "count")
        return "quantity"
    elseif occursin(" \u0647\u0648 ", raw) ||
           occursin(" is ", padded)
        return "definition"
    end
    return "generic"
end

function _polisher_apply_kind_style(prompt::AbstractString, text::AbstractString)
    kind = _polisher_answer_kind(prompt, text)
    t = String(text)

    if kind == "conditional" && _polisher_has_arabic(t)
        result_marker = "\u064A\u062A\u0631\u062A\u0628 \u0639\u0644\u0649 \u0630\u0644\u0643"
        t = replace(t, "\u060C " * result_marker => "\u061B " * result_marker)
        t = replace(t, ", " * result_marker => "\u061B " * result_marker)
    elseif kind == "scene" && _polisher_has_arabic(t)
        purpose_marker = "\u0648\u0645\u0646 \u062C\u0647\u0629 \u0627\u0644\u063A\u0627\u064A\u0629:"
        t = replace(t, "\u060C " * purpose_marker => "\u061B " * purpose_marker)
        t = replace(t, ", " * purpose_marker => "\u061B " * purpose_marker)
    elseif kind == "state" && _polisher_has_arabic(t)
        state_marker = "\u0648\u0643\u0627\u0646 \u0639\u0644\u0649 \u062D\u0627\u0644"
        t = replace(t, "\u060C " * state_marker => "\u061B " * state_marker)
        t = replace(t, ", " * state_marker => "\u061B " * state_marker)
    elseif kind == "quantity"
        t = replace(t, r"\b(is)\s+\1\b"i => s"\1")
    end

    return t
end

"""
    response_polish_profile(prompt, text)

Return a small diagnostic profile for the final wording layer. This keeps the
polisher type-aware without letting it rewrite the source answer.
"""
function response_polish_profile(prompt::AbstractString, text::AbstractString)
    raw = strip(String(text))
    cleaned = _polisher_clean_spacing(raw)
    collapsed = _polisher_collapse_repeated_words(cleaned)
    return (
        kind = _polisher_answer_kind(prompt, cleaned),
        language = _polisher_language(cleaned),
        has_repetition = cleaned != collapsed,
        run_on = _polisher_run_on_words(cleaned),
        preserved = _polisher_should_preserve(raw),
    )
end

"""
    polish_response(prompt, text; enabled=false) -> String

Conservative final wording pass. It does not infer new facts or replace the
answer source; it only normalizes spacing, punctuation, repeated words, and
very long unpunctuated word runs.
"""
function polish_response(prompt::AbstractString, text::AbstractString; enabled::Bool=false)
    raw = strip(String(text))
    enabled || return raw
    _polisher_should_preserve(raw) && return raw
    t = _polisher_clean_spacing(raw)
    isempty(t) && return ""
    t = _polisher_collapse_repeated_words(t)
    t = _polisher_fix_light_arabic_style(t)
    t = _polisher_apply_kind_style(prompt, t)
    _polisher_run_on_words(t) && (t = _polisher_group_run_on_words(t))
    return _polisher_sentence_end(t)
end
