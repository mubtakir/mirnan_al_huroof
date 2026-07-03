"""
al_lisan - Linguistic pattern memory for Mirnan.

This layer remembers sentence shapes separately from semantic matrices. It is
the tongue-side memory: shape, order, slots, and compact examples.
"""
module AlLisan

using JSON

using ..MorphoPhasic: MorphoPhasicEngine, analyze_morpho, PREPOSITIONS,
                      CONJUNCTIONS, PARTICLES

export LinguisticPatternRecord, LinguisticPatternMemory,
       learn_lisan_from_text!, train_lisan_from_texts!,
       select_lisan_pattern, token_role_phase,
       save_lisan, load_lisan, lisan_to_dict,
       has_lisan_patterns, pattern_has_verb, detect_lisan_language,
       load_lisan_markers, default_lisan_marker_path, ROLE_SYNTAX_PHASE

const AL_LISAN_VERSION = 1
const AL_LISAN_MARKERS_VERSION = 1
const AR_VERB = "\u0641\u0639\u0644"
const AR_ADJ = "\u0635\u0641\u0629"
const AR_ADVERB = "\u062d\u0627\u0644"
const AR_TOOL = "\u0623\u062f\u0627\u0629"

const CONDITION_TOOLS = Set([
    "\u0625\u0630\u0627", "\u0627\u0630\u0627", "\u0644\u0648",
    "\u0625\u0646", "\u0627\u0646", "\u0645\u062a\u0649",
    "\u0643\u0644\u0645\u0627",
])

const QUESTION_TOOLS = Set([
    "\u0647\u0644", "\u0645\u0627", "\u0645\u0627\u0630\u0627",
    "\u0645\u0646", "\u0645\u062a\u0649", "\u0623\u064a\u0646",
    "\u0627\u064a\u0646", "\u0643\u064a\u0641", "\u0644\u0645\u0627\u0630\u0627",
])

const SAY_VERBS = Set([
    "\u0642\u0627\u0644", "\u0642\u0627\u0644\u062a", "\u064a\u0642\u0648\u0644",
    "\u062a\u0642\u0648\u0644", "\u0623\u062c\u0627\u0628",
    "\u0627\u062c\u0627\u0628", "\u0633\u0623\u0644", "\u0633\u0627\u0644",
])

const DEFAULT_AR_FIXED_MARKERS = Set([
    "\u0637\u0627\u0644\u0645\u0627", "\u0628\u064a\u0646\u0645\u0627",
    "\u0644\u0637\u0627\u0644\u0645\u0627", "\u0631\u064a\u062b\u0645\u0627",
    "\u062d\u0628\u0630\u0627", "\u0644\u0648\u0644\u0627",
    "\u0644\u0648\u0645\u0627", "\u0625\u0644\u0627",
    "\u0627\u0644\u0627", "\u0639\u062f\u0627",
    "\u062e\u0644\u0627", "\u062d\u0627\u0634\u0627",
    "\u0639\u0646\u062f\u0645\u0627", "\u062d\u064a\u062b\u0645\u0627",
    "\u0625\u0630", "\u0627\u0630", "\u0644\u0643\u0646",
    "\u0628\u0644", "\u0644\u0623\u0646", "\u0644\u0627\u0646",
    "\u0644\u0643\u064a", "\u0643\u064a",
])

const SENTENCE_BOUNDARY_RE = Regex("[\\n\\.\\!\\?" * "\u061F" * "\u061B" * ";]+")
const DEFINITE_ARTICLE_RE = Regex("^\u0627\u0644")

const EN_DETERMINERS = Set(["the", "a", "an", "this", "that", "these", "those"])
const EN_BE_VERBS = Set(["am", "is", "are", "was", "were", "be", "being", "been"])
const EN_AUXILIARIES = Set(["do", "does", "did", "have", "has", "had"])
const EN_MODALS = Set(["can", "could", "may", "might", "must", "shall", "should", "will", "would"])
const EN_NEGATIONS = Set(["not", "never", "no"])
const EN_PREPOSITIONS = Set([
    "of", "to", "in", "on", "at", "by", "for", "from", "with", "about",
    "under", "over", "between", "into", "through", "after", "before", "without",
])
const EN_CONJUNCTIONS = Set(["and", "or", "but", "nor", "yet", "so"])
const EN_CONDITION_TOOLS = Set(["if", "when", "whenever", "unless", "because", "although", "while", "since"])
const EN_QUESTION_TOOLS = Set(["who", "what", "when", "where", "why", "how", "which"])
const EN_RELATIVE_PRONOUNS = Set(["that", "which", "who", "whom", "whose"])
const EN_MARKERS = Set(["however", "therefore", "meanwhile", "otherwise", "then", "there"])
const EN_COMMON_VERBS = Set([
    "say", "says", "said", "ask", "asks", "asked", "answer", "answers", "answered",
    "learn", "learns", "learned", "raise", "raises", "raised", "strengthen", "strengthens",
    "strengthened", "illuminate", "illuminates", "illuminated", "build", "builds", "built",
    "make", "makes", "made", "give", "gives", "gave", "help", "helps", "helped",
    "need", "needs", "needed", "know", "knows", "knew", "understand", "understands",
    "understood", "open", "opens", "opened", "move", "moves", "moved",
    "follow", "follows", "followed", "shape", "shapes", "shaped",
    "guide", "guides", "guided", "create", "creates", "created",
    "change", "changes", "changed",
])
const EN_COMMON_ADJECTIVES = Set([
    "clear", "open", "closed", "strong", "weak", "useful", "bright", "dark",
    "natural", "logical", "semantic", "syntactic", "causal", "new", "old",
])
const EN_COMMON_NOUNS = Set([
    "knowledge", "understanding", "meaning", "language", "student", "students",
    "wisdom", "science", "thought", "sentence", "pattern", "patterns",
    "question", "answer", "answers", "structure", "structures", "grammar",
])

const ROLE_SYNTAX_PHASE = Dict{String,Float64}(
    "verb" => π/2,
    "participle" => 0.35π,
    "adjective" => 3π/4,
    "subject" => π/4,
    "object" => π/4,
    "noun" => π/4,
    "prep_object" => π/4,
    "preposition" => 0.0,
    "conjunction" => π,
    "determiner" => 0.0,
    "be_verb" => 0.2π,
    "auxiliary" => 0.2π,
    "negation" => 0.15π,
    "marker" => 0.0,
    "condition_tool" => 0.0,
    "question_tool" => 0.0,
    "relative_pronoun" => 0.0,
    "particle" => 0.8π,
)

token_role_phase(role::AbstractString) = get(ROLE_SYNTAX_PHASE, String(role), 0.0)

mutable struct LinguisticPatternRecord
    language::String
    shape::String
    roles::Vector{String}
    count::Int
    examples::Vector{String}
end

function LinguisticPatternRecord(shape::String, roles::Vector{String}, count::Int,
                                 examples::Vector{String})
    return LinguisticPatternRecord("ar", shape, roles, count, examples)
end

mutable struct LinguisticPatternMemory
    patterns::Dict{String,LinguisticPatternRecord}
    max_examples::Int
    morpho::MorphoPhasicEngine
    fixed_markers::Set{String}
end

function default_lisan_marker_path()
    return normpath(joinpath(@__DIR__, "..", "..", "..", "config", "al_lisan_markers.json"))
end

function _add_lisan_marker_items!(markers::Set{String}, raw)
    if raw isa AbstractString
        marker = strip(String(raw))
        isempty(marker) || push!(markers, marker)
    elseif raw isa AbstractVector
        for item in raw
            _add_lisan_marker_items!(markers, item)
        end
    elseif raw isa AbstractDict
        for item in values(raw)
            _add_lisan_marker_items!(markers, item)
        end
    end
    return markers
end

function load_lisan_markers(path::String=default_lisan_marker_path())
    markers = Set{String}(DEFAULT_AR_FIXED_MARKERS)
    isfile(path) || return markers
    try
        data = JSON.parsefile(path)
        raw = data isa AbstractDict ? get(data, "arabic_fixed_markers", Any[]) : data
        _add_lisan_marker_items!(markers, raw)
    catch
        return markers
    end
    return markers
end

function LinguisticPatternMemory(; max_examples::Int=5,
                                 marker_path::String=default_lisan_marker_path())
    return LinguisticPatternMemory(
        Dict{String,LinguisticPatternRecord}(),
        max_examples,
        MorphoPhasicEngine(),
        load_lisan_markers(marker_path),
    )
end

function _strip_edges(token::AbstractString)
    return replace(strip(String(token)), r"^[\s\p{P}\p{S}]+|[\s\p{P}\p{S}]+$" => "")
end

function _sentence_tokens(sentence::AbstractString)
    out = String[]
    for raw in split(String(sentence))
        tok = _strip_edges(raw)
        isempty(tok) && continue
        push!(out, tok)
    end
    return out
end

function detect_lisan_language(tokens::Vector{String})
    latin = count(t -> occursin(r"[A-Za-z]", t), tokens)
    arabic = count(t -> occursin(r"[\u0600-\u06FF]", t), tokens)
    latin > 0 && latin >= arabic && return "en"
    return "ar"
end

function _split_sentences(text::AbstractString)
    return [strip(s) for s in split(String(text), SENTENCE_BOUNDARY_RE) if !isempty(strip(s))]
end

function _en_plain(word::String)
    return lowercase(replace(_strip_edges(word), r"^[^A-Za-z]+|[^A-Za-z]+$" => ""))
end

function _token_kind_en(word::String, idx::Int)
    plain = _en_plain(word)
    isempty(plain) && return "PART"
    idx == 1 && plain in EN_QUESTION_TOOLS && return "QUESTION"
    plain in EN_DETERMINERS && return "DET"
    plain in EN_BE_VERBS && return "BE"
    plain in EN_AUXILIARIES && return "AUX"
    plain in EN_MODALS && return "MODAL"
    plain in EN_NEGATIONS && return "NEG"
    plain in EN_PREPOSITIONS && return "PREP"
    plain in EN_CONJUNCTIONS && return "CONJ"
    plain in EN_RELATIVE_PRONOUNS && return "REL"
    plain in EN_CONDITION_TOOLS && return "COND"
    plain in EN_QUESTION_TOOLS && return "QUESTION"
    plain in EN_MARKERS && return "MARKER"
    plain in EN_COMMON_NOUNS && return "NOUN"
    plain in EN_COMMON_VERBS && return "VERB"
    plain in EN_COMMON_ADJECTIVES && return "ADJ"
    (endswith(plain, "ing") || endswith(plain, "ed")) && length(plain) > 4 && return "VERB"
    return "NOUN"
end

function _token_kind(mem::LinguisticPatternMemory, word::String, idx::Int, language::String)
    language == "en" && return _token_kind_en(word, idx)
    plain = _strip_edges(word)
    plain in mem.fixed_markers && return "MARKER"
    plain in CONDITION_TOOLS && return "COND"
    plain in QUESTION_TOOLS && return "QUESTION"
    plain in PREPOSITIONS && return "PREP"
    plain in CONJUNCTIONS && return "CONJ"
    plain in PARTICLES && return "PART"
    plain in SAY_VERBS && return "VERB"
    analysis = try
        analyze_morpho(mem.morpho, plain)
    catch
        Dict("type" => "")
    end
    t = String(get(analysis, "type", ""))
    t == AR_VERB && return "VERB"
    (t == AR_ADJ || t == AR_ADVERB) && return "ADJ"
    t == AR_TOOL && return "PART"
    return "NOUN"
end

function _slot_role_en(kinds::Vector{String}, idx::Int)
    kind = kinds[idx]
    kind == "DET" && return "determiner"
    kind == "BE" && return "be_verb"
    kind == "AUX" && return "auxiliary"
    kind == "MODAL" && return "auxiliary"
    kind == "NEG" && return "negation"
    kind == "COND" && return "condition_tool"
    kind == "QUESTION" && return "question_tool"
    kind == "REL" && return "relative_pronoun"
    kind == "MARKER" && return "marker"
    kind == "PREP" && return "preposition"
    kind == "CONJ" && return "conjunction"
    kind == "PART" && return "particle"
    kind == "ADJ" && return "adjective"
    if kind == "VERB"
        if idx > 1 && kinds[idx - 1] == "BE"
            return "participle"
        end
        return "verb"
    end
    first_verb = findfirst(k -> k in ("VERB", "BE"), kinds)
    first_noun = findfirst(==("NOUN"), kinds)
    if kind == "NOUN"
        if idx > 1 && kinds[idx - 1] == "PREP"
            return "prep_object"
        elseif first_verb === nothing
            return idx == first_noun ? "subject" : "noun"
        elseif idx < first_verb
            return "subject"
        elseif !any(i -> kinds[i] == "NOUN" && i > first_verb && _slot_role_en(kinds, i) == "object",
                    1:(idx - 1))
            return "object"
        else
            return "noun"
        end
    end
    return lowercase(kind)
end

function _slot_role(kinds::Vector{String}, idx::Int, language::String)
    language == "en" && return _slot_role_en(kinds, idx)
    kind = kinds[idx]
    kind == "COND" && return "condition_tool"
    kind == "QUESTION" && return "question_tool"
    kind == "MARKER" && return "marker"
    kind == "PREP" && return "preposition"
    kind == "CONJ" && return "conjunction"
    kind == "PART" && return "particle"
    kind == "ADJ" && return "adjective"
    if kind == "VERB"
        return "verb"
    end
    first_verb = findfirst(==("VERB"), kinds)
    if kind == "NOUN"
        if idx > 1 && kinds[idx - 1] == "PREP"
            return "prep_object"
        elseif first_verb === nothing
            return idx == 1 ? "subject" : "noun"
        elseif idx < first_verb
            return "subject"
        elseif !any(i -> kinds[i] == "NOUN" && i > first_verb && _slot_role(kinds, i, language) == "object",
                    1:(idx - 1))
            return "object"
        else
            return "noun"
        end
    end
    return lowercase(kind)
end

function _shape_label(role::String)
    role == "subject" && return "NOUN"
    role == "verb" && return "VERB"
    role == "object" && return "OBJECT"
    role == "prep_object" && return "NOUN"
    role == "preposition" && return "PREP"
    role == "determiner" && return "DET"
    role == "be_verb" && return "BE"
    role == "auxiliary" && return "AUX"
    role == "negation" && return "NOT"
    role == "participle" && return "PARTICIPLE"
    role == "adjective" && return "ADJ"
    role == "condition_tool" && return "COND"
    role == "question_tool" && return "QUESTION"
    role == "relative_pronoun" && return "REL"
    role == "marker" && return "MARKER"
    role == "conjunction" && return "CONJ"
    role == "particle" && return "PART"
    return uppercase(role)
end

function _extract_pattern(mem::LinguisticPatternMemory, sentence::AbstractString)
    tokens = _sentence_tokens(sentence)
    2 <= length(tokens) <= 16 || return nothing
    language = detect_lisan_language(tokens)
    kinds = [_token_kind(mem, tokens[i], i, language) for i in eachindex(tokens)]
    roles = [_slot_role(kinds, i, language) for i in eachindex(kinds)]
    shape = join(_shape_label.(roles), " ")
    return tokens, roles, shape, language
end

function learn_lisan_from_text!(mem::LinguisticPatternMemory, text::AbstractString;
                                max_sentences::Int=typemax(Int))
    learned = 0
    for sentence in _split_sentences(text)
        learned >= max_sentences && break
        extracted = _extract_pattern(mem, sentence)
        extracted === nothing && continue
        tokens, roles, shape, language = extracted
        key = string(language, "\t", shape)
        rec = get!(mem.patterns, key) do
            LinguisticPatternRecord(language, shape, copy(roles), 0, String[])
        end
        rec.count += 1
        if length(rec.examples) < mem.max_examples && !(sentence in rec.examples)
            push!(rec.examples, String(sentence))
        end
        learned += 1
    end
    return learned
end

function train_lisan_from_texts!(mem::LinguisticPatternMemory, texts::Vector{String};
                                 max_sentences::Int=50_000)
    total = 0
    for text in texts
        total >= max_sentences && break
        total += learn_lisan_from_text!(mem, text; max_sentences=max_sentences - total)
    end
    return total
end

has_lisan_patterns(mem::LinguisticPatternMemory) = !isempty(mem.patterns)

function pattern_has_verb(rec::LinguisticPatternRecord)
    return any(r -> r in ("verb", "be_verb", "participle"), rec.roles)
end

function select_lisan_pattern(mem::LinguisticPatternMemory, prompt_tokens::Vector{String};
                              prefer_verbal::Bool=true)
    isempty(mem.patterns) && return nothing
    language = detect_lisan_language(prompt_tokens)
    scored = Tuple{Float64,LinguisticPatternRecord}[]
    for rec in values(mem.patterns)
        rec.language == language || continue
        score = log(1 + rec.count)
        prefer_verbal && pattern_has_verb(rec) && (score += 0.85)
        length(rec.roles) > 10 && (score -= 0.25)
        push!(scored, (score, rec))
    end
    isempty(scored) && return nothing
    sort!(scored; by=x -> -x[1])
    return scored[1][2]
end

function lisan_to_dict(mem::LinguisticPatternMemory)
    records = sort(collect(values(mem.patterns)); by=r -> (-r.count, r.shape))
    return Dict{String,Any}(
        "version" => AL_LISAN_VERSION,
        "n_patterns" => length(records),
        "patterns" => [Dict{String,Any}(
            "shape" => rec.shape,
            "language" => rec.language,
            "roles" => rec.roles,
            "count" => rec.count,
            "examples" => rec.examples,
        ) for rec in records],
    )
end

function save_lisan(mem::LinguisticPatternMemory, path::String)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, lisan_to_dict(mem))
    end
    return path
end

function load_lisan(path::String)
    mem = LinguisticPatternMemory()
    isfile(path) || return mem
    data = JSON.parsefile(path)
    patterns = get(data, "patterns", Any[])
    for item in patterns
        item isa AbstractDict || continue
        shape = String(get(item, "shape", ""))
        isempty(shape) && continue
        language = String(get(item, "language", "ar"))
        roles = String[String(r) for r in get(item, "roles", String[])]
        examples = String[String(e) for e in get(item, "examples", String[])]
        key = string(language, "\t", shape)
        mem.patterns[key] = LinguisticPatternRecord(
            language,
            shape,
            roles,
            Int(get(item, "count", 0)),
            examples,
        )
    end
    return mem
end

end # module AlLisan
