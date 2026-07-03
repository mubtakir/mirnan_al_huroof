module AlNisba

using JSON

export NisbaRelationRecord, NisbaMemory,
       learn_nisba_fact!, learn_nisba_from_text!, train_nisba_from_texts!,
       select_nisba_relation, nisba_guidance_terms,
       save_nisba, load_nisba, nisba_to_dict,
       has_nisba_relations

const AL_NISBA_VERSION = 1

mutable struct NisbaRelationRecord
    relation_id::String
    concepts::Vector{String}
    relation_type::String
    direction::Vector{Int}
    polarity::Int
    intensity::Float64
    markers::Vector{String}
    evidences::Vector{String}
    slots::Dict{String,Vector{String}}
    count::Int
    source_metadata::Vector{Dict{String,Any}}
end

# Outer constructor for backward compatibility
NisbaRelationRecord(relation_id::String, concepts::Vector{String}, relation_type::String, direction::Vector{Int}, polarity::Int, intensity::Float64, markers::Vector{String}, evidences::Vector{String}, slots::Dict{String,Vector{String}}, count::Int) =
    NisbaRelationRecord(relation_id, concepts, relation_type, direction, polarity, intensity, markers, evidences, slots, count, [Dict{String,Any}() for _ in 1:length(evidences)])

mutable struct NisbaMemory
    relations::Dict{String,NisbaRelationRecord}
    max_evidences::Int
end

NisbaMemory(; max_evidences::Int=5) =
    NisbaMemory(Dict{String,NisbaRelationRecord}(), max_evidences)

has_nisba_relations(mem::NisbaMemory) = !isempty(mem.relations)

_clean_space(s::AbstractString) = replace(strip(String(s)), r"\s+" => " ")

const RELATION_MARKERS = Dict(
    "analogy" => ["يشبه", "شبيه", "مثل", "كمثل", "كـ", "كال", "like", "similar"],
    "causal" => ["لأن", "لان", "بسبب", "لذلك", "يفتح", "يكشف", "يحفظ", "يزيد", "ينتج", "because", "therefore", "raises", "strengthens"],
    "prevention" => ["يمنع", "تمنع", "يحمي", "تحمي", "يوقف", "توقف", "يحجب", "تحجب", "يهذب", "تهذب", "prevents", "protects"],
    "need" => ["يحتاج", "تحتاج", "لا يكفي", "لا تكفي", "يكفي", "تكفي", "needs", "requires", "not enough"],
    "transform" => ["يتحول", "تتحول", "يصير", "تصير", "ينقلب", "تنقلب", "becomes", "turns"],
    "difference" => ["الفرق", "بين", "أما", "اما", "يختلف", "تختلف", "difference", "between"],
    "negation" => ["ليس", "ليست", "لا", "بلا", "دون", "غير", "not", "without"],
)

const BUILD_RELATION_MARKERS = [
    "\u064a\u0628\u0646\u064a", "\u062a\u0628\u0646\u064a",
    "\u064a\u0635\u0646\u0639", "\u062a\u0635\u0646\u0639",
]

const STOPWORDS = Set([
    "ما", "ماذا", "من", "كيف", "لماذا", "هل", "متى", "اين", "أين",
    "في", "من", "عن", "على", "الى", "إلى", "الي", "مع", "او", "أو",
    "هو", "هي", "هذا", "هذه", "ذلك", "تلك", "الذي", "التي", "كل",
    "قد", "ثم", "حين", "عند", "اذا", "إذا", "ان", "أن", "انه", "أنّه", "لان", "لأن",
    "لا", "ليس", "ليست", "بلا", "دون", "فقط", "بل", "اما", "أما",
    "سؤال", "جواب", "س", "ج",
    "what", "why", "how", "when", "is", "are", "the", "a", "an", "to", "of", "and", "or", "because",
])

function _norm_word(s::AbstractString)
    x = lowercase(_clean_space(s))
    x = replace(x, 'أ' => 'ا', 'إ' => 'ا', 'آ' => 'ا', 'ى' => 'ي', 'ة' => 'ه')
    x = replace(x, r"^[\W_]+|[\W_]+$" => "")
    x = replace(x, r"^(?:وال|فال|بال|كال|لل|ال)" => "")
    x = replace(x, r"[\.\,\،\؛\:\!\?\u061F\"'\(\)\[\]\{\}]" => "")
    return strip(x)
end

function _projection_word(s::AbstractString)
    key = _norm_word(s)
    projected = replace(key, r"[\u064B-\u065F\u0670]" => "")
    if occursin('\u064B', key) && endswith(projected, "ا")
        projected = projected[begin:prevind(projected, lastindex(projected))]
    end
    return projected
end

function _token_set(text::AbstractString)
    out = Set{String}()
    for t in _tokens(text)
        push!(out, t)
        p = _projection_word(t)
        isempty(p) || push!(out, p)
    end
    return out
end

function _tokens(text::AbstractString)
    raw = split(_clean_space(text), r"\s+")
    toks = String[]
    for w in raw
        n = _norm_word(w)
        isempty(n) && continue
        n in STOPWORDS && continue
        length(n) < 2 && continue
        push!(toks, n)
    end
    return toks
end

function _unique_keep_order(values)
    seen = Set{String}()
    out = String[]
    for v in values
        n = _norm_word(String(v))
        isempty(n) && continue
        n in seen && continue
        push!(seen, n)
        push!(out, n)
    end
    return out
end

function _sentences(text::AbstractString)
    parts = split(String(text), r"(?<=[\.\!\?\u061F؛;:])\s+|\n+")
    return String[_clean_space(p) for p in parts if !isempty(_clean_space(p))]
end

function _contains_marker(sentence::String, marker::String)
    nmarker = _norm_word(marker)
    return occursin(marker, sentence) || (lastindex(nmarker) >= 4 && occursin(nmarker, _norm_word(sentence)))
end

function _relation_type_and_markers(sentence::String)
    found = Dict{String,Vector{String}}()
    for (rtype, markers) in RELATION_MARKERS
        hits = String[m for m in markers if _contains_marker(sentence, m)]
        isempty(hits) || (found[rtype] = hits)
    end
    build_hits = String[m for m in BUILD_RELATION_MARKERS if _contains_marker(sentence, m)]
    if !isempty(build_hits)
        found["causal"] = unique(vcat(get(found, "causal", String[]), build_hits))
    end
    priority = ["difference", "analogy", "transform", "need", "prevention", "causal", "negation"]
    for rtype in priority
        haskey(found, rtype) && return rtype, found[rtype]
    end
    return "", String[]
end

function _polarity(sentence::String)
    s = _clean_space(sentence)
    occursin("لا ", s) || occursin("ليس", s) || occursin("ليست", s) ||
        occursin("بلا", s) || occursin("دون", s) ? -1 : 1
end

function _concepts_by_pattern(sentence::String, relation_type::String)
    s = _clean_space(sentence)

    if relation_type == "difference"
        m = match(r"بين\s+(.{2,40}?)\s+و\s*(.{2,40}?)(?:[\.\؟\?]|$)", s)
        m !== nothing && return _unique_keep_order([m.captures[1], m.captures[2]])
    end

    if relation_type == "analogy"
        pats = [
            r"(.{2,35}?)\s+(?:يشبه|شبيه\s+ب|مثل)\s+(.{2,35}?)(?:[\s\.\،\؛]|$)",
            r"(.{2,35}?)\s+(?:كـ|كال)(.{2,25}?)(?:[\s\.\،\؛]|$)",
        ]
        for pat in pats
            m = match(pat, s)
            m !== nothing && return _unique_keep_order([m.captures[1], m.captures[2]])
        end
    end

    if relation_type == "transform"
        m = match(r"(?:يتحول|تتحول|ينقلب|تنقلب)\s+(.{2,35}?)\s+(?:الى|إلى|الي)\s+(.{2,35}?)(?:\s+حين|[\.\،\؛]|$)", s)
        if m !== nothing
            left = _tokens(m.captures[1])
            right = _tokens(m.captures[2])
            return _unique_keep_order(vcat(left[1:min(2, length(left))], right[1:min(2, length(right))]))
        end
        m = match(r"(.{2,45}?)\s+(?:يصير|تصير)\s+(.{2,35}?)(?:[\.\،\؛]|$)", s)
        if m !== nothing
            left = _tokens(m.captures[1])
            right = _tokens(m.captures[2])
            chosen_left = isempty(left) ? String[] : left[max(1, length(left)-1):length(left)]
            return _unique_keep_order(vcat(chosen_left, right[1:min(2, length(right))]))
        end
    end

    if relation_type == "need"
        m = match(r"(.{2,35}?)\s+(?:يحتاج|تحتاج)\s+(?:الى|إلى|الي)?\s*(.{2,35}?)(?:[\.\،\؛]|$)", s)
        m !== nothing && return _unique_keep_order([m.captures[1], m.captures[2]])
    end

    if relation_type == "causal" && any(m -> _contains_marker(s, m), BUILD_RELATION_MARKERS)
        toks = _tokens(s)
        marker_idx = findfirst(t -> any(m -> _norm_word(m) == t, BUILD_RELATION_MARKERS), toks)
        if marker_idx !== nothing && marker_idx < length(toks)
            target = toks[marker_idx + 1]
            rest = String[t for (i, t) in enumerate(toks) if i != marker_idx && i != marker_idx + 1]
            return _unique_keep_order(vcat([target], reverse(rest)))
        end
    end

    toks = _tokens(s)
    return toks[1:min(4, length(toks))]
end

function _learned_relation(sentence::String)
    s = _clean_space(sentence)
    (occursin("?", s) || occursin("؟", s)) && return nothing
    occursin(r"^(?:سؤال|س|جواب|ج)\s*[:：]", s) && return nothing
    occursin(r"^(?:ما|ماذا|كيف|لماذا|هل|متى|اين|أين)\b", s) && return nothing
    relation_type, markers = _relation_type_and_markers(sentence)
    isempty(relation_type) && return nothing
    concepts = _concepts_by_pattern(sentence, relation_type)
    length(concepts) < 2 && return nothing
    length(split(sentence)) > 45 && return nothing
    return (relation_type=relation_type, markers=markers,
            concepts=concepts[1:min(4, length(concepts))],
            polarity=_polarity(sentence))
end

function _relation_key(relation_type::String, polarity::Int, concepts::Vector{String})
    core = join(concepts[1:min(3, length(concepts))], "|")
    return "$(relation_type)|$(polarity)|$(core)"
end

function _record!(mem::NisbaMemory, sentence::String, relation_type::String,
                  concepts::AbstractVector, markers::AbstractVector, polarity::Int,
                  source::Dict{String,Any}=Dict{String,Any}())
    clean_concepts = String.(concepts)
    clean_markers = String.(markers)
    key = _relation_key(relation_type, polarity, clean_concepts)
    if !haskey(mem.relations, key)
        mem.relations[key] = NisbaRelationRecord(
            key, clean_concepts, relation_type,
            collect(1:length(clean_concepts)), polarity, 0.35,
            clean_markers, String[], Dict("concepts" => clean_concepts), 0,
            Dict{String,Any}[]
        )
    end
    rec = mem.relations[key]
    rec.count += 1
    rec.intensity = min(1.0, 0.35 + 0.08 * rec.count + 0.04 * length(rec.evidences))
    for m in clean_markers
        nm = _clean_space(m)
        !isempty(nm) && !(nm in rec.markers) && push!(rec.markers, nm)
    end
    if !isempty(strip(sentence)) && !(sentence in rec.evidences)
        push!(rec.evidences, sentence)
        push!(rec.source_metadata, source)
        if length(rec.evidences) > mem.max_evidences
            popfirst!(rec.evidences)
            popfirst!(rec.source_metadata)
        end
    end
    return rec
end

function learn_nisba_fact!(mem::NisbaMemory, relation_type::AbstractString,
                           concepts::AbstractVector;
                           markers::AbstractVector=String[],
                           polarity::Int=1,
                           source::Dict{String,Any}=Dict{String,Any}())
    clean_relation = String(strip(String(relation_type)))
    clean_concepts = _unique_keep_order(concepts)
    isempty(clean_relation) && return 0
    length(clean_concepts) < 2 && return 0
    _record!(mem, "", clean_relation, clean_concepts[1:min(4, length(clean_concepts))],
             String.(markers), polarity < 0 ? -1 : 1, source)
    return 1
end

function learn_nisba_from_text!(mem::NisbaMemory, text::AbstractString, source::Dict{String,Any}=Dict{String,Any}())
    learned = 0
    for sentence in _sentences(text)
        item = _learned_relation(sentence)
        item === nothing && continue
        _record!(mem, sentence, item.relation_type, item.concepts, item.markers, item.polarity, source)
        learned += 1
    end
    return learned
end

function train_nisba_from_texts!(mem::NisbaMemory, texts, metadata=nothing; max_items::Int=50_000)
    learned = 0
    for (i, text) in enumerate(texts)
        learned >= max_items && break
        source = (metadata !== nothing && i <= length(metadata)) ? metadata[i] : Dict{String,Any}()
        learned += learn_nisba_from_text!(mem, String(text), source)
    end
    return learned
end

function _prompt_relation_hint(prompt::AbstractString)
    p = _clean_space(prompt)
    (occursin("العلاقة بين", p) || occursin("العلاقه بين", p)) && return ""
    rtype, _ = _relation_type_and_markers(_clean_space(prompt))
    !isempty(rtype) && return rtype
    occursin("ما الذي يجعل", p) && return "causal"
    occursin("لماذا", p) && return "causal"
    occursin("كيف", p) && return "causal"
    return ""
end

function select_nisba_relation(mem::NisbaMemory, prompt::AbstractString;
                               relation_type::AbstractString="", min_score::Float64=0.25,
                               active_paras::Union{Nothing,Dict{Tuple{String,Int},Float64}}=nothing)
    isempty(mem.relations) && return nothing
    ptoks = _token_set(prompt)
    isempty(ptoks) && return nothing
    hint = isempty(relation_type) ? _prompt_relation_hint(prompt) : String(relation_type)
    scored = Tuple{Float64,NisbaRelationRecord}[]
    for rec in values(mem.relations)
        overlap = count(c -> c in ptoks || _projection_word(c) in ptoks, rec.concepts)
        overlap == 0 && continue
        score = overlap / max(length(rec.concepts), 1)
        score += min(rec.count, 8) * 0.03
        score += rec.intensity * 0.15
        if !isempty(hint)
            rec.relation_type == hint ? (score += 0.55) : (score -= 0.20)
        end
        if active_paras !== nothing
            act = 0.0
            if isempty(rec.source_metadata)
                act = 1.0
            else
                for src in rec.source_metadata
                    f_name = get(src, "file_name", "")
                    p_idx = get(src, "paragraph_index", 0)
                    if !isempty(f_name) && p_idx > 0
                        act = max(act, get(active_paras, (String(f_name), Int(p_idx)), 0.0))
                    end
                end
            end
            act > 0.45 || continue
            score *= act
        end
        push!(scored, (score, rec))
    end
    isempty(scored) && return nothing
    sort!(scored; by=x -> -x[1])
    return scored[1][1] >= min_score ? scored[1][2] : nothing
end

function nisba_guidance_terms(rec::NisbaRelationRecord)
    terms = String[]
    append!(terms, rec.concepts)
    append!(terms, rec.markers[1:min(length(rec.markers), 2)])
    if rec.relation_type == "analogy"
        append!(terms, ["يشبه", "يكشف", "يهدي"])
    elseif rec.relation_type == "causal"
        append!(terms, ["لأن", "ينتج", "يؤدي"])
    elseif rec.relation_type == "prevention"
        append!(terms, ["يمنع", "يحمي", "يوقف"])
    elseif rec.relation_type == "need"
        append!(terms, ["يحتاج", "يكمل", "لا يكفي"])
    elseif rec.relation_type == "transform"
        append!(terms, ["يتحول", "يصير", "حين"])
    elseif rec.relation_type == "difference"
        append!(terms, ["بين", "أما", "يفرق"])
    elseif rec.relation_type == "negation"
        append!(terms, ["لا", "ليس", "بل"])
    end
    return _unique_keep_order(terms)
end

function _record_to_dict(rec::NisbaRelationRecord)
    return Dict(
        "relation_id" => rec.relation_id,
        "concepts" => rec.concepts,
        "relation_type" => rec.relation_type,
        "direction" => rec.direction,
        "polarity" => rec.polarity,
        "intensity" => rec.intensity,
        "markers" => rec.markers,
        "evidences" => rec.evidences,
        "slots" => rec.slots,
        "count" => rec.count,
        "source_metadata" => rec.source_metadata,
    )
end

function nisba_to_dict(mem::NisbaMemory)
    return Dict(
        "version" => AL_NISBA_VERSION,
        "max_evidences" => mem.max_evidences,
        "relations" => [_record_to_dict(r) for r in values(mem.relations)],
    )
end

function save_nisba(mem::NisbaMemory, path::AbstractString)
    open(String(path), "w") do io
        JSON.print(io, nisba_to_dict(mem), 2)
    end
    return String(path)
end

function _string_vec(v)
    return String[String(x) for x in v]
end

function _int_vec(v)
    return Int[Int(x) for x in v]
end

function load_nisba(path::AbstractString)
    isfile(String(path)) || return NisbaMemory()
    data = JSON.parsefile(String(path))
    mem = NisbaMemory(max_evidences=Int(get(data, "max_evidences", 5)))
    for item in get(data, "relations", Any[])
        slots = Dict{String,Vector{String}}()
        for (k, v) in get(item, "slots", Dict{String,Any}())
            slots[String(k)] = _string_vec(v)
        end
        
        evidences = _string_vec(item["evidences"])
        raw_source_meta = get(item, "source_metadata", nothing)
        source_metadata = Dict{String,Any}[]
        if raw_source_meta isa AbstractVector
            for x in raw_source_meta
                if x isa AbstractDict
                    push!(source_metadata, Dict{String,Any}(String(k) => v for (k, v) in x))
                else
                    push!(source_metadata, Dict{String,Any}())
                end
            end
        end
        while length(source_metadata) < length(evidences)
            push!(source_metadata, Dict{String,Any}())
        end
        if length(source_metadata) > length(evidences)
            resize!(source_metadata, length(evidences))
        end
        
        rec = NisbaRelationRecord(
            String(item["relation_id"]),
            _string_vec(item["concepts"]),
            String(item["relation_type"]),
            _int_vec(item["direction"]),
            Int(item["polarity"]),
            Float64(item["intensity"]),
            _string_vec(item["markers"]),
            evidences,
            slots,
            Int(item["count"]),
            source_metadata,
        )
        mem.relations[rec.relation_id] = rec
    end
    return mem
end

end # module
