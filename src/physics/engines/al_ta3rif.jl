module AlTa3rif

using JSON

export Ta3rifRecord, Ta3rifMemory,
       learn_ta3rif_from_text!, train_ta3rif_from_texts!,
       answer_ta3rif, save_ta3rif, load_ta3rif,
       merge_ta3rif!, ta3rif_to_dict, has_ta3rif_records

const AL_TA3RIF_VERSION = 1

mutable struct Ta3rifRecord
    subject::String
    count::Int
    definitions::Dict{String,Int}
    relations::Dict{String,Int}
    attributes::Dict{String,Int}
    examples::Vector{String}
    source_metadata::Vector{Dict{String,Any}}
end

# Outer constructor for backward compatibility
Ta3rifRecord(subject::String, count::Int, definitions::Dict{String,Int}, relations::Dict{String,Int}, attributes::Dict{String,Int}, examples::Vector{String}) =
    Ta3rifRecord(subject, count, definitions, relations, attributes, examples, [Dict{String,Any}() for _ in 1:length(examples)])

mutable struct Ta3rifMemory
    records::Dict{String,Ta3rifRecord}
    max_items::Int
    max_examples::Int
end

Ta3rifMemory(; max_items::Int=80, max_examples::Int=5) =
    Ta3rifMemory(Dict{String,Ta3rifRecord}(), max_items, max_examples)

has_ta3rif_records(mem::Ta3rifMemory) = !isempty(mem.records)

_clean_space(s::AbstractString) = replace(strip(String(s)), r"\s+" => " ")

_looks_english(s::AbstractString) =
    occursin(r"[A-Za-z]", String(s)) && !occursin(r"[\u0600-\u06FF]", String(s))

const FEMININE_SEMANTIC_SUBJECTS = Set([
    "ارض", "الارض", "نفس", "النفس", "روح", "الروح",
    "شمس", "الشمس", "سماء", "السماء", "حياه", "الحياه",
    "حكمه", "الحكمه", "معرفه", "المعرفه", "لغه", "اللغه",
])

function _arabic_copula(subject::AbstractString)
    s = _clean_space(subject)
    n = _norm_subject(s)
    if endswith(s, "ة") || endswith(s, "ى") || endswith(s, "اء") ||
       n in FEMININE_SEMANTIC_SUBJECTS
        return "هي"
    end
    return "هو"
end

function _norm_subject(s::AbstractString)
    x = lowercase(_clean_space(s))
    x = replace(x, 'أ' => 'ا', 'إ' => 'ا', 'آ' => 'ا', 'ى' => 'ي', 'ة' => 'ه')
    x = replace(x, r"^(?:و|ف)+" => "")
    return strip(x)
end

function _projection_subject(s::AbstractString)
    key = _norm_subject(s)
    projected = replace(key, r"[\u064B-\u065F\u0670]" => "")
    if occursin('\u064B', key) && endswith(projected, "ا")
        projected = projected[begin:prevind(projected, lastindex(projected))]
    end
    return projected
end

function _subject_variants(s::AbstractString)
    n = _norm_subject(s)
    p = _projection_subject(s)
    variants = Set{String}([n, p])
    for base in (n, p)
        isempty(base) && continue
        startswith(base, "ال") && push!(variants, base[nextind(base, nextind(base, firstindex(base))):end])
        push!(variants, "ال" * base)
    end
    filter!(!isempty, variants)
    return variants
end

function _record!(mem::Ta3rifMemory, subject::AbstractString)
    key = _norm_subject(subject)
    isempty(key) && return nothing
    if !haskey(mem.records, key)
        mem.records[key] = Ta3rifRecord(key, 0, Dict{String,Int}(),
                                        Dict{String,Int}(), Dict{String,Int}(), String[],
                                        Dict{String,Any}[])
    end
    rec = mem.records[key]
    rec.count += 1
    return rec
end

function _valid_arabic_definition_subject(subject::AbstractString, sentence::AbstractString)
    occursin(r"[\u0600-\u06FF]", String(sentence)) || return true
    n = _norm_subject(subject)
    words = split(n)
    isempty(words) && return false
    first_word = first(words)
    (startswith(first_word, "ال") || startswith(first_word, "Ø§Ù„")) || return false
    return length(words) <= 3
end

function _push_count!(bucket::Dict{String,Int}, value::AbstractString, max_items::Int)
    v = _clean_space(value)
    isempty(v) && return
    length(v) < 2 && return
    bucket[v] = get(bucket, v, 0) + 1
    if length(bucket) > max_items
        ordered = sort(collect(bucket); by=x -> (x[2], x[1]))
        delete!(bucket, ordered[1][1])
    end
end

function _push_example!(rec::Ta3rifRecord, example::AbstractString, max_examples::Int, source::Dict{String,Any}=Dict{String,Any}())
    ex = _clean_space(example)
    isempty(ex) && return
    ex in rec.examples && return
    push!(rec.examples, ex)
    push!(rec.source_metadata, source)
    if length(rec.examples) > max_examples
        popfirst!(rec.examples)
        popfirst!(rec.source_metadata)
    end
end

function _sentences(text::AbstractString)
    parts = split(String(text), r"(?<=[\.\!\?\u061F؛;:])\s+|\n+")
    return String[_clean_space(p) for p in parts if !isempty(_clean_space(p))]
end

function _learn_definition!(mem::Ta3rifMemory, sentence::String, source::Dict{String,Any})
    patterns = [
        r"^(.{2,40}?)\s+(?:هو|هي|يعني|تعني|يقصد به|تسمى|يسمى)\s+(.{3,220})$",
        r"^(.{2,60}?)\s+(?:is|are|means|refers to|is defined as)\s+(.{3,220})$"i,
        r"^(?:معنى|تعريف)\s+(.{2,40}?)\s+(?:هو|هي|:)\s*(.{3,220})$",
        r"^(.{2,40}?)\s*[:]\s*(.{3,220})$",
    ]
    for pat in patterns
        m = match(pat, sentence)
        m === nothing && continue
        subject = _clean_space(m.captures[1])
        definition = _clean_space(m.captures[2])
        occursin(r"^(?:ما|ماذا|كيف|هل|من|اين|أين)\b", subject) && continue
        _valid_arabic_definition_subject(subject, sentence) || continue
        rec = _record!(mem, subject)
        rec === nothing && continue
        _push_count!(rec.definitions, definition, mem.max_items)
        _push_example!(rec, sentence, mem.max_examples, source)
        return true
    end
    return false
end

function _learn_relation!(mem::Ta3rifMemory, sentence::String, source::Dict{String,Any})
    patterns = [
        r"^(.{2,40}?)\s+(?:جزء من|نوع من|يرتبط ب|يرتبط بـ|يدل على|ينتمي إلى|ينتمي الي|رمز ل|رمز لـ)\s+(.{2,80})$",
        r"^(.{2,40}?)\s+(?:له|لها|يملك|تملك)\s+(.{2,80})$",
    ]
    for pat in patterns
        m = match(pat, sentence)
        m === nothing && continue
        subject = _clean_space(m.captures[1])
        object = _clean_space(m.captures[2])
        rec = _record!(mem, subject)
        rec === nothing && continue
        _push_count!(rec.relations, sentence, mem.max_items)
        _push_example!(rec, sentence, mem.max_examples, source)
        return true
    end
    return false
end

function _learn_attribute!(mem::Ta3rifMemory, sentence::String, source::Dict{String,Any})
    m = match(r"^(.{2,40}?)\s+(?:صفة|صفته|صفتها|خصيصته|خاصيته|يمتاز ب|تتميز ب|يتصف ب)\s+(.{2,80})$", sentence)
    m === nothing && return false
    rec = _record!(mem, _clean_space(m.captures[1]))
    rec === nothing && return false
    _push_count!(rec.attributes, _clean_space(m.captures[2]), mem.max_items)
    _push_example!(rec, sentence, mem.max_examples, source)
    return true
end

function learn_ta3rif_from_text!(mem::Ta3rifMemory, text::AbstractString, source::Dict{String,Any}=Dict{String,Any}())
    learned = 0
    for sentence in _sentences(text)
        if _learn_definition!(mem, sentence, source) ||
           _learn_relation!(mem, sentence, source) ||
           _learn_attribute!(mem, sentence, source)
            learned += 1
        end
    end
    return learned
end

function train_ta3rif_from_texts!(mem::Ta3rifMemory, texts, metadata=nothing; max_items::Int=50_000)
    learned = 0
    for (i, text) in enumerate(texts)
        learned >= max_items && break
        source = (metadata !== nothing && i <= length(metadata)) ? metadata[i] : Dict{String,Any}()
        learned += learn_ta3rif_from_text!(mem, String(text), source)
    end
    return learned
end

function _best(bucket::Dict{String,Int}; n::Int=3)
    return String[x[1] for x in sort(collect(bucket); by=x -> (-x[2], length(x[1]), x[1]))[1:min(n, length(bucket))]]
end

function _lookup(mem::Ta3rifMemory, subject::AbstractString)
    for v in _subject_variants(subject)
        haskey(mem.records, v) && return mem.records[v]
    end
    p = _projection_subject(subject)
    if !isempty(p)
        for rec in values(mem.records)
            _projection_subject(rec.subject) == p && return rec
        end
    end
    return nothing
end

function _lookup_loose(mem::Ta3rifMemory, subject::AbstractString)
    n = _norm_subject(subject)
    p = _projection_subject(subject)
    isempty(n) && return nothing
    scored = Tuple{Float64,Ta3rifRecord}[]
    for rec in values(mem.records)
        s = _norm_subject(rec.subject)
        sp = _projection_subject(rec.subject)
        isempty(s) && continue
        score = 0.0
        if startswith(s, n * " ") || (!isempty(p) && startswith(sp, p * " "))
            score = 1.0
        elseif occursin(" " * n * " ", " " * s * " ") ||
               (!isempty(p) && occursin(" " * p * " ", " " * sp * " "))
            score = 0.65
        else
            continue
        end
        length(split(s)) > 7 && (score -= 0.25)
        score += min(rec.count, 10) / 100
        push!(scored, (score, rec))
    end
    isempty(scored) && return nothing
    sort!(scored; by=x -> -x[1])
    return scored[1][2]
end

function _definition_quality(value::AbstractString)
    text = _clean_space(value)
    isempty(text) && return -10.0
    words = split(text)
    score = 1.0
    3 <= length(words) <= 18 && (score += 0.7)
    length(words) > 28 && (score -= 1.0)
    startswith(text, "الذي ") && (score -= 0.8)
    startswith(text, "التي ") && (score -= 0.8)
    occursin("تصنيف:", text) && (score -= 1.5)
    occursin("سؤال", text) && (score -= 0.7)
    occursin("جواب", text) && (score -= 0.7)
    occursin(":", text) && length(words) > 12 && (score -= 0.5)
    return score
end

function _definition_subject(prompt::AbstractString)
    text = _clean_space(prompt)
    patterns = [
        r"^(?:ما|ماذا)\s+(?:هو|هي)\s+(.+?)[\؟\?\.]?$",
        r"^(?:ما|ماذا)\s+معنى\s+(.+?)[\؟\?\.]?$",
        r"^(?:عرف|عرّف|اشرح)\s+(.+?)[\؟\?\.]?$",
        r"^(?:what\s+is|define|explain)\s+(.+?)[\?\.]?$"i,
    ]
    for pat in patterns
        m = match(pat, text)
        m !== nothing && return _clean_space(m.captures[1])
    end
    return ""
end

function answer_ta3rif(mem::Ta3rifMemory, prompt::AbstractString;
                       active_paras::Union{Nothing,Dict{Tuple{String,Int},Float64}}=nothing)
    subject = _definition_subject(prompt)
    isempty(subject) && return ""
    rec = _lookup(mem, subject)
    rec === nothing && (rec = _lookup_loose(mem, subject))
    rec === nothing && return ""
    
    # Filter definitions, relations, attributes if active paragraphs are provided
    filtered_definitions = Dict{String,Int}()
    filtered_relations = Dict{String,Int}()
    filtered_attributes = Dict{String,Int}()
    
    if active_paras === nothing
        filtered_definitions = rec.definitions
        filtered_relations = rec.relations
        filtered_attributes = rec.attributes
    else
        active_sentences = Set{String}()
        for (i, ex) in enumerate(rec.examples)
            src = (i <= length(rec.source_metadata)) ? rec.source_metadata[i] : Dict{String,Any}()
            act = 0.0
            if isempty(src)
                act = 1.0
            else
                f_name = get(src, "file_name", "")
                p_idx = get(src, "paragraph_index", 0)
                if !isempty(f_name) && p_idx > 0
                    act = get(active_paras, (String(f_name), Int(p_idx)), 0.0)
                end
            end
            if act > 0.45
                push!(active_sentences, ex)
            end
        end
        
        if isempty(rec.source_metadata)
            filtered_definitions = rec.definitions
            filtered_relations = rec.relations
            filtered_attributes = rec.attributes
        else
            for (def, count) in rec.definitions
                if any(ex -> occursin(def, ex), active_sentences)
                    filtered_definitions[def] = count
                end
            end
            for (rel, count) in rec.relations
                if rel in active_sentences
                    filtered_relations[rel] = count
                end
            end
            for (attr, count) in rec.attributes
                if any(ex -> occursin(attr, ex), active_sentences)
                    filtered_attributes[attr] = count
                end
            end
        end
    end
    
    defs = String[x[1] for x in sort(collect(filtered_definitions);
                                     by=x -> (-_definition_quality(x[1]), -x[2], length(x[1]), x[1]))]
    defs = defs[1:min(1, length(defs))]
    rels = _best(filtered_relations; n=2)
    attrs = _best(filtered_attributes; n=2)
    parts = String[]
    if !isempty(defs)
        if _looks_english(subject)
            push!(parts, "$(subject) is $(defs[1])")
        else
            push!(parts, "$(subject) $(_arabic_copula(subject)) $(defs[1])")
        end
    end
    append!(parts, rels)
    for attr in attrs
        push!(parts, "ومن صفاته $(attr)")
    end
    isempty(parts) && return ""
    text = join(parts, "، ")
    endswith(text, ".") || (text *= ".")
    return text
end

function merge_ta3rif!(base::Ta3rifMemory, extra::Ta3rifMemory)
    for (subject, incoming) in extra.records
        rec = _record!(base, subject)
        rec === nothing && continue
        rec.count += max(incoming.count - 1, 0)
        for (definition, count) in incoming.definitions
            rec.definitions[definition] = get(rec.definitions, definition, 0) + count
        end
        for (relation, count) in incoming.relations
            rec.relations[relation] = get(rec.relations, relation, 0) + count
        end
        for (attribute, count) in incoming.attributes
            rec.attributes[attribute] = get(rec.attributes, attribute, 0) + count
        end
        for (i, ex) in enumerate(incoming.examples)
            src = (i <= length(incoming.source_metadata)) ? incoming.source_metadata[i] : Dict{String,Any}()
            _push_example!(rec, ex, base.max_examples, src)
        end
    end
    return base
end

function ta3rif_to_dict(mem::Ta3rifMemory)
    records = sort(collect(values(mem.records)); by=r -> (-r.count, r.subject))
    return Dict{String,Any}(
        "version" => AL_TA3RIF_VERSION,
        "n_records" => length(records),
        "records" => [Dict{String,Any}(
            "subject" => rec.subject,
            "count" => rec.count,
            "definitions" => rec.definitions,
            "relations" => rec.relations,
            "attributes" => rec.attributes,
            "examples" => rec.examples,
            "source_metadata" => rec.source_metadata,
        ) for rec in records],
    )
end

function save_ta3rif(mem::Ta3rifMemory, path::String)
    mkpath(dirname(path))
    tmp = string(path, ".tmp")
    open(tmp, "w") do io
        JSON.print(io, ta3rif_to_dict(mem))
    end
    mv(tmp, path; force=true)
    return path
end

function load_ta3rif(path::String)
    mem = Ta3rifMemory()
    isfile(path) || return mem
    filesize(path) == 0 && return mem
    data = try
        JSON.parsefile(path)
    catch e
        @warn "تعذر تحميل al_ta3rif: $path — $e"
        return mem
    end
    for item in get(data, "records", Any[])
        item isa AbstractDict || continue
        subject = String(get(item, "subject", ""))
        isempty(subject) && continue
        
        examples = String[String(x) for x in get(item, "examples", String[])]
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
        while length(source_metadata) < length(examples)
            push!(source_metadata, Dict{String,Any}())
        end
        if length(source_metadata) > length(examples)
            resize!(source_metadata, length(examples))
        end
        
        rec = Ta3rifRecord(
            subject,
            Int(get(item, "count", 0)),
            Dict{String,Int}(String(k) => Int(v) for (k, v) in get(item, "definitions", Dict{String,Any}())),
            Dict{String,Int}(String(k) => Int(v) for (k, v) in get(item, "relations", Dict{String,Any}())),
            Dict{String,Int}(String(k) => Int(v) for (k, v) in get(item, "attributes", Dict{String,Any}())),
            examples,
            source_metadata,
        )
        mem.records[subject] = rec
    end
    return mem
end

end # module AlTa3rif
