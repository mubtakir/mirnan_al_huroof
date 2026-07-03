module AlIstinbat

using JSON
using ..AlHisbanAlDalali: SemanticCalculusMemory
using ..SemanticImagination: SemanticSceneMemory, compare_semantic_scene_with_calculus,
                             semantic_scene_answer

export IstinbatAttentionRecord, IstinbatAttentionMemory, RelationFrame, QuantityFrame,
       QuantityFrameMemory,
       learn_istinbat_fact!, learn_istinbat_from_text!, train_istinbat_from_texts!,
       select_istinbat_attention, istinbat_focus_terms,
       select_causal_anchor_attention, causal_anchor_answer_from_attention,
       select_contradiction_attention, contradiction_answer_from_attention,
       learn_opposition_from_text!, learn_direct_negation_from_text!,
       terms_are_opposed, terms_are_negated,
       merge_istinbat!, save_istinbat, load_istinbat,
       istinbat_to_dict, has_istinbat_records, update_marker_confidence!,
       select_relation_frame_attention, relation_frame_diagnostic,
       relation_type_for_marker, extract_relation_frames,
       relation_type_for_quantity_marker, extract_quantity_frames,
       learn_quantity_frames_from_text!, train_quantity_frames_from_texts!,
       quantity_memory_to_dict, save_quantity_memory, load_quantity_memory,
       has_quantity_records,
       select_quantity_frame, quantity_answer,
       learn_relation_frames_from_text!,
       purpose_answer, conditional_answer, temporal_answer, spatial_answer, state_answer,
       PurposeComparisonRecord, compare_purpose_strategies,
       ConditionalComparisonRecord, compare_conditional_strategies,
       TemporalComparisonRecord, compare_temporal_strategies,
       SpatialComparisonRecord, compare_spatial_strategies,
       StateComparisonRecord, compare_state_strategies,
       QuantityComparisonRecord, compare_quantity_strategies,
       ScenePurposeComparisonRecord, compare_scene_purpose_strategies,
       scene_purpose_answer

const AL_ISTINBAT_VERSION = 1

const DEFAULT_MARKER_TYPES = Pair{String,String}[
    "\u0644\u0623\u0646\u0647" => "causal",
    "\u0644\u0623\u0646\u0647\u0627" => "causal",
    "\u0644\u0623\u0646" => "causal",
    "\u0644\u0627\u0646\u0647" => "causal",
    "\u0644\u0627\u0646\u0647\u0627" => "causal",
    "\u0644\u0627\u0646" => "causal",
    "\u0628\u0633\u0628\u0628" => "causal",
    "\u064a\u0624\u062f\u064a" => "causal",
    "\u062a\u0624\u062f\u064a" => "causal",
    "\u064a\u0632\u064a\u062f" => "causal",
    "\u062a\u0632\u064a\u062f" => "causal",
    "\u062d\u064a\u0646" => "causal_anchor",
    "\u0639\u0646\u062f\u0645\u0627" => "causal_anchor",
    "\u062a\u062a\u062d\u0648\u0644" => "transform",
    "\u064a\u062a\u062d\u0648\u0644" => "transform",
    "\u064a\u062d\u062a\u0627\u062c" => "need",
    "\u062a\u062d\u062a\u0627\u062c" => "need",
    "\u0627\u0644\u0641\u0631\u0642" => "difference",
]

const NEGATIVE_OPERATOR_MARKERS = Set([
    "\u064a\u062d\u062c\u0628", "\u062a\u062d\u062c\u0628",
    "\u064a\u0641\u0633\u062f", "\u062a\u0641\u0633\u062f",
    "\u064a\u0636\u0639\u0641", "\u062a\u0636\u0639\u0641",
    "\u064a\u0647\u062f\u0645", "\u062a\u0647\u062f\u0645",
])

# --- New marker types for RelationFrame extraction ---
const PURPOSE_MARKERS = Pair{String,String}[
    "\u0644\u0643\u064a" => "purpose",
    "\u0643\u064a" => "purpose",
    "\u0644\u0623\u062c\u0644" => "purpose",
    "\u0645\u0646 \u0623\u062c\u0644" => "purpose",
    "\u0628\u063a\u064a\u0629" => "purpose",
    "\u0642\u0635\u062f" => "purpose",
    "in order to" => "purpose",
    "so that" => "purpose",
    "for the purpose of" => "purpose",
]

const CONDITIONAL_MARKERS = Pair{String,String}[
    "\u0625\u0630\u0627" => "conditional",
    "\u0625\u0646" => "conditional",
    "\u0644\u0648" => "conditional",
    "\u0644\u0648\u0644\u0627" => "conditional",
    "\u0643\u0644\u0645\u0627" => "conditional",
    "\u0645\u0647\u0645\u0627" => "conditional",
    "\u062d\u064a\u062b\u0645\u0627" => "conditional",
    "\u0645\u062a\u0649 \u0645\u0627" => "conditional",
    "if" => "conditional",
    "unless" => "conditional",
    "whenever" => "conditional",
]

const TEMPORAL_MARKERS = Pair{String,String}[
    "\u0642\u0628\u0644" => "temporal",
    "\u0628\u0639\u062f" => "temporal",
    "\u0623\u062b\u0646\u0627\u0621" => "temporal",
    "\u062d\u064a\u0646" => "temporal",
    "\u0639\u0646\u062f\u0645\u0627" => "temporal",
    "\u0645\u0646\u0630" => "temporal",
    "\u0637\u0627\u0644\u0645\u0627" => "temporal",
    "\u0628\u064a\u0646\u0645\u0627" => "temporal",
    "\u0625\u0630" => "temporal",
    "\u0639\u0646\u062f" => "temporal",
    "before" => "temporal",
    "after" => "temporal",
    "during" => "temporal",
    "since" => "temporal",
    "when" => "temporal",
]

const SPATIAL_MARKERS = Pair{String,String}[
    "\u062d\u064a\u062b" => "spatial",
    "\u0641\u0648\u0642" => "spatial",
    "\u062a\u062d\u062a" => "spatial",
    "\u062f\u0627\u062e\u0644" => "spatial",
    "\u0639\u0644\u0649" => "spatial",
    "\u0641\u064a" => "spatial",
    "\u0647\u0646\u0627" => "spatial",
    "\u0647\u0646\u0627\u0643" => "spatial",
    "\u0644\u062f\u0649" => "spatial",
    "where" => "spatial",
    "above" => "spatial",
    "over" => "spatial",
    "under" => "spatial",
    "below" => "spatial",
    "inside" => "spatial",
]

const STATE_MARKERS = Pair{String,String}[
    "\u0641\u064a \u062d\u064a\u0646" => "state",
    "\u062d\u0627\u0644" => "state",
    "\u0648\u0647\u0648" => "state",
    "\u0648\u0647\u064a" => "state",
    "while" => "state",
    "being" => "state",
    "as" => "state",
]

const NEW_MARKER_TYPES = vcat(PURPOSE_MARKERS, CONDITIONAL_MARKERS,
                                TEMPORAL_MARKERS, SPATIAL_MARKERS, STATE_MARKERS)

const ALL_MARKER_TYPES = vcat(DEFAULT_MARKER_TYPES, NEW_MARKER_TYPES)

const _HIGH_CONFIDENCE_MARKERS = Set([
    "\u0644\u0643\u064a", "\u0643\u064a", "\u0644\u0623\u062c\u0644", "\u0645\u0646 \u0623\u062c\u0644",
    "\u0628\u063a\u064a\u0629", "\u0642\u0635\u062f",
    "\u0625\u0630\u0627", "\u0642\u0628\u0644", "\u0628\u0639\u062f",
    "\u0641\u0648\u0642", "\u062a\u062d\u062a", "\u062f\u0627\u062e\u0644", "\u0639\u0644\u0649",
    "in order to", "so that", "before", "after", "above", "over", "under", "below", "inside",
])

const _MEDIUM_CONFIDENCE_MARKERS = Set([
    "\u0625\u0646", "\u0644\u0648", "\u0644\u0648\u0644\u0627",
    "\u0643\u0644\u0645\u0627", "\u0645\u0647\u0645\u0627", "\u062d\u064a\u062b\u0645\u0627", "\u0645\u062a\u0649 \u0645\u0627",
    "\u0623\u062b\u0646\u0627\u0621", "\u062d\u064a\u0646", "\u0639\u0646\u062f\u0645\u0627", "\u0645\u0646\u0630",
    "\u0637\u0627\u0644\u0645\u0627", "\u0628\u064a\u0646\u0645\u0627", "\u0625\u0630",
    "\u0648\u0647\u0648", "\u0648\u0647\u064a",
    "for the purpose of", "if", "unless", "whenever", "during", "since", "when", "while", "being",
])

const _LOW_CONFIDENCE_MARKERS = Set([
    "\u0639\u0646\u062f", "\u062d\u064a\u062b", "\u0641\u064a",
    "\u0647\u0646\u0627", "\u0647\u0646\u0627\u0643", "\u0644\u062f\u0649",
    "\u0641\u064a \u062d\u064a\u0646", "\u062d\u0627\u0644",
    "where", "as",
])
# --- End new marker types ---

# --- Quantity marker types for QuantityFrame extraction ---
const QUANTITY_COUNT_MARKERS = Pair{String,String}[
    "\u0643\u0645" => "count",
    "\u0639\u062f\u062f" => "count",
    "how many" => "count",
    "number" => "count",
    "number of" => "count",
]

const QUANTITY_MEASURE_MARKERS = Pair{String,String}[
    "\u0645\u0642\u062f\u0627\u0631" => "measure",
    "\u0643\u0645\u064a\u0629" => "measure",
    "how much" => "measure",
    "amount" => "measure",
    "quantity" => "measure",
    "length" => "measure",
    "weight" => "measure",
    "duration" => "measure",
    "distance" => "measure",
]

const QUANTITY_COMPARISON_MARKERS = Pair{String,String}[
    "\u0623\u0643\u062b\u0631" => "comparison",
    "\u0623\u0642\u0644" => "comparison",
    "\u064a\u0633\u0627\u0648\u064a" => "comparison",
    "\u0646\u0635\u0641" => "comparison",
    "\u0636\u0639\u0641" => "comparison",
    "more than" => "comparison",
    "less than" => "comparison",
    "greater than" => "comparison",
    "equal to" => "comparison",
    "half" => "comparison",
    "double" => "comparison",
]

const QUANTIFIER_SCOPE_MARKERS = Pair{String,String}[
    "\u0643\u0644" => "quantifier_scope",
    "\u0628\u0639\u0636" => "quantifier_scope",
    "all" => "quantifier_scope",
    "some" => "quantifier_scope",
]

const VAGUE_QUANTITY_MARKERS = Pair{String,String}[
    "\u0643\u062b\u064a\u0631" => "vague_quantity",
    "\u0642\u0644\u064a\u0644" => "vague_quantity",
    "many" => "vague_quantity",
    "few" => "vague_quantity",
]

const QUANTITY_MARKER_TYPES = vcat(QUANTITY_COUNT_MARKERS, QUANTITY_MEASURE_MARKERS,
                                    QUANTITY_COMPARISON_MARKERS, QUANTIFIER_SCOPE_MARKERS,
                                    VAGUE_QUANTITY_MARKERS)
# --- End quantity marker types ---

mutable struct IstinbatAttentionRecord
    record_id::String
    relation_type::String
    marker::String
    before_terms::Vector{String}
    after_terms::Vector{String}
    focus_terms::Vector{String}
    polarity::Int
    attention_weight::Float64
    examples::Vector{String}
    count::Int
    source_metadata::Vector{Dict{String,Any}}
end

# Outer constructor for backward compatibility
IstinbatAttentionRecord(record_id::String, relation_type::String, marker::String, before_terms::Vector{String}, after_terms::Vector{String}, focus_terms::Vector{String}, polarity::Int, attention_weight::Float64, examples::Vector{String}, count::Int) =
    IstinbatAttentionRecord(record_id, relation_type, marker, before_terms, after_terms, focus_terms, polarity, attention_weight, examples, count, [Dict{String,Any}() for _ in 1:length(examples)])

mutable struct IstinbatAttentionMemory
    records::Dict{String,IstinbatAttentionRecord}
    max_examples::Int
    discovered_markers::Dict{String, String}
    discovered_confidences::Dict{String, Float64}
end

IstinbatAttentionMemory(; max_examples::Int=5) =
    IstinbatAttentionMemory(Dict{String,IstinbatAttentionRecord}(), max_examples, Dict{String, String}(), Dict{String, Float64}())

has_istinbat_records(mem::IstinbatAttentionMemory) = !isempty(mem.records)

"""
    RelationFrame

هيكل محايد لتمثيل العلاقة المستنبطة من النص.
ليس SVO نحوياً بل left ← marker → right مع توصيف نوع العلاقة ودرجة الثقة.
"""
mutable struct RelationFrame
    left_terms::Vector{String}
    marker::String
    right_terms::Vector{String}
    relation_type::String
    polarity::Int
    direction::Int
    confidence::Float64
    source_sentence::String
end

"""
    QuantityFrame

هيكل مستقل للكمية والعدد، يمثل كمية مستخرجة من النص عبر مفاتيح كمية.
"""
struct QuantityFrame
    marker::String
    quantity_type::String
    target::String
    value::String
    polarity::String
    confidence::Float64
end

mutable struct QuantityFrameMemory
    frames::Vector{QuantityFrame}
    source_metadata::Vector{Dict{String,Any}}
end

QuantityFrameMemory() = QuantityFrameMemory(QuantityFrame[], Dict{String,Any}[])

function _quantity_frame_to_dict(frame::QuantityFrame)
    return Dict{String,Any}(
        "marker" => frame.marker,
        "quantity_type" => frame.quantity_type,
        "target" => frame.target,
        "value" => frame.value,
        "polarity" => frame.polarity,
        "confidence" => frame.confidence,
    )
end

function _quantity_frame_from_dict(d)
    return QuantityFrame(
        String(get(d, "marker", "")),
        String(get(d, "quantity_type", "")),
        String(get(d, "target", "")),
        String(get(d, "value", "")),
        String(get(d, "polarity", "neutral")),
        Float64(get(d, "confidence", 0.0)),
    )
end

function quantity_memory_to_dict(mem::QuantityFrameMemory)
    return Dict{String,Any}(
        "version" => 1,
        "frames" => [_quantity_frame_to_dict(frame) for frame in mem.frames],
        "source_metadata" => mem.source_metadata,
    )
end

function save_quantity_memory(mem::QuantityFrameMemory, path::AbstractString)
    open(String(path), "w") do io
        JSON.print(io, quantity_memory_to_dict(mem), 2)
    end
    return String(path)
end

function load_quantity_memory(path::AbstractString)
    isfile(String(path)) || return QuantityFrameMemory()
    data = JSON.parsefile(String(path))
    frames = QuantityFrame[]
    for item in get(data, "frames", Any[])
        push!(frames, _quantity_frame_from_dict(item))
    end
    raw_meta = get(data, "source_metadata", Any[])
    metadata = Dict{String,Any}[]
    for item in raw_meta
        if item isa Dict
            push!(metadata, Dict{String,Any}(String(k) => v for (k, v) in item))
        else
            push!(metadata, Dict{String,Any}())
        end
    end
    while length(metadata) < length(frames)
        push!(metadata, Dict{String,Any}())
    end
    return QuantityFrameMemory(frames, metadata[1:length(frames)])
end

has_quantity_records(mem::QuantityFrameMemory) = !isempty(mem.frames)

_clean_space(s::AbstractString) = replace(strip(String(s)), r"\s+" => " ")

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

const STOPWORDS = Set([
    "ما", "ماذا", "من", "كيف", "لماذا", "هل", "متى", "اين", "أين",
    "في", "من", "عن", "على", "الى", "إلى", "الي", "مع", "او", "أو",
    "هو", "هي", "هذا", "هذه", "ذلك", "تلك", "الذي", "التي", "كل",
    "قد", "ثم", "حين", "عند", "اذا", "إذا", "ان", "أن", "انه", "أنّه",
    "لا", "ليس", "ليست", "بلا", "دون", "فقط", "بل", "اما", "أما",
    "سؤال", "جواب", "س", "ج",
    "what", "why", "how", "when", "is", "are", "the", "a", "an", "to", "of", "and", "or", "because",
])

const DEFINITE_PREFIXES = ("ال", "Ø§Ù„")

function _has_definite_subject_before_marker(text::AbstractString)
    raw = split(_clean_space(text), r"\s+")
    for tok in raw
        clean = replace(String(tok), r"^[\W_]+|[\W_]+$" => "")
        any(prefix -> startswith(clean, prefix), DEFINITE_PREFIXES) && return true
    end
    return false
end

function _sentences(text::AbstractString)
    parts = split(String(text), r"(?<=[\.\!\?\u061F؛;:])\s+|\n+")
    return String[_clean_space(p) for p in parts if !isempty(_clean_space(p))]
end

function _tokens(text::AbstractString)
    raw = split(_clean_space(text), r"\s+")
    toks = String[]
    seen = Set{String}()
    for w in raw
        n = _norm_word(w)
        isempty(n) && continue
        n in STOPWORDS && continue
        length(n) < 2 && continue
        n in seen && continue
        push!(seen, n)
        push!(toks, n)
    end
    return toks
end

function _ordered_tokens(text::AbstractString)
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

function _marker_hit(sentence::String, discovered_markers::Dict{String, String}=Dict{String, String}();
                     include_defaults::Bool=true)
    s = _clean_space(sentence)
    ns = _norm_word(s)
    ntokens = Set(_tokens(s))
    if include_defaults
        for pair in DEFAULT_MARKER_TYPES
            word = first(pair)
            rtype = last(pair)
            nmarker = _norm_word(word)
            if occursin(word, s) || (!isempty(nmarker) && nmarker in ntokens)
                return rtype, word
            end
        end
    end
    
    # فحص العلامات المكتشفة ذاتياً فقط
    for (word, rtype) in discovered_markers
        nmarker = _norm_word(word)
        if occursin(word, s) || (!isempty(nmarker) && nmarker in ntokens)
            return rtype, word
        end
    end
    return "", ""
end

function _is_marker_boundary_char(c::Char)
    return !(isletter(c) || isnumeric(c))
end

function _marker_span(sentence::AbstractString, marker::AbstractString)
    s = String(sentence)
    m = String(marker)
    isempty(m) && return nothing
    start = firstindex(s)
    while start <= lastindex(s)
        r = findnext(m, s, start)
        r === nothing && return nothing
        before_ok = first(r) == firstindex(s) ||
                    _is_marker_boundary_char(s[prevind(s, first(r))])
        after_ok = last(r) == lastindex(s) ||
                   _is_marker_boundary_char(s[nextind(s, last(r))])
        before_ok && after_ok && return r
        start = nextind(s, first(r))
    end
    return nothing
end

function _split_by_marker(sentence::String, marker::String)
    m = _marker_span(sentence, marker)
    if m !== nothing
        left = first(m) > firstindex(sentence) ? sentence[firstindex(sentence):prevind(sentence, first(m))] : ""
        right = last(m) < lastindex(sentence) ? sentence[nextind(sentence, last(m)):lastindex(sentence)] : ""
        return left, right
    end
    nmarker = _norm_word(marker)
    words = split(_clean_space(sentence))
    idx = findfirst(w -> _norm_word(w) == nmarker, words)
    if idx === nothing
        return sentence, ""
    end
    before = idx > 1 ? join(words[1:idx-1], " ") : ""
    after = idx < length(words) ? join(words[idx+1:end], " ") : ""
    return before, after
end

function _polarity(sentence::String, relation_type::String)
    relation_type in ("negation", "contradiction", "direct_negation", "opposition") && return -1
    s = _clean_space(sentence)
    return occursin("لا ", s) || occursin("ليس", s) || occursin("ليست", s) ||
           occursin("بلا", s) || occursin("دون", s) ? -1 : 1
end

function _focus_weight(relation_type::String, before_terms::Vector{String}, after_terms::Vector{String})
    base = get(Dict(
        "causal" => 0.82,
        "causal_anchor" => 0.88,
        "analogy" => 0.74,
        "transform" => 0.84,
        "need" => 0.80,
        "difference" => 0.78,
        "prevention" => 0.82,
        "contradiction" => 0.86,
        "negation" => 0.76,
        "direct_negation" => 0.86,
        "opposition" => 0.84,
    ), relation_type, 0.60)
    coverage = clamp(0.04 * (length(before_terms) + length(after_terms)), 0.0, 0.18)
    return clamp(base + coverage, 0.0, 1.0)
end

function _record_key(relation_type::String, before_terms::AbstractVector, after_terms::AbstractVector, polarity::Int)
    left = join(before_terms[1:min(2, length(before_terms))], "|")
    right = join(after_terms[1:min(3, length(after_terms))], "|")
    return "$(relation_type)|$(polarity)|$(left)=>$(right)"
end

function _learned_attention(sentence::String, discovered_markers::Dict{String, String}=Dict{String, String}())
    s = _clean_space(sentence)
    isempty(s) && return nothing
    (occursin("?", s) || occursin("؟", s)) && return nothing
    relation_type, marker = _marker_hit(s, discovered_markers)
    isempty(relation_type) && return nothing
    before, after = _split_by_marker(s, marker)
    if relation_type == "causal_anchor" && !_has_definite_subject_before_marker(before)
        return nothing
    end
    before_terms = _tokens(before)
    after_terms = _ordered_tokens(after)
    isempty(before_terms) && isempty(after_terms) && return nothing
    focus_terms = unique(vcat(after_terms, before_terms))
    isempty(focus_terms) && return nothing
    polarity = _polarity(s, relation_type)
    return (relation_type=relation_type, marker=marker,
            before_terms=before_terms[1:min(4, length(before_terms))],
            after_terms=after_terms[1:min(6, length(after_terms))],
            focus_terms=focus_terms[1:min(8, length(focus_terms))],
            polarity=polarity)
end

function _record!(mem::IstinbatAttentionMemory, sentence::String, item, source::Dict{String,Any}=Dict{String,Any}())
    key = _record_key(item.relation_type, item.before_terms, item.after_terms, item.polarity)
    if !haskey(mem.records, key)
        mem.records[key] = IstinbatAttentionRecord(
            key, item.relation_type, item.marker,
            String.(item.before_terms), String.(item.after_terms), String.(item.focus_terms),
            item.polarity,
            _focus_weight(item.relation_type, item.before_terms, item.after_terms),
            String[], 0,
            Dict{String,Any}[]
        )
    end
    rec = mem.records[key]
    rec.count += 1
    rec.attention_weight = clamp(max(rec.attention_weight, _focus_weight(rec.relation_type, rec.before_terms, rec.after_terms)) +
                                 0.02 * min(rec.count, 8), 0.0, 1.0)
    for term in item.focus_terms
        term in rec.focus_terms || push!(rec.focus_terms, term)
    end
    length(rec.focus_terms) > 10 && resize!(rec.focus_terms, 10)
    if !isempty(strip(sentence)) && !(sentence in rec.examples)
        push!(rec.examples, sentence)
        push!(rec.source_metadata, source)
        if length(rec.examples) > mem.max_examples
            popfirst!(rec.examples)
            popfirst!(rec.source_metadata)
        end
    end
    return rec
end

function _unique_nonempty_terms(values)
    seen = Set{String}()
    out = String[]
    for v in values
        n = _norm_word(String(v))
        isempty(n) && continue
        n in STOPWORDS && continue
        n in seen && continue
        push!(seen, n)
        push!(out, n)
    end
    return out
end

function _opposition_pair_candidates(sentence::String)
    s = _clean_space(sentence)
    candidates = Tuple{String,String,String}[]
    patterns = [
        r"(.{2,30}?)\s+(?:ضد|عكس|نقيض|خلاف)\s+(.{2,30}?)(?:[\.\،\؛]|$)",
        r"(.{2,30}?)\s+و\s*(.{2,30}?)\s+(?:ضدان|نقيضان|لا\s+يلتقيان|طريقان\s+لا\s+يلتقيان)(?:[\.\،\؛]|$)",
        r"(?:يميز|يفرق|يفصل|الفرق)\s+(?:بين)?\s*(.{2,30}?)\s+و\s*(.{2,30}?)(?:[\.\،\؛]|$)",
        r"بين\s+(.{2,30}?)\s+و\s*(.{2,30}?)(?:[\.\،\؛]|$)",
    ]
    for pat in patterns
        for m in eachmatch(pat, s)
            left_terms = _unique_nonempty_terms(split(m.captures[1]))
            right_terms = _unique_nonempty_terms(split(m.captures[2]))
            isempty(left_terms) && continue
            isempty(right_terms) && continue
            left = last(left_terms)
            right = first(right_terms)
            left == right && continue
            push!(candidates, (left, right, s))
        end
    end
    return candidates
end

function learn_opposition_from_text!(mem::IstinbatAttentionMemory, text::AbstractString,
                                     source::Dict{String,Any}=Dict{String,Any}())
    learned = 0
    for sentence in _sentences(text)
        for (left, right, example) in _opposition_pair_candidates(sentence)
            item = (
                relation_type="opposition",
                marker="opposition",
                before_terms=[left],
                after_terms=[right],
                focus_terms=[left, right],
                polarity=-1,
            )
            _record!(mem, example, item, source)
            learned += 1
        end
    end
    return learned
end

function _term_keyset(term::AbstractString)
    keys = Set{String}()
    t = _norm_word(term)
    p = _projection_word(term)
    isempty(t) || push!(keys, t)
    isempty(p) || push!(keys, p)
    for k in collect(keys)
        if length(k) > 3 && endswith(k, "ا")
            push!(keys, k[begin:prevind(k, lastindex(k))])
        end
    end
    return keys
end

function _term_matches(term::AbstractString, keys::Set{String})
    tkeys = _term_keyset(term)
    isempty(tkeys) && return false
    return !isempty(intersect(tkeys, keys))
end

function _direct_negation_pair_candidates(sentence::String)
    s = _clean_space(sentence)
    candidates = Tuple{String,String,String}[]
    patterns = [
        r"(.{2,30}?)\s+(?:ليس|ليست)\s+(.{2,30}?)(?:[\.\،\؛]|$)",
        r"(.{2,30}?)\s+غير\s+(.{2,30}?)(?:[\.\،\؛]|$)",
    ]
    for pat in patterns
        for m in eachmatch(pat, s)
            left_terms = _unique_nonempty_terms(split(m.captures[1]))
            right_terms = _unique_nonempty_terms(split(m.captures[2]))
            isempty(left_terms) && continue
            isempty(right_terms) && continue
            left = last(left_terms)
            right = first(right_terms)
            left == right && continue
            push!(candidates, (left, right, s))
        end
    end
    return candidates
end

function learn_direct_negation_from_text!(mem::IstinbatAttentionMemory, text::AbstractString,
                                          source::Dict{String,Any}=Dict{String,Any}())
    learned = 0
    for sentence in _sentences(text)
        for (left, right, example) in _direct_negation_pair_candidates(sentence)
            item = (
                relation_type="direct_negation",
                marker="ليس",
                before_terms=[left],
                after_terms=[right],
                focus_terms=[left, right],
                polarity=-1,
            )
            _record!(mem, example, item, source)
            learned += 1
        end
    end
    return learned
end

function _negative_operator_candidates(sentence::String)
    words = split(_clean_space(sentence), r"\s+")
    candidates = Tuple{String,String,String,String}[]
    for (i, w) in enumerate(words)
        marker = _norm_word(w)
        marker in NEGATIVE_OPERATOR_MARKERS || continue
        i < length(words) || continue
        left_terms = _unique_nonempty_terms(words[1:max(i - 1, 0)])
        right_terms = _unique_nonempty_terms(words[i + 1:end])
        isempty(left_terms) && continue
        isempty(right_terms) && continue
        left = last(left_terms)
        right = first(right_terms)
        left == right && continue
        push!(candidates, (left, right, String(w), _clean_space(sentence)))
    end
    return candidates
end

function _negative_context_candidates(sentence::String)
    s = _clean_space(sentence)
    candidates = Tuple{String,String,String,String}[]
    patterns = [
        r"\u0644\u0627\s+\u064a\u062a\u062d\u0642\u0642\s+(.{2,30}?)\s+(?:\u062d\u064a\u062b|\u0639\u0646\u062f)\s+(?:\u064a\u063a\u0644\u0628|\u062a\u063a\u0644\u0628)\s+(.{2,30}?)(?:[\.\u060C\u061B]|$)",
    ]
    for pat in patterns
        for m in eachmatch(pat, s)
            target_terms = _unique_nonempty_terms(split(m.captures[1]))
            source_terms = _unique_nonempty_terms(split(m.captures[2]))
            isempty(target_terms) && continue
            isempty(source_terms) && continue
            source = first(source_terms)
            target = first(target_terms)
            source == target && continue
            push!(candidates, (source, target, "\u0644\u0627 \u064a\u062a\u062d\u0642\u0642", s))
        end
    end
    return candidates
end

function learn_negative_operator_from_text!(mem::IstinbatAttentionMemory, text::AbstractString,
                                            source::Dict{String,Any}=Dict{String,Any}())
    learned = 0
    for sentence in _sentences(text)
        for (left, right, marker, example) in vcat(_negative_operator_candidates(sentence),
                                                   _negative_context_candidates(sentence))
            item = (
                relation_type="contradiction",
                marker=marker,
                before_terms=[left],
                after_terms=[right],
                focus_terms=[left, right],
                polarity=-1,
            )
            _record!(mem, example, item, source)
            learned += 1
        end
    end
    return learned
end

function terms_are_negated(mem::IstinbatAttentionMemory,
                           left::AbstractString,
                           right::AbstractString;
                           min_weight::Float64=0.20)
    left_keys = _term_keyset(left)
    right_keys = _term_keyset(right)
    (isempty(left_keys) || isempty(right_keys)) && return false
    for rec in values(mem.records)
        rec.relation_type == "direct_negation" || continue
        rec.polarity < 0 || continue
        rec.attention_weight >= min_weight || continue
        before_hit_left = any(t -> _term_matches(t, left_keys), rec.before_terms)
        after_hit_right = any(t -> _term_matches(t, right_keys), rec.after_terms)
        before_hit_left && after_hit_right && return true
    end
    return false
end

function _term_matches_old(term::AbstractString, keys::Set{String})
    t = _norm_word(term)
    p = _projection_word(term)
    return (!isempty(t) && t in keys) || (!isempty(p) && p in keys)
end

function _has_explicit_opposition_cue(text::AbstractString)
    s = _clean_space(text)
    cues = (
        "\u0636\u062f", "\u0639\u0643\u0633", "\u0646\u0642\u064a\u0636", "\u062e\u0644\u0627\u0641",
        "\u0636\u062f\u0627\u0646", "\u0646\u0642\u064a\u0636\u0627\u0646",
        "\u0644\u0627 \u064a\u0644\u062a\u0642\u064a\u0627\u0646",
        "\u0637\u0631\u064a\u0642\u0627\u0646 \u0644\u0627 \u064a\u0644\u062a\u0642\u064a\u0627\u0646",
        "\u064a\u0645\u064a\u0632", "\u064a\u0641\u0631\u0642", "\u064a\u0641\u0635\u0644", "\u0627\u0644\u0641\u0631\u0642",
    )
    return any(c -> occursin(c, s), cues)
end

function terms_are_opposed(mem::IstinbatAttentionMemory,
                           left::AbstractString,
                           right::AbstractString;
                           min_weight::Float64=0.20)
    left_keys = _term_keyset(left)
    right_keys = _term_keyset(right)
    (isempty(left_keys) || isempty(right_keys)) && return false
    for rec in values(mem.records)
        rec.relation_type == "opposition" || continue
        rec.polarity < 0 || continue
        rec.attention_weight >= min_weight || continue
        any(_has_explicit_opposition_cue, rec.examples) || continue
        before_hit_left = any(t -> _term_matches(t, left_keys), rec.before_terms)
        after_hit_right = any(t -> _term_matches(t, right_keys), rec.after_terms)
        before_hit_right = any(t -> _term_matches(t, right_keys), rec.before_terms)
        after_hit_left = any(t -> _term_matches(t, left_keys), rec.after_terms)
        ((before_hit_left && after_hit_right) || (before_hit_right && after_hit_left)) && return true
    end
    return false
end

function learn_istinbat_fact!(mem::IstinbatAttentionMemory, relation_type::AbstractString;
                              marker::AbstractString="",
                              before_terms::AbstractVector=String[],
                              after_terms::AbstractVector=String[],
                              focus_terms::AbstractVector=String[],
                              polarity::Int=1,
                              source::Dict{String,Any}=Dict{String,Any}())
    clean_relation = String(strip(String(relation_type)))
    isempty(clean_relation) && return 0
    before = _tokens(join(before_terms, " "))
    after = _tokens(join(after_terms, " "))
    focus = _tokens(join(isempty(focus_terms) ? vcat(after, before) : focus_terms, " "))
    isempty(before) && isempty(after) && return 0
    isempty(focus) && (focus = unique(vcat(after, before)))
    item = (
        relation_type=clean_relation,
        marker=String(strip(String(marker))),
        before_terms=before[1:min(4, length(before))],
        after_terms=after[1:min(6, length(after))],
        focus_terms=focus[1:min(8, length(focus))],
        polarity=polarity < 0 ? -1 : 1,
    )
    _record!(mem, "", item, source)
    return 1
end

function learn_istinbat_from_text!(mem::IstinbatAttentionMemory, text::AbstractString, source::Dict{String,Any}=Dict{String,Any}())
    learned = 0
    for sentence in _sentences(text)
        item = _learned_attention(sentence, mem.discovered_markers)
        item === nothing && continue
        _record!(mem, sentence, item, source)
        learned += 1
    end
    learned += learn_opposition_from_text!(mem, text, source)
    learned += learn_direct_negation_from_text!(mem, text, source)
    learned += learn_negative_operator_from_text!(mem, text, source)
    learned += learn_relation_frames_from_text!(mem, text, source)
    return learned
end

function train_istinbat_from_texts!(mem::IstinbatAttentionMemory, texts, metadata=nothing;
                                    max_items::Int=50_000)
    learned = 0
    for (i, text) in enumerate(texts)
        learned >= max_items && break
        source = (metadata !== nothing && i <= length(metadata)) ? metadata[i] : Dict{String,Any}()
        learned += learn_istinbat_from_text!(mem, text, source)
    end
    return learned
end

function _query_terms(prompt::AbstractString)
    terms = Set{String}()
    for t in _tokens(prompt)
        push!(terms, t)
        p = _projection_word(t)
        isempty(p) || push!(terms, p)
    end
    return terms
end

function _overlap_score(query::Set{String}, rec::IstinbatAttentionRecord)
    isempty(query) && return 0.0
    terms = Set{String}()
    for t in vcat(rec.before_terms, rec.after_terms, rec.focus_terms)
        push!(terms, t)
        p = _projection_word(t)
        isempty(p) || push!(terms, p)
    end
    hits = length(intersect(query, terms))
    hits == 0 && return 0.0
    return hits / max(1, min(length(query), length(terms)))
end

function select_istinbat_attention(mem::IstinbatAttentionMemory, prompt::AbstractString;
                                   min_score::Float64=0.18,
                                   active_paras::Union{Nothing,Dict{Tuple{String,Int},Float64}}=nothing)
    query = _query_terms(prompt)
    best = nothing
    best_score = 0.0
    for rec in values(mem.records)
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
        end
        score = 0.72 * _overlap_score(query, rec) + 0.28 * rec.attention_weight
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
            score *= act
        end
        if score > best_score
            best = rec
            best_score = score
        end
    end
    best_score >= min_score ? best : nothing
end

function istinbat_focus_terms(rec::IstinbatAttentionRecord; limit::Int=8)
    terms = unique(vcat(rec.focus_terms, rec.after_terms, rec.before_terms))
    return terms[1:min(limit, length(terms))]
end

function select_causal_anchor_attention(mem::IstinbatAttentionMemory, prompt::AbstractString;
                                        min_score::Float64=0.14,
                                        active_paras::Union{Nothing,Dict{Tuple{String,Int},Float64}}=nothing)
    query = _query_terms(prompt)
    isempty(query) && return nothing
    best = nothing
    best_score = 0.0
    for rec in values(mem.records)
        rec.relation_type == "causal_anchor" || continue
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
        end
        score = 0.80 * _overlap_score(query, rec) + 0.20 * rec.attention_weight
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
            score *= act
        end
        if score > best_score
            best = rec
            best_score = score
        end
    end
    best_score >= min_score ? best : nothing
end

function causal_anchor_answer_from_attention(rec::IstinbatAttentionRecord)
    rec.relation_type == "causal_anchor" || return ""
    isempty(rec.examples) && return ""
    ex = _clean_space(first(rec.examples))
    return endswith(ex, ".") || endswith(ex, "؟") || endswith(ex, "!") ? ex : ex * "."
end

function select_contradiction_attention(mem::IstinbatAttentionMemory, prompt::AbstractString;
                                        min_score::Float64=0.16,
                                        active_paras::Union{Nothing,Dict{Tuple{String,Int},Float64}}=nothing)
    query = _query_terms(prompt)
    isempty(query) && return nothing
    best = nothing
    best_score = 0.0
    for rec in values(mem.records)
        rec.polarity < 0 || continue
        rec.relation_type in ("contradiction", "negation", "need", "direct_negation", "opposition") || continue
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
        end
        score = 0.78 * _overlap_score(query, rec) + 0.22 * rec.attention_weight
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
            score *= act
        end
        if score > best_score
            best = rec
            best_score = score
        end
    end
    best_score >= min_score ? best : nothing
end

"""
    select_relation_frame_attention(mem::IstinbatAttentionMemory, prompt::AbstractString;
                                     min_score::Float64=0.15,
                                     active_paras=nothing) -> Union{IstinbatAttentionRecord,Nothing}

اختيار داخلي لسجلات العلاقات الجديدة (purpose, conditional, temporal, spatial, state)
من ذاكرة الاستنباط. لا يُستخدم في التوليد — طبقة رصد واختيار فقط.
"""
function select_relation_frame_attention(mem::IstinbatAttentionMemory, prompt::AbstractString;
                                          min_score::Float64=0.15,
                                          active_paras::Union{Nothing,Dict{Tuple{String,Int},Float64}}=nothing)
    query = _query_terms(prompt)
    isempty(query) && return nothing
    best = nothing
    best_score = 0.0
    for rec in values(mem.records)
        rec.relation_type in ("purpose", "conditional", "temporal", "spatial", "state") || continue
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
        end
        score = 0.75 * _overlap_score(query, rec) + 0.25 * rec.attention_weight
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
            score *= act
        end
        if score > best_score
            best = rec
            best_score = score
        end
    end
    best_score >= min_score ? best : nothing
end

function contradiction_answer_from_attention(rec::IstinbatAttentionRecord)
    rec.polarity < 0 || return ""
    if !isempty(rec.examples)
        ex = _clean_space(first(rec.examples))
        ex = replace(ex, r"^[\s\-–—]*" => "")
        ex = replace(ex, r"^(?:لا[،,\s]+)+" => "")
        return endswith(ex, ".") || endswith(ex, "؟") || endswith(ex, "!") ? "لا، " * ex : "لا، " * ex * "."
    end
    terms = istinbat_focus_terms(rec; limit=6)
    isempty(terms) && return ""
    return "لا، تظهر علاقة نفي أو تضاد حول " * join(terms, " و") * "."
end

function merge_istinbat!(base::IstinbatAttentionMemory, extra::IstinbatAttentionMemory)
    for (id, incoming) in extra.records
        if !haskey(base.records, id)
            base.records[id] = incoming
            continue
        end
        rec = base.records[id]
        rec.count += incoming.count
        rec.attention_weight = max(rec.attention_weight, incoming.attention_weight)
        for term in incoming.focus_terms
            term in rec.focus_terms || push!(rec.focus_terms, term)
        end
        length(rec.focus_terms) > 10 && resize!(rec.focus_terms, 10)
        for (i, ex) in enumerate(incoming.examples)
            src = (i <= length(incoming.source_metadata)) ? incoming.source_metadata[i] : Dict{String,Any}()
            if !(ex in rec.examples)
                push!(rec.examples, ex)
                push!(rec.source_metadata, src)
                if length(rec.examples) > base.max_examples
                    popfirst!(rec.examples)
                    popfirst!(rec.source_metadata)
                end
            end
        end
    end
    return base
end

function _record_to_dict(rec::IstinbatAttentionRecord)
    return Dict(
        "record_id" => rec.record_id,
        "relation_type" => rec.relation_type,
        "marker" => rec.marker,
        "before_terms" => rec.before_terms,
        "after_terms" => rec.after_terms,
        "focus_terms" => rec.focus_terms,
        "polarity" => rec.polarity,
        "attention_weight" => rec.attention_weight,
        "examples" => rec.examples,
        "count" => rec.count,
        "source_metadata" => rec.source_metadata,
    )
end

function istinbat_to_dict(mem::IstinbatAttentionMemory)
    return Dict(
        "version" => AL_ISTINBAT_VERSION,
        "max_examples" => mem.max_examples,
        "records" => [_record_to_dict(rec) for rec in values(mem.records)],
        "discovered_markers" => mem.discovered_markers,
        "discovered_confidences" => mem.discovered_confidences,
    )
end

function save_istinbat(mem::IstinbatAttentionMemory, path::AbstractString)
    open(String(path), "w") do io
        JSON.print(io, istinbat_to_dict(mem), 2)
    end
    return String(path)
end

function _string_vec(v)
    return String[String(x) for x in v]
end

function load_istinbat(path::AbstractString)
    isfile(String(path)) || return IstinbatAttentionMemory()
    data = JSON.parsefile(String(path))
    mem = IstinbatAttentionMemory(max_examples=Int(get(data, "max_examples", 5)))
    if haskey(data, "discovered_markers")
        for (k, v) in data["discovered_markers"]
            mem.discovered_markers[String(k)] = String(v)
        end
    end
    if haskey(data, "discovered_confidences")
        for (k, v) in data["discovered_confidences"]
            mem.discovered_confidences[String(k)] = Float64(v)
        end
    end
    for item in get(data, "records", Any[])
        examples = _string_vec(item["examples"])
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
        
        rec = IstinbatAttentionRecord(
            String(item["record_id"]),
            String(item["relation_type"]),
            String(item["marker"]),
            _string_vec(item["before_terms"]),
            _string_vec(item["after_terms"]),
            _string_vec(item["focus_terms"]),
            Int(item["polarity"]),
            Float64(item["attention_weight"]),
            examples,
            Int(item["count"]),
            source_metadata,
        )
        mem.records[rec.record_id] = rec
    end
    return mem
end

function update_marker_confidence!(mem::IstinbatAttentionMemory, active_markers::Vector{String}, reward::Float64)
    for w in active_markers
        if haskey(mem.discovered_markers, w)
            current_conf = get(mem.discovered_confidences, w, 0.5)
            mem.discovered_confidences[w] = 0.9 * current_conf + 0.1 * reward
        end
    end
end

# ---------------------------------------------------------------------------
# RelationFrame API: standalone extraction (does NOT affect existing memory)
# ---------------------------------------------------------------------------

"""
    relation_type_for_marker(marker::AbstractString) -> String

تحديد نوع العلاقة بناءً على الكلمة المفتاحية.
تبحث في DEFAULT_MARKER_TYPES أولاً (السلوك الحالي)، ثم في الأنواع الجديدة.
"""
function relation_type_for_marker(marker::AbstractString)
    m = String(strip(String(marker)))
    for pair in NEW_MARKER_TYPES
        first(pair) == m && return last(pair)
    end
    for pair in DEFAULT_MARKER_TYPES
        first(pair) == m && return last(pair)
    end
    return ""
end

"""
    _confidence_for_marker(marker::AbstractString) -> Float64

تحديد درجة الثقة الافتراضية للمفتاح بناءً على وضوح معناه.
المفاتيح الصريحة (لكي، إذا، قبل) تحصل على ثقة أعلى من المفاتيح الملتبسة.
"""
function _confidence_for_marker(marker::AbstractString)
    m = String(strip(String(marker)))
    m in _HIGH_CONFIDENCE_MARKERS && return 0.90
    m in _MEDIUM_CONFIDENCE_MARKERS && return 0.75
    m in _LOW_CONFIDENCE_MARKERS && return 0.60
    return 0.50
end

function _conditional_split_terms(terms::Vector{String})
    length(terms) < 2 && return terms, String[]
    then_idx = findfirst(t -> lowercase(String(t)) == "then", terms)
    if then_idx !== nothing && then_idx > 1 && then_idx < length(terms)
        return terms[1:then_idx-1], terms[then_idx+1:end]
    end
    if length(terms) >= 4 && _norm_word(terms[1]) == _norm_word(terms[3])
        return terms[1:2], terms[3:end]
    end
    for i in 2:length(terms)
        if _looks_like_event_term(terms[i])
            return terms[1:i-1], terms[i:end]
        end
    end
    mid = max(1, length(terms) ÷ 2)
    return terms[1:mid], terms[mid+1:end]
end

function _cut_at_next_relation_marker(text::AbstractString)
    s = String(text)
    best = nothing
    for pair in ALL_MARKER_TYPES
        marker = first(pair)
        isempty(marker) && continue
        r = _marker_span(s, marker)
        r === nothing && continue
        first(r) == firstindex(s) && continue
        if best === nothing || first(r) < best
            best = first(r)
        end
    end
    best === nothing && return s
    return strip(s[firstindex(s):prevind(s, best)])
end

"""
    extract_relation_frames(text::AbstractString) -> Vector{RelationFrame}

استخراج جميع أطر العلاقات (RelationFrame) من النص.
تبحث عن كل المفاتيح المعروفة (القديمة والجديدة) في كل جملة،
وتُعيد إطاراً لكل مفتاح يُعثر عليه مع left/right terms ونوع العلاقة ودرجة الثقة.
لا تؤثر على ذاكرة الاستنباط الحالية. تستخدم للرصد والاستنباط فقط.
"""
function extract_relation_frames(text::AbstractString)
    frames = RelationFrame[]
    for sentence in _sentences(String(text))
        s = _clean_space(sentence)
        isempty(s) && continue
        for pair in ALL_MARKER_TYPES
            marker, rtype = first(pair), last(pair)
            _marker_span(s, marker) === nothing && continue
            before, after = _split_by_marker(s, marker)
            after = _cut_at_next_relation_marker(after)
            left_terms = _tokens(before)
            right_terms = _tokens(after)
            if rtype == "conditional" && isempty(left_terms)
                left_terms, right_terms = _conditional_split_terms(_ordered_tokens(after))
            end
            isempty(left_terms) && isempty(right_terms) && continue
            pol = _polarity(s, rtype)
            dir = 1
            conf = _confidence_for_marker(marker)
            push!(frames, RelationFrame(left_terms, marker, right_terms, rtype, pol, dir, conf, s))
        end
    end
    return frames
end
"""

    relation_type_for_quantity_marker(marker::AbstractString) -> String

تحديد نوع الكمية بناءً على الكلمة المفتاحية.
"""
function relation_type_for_quantity_marker(marker::AbstractString)
    m = String(strip(String(marker)))
    for pair in QUANTITY_MARKER_TYPES
        first(pair) == m && return last(pair)
    end
    return ""
end

function _clean_quantity_piece(x::AbstractString)
    return replace(_clean_space(String(x)), r"^[\s\.,،؛:!\?؟]+|[\s\.,،؛:!\?؟]+$" => "")
end

const QUANTITY_NUMERAL_WORDS = Set([
    "\u0635\u0641\u0631", "\u0648\u0627\u062d\u062f", "\u0648\u0627\u062d\u062f\u0629", "\u0627\u062b\u0646\u0627\u0646", "\u0627\u062b\u0646\u064a\u0646",
    "\u0627\u062b\u0646\u062a\u0627\u0646", "\u0627\u062b\u0646\u062a\u064a\u0646", "\u062b\u0644\u0627\u062b\u0629", "\u0623\u0631\u0628\u0639\u0629", "\u0627\u0631\u0628\u0639\u0629",
    "\u062e\u0645\u0633\u0629", "\u0633\u062a\u0629", "\u0633\u0628\u0639\u0629", "\u062b\u0645\u0627\u0646\u064a\u0629", "\u062a\u0633\u0639\u0629", "\u0639\u0634\u0631\u0629",
    "\u0639\u0634\u0631\u0648\u0646", "\u062b\u0644\u0627\u062b\u0648\u0646", "\u0623\u0631\u0628\u0639\u0648\u0646", "\u0627\u0631\u0628\u0639\u0648\u0646",
    "\u062e\u0645\u0633\u0648\u0646", "\u0633\u062a\u0648\u0646", "\u0633\u0628\u0639\u0648\u0646", "\u062b\u0645\u0627\u0646\u0648\u0646", "\u062a\u0633\u0639\u0648\u0646",
    "\u0645\u0626\u0629", "\u0645\u0627\u0626\u0629", "\u0623\u0644\u0641", "\u0627\u0644\u0641",
])

const QUANTITY_UNIT_WORDS = Set([
    "\u0645\u062a\u0631", "\u0645\u062a\u0631\u0627", "\u0645\u062a\u0631\u064b\u0627", "\u0643\u064a\u0644\u0648", "\u0643\u064a\u0644\u0648\u063a\u0631\u0627\u0645",
    "\u063a\u0631\u0627\u0645", "\u0644\u062a\u0631", "\u062c\u0648\u0644", "\u0648\u0627\u0637", "\u062f\u0631\u062c\u0629", "\u0633\u0627\u0639\u0629",
    "\u062f\u0642\u064a\u0642\u0629", "\u062b\u0627\u0646\u064a\u0629", "\u064a\u0648\u0645", "\u0634\u0647\u0631", "\u0633\u0646\u0629", "meter",
    "metre", "kg", "gram", "liter", "litre", "joule", "watt", "hour", "day",
])

const QUANTITY_GENERIC_VALUES = Set([
    "\u0643\u0644 \u0634\u064a\u0621", "\u0643\u0644 \u0634\u064a", "all", "everything", "nothing",
    "thanks", "thank you", "many thanks", "you are welcome", "welcome",
])

function _quantity_piece_words(s::AbstractString)
    return [t for t in split(_clean_quantity_piece(s)) if !isempty(t)]
end

function _quantity_has_numeric_value(s::AbstractString)
    text = lowercase(_clean_quantity_piece(s))
    occursin(r"[0-9٠-٩]", text) && return true
    words = Set(_quantity_piece_words(text))
    !isempty(intersect(words, QUANTITY_NUMERAL_WORDS)) && return true
    !isempty(intersect(words, QUANTITY_UNIT_WORDS)) && return true
    return false
end

function _quantity_piece_is_generic(s::AbstractString)
    text = lowercase(_clean_quantity_piece(s))
    isempty(text) && return false
    occursin(r"[\p{L}\p{N}]", text) || return true
    text in QUANTITY_GENERIC_VALUES && return true
    return text in (
        "\u0648\u0639\u0644\u064a", "\u0648\u0639\u0644\u0649", "\u0639\u0644\u064a", "\u0639\u0644\u0649",
        "\u0644", "\u0628", "\u0641\u064a", "\u0645\u0646", "\u0625\u0644\u0649", "\u0627\u0644\u0649",
        "\u0625\u0630\u0627", "\u0627\u0630\u0627",
        "\u0647\u0644", "\u0648\u062d\u062f\u0629", "\u0623\u0645\u062b\u0644\u0629", "\u0627\u0645\u062b\u0644\u0629",
        "\u0644\u0645\u0627\u0630\u0627", "\u0643\u064a\u0641", "\u0645\u0627", "\u0634\u064a\u0621",
    )
end

function _quantity_piece_has_bad_glue(s::AbstractString)
    text = " " * lowercase(_clean_quantity_piece(s)) * " "
    return occursin(" \u0644\u0623\u0646 ", text) ||
           occursin(" \u0644\u0627\u0646 ", text) ||
           occursin(" \u0644\u0643\u064a ", text) ||
           occursin(" \u0643\u0644\u0645\u0627 ", text) ||
           occursin(" \u0647\u0648 ", text) ||
           occursin(" \u0647\u064a ", text) ||
           occursin("=", text) ||
           occursin("|", text) ||
           occursin("`", text) ||
           occursin("*", text) ||
           occursin("[", text) ||
           occursin("]", text)
end

function _quantity_piece_starts_question(s::AbstractString)
    text = lowercase(_clean_quantity_piece(s))
    return startswith(text, "\u0644\u0645\u0627\u0630\u0627") ||
           startswith(text, "\u0643\u064a\u0641") ||
           startswith(text, "\u0647\u0644") ||
           startswith(text, "\u0645\u0627 ")
end

function _quantity_prompt_kind(s::AbstractString)
    x = lowercase(_clean_space(String(s)))
    if occursin("\u0623\u064a\u0647\u0645\u0627 \u0623\u0643\u062b\u0631", x) || occursin("\u0627\u064a\u0647\u0645\u0627 \u0627\u0643\u062b\u0631", x)
        return "comparison"
    elseif occursin("\u0623\u064a\u0647\u0645\u0627 \u0623\u0642\u0644", x) || occursin("\u0627\u064a\u0647\u0645\u0627 \u0627\u0642\u0644", x)
        return "comparison_less"
    elseif occursin("which is more", x) || occursin("which is greater", x)
        return "comparison"
    elseif occursin("which is less", x)
        return "comparison_less"
    elseif occursin("\u0645\u0627 \u0646\u0637\u0627\u0642", x) || occursin("\u0646\u0637\u0627\u0642", x)
        return "quantifier_scope"
    elseif occursin("\u0645\u0627 \u0643\u0645\u064a\u0629", x) || occursin("\u0643\u0645\u064a\u0629", x) || occursin("what quantity", x)
        return "quantity"
    elseif occursin("\u0645\u0642\u062f\u0627\u0631", x) || occursin("\u0643\u0645 \u0637\u0648\u0644", x) ||
           occursin("\u0643\u0645 \u0648\u0632\u0646", x) || occursin("\u0643\u0645 \u0645\u062f\u0629", x) ||
           occursin("\u0643\u0645 \u0645\u0633\u0627\u0641\u0629", x) || occursin("how much", x) ||
           occursin("what amount", x)
        return "measure"
    elseif occursin("\u0643\u0645", x) || occursin("\u0639\u062f\u062f", x) || occursin("how many", x)
        return "count"
    end
    return ""
end

function _quantity_frame_answerable(frame::QuantityFrame)
    target = _clean_quantity_piece(frame.target)
    value = _clean_quantity_piece(frame.value)
    qtype = frame.quantity_type
    isempty(target) && isempty(value) && return false
    _quantity_piece_is_generic(target) && return false
    _quantity_piece_is_generic(value) && qtype != "quantifier_scope" && return false

    target_words = length(_quantity_piece_words(target))
    value_words = length(_quantity_piece_words(value))

    if qtype == "count" || qtype == "measure"
        isempty(target) && return false
        ncodeunits(target) > 6 || return false
        startswith(lowercase(target), "\u0628\u0623\u0646") && return false
        _quantity_piece_starts_question(target) && return false
        _quantity_has_numeric_value(value) || return false
        target_words <= 8 && value_words <= 8 || return false
        return !_quantity_piece_has_bad_glue(target) && !_quantity_piece_has_bad_glue(value)
    elseif qtype == "comparison"
        (isempty(target) || isempty(value)) && return false
        _quantity_piece_is_generic(target) && return false
        startswith(lowercase(target), "\u0628\u0623\u0646") && return false
        _quantity_piece_starts_question(target) && return false
        startswith(lowercase(target), "\u0623\u0645\u062b\u0644\u0629") && return false
        startswith(lowercase(target), "\u0627\u0645\u062b\u0644\u0629") && return false
        occursin(" \u0639\u0644\u0649 ", " " * lowercase(target) * " ") && return false
        target_words <= 8 && value_words <= 8 || return false
        (_quantity_piece_has_bad_glue(target) || _quantity_piece_has_bad_glue(value)) && return false
        v = " " * lowercase(value) * " "
        marker = lowercase(_clean_quantity_piece(frame.marker))
        marker in ("more than", "less than", "greater than", "equal to", "half", "double") && return true
        return startswith(lowercase(value), "\u0645\u0646 ") || occursin(" than ", v)
    elseif qtype == "quantifier_scope"
        target_words <= 6 && value_words <= 10 || return false
        (_quantity_piece_has_bad_glue(target) || _quantity_piece_has_bad_glue(value)) && return false
        occursin("\u0634\u064a\u0621", lowercase(target)) && return false
        occursin("\u0628\u0639\u062f\u062f", lowercase(target)) && return false
        return true
    elseif qtype == "vague_quantity"
        isempty(target) && return false
        startswith(lowercase(target), "\u0628\u0623\u0646") && return false
        _quantity_piece_starts_question(target) && return false
        target_words <= 6 && value_words <= 6 || return false
        (_quantity_piece_has_bad_glue(target) || _quantity_piece_has_bad_glue(value)) && return false
        _quantity_piece_is_generic(value) && return false
        return true
    end
    return false
end

function _quantity_frame_matches_prompt(frame::QuantityFrame, prompt_kind::String)
    isempty(prompt_kind) && return false
    qtype = frame.quantity_type
    prompt_kind == "quantity" && return qtype in ("measure", "vague_quantity", "quantifier_scope", "count")
    prompt_kind == qtype && return true
    prompt_kind == "comparison_less" && return qtype == "comparison" && _clean_quantity_piece(frame.marker) in ("\u0623\u0642\u0644", "\u0627\u0642\u0644")
    prompt_kind == "measure" && return qtype in ("measure", "vague_quantity")
    return false
end

function _quantity_prompt_payload(s::AbstractString)
    text = lowercase(_clean_space(String(s)))
    for phrase in (
        "\u0643\u0645 \u0639\u062f\u062f", "\u0645\u0627 \u0645\u0642\u062f\u0627\u0631", "\u0645\u0627 \u0643\u0645\u064a\u0629",
        "\u0623\u064a\u0647\u0645\u0627 \u0623\u0643\u062b\u0631", "\u0627\u064a\u0647\u0645\u0627 \u0627\u0643\u062b\u0631",
        "\u0623\u064a\u0647\u0645\u0627 \u0623\u0642\u0644", "\u0627\u064a\u0647\u0645\u0627 \u0627\u0642\u0644",
        "\u0645\u0627 \u0646\u0637\u0627\u0642", "\u0643\u0645", "\u0639\u062f\u062f", "\u0645\u0642\u062f\u0627\u0631",
        "\u0643\u0645\u064a\u0629", "\u0646\u0637\u0627\u0642", "\u0637\u0648\u0644", "\u0648\u0632\u0646", "\u0645\u062f\u0629", "\u0645\u0633\u0627\u0641\u0629", "\u0623\u0645", "\u0627\u0645",
        "how many", "how much", "what amount", "what quantity",
        "which is more", "which is greater", "which is less",
    )
        text = replace(text, phrase => " ")
    end
    return _clean_quantity_piece(text)
end

function _quantity_prompt_is_answerable(s::AbstractString)
    payload = _quantity_prompt_payload(s)
    isempty(payload) && return false
    ncodeunits(payload) > 6 || return false
    _quantity_piece_is_generic(payload) && return false
    startswith(lowercase(payload), "\u0628\u0623\u0646") && return false
    startswith(lowercase(payload), "\u0644\u0645\u0627\u0630\u0627") && return false
    startswith(lowercase(payload), "\u0643\u064a\u0641") && return false
    _quantity_piece_has_bad_glue(payload) && return false
    length(_quantity_piece_words(payload)) <= 5 || return false
    p = " " * lowercase(payload) * " "
    (occursin(" \u064a\u0636\u064a\u0642 ", p) ||
     occursin(" \u0641\u064a\u0638\u0646 ", p) ||
     occursin(" \u064a\u062d\u062f\u062b ", p) ||
     occursin(" \u064a\u0635\u0628\u062d ", p) ||
     occursin(" \u0623\u0635\u0628\u062d ", p) ||
     occursin(" \u062a\u0642\u0648\u062f ", p) ||
     occursin(" \u064a\u0639\u0631\u0641 ", p) ||
     occursin(" \u064a\u0639\u0631\u0641\u0647 ", p) ||
     occursin(" \u0646\u0633\u062a\u0639\u0645\u0644 ", p) ||
     occursin(" \u0643\u0627\u0644\u0645\u0631\u0636 ", p)) && return false
    return true
end

"""
    extract_quantity_frames(text::AbstractString) -> Vector{QuantityFrame}

استخراج جميع أطر الكمية (QuantityFrame) من النص.
تبحث عن مفاتيح الكمية المعروفة في كل جملة، وتُعيد إطاراً لكل مفتاح يُعثر عليه.
لا تؤثر على ذاكرة الاستنباط الحالية.
"""
function _quantity_push_frame!(frames::Vector{QuantityFrame}, marker::String, qtype::String,
                               target::AbstractString, value::AbstractString,
                               confidence::Float64; require_answerable::Bool=false)
    frame = QuantityFrame(marker, qtype, _clean_quantity_target_piece(target), _clean_quantity_piece(value),
                          "neutral", confidence)
    isempty(frame.target) && isempty(frame.value) && return false
    require_answerable && !_quantity_frame_answerable(frame) && return false
    push!(frames, frame)
    return true
end

function _quantity_after_phrase(s::AbstractString, phrase::AbstractString)
    idx = findfirst(phrase, s)
    idx === nothing && return ""
    return _clean_quantity_piece(s[nextind(s, last(idx)):end])
end

function _quantity_split_statement(s::AbstractString, marker::AbstractString)
    before, after = _split_by_marker(s, marker)
    return _clean_quantity_piece(before), _clean_quantity_piece(after)
end

function _quantity_strip_answer_prefix(s::AbstractString)
    text = _clean_quantity_piece(s)
    return _clean_quantity_piece(replace(text, r"^(?:\u0646\u0639\u0645|\u0644\u0627)\s*[\u060C,]?\s+" => ""))
end

function _quantity_marker_word_hit(s::AbstractString, marker::AbstractString)
    padded = " " * _clean_space(String(s)) * " "
    return occursin(" " * String(marker) * " ", padded)
end

function _quantity_count_question_targets(s::AbstractString)
    x = lowercase(_clean_space(String(s)))
    if startswith(x, "how many ")
        rest = _quantity_after_phrase(x, "how many")
        return [_clean_quantity_piece(rest)]
    elseif startswith(x, "number of ")
        rest = _quantity_after_phrase(x, "number of")
        return [_clean_quantity_piece(rest)]
    end
    startswith(s, "\u0643\u0645 \u0639\u062f\u062f") || return String[]
    rest = _quantity_after_phrase(s, "\u0643\u0645 \u0639\u062f\u062f")
    parts = split(rest, "\u0648\u0643\u0645 \u0639\u062f\u062f")
    return [_clean_quantity_piece(p) for p in parts if !isempty(_clean_quantity_piece(p))]
end

function _clean_quantity_target_piece(s::AbstractString)
    text = _clean_quantity_piece(s)
    text = replace(text, r"\s+(?:\u0641\u064a|\u0645\u0646|\u0639\u0646|\u0639\u0646\u062f|\u0625\u0644\u0649|\u0627\u0644\u0649|in|of|to|at)$"i => "")
    return _clean_quantity_piece(text)
end

function extract_quantity_frames(text::AbstractString)
    frames = QuantityFrame[]
    for sentence in _sentences(String(text))
        s = _clean_space(sentence)
        isempty(s) && continue

        if occursin("\u0643\u0645 \u0639\u062f\u062f", s)
            for target in _quantity_count_question_targets(s)
                _quantity_push_frame!(frames, "\u0643\u0645", "count", target, "", 0.90)
            end
            continue
        elseif startswith(lowercase(s), "how many ")
            target = _quantity_after_phrase(lowercase(s), "how many")
            _quantity_push_frame!(frames, "how many", "count", target, "", 0.90)
            continue
        elseif startswith(lowercase(s), "how much ")
            target = _quantity_after_phrase(lowercase(s), "how much")
            _quantity_push_frame!(frames, "how much", "measure", target, "", 0.90)
            continue
        elseif startswith(lowercase(s), "what amount ")
            target = _quantity_after_phrase(lowercase(s), "what amount")
            _quantity_push_frame!(frames, "amount", "measure", target, "", 0.90)
            continue
        elseif startswith(lowercase(s), "what quantity ")
            target = _quantity_after_phrase(lowercase(s), "what quantity")
            _quantity_push_frame!(frames, "quantity", "measure", target, "", 0.90)
            continue
        elseif startswith(s, "\u0643\u0645 \u0637\u0648\u0644 ")
            target = _quantity_after_phrase(s, "\u0643\u0645 \u0637\u0648\u0644")
            _quantity_push_frame!(frames, "\u0637\u0648\u0644", "measure", target, "", 0.90)
            continue
        elseif startswith(s, "\u0643\u0645 \u0648\u0632\u0646 ")
            target = _quantity_after_phrase(s, "\u0643\u0645 \u0648\u0632\u0646")
            _quantity_push_frame!(frames, "\u0648\u0632\u0646", "measure", target, "", 0.90)
            continue
        elseif startswith(s, "\u0643\u0645 \u0645\u062f\u0629 ")
            target = _quantity_after_phrase(s, "\u0643\u0645 \u0645\u062f\u0629")
            _quantity_push_frame!(frames, "\u0645\u062f\u0629", "measure", target, "", 0.90)
            continue
        elseif startswith(s, "\u0643\u0645 \u0645\u0633\u0627\u0641\u0629 ")
            target = _quantity_after_phrase(s, "\u0643\u0645 \u0645\u0633\u0627\u0641\u0629")
            _quantity_push_frame!(frames, "\u0645\u0633\u0627\u0641\u0629", "measure", target, "", 0.90)
            continue
        elseif startswith(s, "\u0643\u0645 ")
            target = _quantity_after_phrase(s, "\u0643\u0645")
            _quantity_push_frame!(frames, "\u0643\u0645", "count", target, "", 0.90)
            continue
        elseif occursin("\u0645\u0627 \u0645\u0642\u062f\u0627\u0631", s)
            target = _quantity_after_phrase(s, "\u0645\u0627 \u0645\u0642\u062f\u0627\u0631")
            _quantity_push_frame!(frames, "\u0645\u0642\u062f\u0627\u0631", "measure", target, "", 0.90)
            continue
        elseif occursin("\u0645\u0627 \u0643\u0645\u064a\u0629", s)
            target = _quantity_after_phrase(s, "\u0645\u0627 \u0643\u0645\u064a\u0629")
            _quantity_push_frame!(frames, "\u0643\u0645\u064a\u0629", "measure", target, "", 0.90)
            continue
        end

        if _quantity_marker_word_hit(s, "\u0639\u062f\u062f")
            target, value = _quantity_split_statement(s, "\u0639\u062f\u062f")
            _quantity_push_frame!(frames, "\u0639\u062f\u062f", "count", target, value, 0.90; require_answerable=true)
        end
        if _quantity_marker_word_hit(lowercase(s), "number")
            target, value = _quantity_split_statement(lowercase(s), "number")
            _quantity_push_frame!(frames, "number", "count", target, value, 0.90; require_answerable=true)
        end
        if occursin("number of", lowercase(s))
            target, value = _quantity_split_statement(lowercase(s), "number of")
            _quantity_push_frame!(frames, "number of", "count", target, value, 0.90; require_answerable=true)
        end
        if _quantity_marker_word_hit(s, "\u0645\u0642\u062f\u0627\u0631")
            target, value = _quantity_split_statement(s, "\u0645\u0642\u062f\u0627\u0631")
            _quantity_push_frame!(frames, "\u0645\u0642\u062f\u0627\u0631", "measure", target, value, 0.90; require_answerable=true)
        end
        if _quantity_marker_word_hit(s, "\u0643\u0645\u064a\u0629")
            target, value = _quantity_split_statement(s, "\u0643\u0645\u064a\u0629")
            _quantity_push_frame!(frames, "\u0643\u0645\u064a\u0629", "measure", target, value, 0.90; require_answerable=true)
        end
        if _quantity_marker_word_hit(s, "\u062f\u0631\u062c\u0629")
            target, value = _quantity_split_statement(s, "\u062f\u0631\u062c\u0629")
            _quantity_push_frame!(frames, "\u062f\u0631\u062c\u0629", "measure", target, value, 0.90; require_answerable=true)
        end
        if _quantity_marker_word_hit(s, "\u062f\u0631\u062c\u0629_\u062d\u0631\u0627\u0631\u0629")
            target, value = _quantity_split_statement(s, "\u062f\u0631\u062c\u0629_\u062d\u0631\u0627\u0631\u0629")
            _quantity_push_frame!(frames, "\u062f\u0631\u062c\u0629_\u062d\u0631\u0627\u0631\u0629", "measure", target, value, 0.90; require_answerable=true)
        end
        if _quantity_marker_word_hit(s, "\u062f\u0631\u062c\u0629 \u062d\u0631\u0627\u0631\u0629")
            target, value = _quantity_split_statement(s, "\u062f\u0631\u062c\u0629 \u062d\u0631\u0627\u0631\u0629")
            _quantity_push_frame!(frames, "\u062f\u0631\u062c\u0629 \u062d\u0631\u0627\u0631\u0629", "measure", target, value, 0.90; require_answerable=true)
        end
        for marker in ("amount", "quantity", "length", "weight", "duration", "distance", "temperature", "degree")
            _quantity_marker_word_hit(lowercase(s), marker) || continue
            target, value = _quantity_split_statement(lowercase(s), marker)
            _quantity_push_frame!(frames, marker, "measure", target, value, 0.90; require_answerable=true)
        end

        for marker in ("\u0623\u0643\u062b\u0631", "\u0623\u0642\u0644", "\u0636\u0639\u0641", "\u0646\u0635\u0641")
            _quantity_marker_word_hit(s, marker) || continue
            target, value = _quantity_split_statement(s, marker)
            target = _quantity_strip_answer_prefix(target)
            _quantity_push_frame!(frames, marker, "comparison", target, value, 0.75; require_answerable=true)
        end
        for marker in ("more than", "less than", "greater than", "equal to", "half", "double")
            occursin(marker, lowercase(s)) || continue
            target, value = _quantity_split_statement(lowercase(s), marker)
            target = _quantity_strip_answer_prefix(target)
            _quantity_push_frame!(frames, marker, "comparison", target, value, 0.75; require_answerable=true)
        end

        if startswith(s, "\u0643\u0644 ")
            target = _clean_quantity_piece(replace(s, r"^\s*\u0643\u0644\s+" => ""))
            length(_quantity_piece_words(target)) <= 6 || continue
            _quantity_push_frame!(frames, "\u0643\u0644", "quantifier_scope", target, "", 0.75; require_answerable=true)
        elseif startswith(s, "\u0628\u0639\u0636 ")
            target = _clean_quantity_piece(replace(s, r"^\s*\u0628\u0639\u0636\s+" => ""))
            length(_quantity_piece_words(target)) <= 6 || continue
            _quantity_push_frame!(frames, "\u0628\u0639\u0636", "quantifier_scope", target, "", 0.75; require_answerable=true)
        elseif startswith(lowercase(s), "all ")
            target = _clean_quantity_piece(replace(lowercase(s), r"^\s*all\s+" => ""))
            length(_quantity_piece_words(target)) <= 6 || continue
            _quantity_push_frame!(frames, "all", "quantifier_scope", target, "", 0.75; require_answerable=true)
        elseif startswith(lowercase(s), "some ")
            target = _clean_quantity_piece(replace(lowercase(s), r"^\s*some\s+" => ""))
            length(_quantity_piece_words(target)) <= 6 || continue
            _quantity_push_frame!(frames, "some", "quantifier_scope", target, "", 0.75; require_answerable=true)
        end

        for marker in ("\u0643\u062b\u064a\u0631", "\u0642\u0644\u064a\u0644")
            _quantity_marker_word_hit(s, marker) || continue
            target, value = _quantity_split_statement(s, marker)
            if isempty(target) && startswith(s, "\u0647\u0646\u0627\u0643 ")
                target = _clean_quantity_piece(replace(s, r"^\s*\u0647\u0646\u0627\u0643\s+" => ""))
                value = ""
            end
            _quantity_push_frame!(frames, marker, "vague_quantity", target, value, 0.60; require_answerable=true)
        end
        for marker in ("many", "few")
            _quantity_marker_word_hit(lowercase(s), marker) || continue
            target, value = _quantity_split_statement(lowercase(s), marker)
            _quantity_push_frame!(frames, marker, "vague_quantity", target, value, 0.60; require_answerable=true)
        end
    end
    return frames
end

"""
    learn_quantity_frames_from_text!(mem::QuantityFrameMemory,
                                     text::AbstractString,
                                     source::Dict{String,Any}=Dict{String,Any}()) -> Int

Append QuantityFrame observations extracted from text to a lightweight quantity
memory. This does not touch IstinbatAttentionMemory.
"""
function learn_quantity_frames_from_text!(mem::QuantityFrameMemory,
                                          text::AbstractString,
                                          source::Dict{String,Any}=Dict{String,Any}())
    learned = 0
    for frame in extract_quantity_frames(text)
        isempty(strip(frame.target)) && isempty(strip(frame.value)) && continue
        push!(mem.frames, frame)
        push!(mem.source_metadata, copy(source))
        learned += 1
    end
    return learned
end

function train_quantity_frames_from_texts!(mem::QuantityFrameMemory, texts,
                                           metadata=nothing; max_items::Int=50_000)
    learned = 0
    for (i, text) in enumerate(texts)
        learned >= max_items && break
        source = (metadata !== nothing && i <= length(metadata)) ? metadata[i] : Dict{String,Any}()
        learned += learn_quantity_frames_from_text!(mem, text, source)
    end
    return learned
end

function _is_quantity_question(s::AbstractString)
    x = lowercase(_clean_space(String(s)))
    isempty(x) && return false
    return occursin("\u0643\u0645", x) ||
           occursin("\u0639\u062f\u062f", x) ||
           occursin("\u0645\u0642\u062f\u0627\u0631", x) ||
           occursin("\u0643\u0645\u064a\u0629", x) ||
           occursin("\u0646\u0637\u0627\u0642", x) ||
           occursin("\u0623\u064a\u0647\u0645\u0627 \u0623\u0643\u062b\u0631", x) ||
           occursin("\u0627\u064a\u0647\u0645\u0627 \u0627\u0643\u062b\u0631", x) ||
           occursin("\u0623\u064a\u0647\u0645\u0627 \u0623\u0642\u0644", x) ||
           occursin("\u0627\u064a\u0647\u0645\u0627 \u0627\u0642\u0644", x) ||
           occursin("how many", x) ||
           occursin("how much", x) ||
           occursin("what amount", x) ||
           occursin("what quantity", x) ||
           occursin("which is more", x) ||
           occursin("which is greater", x) ||
           occursin("which is less", x) ||
           occursin("درجة", x) ||
           occursin("درجه", x) ||
           occursin("حرارة", x) ||
           occursin("حراره", x) ||
           occursin("temperature", x) ||
           occursin("degree", x)
end

function _quantity_frame_terms(frame::QuantityFrame)
    terms = Set{String}()
    for t in vcat(_tokens(frame.target), _tokens(frame.value), _tokens(frame.marker))
        isempty(t) || push!(terms, t)
        p = _projection_word(t)
        isempty(p) || push!(terms, p)
    end
    return terms
end

function _quantity_overlap_score(query::Set{String}, frame::QuantityFrame)
    isempty(query) && return 0.0
    terms = _quantity_frame_terms(frame)
    isempty(terms) && return 0.0
    hits = length(intersect(query, terms))
    hits == 0 && return 0.0
    return hits / max(1, min(length(query), length(terms)))
end

"""
    select_quantity_frame(frames::Vector{QuantityFrame}, prompt::AbstractString;
                          min_score::Float64=0.15)

Select the best quantity frame for a quantity prompt. This is diagnostic and
does not read or mutate Istinbat memory.
"""
function select_quantity_frame(frames::Vector{QuantityFrame}, prompt::AbstractString;
                               min_score::Float64=0.45)
    query = _query_terms(prompt)
    prompt_kind = _quantity_prompt_kind(prompt)
    best = nothing
    best_score = 0.0
    for frame in frames
        _quantity_frame_answerable(frame) || continue
        _quantity_frame_matches_prompt(frame, prompt_kind) || continue
        score = 0.75 * _quantity_overlap_score(query, frame) + 0.25 * frame.confidence
        if score > best_score
            best = frame
            best_score = score
        end
    end
    best_score >= min_score || return nothing
    return best
end

function select_quantity_frame(mem::QuantityFrameMemory, prompt::AbstractString;
                               min_score::Float64=0.45)
    return select_quantity_frame(mem.frames, prompt; min_score=min_score)
end

function _format_quantity_answer(frame::QuantityFrame)
    target = _clean_quantity_piece(frame.target)
    value = _clean_quantity_piece(frame.value)
    marker = _clean_quantity_piece(frame.marker)
    qtype = frame.quantity_type

    if qtype == "count"
        isempty(target) && return isempty(value) ? "" : "\u0627\u0644\u0639\u062f\u062f \u0647\u0648 $(value)."
        isempty(value) && return "\u0627\u0644\u0639\u062f\u062f \u064a\u062a\u0639\u0644\u0642 \u0628\u0640 $(target)."
        if endswith(target, "\u064a\u062f\u0644 \u0639\u0644\u0649")
            subject = strip(replace(target, r"\s*\u064a\u062f\u0644\s+\u0639\u0644\u0649$" => ""))
            !isempty(subject) && return "\u064a\u062f\u0644 $(subject) \u0639\u0644\u0649 $(value)."
        end
        return "\u0639\u062f\u062f $(target) \u0647\u0648 $(value)."
    elseif qtype == "measure"
        isempty(target) && return isempty(value) ? "" : "\u0627\u0644\u0645\u0642\u062f\u0627\u0631 \u0647\u0648 $(value)."
        isempty(value) && return "\u0645\u0642\u062f\u0627\u0631 $(target) \u063a\u064a\u0631 \u0645\u062d\u062f\u062f."
        if marker in ("درجة", "درجة_حرارة", "درجة حرارة", "degree", "temperature")
            return "درجة حرارة $(target) هي $(value)."
        end
        return "\u0645\u0642\u062f\u0627\u0631 $(target) \u0647\u0648 $(value)."
    elseif qtype == "comparison"
        if isempty(target)
            return isempty(value) ? "" : "\u0627\u0644\u0645\u0642\u0627\u0631\u0646\u0629: $(marker) $(value)."
        end
        isempty(value) && return "\u0627\u0644\u0645\u0642\u0627\u0631\u0646\u0629 \u062a\u062a\u0639\u0644\u0642 \u0628\u0640 $(target)."
        return "$(target) $(marker) $(value)."
    elseif qtype == "quantifier_scope"
        isempty(value) && return isempty(target) ? "" : "\u0627\u0644\u0646\u0637\u0627\u0642: $(marker) $(target)."
        isempty(target) && return "\u0627\u0644\u0646\u0637\u0627\u0642: $(marker) $(value)."
        return "\u0627\u0644\u0646\u0637\u0627\u0642: $(marker) $(target) $(value)."
    elseif qtype == "vague_quantity"
        isempty(value) && return isempty(target) ? "" : "\u0627\u0644\u0643\u0645\u064a\u0629 \u0627\u0644\u062a\u0642\u0631\u064a\u0628\u064a\u0629 \u0641\u064a $(target): $(marker)."
        isempty(target) && return "\u0627\u0644\u0643\u0645\u064a\u0629 \u0627\u0644\u062a\u0642\u0631\u064a\u0628\u064a\u0629: $(marker) \u0641\u064a $(value)."
        return "\u0627\u0644\u0643\u0645\u064a\u0629 \u0627\u0644\u062a\u0642\u0631\u064a\u0628\u064a\u0629 \u0641\u064a $(target): $(marker) $(value)."
    end
    return ""
end

"""
    quantity_answer(frames::Vector{QuantityFrame}, prompt::AbstractString) -> String

Generate an independent answer from QuantityFrame records. This is a diagnostic
quantity layer only; it is not inserted into `generate!`.
"""
function quantity_answer(frames::Vector{QuantityFrame}, prompt::AbstractString)
    s = _clean_space(String(prompt))
    isempty(s) && return ""
    _is_yesno_question(s) && return ""
    _is_quantity_question(s) || return ""
    _quantity_prompt_is_answerable(s) || return ""

    frame = select_quantity_frame(frames, s)
    frame === nothing && return ""
    frame.confidence >= 0.30 || return ""
    _quantity_frame_answerable(frame) || return ""
    isempty(strip(frame.target)) && isempty(strip(frame.value)) && return ""
    return _format_quantity_answer(frame)
end

function quantity_answer(mem::QuantityFrameMemory, prompt::AbstractString)
    return quantity_answer(mem.frames, prompt)
end

"""
    QuantityComparisonRecord

Independent comparison record between quantity_answer and a caller-supplied
generation function. This is diagnostic only and does not enter generate!.
"""
struct QuantityComparisonRecord
    prompt::String
    quantity_answer::String
    generate_answer::String
    memory_has_quantity::Bool
    quantity_confidence::Float64
    overlap_score::Float64
    has_marker::Bool
    quantity_type::String
end

"""
    compare_quantity_strategies(frames::Vector{QuantityFrame},
                                generate_func::Function,
                                prompt::AbstractString) -> QuantityComparisonRecord

Diagnostic comparison between quantity_answer and a caller-supplied generation
function. This keeps quantity reasoning measurable without changing generation.
"""
function compare_quantity_strategies(frames::Vector{QuantityFrame},
                                     generate_func::Function,
                                     prompt::AbstractString)
    s = _clean_space(String(prompt))
    q_ans = quantity_answer(frames, s)

    generate_ans = try
        String(generate_func(s))
    catch err
        "ERROR: $(typeof(err))"
    end

    frame = select_quantity_frame(frames, s)
    found = frame !== nothing && !isempty(q_ans)
    confidence = found ? frame.confidence : 0.0
    overlap = found ? _quantity_overlap_score(_query_terms(s), frame) : 0.0
    marker_found = found ? !isempty(strip(frame.marker)) : false
    qtype = found ? frame.quantity_type : "none"

    return QuantityComparisonRecord(s, q_ans, generate_ans,
                                    found, confidence, overlap,
                                    marker_found, qtype)
end

function compare_quantity_strategies(mem::QuantityFrameMemory,
                                     generate_func::Function,
                                     prompt::AbstractString)
    return compare_quantity_strategies(mem.frames, generate_func, prompt)
end

"""
    learn_relation_frames_from_text!(mem::IstinbatAttentionMemory, text::AbstractString,
                                      source::Dict{String,Any}=Dict{String,Any}()) -> Int

تخزين أطر العلاقات (RelationFrame) المستخرجة من النص في ذاكرة الاستنباط.
تستخدم `extract_relation_frames` لرصد جميع المفاتيح (old + new) وتحفظها كسجلات.
طبقة رصد فقط — لا تؤثر على التوليد أو الإجابات النهائية.
"""
function learn_relation_frames_from_text!(mem::IstinbatAttentionMemory, text::AbstractString,
                                           source::Dict{String,Any}=Dict{String,Any}())
    learned = 0
    frames = extract_relation_frames(text)
    for frame in frames
        left_clean = String[strip(t) for t in frame.left_terms if !isempty(strip(t))]
        right_clean = String[strip(t) for t in frame.right_terms if !isempty(strip(t))]
        isempty(left_clean) && isempty(right_clean) && continue
        focus = unique(vcat(left_clean, right_clean))
        item = (
            relation_type=frame.relation_type,
            marker=frame.marker,
            before_terms=left_clean,
            after_terms=right_clean,
            focus_terms=focus,
            polarity=frame.polarity,
        )
        _record!(mem, frame.source_sentence, item, source)
        learned += 1
    end
    return learned
end

"""
    relation_frame_diagnostic(mem::IstinbatAttentionMemory, prompt::AbstractString) -> String

تقرير تشخيصي يبين الإطار العلائقي الذي اختاره `select_relation_frame_attention`
دون استخدامه في الإجابة. الهدف: رؤية ما يراه مرنان داخلياً (للمطور/الباحث فقط).
"""
function relation_frame_diagnostic(mem::IstinbatAttentionMemory, prompt::AbstractString)
    rec = select_relation_frame_attention(mem, prompt)
    rec === nothing && return "لا يوجد إطار علائقي مناسب."
    lines = String[]
    push!(lines, "نوع العلاقة: $(rec.relation_type)")
    push!(lines, "المفتاح: $(rec.marker)")
    push!(lines, "الطرف الأيسر: $(join(rec.before_terms, "، "))")
    push!(lines, "الطرف الأيمن: $(join(rec.after_terms, "، "))")
    push!(lines, "الثقة: $(rec.attention_weight)")
    if !isempty(rec.examples)
        push!(lines, "المثال: $(first(rec.examples))")
    end
    return join(lines, "\n")
end

# ---------------------------------------------------------------------------
# Purpose-specific answer generation (internal, does NOT touch yesno_relations)
# ---------------------------------------------------------------------------

"""
    _is_purpose_question(s::AbstractString) -> Bool

تحديد إذا كان السؤال من نوع "لماذا/لأي غاية/من أجل ماذا" (غائي).
"""
function _is_purpose_question(s::AbstractString)
    q = lowercase(_clean_space(String(s)))
    (startswith(q, "why ") ||
     occursin("for what purpose", q) ||
     occursin("what purpose", q) ||
     occursin("what goal", q) ||
     occursin("what benefit", q)) && return true
    return occursin(r"^لماذا\s+", s) ||
           occursin(r"لأي\s+غاية", s) ||
           occursin(r"من\s+أجل\s+ماذا", s) ||
           occursin(r"لأي\s+هدف", s) ||
           occursin(r"ما\s+(?:هي\s+)?غاية", s) ||
           occursin(r"ما\s+(?:هو\s+)?هدف", s) ||
           occursin(r"ما\s+فائدة", s)
end

"""
    _is_yesno_question(s::AbstractString) -> Bool

تحديد إذا كان السؤال من نوع "هل" (مباشر بنعم/لا).
"""
function _is_yesno_question(s::AbstractString)
    q = lowercase(strip(String(s)))
    (startswith(q, "does ") || startswith(q, "do ") ||
     startswith(q, "did ") || startswith(q, "is ") ||
     startswith(q, "are ") || startswith(q, "was ") ||
     startswith(q, "were ") || startswith(q, "can ") ||
     startswith(q, "should ")) && return true
    return startswith(strip(s), "هل") || startswith(strip(s), "\u0647\u0644")
end

"""
    _format_purpose_answer(rec::IstinbatAttentionRecord) -> String

تنسيق جواب غائي من RelationFrame من نوع purpose.
"""
function _looks_like_event_term(term::AbstractString)
    t = lowercase(strip(String(term)))
    isempty(t) && return false

    english_events = Set([
        "hit", "hits", "pushed", "push", "pushes", "broke", "break", "breaks",
        "illuminated", "illuminate", "illuminates", "study", "studies", "studied",
        "succeed", "succeeds", "succeeded",
        "work", "works", "worked", "open", "opens", "opened", "move", "moves",
        "moved", "learn", "learns", "learned", "read", "reads", "write", "writes",
    ])
    t in english_events && return true
    endswith(t, "ing") && length(t) > 5 && return true

    arabic_events = Set([
        "\u062f\u0631\u0633", "\u064a\u062f\u0631\u0633", "\u0627\u062c\u062a\u0647\u062f", "\u064a\u062c\u062a\u0647\u062f",
        "\u0639\u0645\u0644", "\u064a\u0639\u0645\u0644", "\u0641\u062a\u062d", "\u064a\u0641\u062a\u062d",
        "\u0646\u0642\u0644", "\u064a\u0646\u0642\u0644", "\u062d\u0641\u0638", "\u064a\u062d\u0641\u0638",
        "\u0636\u0631\u0628", "\u064a\u0636\u0631\u0628", "\u062f\u0641\u0639", "\u064a\u062f\u0641\u0639",
        "\u0643\u0633\u0631", "\u064a\u0643\u0633\u0631", "\u0627\u0636\u0627\u0621", "\u064a\u0636\u064a\u0621",
        "\u062a\u0639\u0644\u0645", "\u064a\u062a\u0639\u0644\u0645", "\u0642\u0631\u0623", "\u064a\u0642\u0631\u0623",
        "\u0643\u062a\u0628", "\u064a\u0643\u062a\u0628", "\u0633\u0639\u0649", "\u064a\u0633\u0639\u0649",
    ])
    t in arabic_events && return true

    if occursin(r"^[\u0600-\u06FF]+$", t)
        startswith(t, "\u064a") && length(t) >= 4 && return true
        startswith(t, "\u062a") && length(t) >= 4 && return true
        startswith(t, "\u0646") && length(t) >= 4 && return true
        startswith(t, "\u0623") && length(t) >= 4 && return true
    end

    return false
end

function _purpose_record_quality_ok(rec::IstinbatAttentionRecord)
    rec.relation_type == "purpose" || return true
    isempty(strip(join(rec.before_terms, " "))) && return false
    isempty(strip(join(rec.after_terms, " "))) && return false
    return any(_looks_like_event_term, rec.before_terms)
end

function _has_arabic_chars(s::AbstractString)
    return occursin(r"[\u0600-\u06FF]", String(s))
end

function _arabic_definite(term::AbstractString)
    t = strip(String(term))
    isempty(t) && return t
    _has_arabic_chars(t) || return t
    startswith(t, "\u0627\u0644") && return t
    startswith(t, "\u0644\u0644") && return t
    _looks_like_event_term(t) && return t
    return "\u0627\u0644$(t)"
end

function _format_purpose_left_terms(terms::Vector{String})
    isempty(terms) && return ""
    length(terms) == 1 && return first(terms)
    length(terms) > 4 && return join(terms, " ")
    first_term = first(terms)
    rest = [_arabic_definite(t) for t in terms[2:end]]
    return join(vcat([first_term], rest), " ")
end

function _format_purpose_right_terms(terms::Vector{String})
    isempty(terms) && return ""
    if 3 <= length(terms) <= 5 && _has_arabic_chars(first(terms))
        verb = first(terms)
        obj = _arabic_definite(terms[2])
        target = _arabic_definite(terms[3])
        if startswith(verb, "\u064a\u0628\u0639\u062f") || startswith(verb, "\u062a\u0628\u0639\u062f")
            return "\u0625\u0628\u0639\u0627\u062f $(obj) \u0639\u0646 $(target)"
        elseif startswith(verb, "\u064a\u0638\u0647\u0631") || startswith(verb, "\u062a\u0638\u0647\u0631")
            return "\u0625\u0638\u0647\u0627\u0631 $(obj) \u0628\u0648\u0636\u0648\u062d"
        elseif startswith(verb, "\u064a\u062d\u0641\u0638") || startswith(verb, "\u062a\u062d\u0641\u0638")
            return "\u062d\u0641\u0638 $(obj)"
        end
    end
    return join(terms, " ")
end

function _format_purpose_answer(rec::IstinbatAttentionRecord)
    left = _format_purpose_left_terms(rec.before_terms)
    right = _format_purpose_right_terms(rec.after_terms)
    length(split(left)) > 12 && return ""
    length(split(right)) > 16 && return ""
    if !_has_arabic_chars(left) && !_has_arabic_chars(right)
        !isempty(left) && !isempty(right) && return "The purpose of $(left) is $(right)."
        !isempty(right) && return "The purpose is $(right)."
    end
    if !isempty(left) && !isempty(right)
        return "الغاية من $left هي $right."
    elseif !isempty(right)
        return "الغاية هي $right."
    end
    return ""
end

function _select_purpose_attention(mem::IstinbatAttentionMemory, prompt::AbstractString)
    strict = _select_typed_relation_frame_attention(mem, prompt, "purpose")
    strict !== nothing && return strict

    query = _query_terms(prompt)
    isempty(query) && return nothing
    best = nothing
    best_score = -Inf
    for rec in values(mem.records)
        rec.relation_type == "purpose" || continue
        rec.attention_weight < 0.30 && continue
        _purpose_record_quality_ok(rec) || continue
        isempty(strip(join(rec.before_terms, " "))) && continue
        isempty(strip(join(rec.after_terms, " "))) && continue
        overlap = _overlap_score(query, rec)
        overlap < 0.15 && continue
        score = 0.75 * overlap + 0.25 * rec.attention_weight
        if score > best_score
            best = rec
            best_score = score
        end
    end
    return best
end

"""
    purpose_answer(mem::IstinbatAttentionMemory, prompt::AbstractString) -> String

توليد جواب لسؤال غائي (لماذا/لأي غاية/من أجل ماذا) باستخدام RelationFrame من نوع purpose.
لا تعمل مع أسئلة "هل"، ولا مع العلاقات السببية.
تُعيد نصاً فارغاً إذا لم تجد تطابقاً كافياً.
"""
function purpose_answer(mem::IstinbatAttentionMemory, prompt::AbstractString)
    s = _clean_space(String(prompt))
    isempty(s) && return ""

    _is_purpose_question(s) || return ""
    _is_yesno_question(s) && return ""

    rec = _select_purpose_attention(mem, s)
    rec === nothing && return ""
    _purpose_record_quality_ok(rec) || return ""

    rec.attention_weight < 0.30 && return ""

    query = _query_terms(s)
    overlap = _overlap_score(query, rec)
    overlap < 0.15 && return ""
    isempty(strip(join(rec.before_terms, " "))) && return ""
    isempty(strip(join(rec.after_terms, " "))) && return ""

    return _format_purpose_answer(rec)
end

function _is_conditional_question(s::AbstractString)
    q = _clean_space(String(s))
    return occursin("\u0625\u0630\u0627", q) ||
           occursin("\u0627\u0630\u0627", q) ||
           occursin("\u0625\u0646 ", q) ||
           occursin("\u0644\u0648 ", q) ||
           occursin("\u0645\u0627\u0630\u0627 \u064a\u062d\u062f\u062b", q) ||
           occursin("\u0645\u0627 \u0627\u0644\u0646\u062a\u064a\u062c\u0629", q) ||
           occursin("\u0645\u0627 \u064a\u062d\u062f\u062b", q) ||
           occursin("what happens", lowercase(q)) ||
           occursin("if ", lowercase(q))
end

function _conditional_prompt_terms(prompt::AbstractString)
    s = _clean_space(String(prompt))
    ls = lowercase(s)
    markers = [
        "\u0625\u0630\u0627", "\u0627\u0630\u0627", "\u0625\u0646", "\u0644\u0648",
        "if", "when",
    ]
    for marker in markers
        haystack = all(isascii, marker) ? ls : s
        needle = all(isascii, marker) ? lowercase(marker) : marker
        occursin(needle, haystack) || continue
        _, after = _split_by_marker(haystack, needle)
        terms = _ordered_tokens(after)
        terms = [t for t in terms if !(t in Set([
            "\u0645\u0627", "\u0645\u0627\u0630\u0627", "\u064a\u062d\u062f\u062b", "\u0627\u0644\u0646\u062a\u064a\u062c\u0629",
            "what", "happens", "result", "the",
        ]))]
        !isempty(terms) && return terms
    end
    return _query_terms(prompt)
end

function _format_condition_terms(terms::Vector{String})
    isempty(terms) && return ""
    any(_has_arabic_chars, terms) && return _format_purpose_left_terms(terms)
    return join(terms, " ")
end

function _format_result_terms(terms::Vector{String})
    isempty(terms) && return ""
    any(_has_arabic_chars, terms) && return _format_purpose_left_terms(terms)
    return join(terms, " ")
end

function _complete_conditional_from_example(rec::IstinbatAttentionRecord)
    isempty(rec.examples) && return nothing
    example = _clean_space(first(rec.examples))
    isempty(example) && return nothing

    marker = String(strip(String(rec.marker)))
    isempty(marker) && (marker = "\u0625\u0630\u0627")
    before, after = _split_by_marker(example, marker)
    after_terms = _ordered_tokens(after)
    length(after_terms) >= 4 || return nothing

    # Arabic conditional records often arrive as "verb subject verb result".
    # Keep this conservative so it only repairs clearly paired clauses.
    if _has_arabic_chars(after)
        a = after_terms
        if length(a) >= 4 && a[1] == a[3]
            return String[a[1], a[2]], String[a[3], a[4]]
        elseif length(a) == 4
            return String[a[1], a[2]], String[a[3], a[4]]
        end
    else
        left, right = _conditional_split_terms(after_terms)
        !isempty(left) && !isempty(right) && return left, right
    end
    return nothing
end

function _conditional_terms_for_answer(rec::IstinbatAttentionRecord)
    before_terms = rec.before_terms
    after_terms = rec.after_terms
    if length(before_terms) < 2 || length(after_terms) < 2
        completed = _complete_conditional_from_example(rec)
        if completed !== nothing
            before_terms, after_terms = completed
        end
    end
    isempty(strip(join(before_terms, " "))) && return nothing
    isempty(strip(join(after_terms, " "))) && return nothing
    (length(before_terms) < 2 || length(after_terms) < 2) && return nothing
    return before_terms, after_terms
end

function _conditional_record_answerable(rec::IstinbatAttentionRecord)
    rec.relation_type == "conditional" || return false
    return _conditional_terms_for_answer(rec) !== nothing
end

function _select_conditional_attention(mem::IstinbatAttentionMemory, prompt::AbstractString)
    query = _query_terms(prompt)
    condition_query = _conditional_prompt_terms(prompt)
    condition_keys = Set{String}()
    for term in condition_query
        union!(condition_keys, _term_keyset(term))
    end
    required_condition_hits = length(condition_query) >= 2 ? 2 : length(condition_query)
    best = nothing
    best_score = -Inf
    for rec in values(mem.records)
        rec.relation_type == "conditional" || continue
        rec.attention_weight < 0.30 && continue
        completed_terms = _conditional_terms_for_answer(rec)
        completed_terms === nothing && continue
        before_terms, after_terms = completed_terms
        length(before_terms) <= 12 && length(after_terms) <= 16 || continue
        if required_condition_hits > 0
            condition_hits = count(t -> _term_matches(t, condition_keys), before_terms)
            condition_hits >= required_condition_hits || continue
        end
        overlap = _overlap_score(query, rec)
        overlap < 0.15 && continue
        score = 0.75 * overlap + 0.25 * rec.attention_weight
        if score > best_score
            best = rec
            best_score = score
        end
    end
    return best
end

function _format_conditional_answer(rec::IstinbatAttentionRecord)
    completed_terms = _conditional_terms_for_answer(rec)
    completed_terms === nothing && return ""
    before_terms, after_terms = completed_terms

    condition = _format_condition_terms(before_terms)
    result = _format_result_terms(after_terms)
    isempty(condition) && return ""
    isempty(result) && return ""
    length(split(condition)) > 12 && return ""
    length(split(result)) > 16 && return ""
    if _has_arabic_chars(condition) || _has_arabic_chars(result)
        return "\u0625\u0630\u0627 $(condition)\u061b \u0641\u0627\u0644\u0646\u062a\u064a\u062c\u0629: $(result)."
    end
    return "If $(condition) happens, the result is $(result)."
end

"""
    conditional_answer(mem::IstinbatAttentionMemory, prompt::AbstractString) -> String

Generate an independent answer for conditional RelationFrame records. This is a
diagnostic/rational layer only; it is not inserted into `generate!`.
"""
function conditional_answer(mem::IstinbatAttentionMemory, prompt::AbstractString)
    s = _clean_space(String(prompt))
    isempty(s) && return ""
    _is_yesno_question(s) && return ""
    _is_conditional_question(s) || return ""

    rec = _select_conditional_attention(mem, s)
    rec === nothing && return ""

    return _format_conditional_answer(rec)
end

function _is_temporal_question(s::AbstractString)
    q = _clean_space(String(s))
    lq = lowercase(q)
    return occursin("\u0645\u062a\u0649", q) ||
           occursin("\u0642\u0628\u0644 \u0645\u0627\u0630\u0627", q) ||
           occursin("\u0628\u0639\u062f \u0645\u0627\u0630\u0627", q) ||
           occursin("\u0645\u062a\u0649 \u064a\u062d\u062f\u062b", q) ||
           occursin("\u0645\u062a\u0649 \u062d\u062f\u062b", q) ||
           occursin("when", lq) ||
           occursin("before what", lq) ||
           occursin("after what", lq)
end

function _format_temporal_event_terms(terms::Vector{String})
    isempty(terms) && return ""
    any(_has_arabic_chars, terms) && return _format_purpose_left_terms(terms)
    return join(terms, " ")
end

function _format_temporal_time_terms(terms::Vector{String})
    isempty(terms) && return ""
    any(_has_arabic_chars, terms) && return join([_arabic_definite(t) for t in terms], " ")
    return join(terms, " ")
end

function _format_temporal_marker(marker::AbstractString)
    m = strip(String(marker))
    isempty(m) && return ""
    return m
end

const FRAME_PROMPT_NOISE_TERMS = Set([
    "\u0645\u062a\u0649", "\u0627\u064a\u0646", "\u0623\u064a\u0646", "\u0643\u064a\u0641", "\u0645\u0627", "\u0645\u0627\u0630\u0627",
    "\u064a\u062d\u062f\u062b", "\u062d\u062f\u062b", "\u0645\u0643\u0627\u0646", "\u062d\u0627\u0644", "\u0647\u064a\u0626\u0629",
    "\u0642\u0628\u0644", "\u0628\u0639\u062f", "\u0627\u0644\u0632\u0645\u0646", "\u0627\u0644\u0645\u0643\u0627\u0646",
    "when", "where", "how", "what", "happens", "happened", "state", "place",
    "before", "after", "time", "the", "of", "in",
])

const RELATION_FRAME_BAD_TERMS = Set([
    "\u0627\u0644\u0644\u0645", "\u0627\u0644\u0644\u0643\u0646", "\u064a\u0644", "\u062a\u064a", "\u0627\u0644\u0643\u0627\u0646",
    "\u0627\u0644\u0642\u0628\u0644\u064b\u0627", "\u0642\u0628\u0644\u064b\u0627", "\u0627\u0644\u0637\u0645\u0627\u0646\u0647\u0627",
    "\u0627\u0644\u0648\u0628\u0627\u0646", "\u0648\u0628\u0627\u0646", "\u0644\u0645", "\u0644\u0643\u0646",
])

const RELATION_FRAME_GENERIC_EVENT_TERMS = Set([
    "\u0643\u0627\u0646", "\u0643\u0627\u0646\u062a", "\u064a\u0643\u0648\u0646", "\u062a\u0643\u0648\u0646",
    "\u064a\u0639\u062f", "\u062a\u0639\u062f", "\u0645\u0641\u0647\u0648\u0645", "\u0634\u064a\u0621",
    "was", "were", "is", "are", "thing", "concept",
])

const TEMPORAL_STRONG_MARKERS = Set([
    "\u0642\u0628\u0644", "\u0628\u0639\u062f", "\u0623\u062b\u0646\u0627\u0621", "\u0645\u0646\u0630", "\u0637\u0627\u0644\u0645\u0627",
    "before", "after", "during", "since",
])

const TEMPORAL_TIME_WORDS = Set([
    "\u0641\u062c\u0631", "\u0627\u0644\u0641\u062c\u0631", "\u0635\u0628\u0627\u062d", "\u0627\u0644\u0635\u0628\u0627\u062d",
    "\u0645\u0633\u0627\u0621", "\u0627\u0644\u0645\u0633\u0627\u0621", "\u0644\u064a\u0644", "\u0627\u0644\u0644\u064a\u0644",
    "\u0646\u0647\u0627\u0631", "\u0627\u0644\u0646\u0647\u0627\u0631", "\u064a\u0648\u0645", "\u0627\u0644\u064a\u0648\u0645",
    "\u0633\u0627\u0639\u0629", "\u062f\u0642\u064a\u0642\u0629", "\u062b\u0627\u0646\u064a\u0629", "\u0634\u0647\u0631", "\u0633\u0646\u0629",
    "dawn", "morning", "evening", "night", "day", "hour", "minute", "second", "month", "year",
])

function _frame_prompt_subject_terms(prompt::AbstractString)
    terms = _ordered_tokens(prompt)
    filtered = [t for t in terms if !(t in FRAME_PROMPT_NOISE_TERMS)]
    isempty(filtered) && return terms
    return filtered
end

function _required_subject_hits(terms::Vector{String})
    length(terms) >= 2 && return 2
    return length(terms)
end

function _subject_terms_useful(terms::Vector{String})
    isempty(terms) && return false
    useful = [t for t in terms if !(_norm_word(t) in RELATION_FRAME_GENERIC_EVENT_TERMS)]
    return !isempty(useful)
end

function _subject_hit_count(subject_terms::Vector{String}, candidate_terms::Vector{String})
    keys = Set{String}()
    for term in subject_terms
        union!(keys, _term_keyset(term))
    end
    isempty(keys) && return 0
    return count(t -> _term_matches(t, keys), candidate_terms)
end

function _leading_subject_term_hit(subject_terms::Vector{String}, candidate_terms::Vector{String})
    length(subject_terms) >= 2 || return true
    keys = _term_keyset(first(subject_terms))
    isempty(keys) && return true
    return any(t -> _term_matches(t, keys), candidate_terms)
end

function _relation_frame_terms_clean_enough(terms::Vector{String}; event_side::Bool=false)
    isempty(terms) && return false
    length(terms) <= 12 || return false
    joined = " " * _clean_space(join(terms, " ")) * " "
    norm_joined = " " * _norm_word(joined) * " "
    for bad in RELATION_FRAME_BAD_TERMS
        occursin(bad, joined) && return false
        occursin(bad, norm_joined) && return false
    end
    useful = 0
    for term in terms
        n = _norm_word(term)
        isempty(n) && continue
        n in RELATION_FRAME_BAD_TERMS && return false
        n in STOPWORDS && continue
        length(n) < 2 && return false
        useful += 1
    end
    useful > 0 || return false
    if event_side
        all(t -> _norm_word(t) in RELATION_FRAME_GENERIC_EVENT_TERMS, terms) && return false
    end
    return true
end

function _temporal_terms_plausible(marker::AbstractString, terms::Vector{String})
    m = _norm_word(marker)
    m in TEMPORAL_STRONG_MARKERS && return true
    for term in terms
        n = _norm_word(term)
        n in TEMPORAL_TIME_WORDS && return true
        occursin(r"^[0-9]+$", n) && return true
    end
    return false
end

function _select_typed_relation_frame_attention(mem::IstinbatAttentionMemory,
                                                prompt::AbstractString,
                                                relation_type::AbstractString)
    query = _query_terms(prompt)
    subject_terms = _frame_prompt_subject_terms(prompt)
    _subject_terms_useful(subject_terms) || return nothing
    required_hits = _required_subject_hits(subject_terms)
    best = nothing
    best_score = -Inf
    for rec in values(mem.records)
        rec.relation_type == relation_type || continue
        rec.attention_weight < 0.30 && continue
        _relation_frame_terms_clean_enough(rec.before_terms; event_side=true) || continue
        _relation_frame_terms_clean_enough(rec.after_terms; event_side=false) || continue
        relation_type == "temporal" && !_temporal_terms_plausible(rec.marker, rec.after_terms) && continue
        if required_hits > 0
            _subject_hit_count(subject_terms, rec.before_terms) >= required_hits || continue
            relation_type == "spatial" && !_leading_subject_term_hit(subject_terms, rec.before_terms) && continue
        end
        overlap = _overlap_score(query, rec)
        overlap < 0.15 && continue
        score = 0.75 * overlap + 0.25 * rec.attention_weight
        if score > best_score
            best = rec
            best_score = score
        end
    end
    return best
end

function _arabic_event_nominal_phrase(event::AbstractString)
    e = _clean_space(String(event))
    isempty(e) && return ""
    parts = split(e)
    isempty(parts) && return e
    verb = first(parts)
    rest = length(parts) > 1 ? join(parts[2:end], " ") : ""
    nouns = Dict(
        "\u0633\u0627\u0641\u0631" => "\u0633\u0641\u0631",
        "\u062c\u0644\u0633" => "\u062c\u0644\u0648\u0633",
        "\u062f\u062e\u0644" => "\u062f\u062e\u0648\u0644",
        "\u062e\u0631\u062c" => "\u062e\u0631\u0648\u062c",
        "\u0632\u0627\u062f" => "\u0632\u064a\u0627\u062f\u0629",
        "\u0646\u062c\u062d" => "\u0646\u062c\u0627\u062d",
        "\u062f\u0641\u0639" => "\u062f\u0641\u0639",
        "\u0636\u0631\u0628" => "\u0636\u0631\u0628",
        "\u0643\u0633\u0631" => "\u0643\u0633\u0631",
        "\u0627\u0636\u0627\u0621" => "\u0625\u0636\u0627\u0621\u0629",
        "\u0623\u0636\u0627\u0621" => "\u0625\u0636\u0627\u0621\u0629",
    )
    base = get(nouns, verb, "")
    isempty(base) && return e
    return isempty(rest) ? base : "$(base) $(rest)"
end

function _format_temporal_answer(rec::IstinbatAttentionRecord)
    event = _format_temporal_event_terms(rec.before_terms)
    time = _format_temporal_time_terms(rec.after_terms)
    marker = _format_temporal_marker(rec.marker)
    isempty(time) && return ""
    length(split(event)) > 12 && return ""
    length(split(time)) > 12 && return ""
    if _has_arabic_chars(event) || _has_arabic_chars(time) || _has_arabic_chars(marker)
        if !isempty(event) && !isempty(marker)
            nominal = _arabic_event_nominal_phrase(event)
            return "\u0643\u0627\u0646 $(nominal) $(marker) $(time)."
        elseif !isempty(marker)
            return "\u0627\u0644\u0632\u0645\u0646: $(marker) $(time)."
        end
        return "\u0627\u0644\u0632\u0645\u0646: $(time)."
    end
    if !isempty(event) && !isempty(marker)
        return "$(event) $(marker) $(time)."
    elseif !isempty(marker)
        return "Time: $(marker) $(time)."
    end
    return "Time: $(time)."
end

"""
    temporal_answer(mem::IstinbatAttentionMemory, prompt::AbstractString) -> String

Generate an independent answer for temporal RelationFrame records. This is a
diagnostic RelationFrame layer only; it is not inserted into `generate!`.
"""
function temporal_answer(mem::IstinbatAttentionMemory, prompt::AbstractString)
    s = _clean_space(String(prompt))
    isempty(s) && return ""
    _is_yesno_question(s) && return ""
    _is_temporal_question(s) || return ""

    rec = _select_typed_relation_frame_attention(mem, s, "temporal")
    rec === nothing && return ""

    return _format_temporal_answer(rec)
end

function _is_spatial_question(s::AbstractString)
    q = _clean_space(String(s))
    lq = lowercase(q)
    return occursin("\u0623\u064a\u0646", q) ||
           occursin("\u0627\u064a\u0646", q) ||
           occursin("\u0641\u064a \u0623\u064a \u0645\u0643\u0627\u0646", q) ||
           occursin("\u0641\u064a \u0627\u064a \u0645\u0643\u0627\u0646", q) ||
           occursin("\u0645\u0627 \u0645\u0643\u0627\u0646", q) ||
           occursin("where", lq)
end

function _format_spatial_event_terms(terms::Vector{String})
    isempty(terms) && return ""
    any(_has_arabic_chars, terms) && return _format_purpose_left_terms(terms)
    return join(terms, " ")
end

function _format_spatial_place_terms(terms::Vector{String})
    isempty(terms) && return ""
    if any(_has_arabic_chars, terms)
        return join([replace(_arabic_definite(t), r"\u0647$" => "\u0629") for t in terms], " ")
    end
    return join(terms, " ")
end

function _format_spatial_marker(marker::AbstractString)
    m = strip(String(marker))
    n = _norm_word(m)
    if n == "\u062d\u064a\u062b"
        return "\u0641\u064a"
    end
    return m
end

function _format_spatial_answer(rec::IstinbatAttentionRecord)
    event = _format_spatial_event_terms(rec.before_terms)
    place = _format_spatial_place_terms(rec.after_terms)
    marker = _format_spatial_marker(rec.marker)
    isempty(place) && return ""
    length(split(event)) > 12 && return ""
    length(split(place)) > 12 && return ""
    if _has_arabic_chars(event) || _has_arabic_chars(place) || _has_arabic_chars(marker)
        if !isempty(event) && !isempty(marker)
            nominal = _arabic_event_nominal_phrase(event)
            return "\u0643\u0627\u0646 \u0645\u0643\u0627\u0646 $(nominal) $(marker) $(place)."
        elseif !isempty(event)
            return "\u0645\u0643\u0627\u0646 $(event) \u0647\u0648 $(place)."
        end
        return "\u0627\u0644\u0645\u0643\u0627\u0646: $(place)."
    end
    if !isempty(event) && !isempty(marker)
        lowercase(marker) == "where" && return "The place of $(event) is $(place)."
        return "$(event) $(marker) $(place)."
    elseif !isempty(event)
        return "The place of $(event) is $(place)."
    end
    return "Place: $(place)."
end

"""
    spatial_answer(mem::IstinbatAttentionMemory, prompt::AbstractString) -> String

Generate an independent answer for spatial RelationFrame records. This is a
diagnostic RelationFrame layer only; it is not inserted into `generate!`.
"""
function spatial_answer(mem::IstinbatAttentionMemory, prompt::AbstractString)
    s = _clean_space(String(prompt))
    isempty(s) && return ""
    _is_yesno_question(s) && return ""
    _is_spatial_question(s) || return ""

    rec = _select_typed_relation_frame_attention(mem, s, "spatial")
    rec === nothing && return ""

    return _format_spatial_answer(rec)
end

function _is_state_question(s::AbstractString)
    q = _clean_space(String(s))
    lq = lowercase(q)
    return occursin("\u0643\u064a\u0641", q) ||
           occursin("\u0628\u0623\u064a \u062d\u0627\u0644", q) ||
           occursin("\u0628\u0627\u064a \u062d\u0627\u0644", q) ||
           occursin("\u0645\u0627 \u062d\u0627\u0644", q) ||
           occursin("\u0645\u0627 \u0647\u064a\u0626\u0629", q) ||
           occursin("how", lq) ||
           occursin("in what state", lq)
end

function _format_state_event_terms(terms::Vector{String})
    isempty(terms) && return ""
    any(_has_arabic_chars, terms) && return _format_purpose_left_terms(terms)
    return join(terms, " ")
end

function _format_state_terms(terms::Vector{String})
    isempty(terms) && return ""
    return join(terms, " ")
end

function _format_state_answer(rec::IstinbatAttentionRecord)
    event = _format_state_event_terms(rec.before_terms)
    state = _format_state_terms(rec.after_terms)
    marker = strip(String(rec.marker))
    isempty(state) && return ""
    length(split(event)) > 12 && return ""
    length(split(state)) > 12 && return ""
    if _has_arabic_chars(event) || _has_arabic_chars(state) || _has_arabic_chars(marker)
        if !isempty(event) && !isempty(marker)
            return "$(event)\u060c \u0648\u0643\u0627\u0646 \u0639\u0644\u0649 \u062d\u0627\u0644 $(state)."
        elseif !isempty(event)
            return "\u062d\u0627\u0644 $(event) \u0647\u0648 $(state)."
        end
        return "\u0627\u0644\u062d\u0627\u0644: $(state)."
    end
    if !isempty(event) && !isempty(marker)
        return "$(event) $(marker) $(state)."
    elseif !isempty(event)
        return "The state of $(event) is $(state)."
    end
    return "State: $(state)."
end

"""
    state_answer(mem::IstinbatAttentionMemory, prompt::AbstractString) -> String

Generate an independent answer for state RelationFrame records. This is a
diagnostic RelationFrame layer only; it is not inserted into `generate!`.
"""
function state_answer(mem::IstinbatAttentionMemory, prompt::AbstractString)
    s = _clean_space(String(prompt))
    isempty(s) && return ""
    _is_yesno_question(s) && return ""
    _is_state_question(s) || return ""

    rec = _select_typed_relation_frame_attention(mem, s, "state")
    rec === nothing && return ""

    return _format_state_answer(rec)
end

"""
    PurposeComparisonRecord

سجل مقارنة بين جواب purpose_answer وجواب generate! لسؤال غائي.
"""
struct PurposeComparisonRecord
    prompt::String
    purpose_answer::String
    generate_answer::String
    memory_has_purpose::Bool
    purpose_confidence::Float64
    overlap_score::Float64
    has_marker::Bool
    relation_type::String
end

"""
    ConditionalComparisonRecord

Independent comparison record between conditional_answer and a caller-supplied
generation function. This is diagnostic only and does not enter generate!.
"""
struct ConditionalComparisonRecord
    prompt::String
    conditional_answer::String
    generate_answer::String
    memory_has_conditional::Bool
    conditional_confidence::Float64
    overlap_score::Float64
    has_marker::Bool
    relation_type::String
end

"""
    TemporalComparisonRecord

Independent comparison record between temporal_answer and a caller-supplied
generation function. This is diagnostic only and does not enter generate!.
"""
struct TemporalComparisonRecord
    prompt::String
    temporal_answer::String
    generate_answer::String
    memory_has_temporal::Bool
    temporal_confidence::Float64
    overlap_score::Float64
    has_marker::Bool
    relation_type::String
end

"""
    SpatialComparisonRecord

Independent comparison record between spatial_answer and a caller-supplied
generation function. This is diagnostic only and does not enter generate!.
"""
struct SpatialComparisonRecord
    prompt::String
    spatial_answer::String
    generate_answer::String
    memory_has_spatial::Bool
    spatial_confidence::Float64
    overlap_score::Float64
    has_marker::Bool
    relation_type::String
end

"""
    StateComparisonRecord

Independent comparison record between state_answer and a caller-supplied
generation function. This is diagnostic only and does not enter generate!.
"""
struct StateComparisonRecord
    prompt::String
    state_answer::String
    generate_answer::String
    memory_has_state::Bool
    state_confidence::Float64
    overlap_score::Float64
    has_marker::Bool
    relation_type::String
end

"""
    ScenePurposeComparisonRecord

سجل مقارنة مستقل بين التخيل الدلالي الحسي و RelationFrame الغائي.
لا يدخل في `generate!` ولا يغير بوابات الإجابة.
"""
struct ScenePurposeComparisonRecord
    prompt::String
    scene_answer::String
    scene_summary::String
    purpose_answer::String
    scene_has_event::Bool
    memory_has_purpose::Bool
    agreement::String
    scene_confidence::Float64
    purpose_confidence::Float64
    scene_relation::String
    purpose_relation::String
end

"""
    compare_purpose_strategies(mem::IstinbatAttentionMemory,
                               generate_func::Function,
                               prompt::AbstractString) -> PurposeComparisonRecord

مقارنة خارجية بين purpose_answer و generate! لسؤال غائي.
تأخذ `generate_func` دالة تقبل النص وتُعيد الجواب (مثلاً `p -> generate!(gen, p)`).
لا تؤثر على مسارات التوليد الحالية ولا على `_learned_attention`.
"""
function compare_purpose_strategies(mem::IstinbatAttentionMemory,
                                     generate_func::Function,
                                     prompt::AbstractString)
    s = _clean_space(String(prompt))
    purpose_ans = purpose_answer(mem, s)

    generate_ans = try
        String(generate_func(s))
    catch
        ""
    end

    rec = select_relation_frame_attention(mem, s)
    purpose_found = rec !== nothing && rec.relation_type == "purpose" && !isempty(strip(purpose_ans))
    confidence = purpose_found ? rec.attention_weight : 0.0
    query = _query_terms(s)
    overlap = purpose_found ? _overlap_score(query, rec) : 0.0
    marker_found = purpose_found && !isempty(rec.marker)
    rtype = purpose_found ? rec.relation_type : "none"

    return PurposeComparisonRecord(s, purpose_ans, generate_ans,
                                   purpose_found, confidence,
                                   overlap, marker_found, rtype)
end

"""
    compare_conditional_strategies(mem::IstinbatAttentionMemory,
                                   generate_func::Function,
                                   prompt::AbstractString) -> ConditionalComparisonRecord

Diagnostic comparison between conditional_answer and a caller-supplied
generation function. This keeps conditional reasoning measurable without
changing the generation strategy list.
"""
function compare_conditional_strategies(mem::IstinbatAttentionMemory,
                                        generate_func::Function,
                                        prompt::AbstractString)
    s = _clean_space(String(prompt))
    conditional_ans = conditional_answer(mem, s)

    generate_ans = try
        String(generate_func(s))
    catch
        ""
    end

    rec = select_relation_frame_attention(mem, s)
    conditional_found = rec !== nothing &&
                        rec.relation_type == "conditional" &&
                        !isempty(strip(conditional_ans))
    confidence = conditional_found ? rec.attention_weight : 0.0
    query = _query_terms(s)
    overlap = conditional_found ? _overlap_score(query, rec) : 0.0
    marker_found = conditional_found && !isempty(rec.marker)
    rtype = conditional_found ? rec.relation_type : "none"

    return ConditionalComparisonRecord(s, conditional_ans, generate_ans,
                                       conditional_found, confidence,
                                       overlap, marker_found, rtype)
end

"""
    compare_temporal_strategies(mem::IstinbatAttentionMemory,
                                generate_func::Function,
                                prompt::AbstractString) -> TemporalComparisonRecord

Diagnostic comparison between temporal_answer and a caller-supplied generation
function. This keeps temporal reasoning measurable without changing generation.
"""
function compare_temporal_strategies(mem::IstinbatAttentionMemory,
                                     generate_func::Function,
                                     prompt::AbstractString)
    s = _clean_space(String(prompt))
    temporal_ans = temporal_answer(mem, s)

    generate_ans = try
        String(generate_func(s))
    catch
        ""
    end

    rec = select_relation_frame_attention(mem, s)
    temporal_found = rec !== nothing &&
                     rec.relation_type == "temporal" &&
                     !isempty(strip(temporal_ans))
    confidence = temporal_found ? rec.attention_weight : 0.0
    query = _query_terms(s)
    overlap = temporal_found ? _overlap_score(query, rec) : 0.0
    marker_found = temporal_found && !isempty(rec.marker)
    rtype = temporal_found ? rec.relation_type : "none"

    return TemporalComparisonRecord(s, temporal_ans, generate_ans,
                                    temporal_found, confidence,
                                    overlap, marker_found, rtype)
end

"""
    compare_spatial_strategies(mem::IstinbatAttentionMemory,
                               generate_func::Function,
                               prompt::AbstractString) -> SpatialComparisonRecord

Diagnostic comparison between spatial_answer and a caller-supplied generation
function. This keeps spatial reasoning measurable without changing generation.
"""
function compare_spatial_strategies(mem::IstinbatAttentionMemory,
                                    generate_func::Function,
                                    prompt::AbstractString)
    s = _clean_space(String(prompt))
    spatial_ans = spatial_answer(mem, s)

    generate_ans = try
        String(generate_func(s))
    catch
        ""
    end

    rec = select_relation_frame_attention(mem, s)
    spatial_found = rec !== nothing &&
                    rec.relation_type == "spatial" &&
                    !isempty(strip(spatial_ans))
    confidence = spatial_found ? rec.attention_weight : 0.0
    query = _query_terms(s)
    overlap = spatial_found ? _overlap_score(query, rec) : 0.0
    marker_found = spatial_found && !isempty(rec.marker)
    rtype = spatial_found ? rec.relation_type : "none"

    return SpatialComparisonRecord(s, spatial_ans, generate_ans,
                                   spatial_found, confidence,
                                   overlap, marker_found, rtype)
end

"""
    compare_state_strategies(mem::IstinbatAttentionMemory,
                             generate_func::Function,
                             prompt::AbstractString) -> StateComparisonRecord

Diagnostic comparison between state_answer and a caller-supplied generation
function. This keeps state reasoning measurable without changing generation.
"""
function compare_state_strategies(mem::IstinbatAttentionMemory,
                                  generate_func::Function,
                                  prompt::AbstractString)
    s = _clean_space(String(prompt))
    state_ans = state_answer(mem, s)

    generate_ans = try
        String(generate_func(s))
    catch
        ""
    end

    rec = select_relation_frame_attention(mem, s)
    state_found = rec !== nothing &&
                  rec.relation_type == "state" &&
                  !isempty(strip(state_ans))
    confidence = state_found ? rec.attention_weight : 0.0
    query = _query_terms(s)
    overlap = state_found ? _overlap_score(query, rec) : 0.0
    marker_found = state_found && !isempty(rec.marker)
    rtype = state_found ? rec.relation_type : "none"

    return StateComparisonRecord(s, state_ans, generate_ans,
                                 state_found, confidence,
                                 overlap, marker_found, rtype)
end

function _scene_summary_from_comparison(cmp)
    cmp.scene === nothing && return ""
    scene = cmp.scene
    effects = _clean_scene_summary_effects(isempty(cmp.scene_effect_terms) ? cmp.guidance_terms : cmp.scene_effect_terms)
    effect_text = isempty(effects) ? "" : join(effects[1:min(length(effects), 3)], ", ")
    parts = String[]
    isempty(scene.actor) || push!(parts, "actor=$(scene.actor)")
    isempty(scene.action) || push!(parts, "action=$(scene.action)")
    isempty(scene.patient) || push!(parts, "patient=$(scene.patient)")
    isempty(scene.instrument) || push!(parts, "instrument=$(scene.instrument)")
    isempty(scene.place) || push!(parts, "place=$(scene.place)")
    isempty(scene.time_marker) || push!(parts, "time=$(scene.time_marker)")
    isempty(scene.state_before) || push!(parts, "before=$(scene.state_before)")
    isempty(scene.state_after) || push!(parts, "after=$(scene.state_after)")
    isempty(effect_text) || push!(parts, "effects=$(effect_text)")
    return join(parts, " | ")
end

const SCENE_SUMMARY_EFFECT_NOISE = Set([
    "كل", "ولا", "لا", "ليس", "علي", "على", "جواب", "سؤال", "بين",
    "الانسان", "إنسان", "الإنسان",
    "all", "no", "not", "answer", "question", "between",
])

function _clean_scene_summary_effects(items)
    out = String[]
    seen = Set{String}()
    for item in items
        clean = _norm_word(strip(String(item)))
        isempty(clean) && continue
        clean in STOPWORDS && continue
        clean in SCENE_SUMMARY_EFFECT_NOISE && continue
        clean in seen && continue
        push!(seen, clean)
        push!(out, strip(String(item)))
        length(out) >= 3 && break
    end
    return out
end

function _scene_probe_terms(scene)
    scene === nothing && return Set{String}()
    terms = Set{String}()
    for t in (scene.action, scene.patient, scene.actor, scene.instrument)
        raw = strip(String(t))
        isempty(raw) && continue
        push!(terms, _norm_word(raw))
        p = _projection_word(raw)
        isempty(p) || push!(terms, p)
    end
    return Set(filter(!isempty, terms))
end

function _record_probe_terms(rec::IstinbatAttentionRecord)
    terms = Set{String}()
    for t in vcat(rec.before_terms, rec.focus_terms)
        raw = strip(String(t))
        isempty(raw) && continue
        push!(terms, _norm_word(raw))
        p = _projection_word(raw)
        isempty(p) || push!(terms, p)
    end
    return Set(filter(!isempty, terms))
end

function _scene_matches_prompt(scene, prompt::AbstractString)
    scene_terms = _scene_probe_terms(scene)
    isempty(scene_terms) && return false
    return !isempty(intersect(scene_terms, _query_terms(prompt)))
end

function _scene_matches_purpose(scene, rec::Union{IstinbatAttentionRecord,Nothing})
    rec === nothing && return false
    scene_terms = _scene_probe_terms(scene)
    rec_terms = _record_probe_terms(rec)
    isempty(scene_terms) && return false
    isempty(rec_terms) && return false
    return !isempty(intersect(scene_terms, rec_terms))
end

"""
    compare_scene_purpose_strategies(scene_mem, calculus, istinbat_mem, prompt)

يقارن بين طبقة الحس الدلالي (`SemanticScene`) وطبقة الاستنباط الغائي
(`RelationFrame` من نوع purpose) من غير إدخال النتيجة في التوليد.
"""
function compare_scene_purpose_strategies(scene_mem::SemanticSceneMemory,
                                          calculus::SemanticCalculusMemory,
                                          istinbat_mem::IstinbatAttentionMemory,
                                          prompt::AbstractString)
    s = _clean_space(String(prompt))
    scene_ans = semantic_scene_answer(scene_mem, calculus, s)
    scene_cmp = compare_semantic_scene_with_calculus(scene_mem, calculus, s)
    scene_summary = _scene_summary_from_comparison(scene_cmp)

    purpose_ans = purpose_answer(istinbat_mem, s)
    rec = _select_purpose_attention(istinbat_mem, s)
    purpose_found = rec !== nothing && !isempty(strip(purpose_ans))
    purpose_confidence = purpose_found ? rec.attention_weight : 0.0
    purpose_relation = purpose_found ? rec.relation_type : "none"
    scene_found = scene_cmp.scene !== nothing && _scene_matches_prompt(scene_cmp.scene, s)
    cooperative_ready = scene_found && purpose_found && _scene_matches_purpose(scene_cmp.scene, rec)
    scene_relation = isempty(scene_cmp.guidance_relation) ? scene_cmp.agreement : scene_cmp.guidance_relation

    agreement = if cooperative_ready
        "cooperative"
    elseif scene_found
        "scene_only"
    elseif purpose_found
        "purpose_only"
    else
        "none"
    end

    return ScenePurposeComparisonRecord(
        s,
        scene_ans,
        scene_summary,
        purpose_ans,
        scene_found,
        purpose_found,
        agreement,
        scene_cmp.scene_confidence,
        purpose_confidence,
        scene_relation,
        purpose_relation,
    )
end

function _strip_terminal_sentence_punc(s::AbstractString)
    return replace(strip(String(s)), r"[\.\!\?\u061F\u06D4]+$" => "")
end

function _scene_summary_fields(summary::AbstractString)
    fields = Dict{String,String}()
    for part in split(String(summary), " | ")
        idx = findfirst(==('='), part)
        idx === nothing && continue
        key = strip(part[1:prevind(part, idx)])
        val = strip(part[nextind(part, idx):end])
        isempty(key) && continue
        isempty(val) && continue
        fields[key] = val
    end
    return fields
end

function _scene_summary_sentence(summary::AbstractString)
    raw = strip(String(summary))
    isempty(raw) && return ""
    occursin("=", raw) || return raw
    fields = _scene_summary_fields(raw)
    action = get(fields, "action", "")
    patient = get(fields, "patient", "")
    before = get(fields, "before", "")
    after = get(fields, "after", "")
    effects = get(fields, "effects", "")
    isempty(action) && isempty(patient) && return raw
    arabic = _has_arabic_chars(raw)
    if arabic
        target = isempty(patient) ? "\u0627\u0644\u0645\u062a\u0623\u062b\u0631" : patient
        base = "\u0639\u0646\u062f $(action) $(target)"
        if !isempty(before) && !isempty(after)
            base *= " \u064a\u0646\u062a\u0642\u0644 \u0627\u0644\u0645\u062a\u0623\u062b\u0631 \u0645\u0646 $(before) \u0625\u0644\u0649 $(after)"
        end
        isempty(effects) || (base *= "\u060c \u0648\u062a\u0638\u0647\u0631 \u0622\u062b\u0627\u0631 \u0645\u062b\u0644 $(effects)")
        return base
    end
    target = isempty(patient) ? "the affected thing" : patient
    base = "When $(action) affects $(target)"
    if !isempty(before) && !isempty(after)
        base *= ", the affected thing shifts from $(before) to $(after)"
    end
    isempty(effects) || (base *= ", with effects such as $(effects)")
    return base
end

function _arabic_list_separators(s::AbstractString)
    return replace(String(s), r"\s*,\s*" => "\u060c ")
end

function _purpose_fragment_for_composite(purpose::AbstractString)
    p = _strip_terminal_sentence_punc(purpose)
    m = match(r"^الغاية\s+من\s+.+?\s+هي\s+(.+)$", p)
    m !== nothing && return strip(m.captures[1])
    m = match(r"^الغاية\s+هي\s+(.+)$", p)
    m !== nothing && return strip(m.captures[1])
    return p
end

function _natural_arabic_scene_purpose_sentence(summary::AbstractString,
                                                purpose_clean::AbstractString)
    fields = _scene_summary_fields(summary)
    action = get(fields, "action", "")
    actor = get(fields, "actor", "")
    patient = get(fields, "patient", "")
    before = get(fields, "before", "")
    after = get(fields, "after", "")
    effects = get(fields, "effects", "")

    scene_text = join([action, actor, patient, before, after, effects], " ")
    _has_arabic_chars(scene_text) || return ""
    isempty(action) && isempty(patient) && return ""

    target = isempty(patient) ? "\u0627\u0644\u0645\u062a\u0623\u062b\u0631" : patient
    event = strip(isempty(actor) ? "$(action) $(target)" : "$(action) $(actor) $(target)")
    clauses = String[event]

    if !isempty(before) && !isempty(after)
        push!(clauses, "\u0641\u0627\u0646\u062a\u0642\u0644 $(target) \u0645\u0646 $(before) \u0625\u0644\u0649 $(after)")
    end
    if !isempty(effects)
        push!(clauses, "\u0648\u0638\u0647\u0631\u062a \u0622\u062b\u0627\u0631 \u0645\u062b\u0644 $(_arabic_list_separators(effects))")
    end

    purpose = _arabic_list_separators(purpose_clean)
    isempty(strip(purpose)) && return ""
    return join(clauses, "\u061b ") * "\u061b \u0648\u0643\u0627\u0646\u062a \u0627\u0644\u063a\u0627\u064a\u0629 $(purpose)."
end

function _format_scene_purpose_answer(cmp::ScenePurposeComparisonRecord)
    cmp.agreement == "cooperative" || return ""
    scene_part = !isempty(strip(cmp.scene_answer)) ? cmp.scene_answer : _scene_summary_sentence(cmp.scene_summary)
    purpose_part = cmp.purpose_answer
    isempty(strip(scene_part)) && return ""
    isempty(strip(purpose_part)) && return ""
    scene_clean = _strip_terminal_sentence_punc(scene_part)
    purpose_clean = _purpose_fragment_for_composite(purpose_part)
    natural = _natural_arabic_scene_purpose_sentence(cmp.scene_summary, purpose_clean)
    isempty(natural) || return natural
    arabic = _has_arabic_chars(scene_clean) || _has_arabic_chars(purpose_clean)
    if arabic
        _has_arabic_chars(scene_clean) && (scene_clean = _arabic_list_separators(scene_clean))
        _has_arabic_chars(purpose_clean) && (purpose_clean = _arabic_list_separators(purpose_clean))
        if _has_arabic_chars(scene_clean)
            return "$(scene_clean)\u061b \u0648\u0627\u0644\u063a\u0627\u064a\u0629 $(purpose_clean)."
        end
        return "$(scene_clean). \u0648\u0627\u0644\u063a\u0627\u064a\u0629 $(purpose_clean)."
    end
    return "$(scene_clean). In terms of purpose: $(purpose_clean)."
end

"""
    scene_purpose_answer(scene_mem, calculus, istinbat_mem, prompt) -> String

Build a diagnostic composite answer from the sensory semantic scene and the rational
purpose frame. It returns a non-empty answer only when both layers cooperate.
It does not enter `generate!` and does not alter strategy order.
"""
function scene_purpose_answer(scene_mem::SemanticSceneMemory,
                              calculus::SemanticCalculusMemory,
                              istinbat_mem::IstinbatAttentionMemory,
                              prompt::AbstractString)
    cmp = compare_scene_purpose_strategies(scene_mem, calculus, istinbat_mem, prompt)
    return _format_scene_purpose_answer(cmp)
end

end # module
