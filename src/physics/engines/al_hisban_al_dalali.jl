"""
al_hisban_al_dalali - Clifford semantic calculus memory.

This layer stores semantic transitions, not surface sentences:
source text -> target text, encoded as compact Clifford/phase signatures.
It is a guidance memory for meaning movement.
"""
module AlHisbanAlDalali

using JSON, LinearAlgebra

using ..WordPhysics: compute_extended_phase_vector
using ..CliffordMath: Multivector22, from_vector, word_to_multivector,
                      clifford_similarity

export SemanticCalculusRecord, SemanticCalculusMemory,
       sentence_semantic_signature, semantic_transform_signature,
       learn_semantic_calculus_from_pair!, learn_semantic_calculus_from_text!,
       train_semantic_calculus_from_texts!,
       select_semantic_transform, semantic_relation_movement,
       semantic_guidance, semantic_answer_plan, semantic_guidance_terms,
       save_semantic_calculus, load_semantic_calculus,
       semantic_calculus_to_dict, has_semantic_calculus

const AL_HISBAN_VERSION = 1
const SIGNATURE_DIM = 22
const SENTENCE_SPLIT_RE = r"[\n.!?\u061F;]+"

function _finite_float(x, default::Float64=0.0)
    value = try
        Float64(x)
    catch
        default
    end
    return isfinite(value) ? value : default
end

function _finite_vector(v; dim::Int=SIGNATURE_DIM)
    out = zeros(Float64, dim)
    v isa AbstractVector || return out
    n = min(length(v), dim)
    for i in 1:n
        out[i] = _finite_float(v[i])
    end
    return out
end

function _safe_norm(v)
    value = try
        norm(v)
    catch
        0.0
    end
    return isfinite(value) ? Float64(value) : 0.0
end

function _normalize_finite!(v::Vector{Float64})
    n = _safe_norm(v)
    n > 1e-10 && (v ./= n)
    for i in eachindex(v)
        isfinite(v[i]) || (v[i] = 0.0)
    end
    return v
end

mutable struct SemanticCalculusRecord
    relation::String
    source_signature::Vector{Float64}
    target_signature::Vector{Float64}
    transform_signature::Vector{Float64}
    scalar_shift::Float64
    bivector_shift::Float64
    count::Int
    examples::Vector{Dict{String,String}}
    target_terms::Dict{String,Int}
end

mutable struct SemanticCalculusMemory
    records::Dict{String,SemanticCalculusRecord}
    max_examples::Int
    max_terms::Int
end

function SemanticCalculusMemory(; max_examples::Int=5, max_terms::Int=120)
    return SemanticCalculusMemory(Dict{String,SemanticCalculusRecord}(), max_examples, max_terms)
end

function _words(text::AbstractString)
    return String[m.match for m in eachmatch(r"[\p{L}\p{N}_]+", String(text))]
end

function _clean_word(word::AbstractString)
    return strip(String(word))
end

function _is_useful_term(word::AbstractString)
    w = _clean_word(word)
    length(w) >= 2 || return false
    all(isdigit, w) && return false
    return true
end

function _safe_phase22(word::AbstractString)
    v = Float64.(compute_extended_phase_vector(String(word)))
    return _finite_vector(v; dim=SIGNATURE_DIM)
end

function _word_mv(word::AbstractString)
    mv = try
        word_to_multivector(String(word))
    catch
        from_vector(_safe_phase22(word))
    end
    if _safe_norm(mv) < 1e-10
        mv = from_vector(_safe_phase22(word))
    end
    return mv
end

function _mv_signature(mv::Multivector22)
    vec = _normalize_finite!(_finite_vector(mv.v; dim=SIGNATURE_DIM))
    return Dict{String,Any}(
        "vector" => vec,
        "scalar" => _finite_float(mv.s),
        "bivector_norm" => _safe_norm(mv.b),
    )
end

function sentence_semantic_signature(text::AbstractString)
    terms = [_clean_word(w) for w in _words(text) if _is_useful_term(w)]
    if isempty(terms)
        return Dict{String,Any}(
            "vector" => zeros(Float64, SIGNATURE_DIM),
            "scalar" => 0.0,
            "bivector_norm" => 0.0,
            "terms" => String[],
        )
    end

    scalar = 0.0
    vector = zeros(Float64, SIGNATURE_DIM)
    bivector_norm = 0.0
    ordered = nothing
    for (i, word) in enumerate(terms)
        mv = _word_mv(word)
        ordered = try
            ordered === nothing ? mv : ordered * mv
        catch
            nothing
        end
        weight = 1.0 / sqrt(i)
        scalar += weight * _finite_float(mv.s)
        vector .+= weight .* _finite_vector(mv.v; dim=SIGNATURE_DIM)
        bivector_norm += weight * _safe_norm(mv.b)
    end

    ordered !== nothing && begin
        scalar = 0.5 * scalar + 0.5 * _finite_float(ordered.s)
        vector .+= 0.25 .* _finite_vector(ordered.v; dim=SIGNATURE_DIM)
        bivector_norm = 0.5 * bivector_norm + 0.5 * _safe_norm(ordered.b)
    end

    _normalize_finite!(vector)
    return Dict{String,Any}(
        "vector" => vector,
        "scalar" => _finite_float(scalar / max(length(terms), 1)),
        "bivector_norm" => _finite_float(bivector_norm / max(length(terms), 1)),
        "terms" => terms,
    )
end

function semantic_transform_signature(source::AbstractString, target::AbstractString)
    src = sentence_semantic_signature(source)
    tgt = sentence_semantic_signature(target)
    delta = _finite_vector(tgt["vector"]; dim=SIGNATURE_DIM) .- _finite_vector(src["vector"]; dim=SIGNATURE_DIM)
    _normalize_finite!(delta)
    return Dict{String,Any}(
        "source" => src,
        "target" => tgt,
        "transform" => delta,
        "scalar_shift" => _finite_float(tgt["scalar"]) - _finite_float(src["scalar"]),
        "bivector_shift" => _finite_float(tgt["bivector_norm"]) - _finite_float(src["bivector_norm"]),
    )
end

function _detect_relation(source::AbstractString, target::AbstractString="")
    s = lowercase(strip(String(source)))
    if occursin(r"^(هل|أهل|اهل|is|are|do|does|did|can|could|will|would)\b", s)
        return "yes_no_answer"
    elseif occursin(r"^(كيف|how)\b", s)
        return "method_answer"
    elseif occursin(r"^(لماذا|لما|why)\b", s)
        return "reason_answer"
    elseif occursin(r"^(ما|ماذا|what)\b", s)
        return "definition_answer"
    elseif occursin(r"^(متى|when)\b", s)
        return "time_answer"
    elseif occursin(r"^(أين|اين|where)\b", s)
        return "place_answer"
    elseif occursin(r"(لأن|لان|because|therefore|لذلك|ومن ثم)", lowercase(String(target)))
        return "cause_result"
    end
    return "semantic_continuation"
end

function semantic_relation_movement(relation::AbstractString)
    rel = String(relation)
    rel == "yes_no_answer" && return "judgment"
    rel == "method_answer" && return "method"
    rel == "reason_answer" && return "reason"
    rel == "definition_answer" && return "definition"
    rel == "time_answer" && return "time"
    rel == "place_answer" && return "place"
    rel == "cause_result" && return "cause_result"
    rel == "semantic_continuation" && return "continuation"
    return "semantic_shift"
end

function _answer_frame(relation::AbstractString)
    rel = String(relation)
    rel == "yes_no_answer" && return "judgment_then_support"
    rel == "method_answer" && return "mechanism_or_steps"
    rel == "reason_answer" && return "cause_then_explanation"
    rel == "definition_answer" && return "definition_then_feature"
    rel == "time_answer" && return "time_context"
    rel == "place_answer" && return "place_context"
    rel == "cause_result" && return "cause_to_result"
    rel == "semantic_continuation" && return "continue_same_field"
    return "shift_toward_target"
end

function _plan_roles_for_movement(movement::AbstractString)
    mv = String(movement)
    mv == "method" && return ["mechanism", "means", "result"]
    mv == "definition" && return ["definition", "feature", "implication"]
    mv == "reason" && return ["cause", "explanation", "consequence"]
    mv == "cause_result" && return ["cause", "transition", "result"]
    mv == "judgment" && return ["judgment", "support", "boundary"]
    mv == "time" && return ["time_anchor", "event", "effect"]
    mv == "place" && return ["place_anchor", "entity", "relation"]
    mv == "continuation" && return ["same_field", "extension", "closure"]
    return ["source", "shift", "target"]
end

function _terms_for_plan_step(terms::Vector{String}, idx::Int)
    isempty(terms) && return String[]
    n = length(terms)
    start = min(n, max(1, idx))
    out = String[]
    push!(out, terms[start])
    idx + 3 <= n && push!(out, terms[idx + 3])
    return unique(out)
end

function _record_key(relation::String)
    return relation
end

function _merge_vector(old::Vector{Float64}, new::Vector{Float64}, old_count::Int)
    old_v = _finite_vector(old; dim=SIGNATURE_DIM)
    new_v = _finite_vector(new; dim=SIGNATURE_DIM)
    merged = (old_v .* old_count .+ new_v) ./ (old_count + 1)
    return _normalize_finite!(merged)
end

function _push_term!(bucket::Dict{String,Int}, term::String, max_terms::Int)
    _is_useful_term(term) || return
    bucket[term] = get(bucket, term, 0) + 1
    if length(bucket) > max_terms
        ordered = sort(collect(bucket); by=x -> (x[2], x[1]))
        delete!(bucket, ordered[1][1])
    end
end

function _remember_example!(rec::SemanticCalculusRecord, source::AbstractString, target::AbstractString,
                            max_examples::Int)
    src = String(source)
    tgt = String(target)
    item = Dict("source" => src, "target" => tgt)
    if !any(e -> get(e, "source", "") == src && get(e, "target", "") == tgt, rec.examples)
        push!(rec.examples, item)
        while length(rec.examples) > max_examples
            popfirst!(rec.examples)
        end
    end
end

function learn_semantic_calculus_from_pair!(mem::SemanticCalculusMemory,
                                            source::AbstractString,
                                            target::AbstractString;
                                            relation::String="")
    src = strip(String(source))
    tgt = strip(String(target))
    (length(src) >= 2 && length(tgt) >= 2) || return false

    rel = isempty(strip(relation)) ? _detect_relation(src, tgt) : String(relation)
    sig = semantic_transform_signature(src, tgt)
    key = _record_key(rel)
    source_vec = _finite_vector(sig["source"]["vector"]; dim=SIGNATURE_DIM)
    target_vec = _finite_vector(sig["target"]["vector"]; dim=SIGNATURE_DIM)
    transform_vec = _finite_vector(sig["transform"]; dim=SIGNATURE_DIM)
    scalar_shift = _finite_float(sig["scalar_shift"])
    bivector_shift = _finite_float(sig["bivector_shift"])

    if haskey(mem.records, key)
        rec = mem.records[key]
        old_count = rec.count
        rec.source_signature = _merge_vector(rec.source_signature, source_vec, old_count)
        rec.target_signature = _merge_vector(rec.target_signature, target_vec, old_count)
        rec.transform_signature = _merge_vector(rec.transform_signature, transform_vec, old_count)
        rec.scalar_shift = _finite_float((rec.scalar_shift * old_count + scalar_shift) / (old_count + 1))
        rec.bivector_shift = _finite_float((rec.bivector_shift * old_count + bivector_shift) / (old_count + 1))
        rec.count += 1
        _remember_example!(rec, src, tgt, mem.max_examples)
        for term in String.(sig["target"]["terms"])
            _push_term!(rec.target_terms, term, mem.max_terms)
        end
    else
        terms = Dict{String,Int}()
        for term in String.(sig["target"]["terms"])
            _push_term!(terms, term, mem.max_terms)
        end
        mem.records[key] = SemanticCalculusRecord(
            rel,
            source_vec,
            target_vec,
            transform_vec,
            scalar_shift,
            bivector_shift,
            1,
            [Dict("source" => src, "target" => tgt)],
            terms,
        )
    end
    return true
end

function _split_sentences(text::AbstractString)
    return String[strip(s) for s in split(String(text), SENTENCE_SPLIT_RE) if length(strip(s)) >= 2]
end

function learn_semantic_calculus_from_text!(mem::SemanticCalculusMemory,
                                            text::AbstractString)
    learned = 0
    lines = String[strip(l) for l in split(String(text), '\n') if length(strip(l)) >= 2]
    for i in 1:(length(lines) - 1)
        if occursin(r"[\?\u061F]\s*$", lines[i])
            learn_semantic_calculus_from_pair!(mem, lines[i], lines[i + 1]) && (learned += 1)
        end
    end

    sentences = _split_sentences(text)
    for i in 1:(length(sentences) - 1)
        learn_semantic_calculus_from_pair!(
            mem,
            sentences[i],
            sentences[i + 1];
            relation="semantic_continuation",
        ) && (learned += 1)
    end
    return learned
end

function train_semantic_calculus_from_texts!(mem::SemanticCalculusMemory,
                                             texts::Vector{String};
                                             max_pairs::Int=50_000)
    learned = 0
    for text in texts
        learned += learn_semantic_calculus_from_text!(mem, text)
        learned >= max_pairs && break
    end
    return learned
end

function _cosine(a::AbstractVector, b::AbstractVector)
    d = min(length(a), length(b))
    d == 0 && return 0.0
    na = norm(view(a, 1:d)); nb = norm(view(b, 1:d))
    (na < 1e-10 || nb < 1e-10) && return 0.0
    return clamp(dot(view(a, 1:d), view(b, 1:d)) / (na * nb), -1.0, 1.0)
end

function select_semantic_transform(mem::SemanticCalculusMemory,
                                   prompt::AbstractString;
                                   relation::String="")
    isempty(mem.records) && return nothing
    scored = _rank_semantic_transforms(mem, prompt; relation=relation)
    isempty(scored) && return nothing
    return scored[1][2]
end

function _rank_semantic_transforms(mem::SemanticCalculusMemory,
                                   prompt::AbstractString;
                                   relation::String="")
    sig = sentence_semantic_signature(prompt)
    prompt_vec = _finite_vector(sig["vector"]; dim=SIGNATURE_DIM)
    inferred = isempty(strip(relation)) ? _detect_relation(prompt) : relation
    scored = Tuple{Float64,SemanticCalculusRecord}[]
    for rec in values(mem.records)
        score = 0.55 * _cosine(prompt_vec, rec.source_signature)
        rec.relation == inferred && (score += 0.35)
        score += 0.10 * log(1 + rec.count)
        push!(scored, (score, rec))
    end
    isempty(scored) && return scored
    sort!(scored; by=x -> -x[1])
    return scored
end

function _preferred_terms(rec::SemanticCalculusRecord; limit::Int=12)
    vals = sort(collect(rec.target_terms); by=x -> (-x[2], x[1]))
    return String[v[1] for v in vals[1:min(limit, length(vals))]]
end

function _projected_signature(prompt::AbstractString, rec::SemanticCalculusRecord)
    sig = sentence_semantic_signature(prompt)
    projected = _finite_vector(sig["vector"]; dim=SIGNATURE_DIM) .+
                _finite_vector(rec.transform_signature; dim=SIGNATURE_DIM)
    return _normalize_finite!(projected)
end

function semantic_guidance(mem::SemanticCalculusMemory,
                           prompt::AbstractString;
                           relation::String="", limit::Int=12)
    scored = _rank_semantic_transforms(mem, prompt; relation=relation)
    if isempty(scored)
        return Dict{String,Any}(
            "active" => false,
            "relation" => "",
            "movement" => "none",
            "answer_frame" => "none",
            "confidence" => 0.0,
            "target_terms" => String[],
            "transform_signature" => zeros(Float64, SIGNATURE_DIM),
            "projected_signature" => zeros(Float64, SIGNATURE_DIM),
            "scalar_shift" => 0.0,
            "bivector_shift" => 0.0,
            "answer_plan" => Dict{String,Any}(
                "active" => false,
                "steps" => Any[],
                "plan_signature" => "",
            ),
            "example" => nothing,
        )
    end
    score, rec = scored[1]
    confidence = clamp(0.5 + 0.5 * score, 0.0, 1.0)
    movement = semantic_relation_movement(rec.relation)
    frame = _answer_frame(rec.relation)
    terms = _preferred_terms(rec; limit=limit)
    plan = _build_answer_plan(movement, frame, terms, confidence)
    return Dict{String,Any}(
        "active" => true,
        "relation" => rec.relation,
        "movement" => movement,
        "answer_frame" => frame,
        "confidence" => confidence,
        "target_terms" => terms,
        "transform_signature" => rec.transform_signature,
        "projected_signature" => _projected_signature(prompt, rec),
        "scalar_shift" => rec.scalar_shift,
        "bivector_shift" => rec.bivector_shift,
        "answer_plan" => plan,
        "example" => isempty(rec.examples) ? nothing : rec.examples[end],
    )
end

function _build_answer_plan(movement::String, frame::String,
                            terms::Vector{String}, confidence::Float64)
    roles = _plan_roles_for_movement(movement)
    steps = Vector{Dict{String,Any}}()
    for (i, role) in enumerate(roles)
        push!(steps, Dict{String,Any}(
            "index" => i,
            "role" => role,
            "intent" => string(frame, "::", role),
            "terms" => _terms_for_plan_step(terms, i),
        ))
    end
    return Dict{String,Any}(
        "active" => true,
        "movement" => movement,
        "answer_frame" => frame,
        "confidence" => confidence,
        "steps" => steps,
        "plan_signature" => string(movement, ":", frame, ":", join(roles, ">")),
    )
end

function semantic_answer_plan(mem::SemanticCalculusMemory,
                              prompt::AbstractString;
                              relation::String="", limit::Int=12)
    guidance = semantic_guidance(mem, prompt; relation=relation, limit=limit)
    return get(guidance, "answer_plan", Dict{String,Any}(
        "active" => false,
        "steps" => Any[],
        "plan_signature" => "",
    ))
end

function semantic_guidance_terms(mem::SemanticCalculusMemory,
                                 prompt::AbstractString; limit::Int=12)
    guidance = semantic_guidance(mem, prompt; limit=limit)
    return String[String(t) for t in get(guidance, "target_terms", String[])]
end

has_semantic_calculus(mem::SemanticCalculusMemory) = !isempty(mem.records)

function semantic_calculus_to_dict(mem::SemanticCalculusMemory)
    records = sort(collect(values(mem.records)); by=r -> (-r.count, r.relation))
    return Dict{String,Any}(
        "version" => AL_HISBAN_VERSION,
        "n_records" => length(records),
        "records" => [Dict{String,Any}(
            "relation" => rec.relation,
            "source_signature" => _finite_vector(rec.source_signature; dim=SIGNATURE_DIM),
            "target_signature" => _finite_vector(rec.target_signature; dim=SIGNATURE_DIM),
            "transform_signature" => _finite_vector(rec.transform_signature; dim=SIGNATURE_DIM),
            "scalar_shift" => _finite_float(rec.scalar_shift),
            "bivector_shift" => _finite_float(rec.bivector_shift),
            "count" => rec.count,
            "examples" => rec.examples,
            "target_terms" => rec.target_terms,
        ) for rec in records],
    )
end

function save_semantic_calculus(mem::SemanticCalculusMemory, path::String)
    mkpath(dirname(path))
    tmp = string(path, ".tmp")
    open(tmp, "w") do io
        JSON.print(io, semantic_calculus_to_dict(mem))
    end
    mv(tmp, path; force=true)
    return path
end

function load_semantic_calculus(path::String)
    mem = SemanticCalculusMemory()
    isfile(path) || return mem
    filesize(path) == 0 && return mem
    data = try
        JSON.parsefile(path)
    catch e
        @warn "تعذر تحميل al_hisban_al_dalali: $path — $e"
        return mem
    end
    for item in get(data, "records", Any[])
        item isa AbstractDict || continue
        relation = String(get(item, "relation", "semantic_continuation"))
        source_signature = _finite_vector(get(item, "source_signature", zeros(Float64, SIGNATURE_DIM)); dim=SIGNATURE_DIM)
        target_signature = _finite_vector(get(item, "target_signature", zeros(Float64, SIGNATURE_DIM)); dim=SIGNATURE_DIM)
        transform_signature = _finite_vector(get(item, "transform_signature", zeros(Float64, SIGNATURE_DIM)); dim=SIGNATURE_DIM)
        examples = Dict{String,String}[]
        for ex in get(item, "examples", Any[])
            ex isa AbstractDict || continue
            push!(examples, Dict(
                "source" => String(get(ex, "source", "")),
                "target" => String(get(ex, "target", "")),
            ))
        end
        raw_terms = get(item, "target_terms", Dict{String,Any}())
        terms = Dict{String,Int}()
        for (k, v) in raw_terms
            count = try
                Int(v)
            catch
                0
            end
            count > 0 && (terms[String(k)] = count)
        end
        mem.records[_record_key(relation)] = SemanticCalculusRecord(
            relation,
            source_signature,
            target_signature,
            transform_signature,
            _finite_float(get(item, "scalar_shift", 0.0)),
            _finite_float(get(item, "bivector_shift", 0.0)),
            Int(get(item, "count", 0)),
            examples,
            terms,
        )
    end
    return mem
end

end # module AlHisbanAlDalali
