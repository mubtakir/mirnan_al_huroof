module DSL

using ..Entities: Thing, PROPERTY_MAP, get_attribute, set_attribute!, set_property!
using ..CausalSpace: SimulationSpace, CausalFrame, PropertyEffect, DynamicVerb, CausalRule, CausalTemplate,
                     ProcessStep, ProcessConcept, ChainStep, EventChain,
                     register_entity!, register_verb!, add_rule!, evaluate_idea!,
                     register_class!, assign_class!, known_classes, register_template!, record_frame!,
                     register_opposite!, apply_signed_action!, set_circumstance!, relate_events!
using ..CausalSpace: register_process!, register_event_chain!, assert_relation!,
                     assert_quantified_fact!, assert_comparison!, compare_entities!
using ..CausalSpace: assert_metaphor!, assert_intent!, assert_speech_act!, add_exception_rule!
using ..CausalSpace: curate_rule!, propose_rule!, approve_rule!, reject_rule!,
                     learn_rejection_lesson!, annotate_corpus_sentence!,
                     critical_corpus_pass!

export property_key, compile_adl!, train_from_text!

const PROPERTY_ALIASES = Dict{String,String}(
    "حرارة" => "temperature",
    "درجة" => "temperature",
    "غليان" => "boiling_point",
    "الغليان" => "boiling_point",
    "درجة_الغليان" => "boiling_point",
    "نقطة_الغليان" => "boiling_point",
    "حركة" => "motion",
    "وعي" => "awareness",
    "خوف" => "fear",
    "استقرار" => "stability",
    "ثبات" => "stability",
    "تماسك" => "integrity",
    "نور" => "light",
    "ضوء" => "light",
    "طاقة" => "energy",
    "جمال" => "beauty",
    "غضب" => "anger",
    "رائحة" => "scent",
    "لون" => "color",
    "كتلة" => "mass",
    "الهدف" => "target",
    "هدف" => "target",
    "مقياس" => "scale",
    "المقياس" => "scale",
    "مصدر" => "source",
    "المصدر" => "source",
)

const PREPOSITIONS = Set(["على", "في", "إلى", "الى", "من", "عن", "ب", "ل",
                          "عند", "لدى", "مع", "فوق", "تحت", "داخل", "خارج", "بين",
                          "on", "in", "into", "to", "from", "with", "at", "inside", "outside",
                          "under", "over", "between"])

const IMPLICIT_ADJECTIVES = Dict{String,Pair{String,Any}}(
    "جميل" => "beauty" => 0.8,
    "جميلة" => "beauty" => 0.8,
    "حسن" => "beauty" => 0.7,
    "قبيح" => "beauty" => -0.7,
    "ساخن" => "temperature" => 0.8,
    "ساخنة" => "temperature" => 0.8,
    "بارد" => "temperature" => -0.6,
    "باردة" => "temperature" => -0.6,
    "كبير" => "size" => 0.8,
    "كبيرة" => "size" => 0.8,
    "صغير" => "size" => -0.6,
    "صغيرة" => "size" => -0.6,
    "سريع" => "motion" => 0.7,
    "سريعة" => "motion" => 0.7,
    "beautiful" => "beauty" => 0.8,
    "ugly" => "beauty" => -0.7,
    "hot" => "temperature" => 0.8,
    "cold" => "temperature" => -0.6,
    "large" => "size" => 0.8,
    "big" => "size" => 0.8,
    "small" => "size" => -0.6,
    "fast" => "motion" => 0.7,
    "slow" => "motion" => -0.5,
)

function property_key(name::AbstractString)
    key = strip(String(name))
    mapped = get(PROPERTY_ALIASES, key, nothing)
    mapped !== nothing && return mapped
    if startswith(key, "ال") && lastindex(key) > 2
        bare = key[nextind(key, firstindex(key), 2):end]
        return get(PROPERTY_ALIASES, bare, bare)
    end
    return key
end

function _parse_number(value::AbstractString, default::Float64=0.0)
    try
        return parse(Float64, strip(String(value)))
    catch
        return default
    end
end

function _split_fields(body::AbstractString)
    fields = Pair{String,String}[]
    for part in split(String(body), r"[;,،؛]")
        part = strip(part)
        isempty(part) && continue
        pieces = split(part, ":"; limit=2)
        length(pieces) == 2 || continue
        push!(fields, strip(pieces[1]) => strip(pieces[2]))
    end
    return fields
end

function _parse_entity!(space::SimulationSpace, name::String, body::String)
    attrs = Dict{String,Any}()
    mass = 1.0
    for (raw_key, raw_val) in _split_fields(body)
        key = property_key(raw_key)
        if key == "mass"
            mass = _parse_number(raw_val, 1.0)
        else
            value = try
                parse(Float64, raw_val)
            catch
                raw_val
            end
            attrs[key] = value
        end
    end
    entity = Thing(name; kind="adl", mass=mass, attributes=attrs)
    for (key, value) in attrs
        value isa Number && set_property!(entity, key, Float64(value), 0.0)
    end
    register_entity!(space, entity)
    return entity
end

_parse_entity!(space::SimulationSpace, name::AbstractString, body::AbstractString) =
    _parse_entity!(space, String(name), String(body))

function _parse_verb!(space::SimulationSpace, name::String, body::String)
    effects = PropertyEffect[]
    current = Dict{String,String}()
    for (raw_key, raw_val) in _split_fields(body)
        key = property_key(raw_key)
        current[key] = raw_val
        if haskey(current, "target") && haskey(current, "scale")
            target = property_key(current["target"])
            scale = _parse_number(current["scale"], 1.0)
            source = haskey(current, "source") ? property_key(current["source"]) : "self"
            push!(effects, PropertyEffect(target, scale, source))
            empty!(current)
        end
    end
    isempty(effects) && push!(effects, PropertyEffect("awareness", 1.0, "self"))
    return register_verb!(space, DynamicVerb(name, effects))
end

_parse_verb!(space::SimulationSpace, name::AbstractString, body::AbstractString) =
    _parse_verb!(space, String(name), String(body))

function _parse_rule!(space::SimulationSpace, body::String)
    m = match(r"^\s*([^<>]+?)\s*([<>])\s*([0-9.]+)\s*->\s*(\S+)\s*$", body)
    m === nothing && return nothing
    prop = property_key(m.captures[1])
    op = m.captures[2] == ">" ? :> : :<
    threshold = _parse_number(m.captures[3], 0.0)
    action = strip(m.captures[4])
    return add_rule!(space, CausalRule(prop, op, threshold, action))
end

_parse_rule!(space::SimulationSpace, body::AbstractString) = _parse_rule!(space, String(body))

function _parse_idea!(space::SimulationSpace, body::String)
    _infer_implicit_state!(space, body) && return true
    words = [w for w in split(strip(body)) if !(w in PREPOSITIONS)]
    length(words) < 2 && return false
    source = String(words[1])
    action = String(words[2])
    target = length(words) >= 3 ? String(words[3]) : source
    if startswith(action, "-")
        return apply_signed_action!(space, source, action[nextind(action, firstindex(action)):end], target; polarity=-1)
    end
    return evaluate_idea!(space, source, action, target)
end

_parse_idea!(space::SimulationSpace, body::AbstractString) = _parse_idea!(space, String(body))

function _parse_property!(space::SimulationSpace, body::String)
    parts = split(strip(body))
    length(parts) >= 3 || return false
    name, prop, raw_val = parts[1], property_key(parts[2]), parts[3]
    haskey(space.entities, name) || return false
    entity = space.entities[name]
    value = _parse_number(raw_val, 0.0)
    if !set_property!(entity, prop, value, 0.0) && entity isa Thing
        set_attribute!(entity, prop, value)
    end
    return true
end

function _strip_article(word::AbstractString)
    text = String(word)
    if startswith(text, "ال") && lastindex(text) > 2
        return text[nextind(text, firstindex(text), 2):end]
    end
    return text
end

function _class_candidates(token::AbstractString)
    raw = String(token)
    bare = _strip_article(raw)
    candidates = String[raw, bare]
    bare == "عاشب" && push!(candidates, "آكل_عشب")
    bare == "عشبي" && push!(candidates, "آكل_عشب")
    bare == "مفترسة" && push!(candidates, "مفترس")
    bare == "حية" && push!(candidates, "حي")
    bare == "عاقلة" && push!(candidates, "عاقل")
    bare == "نباتية" && push!(candidates, "نبات")
    return unique(candidates)
end

function _sentence_classes(space::SimulationSpace, words)
    known = Set(known_classes(space))
    classes = String[]
    for word in words
        clean = strip(String(word), ['،', ',', '.', '؛', ';', ':'])
        isempty(clean) && continue
        for candidate in _class_candidates(clean)
            candidate in known && !(candidate in classes) && push!(classes, candidate)
        end
    end
    return classes
end

function _register_or_classify_entity!(space::SimulationSpace, name::AbstractString,
                                       attrs::Dict{String,Any}, classes::Vector{String})
    entity_name = String(name)
    if haskey(space.entities, entity_name)
        entity = space.entities[entity_name]
        if entity isa Thing
            for (key, value) in attrs
                !haskey(entity.attributes, key) && set_attribute!(entity, key, value)
            end
        end
    else
        register_entity!(space, Thing(entity_name; attributes=attrs); classes=classes)
    end

    for class_name in classes
        assign_class!(space, entity_name, class_name)
    end
    return space.entities[entity_name]
end

function _ensure_thing!(space::SimulationSpace, name::AbstractString)
    entity_name = String(strip(String(name)))
    isempty(entity_name) && return nothing
    haskey(space.entities, entity_name) || register_entity!(space, Thing(entity_name))
    return space.entities[entity_name]
end

function _record_implicit_be!(space::SimulationSpace, source::AbstractString,
                              relation::AbstractString, target::AbstractString)
    src = String(strip(String(source)))
    rel = String(strip(String(relation)))
    tgt = String(strip(String(target)))
    (isempty(src) || isempty(rel) || isempty(tgt)) && return false

    _ensure_thing!(space, src)
    _ensure_thing!(space, tgt)
    if space.entities[src] isa Thing
        set_attribute!(space.entities[src], "implicit_relation", rel)
        set_attribute!(space.entities[src], "implicit_target", tgt)
    end
    rel in ("على", "في", "عند", "لدى", "فوق", "تحت", "داخل", "خارج", "بين", "مع") &&
        set_circumstance!(space, src; location="$rel $tgt")
    assert_relation!(space, src, rel, tgt; evidence="implicit_be")
    record_frame!(space, CausalFrame([src, tgt], "يكون", "relation=$rel | target=$tgt"))
    return true
end

function _record_construct_state!(space::SimulationSpace, head::AbstractString, owner::AbstractString)
    h = String(strip(String(head)))
    o = String(strip(String(owner)))
    (isempty(h) || isempty(o)) && return false
    compound = "$h $o"

    _ensure_thing!(space, h)
    _ensure_thing!(space, o)
    _ensure_thing!(space, compound)

    if space.entities[h] isa Thing
        set_attribute!(space.entities[h], "possessor", o)
        set_attribute!(space.entities[h], "compound_name", compound)
    end
    if space.entities[o] isa Thing
        set_attribute!(space.entities[o], "has_part", h)
    end
    if space.entities[compound] isa Thing
        set_attribute!(space.entities[compound], "head", h)
        set_attribute!(space.entities[compound], "possessor", o)
        set_attribute!(space.entities[compound], "implicit_relation", "له")
    end

    assert_relation!(space, o, "له", h; evidence="construct_state")
    assert_relation!(space, h, "جزء_من", o; evidence="construct_state")
    record_frame!(space, CausalFrame([o, h], "يكون", "actor=$o | relation=له | target=$h | focus=$compound"))
    return true
end

function _record_implicit_attribute!(space::SimulationSpace, entity::AbstractString, adjective::AbstractString)
    name = String(strip(String(entity)))
    adj = String(strip(String(adjective)))
    haskey(IMPLICIT_ADJECTIVES, adj) || return false

    pair = IMPLICIT_ADJECTIVES[adj]
    key, value = pair.first, pair.second
    _ensure_thing!(space, name)
    if space.entities[name] isa Thing
        set_attribute!(space.entities[name], key, value)
        set_attribute!(space.entities[name], "implicit_attribute", adj)
        value isa Number && set_property!(space.entities[name], key, Float64(max(value, 0.0)), 0.0)
    end
    record_frame!(space, CausalFrame([name], "يكون", "attribute=$key | value=$value | adjective=$adj"))
    return true
end

function _infer_implicit_state!(space::SimulationSpace, sentence::AbstractString)
    words = [String(w) for w in split(strip(String(sentence))) if !isempty(strip(String(w)))]
    length(words) >= 2 || return false

    prep_idx = findfirst(w -> w in PREPOSITIONS, words)
    if prep_idx !== nothing && prep_idx > 1 && prep_idx < length(words)
        source = join(words[1:prep_idx-1], " ")
        target = join(words[prep_idx+1:end], " ")
        return _record_implicit_be!(space, source, words[prep_idx], target)
    end

    if length(words) == 2 && _record_implicit_attribute!(space, words[1], words[2])
        return true
    end

    if length(words) == 2 && isempty(_sentence_classes(space, [words[2]]))
        return _record_construct_state!(space, words[1], words[2])
    end

    return false
end

function _split_list(value::AbstractString)
    return [strip(String(v)) for v in split(String(value), r"[|،,]") if !isempty(strip(String(v)))]
end

function _parse_value(value::AbstractString)
    raw = strip(String(value))
    raw in ("true", "صحيح", "نعم") && return true
    raw in ("false", "خطأ", "لا") && return false
    try
        return parse(Float64, raw)
    catch
        return raw
    end
end

function _parse_class_body(body::String)
    parents = String[]
    attrs = Dict{String,Any}()
    props = Dict{String,Float64}()
    members = String[]
    abstract_key = true

    for (raw_key, raw_val) in _split_fields(body)
        key = strip(String(raw_key))
        canonical = property_key(key)

        if key in ("أب", "آباء", "والد", "فوق", "parent", "parents")
            append!(parents, _split_list(raw_val))
            continue
        end

        if key in ("عضو", "أعضاء", "member", "members")
            append!(members, _split_list(raw_val))
            continue
        end

        if key in ("مجرد", "abstract", "abstract_key")
            abstract_key = Bool(_parse_value(raw_val))
            continue
        end

        value = _parse_value(raw_val)
        if value isa Number && haskey(PROPERTY_MAP, canonical)
            props[canonical] = Float64(value)
        else
            attrs[canonical] = value
        end
    end

    return unique(parents), attrs, props, unique(members), abstract_key
end

function _parse_class!(space::SimulationSpace, name::AbstractString,
                       parent_expr::Union{Nothing,AbstractString}, body::AbstractString)
    parents, attrs, props, members, abstract_key = _parse_class_body(String(body))
    if parent_expr !== nothing && !isempty(strip(String(parent_expr)))
        append!(parents, _split_list(parent_expr))
    end
    return register_class!(space, strip(String(name)); parents=unique(parents),
                           attributes=attrs, properties=props, members=members,
                           abstract_key=abstract_key)
end

function _parse_classification!(space::SimulationSpace, entity_name::AbstractString, class_expr::AbstractString)
    name = strip(String(entity_name))
    classes = _split_list(class_expr)
    isempty(classes) && return false
    haskey(space.entities, name) || register_entity!(space, Thing(String(name)); classes=classes)
    for class_name in classes
        assign_class!(space, name, class_name)
    end
    return true
end

function _parse_typed_entity!(space::SimulationSpace, name::AbstractString,
                              class_expr::AbstractString, body::AbstractString)
    attrs = Dict{String,Any}()
    mass = 1.0
    for (raw_key, raw_val) in _split_fields(String(body))
        key = property_key(raw_key)
        if key == "mass"
            mass = _parse_number(raw_val, 1.0)
        else
            attrs[key] = _parse_value(raw_val)
        end
    end
    classes = _split_list(class_expr)
    entity = Thing(String(strip(String(name))); kind=isempty(classes) ? "adl" : String(classes[1]),
                   mass=mass, attributes=attrs)
    for (key, value) in attrs
        value isa Number && set_property!(entity, key, Float64(value), 0.0)
    end
    return register_entity!(space, entity; classes=classes)
end

function _parse_template!(space::SimulationSpace, name::AbstractString, body::AbstractString)
    values = Dict{String,String}()
    for (raw_key, raw_val) in _split_fields(String(body))
        values[strip(String(raw_key))] = strip(String(raw_val))
    end

    function pick(names...; default="")
        for name in names
            haskey(values, name) && return values[name]
        end
        return default
    end

    template = CausalTemplate(
        String(strip(String(name))),
        pick("مجال", "domain"),
        pick("مصدر", "فاعل", "source"),
        pick("فعل", "حدث", "action"),
        pick("هدف", "مفعول", "target"),
        pick("نوع", "نوع_الفعل", "kind"),
        pick("فاعل_النتيجة", "نتيجة_الفاعل", "result_actor"; default="كلاهما"),
        pick("فعل_النتيجة", "result_action"),
        pick("هدف_النتيجة", "result_target"),
        pick("نوع_النتيجة", "result_kind"),
        pick("نتيجة", "حالة_النتيجة", "result_state"),
        pick("شرط", "إذا", "condition"),
        _parse_number(pick("ثقة", "احتمال", "confidence"; default="1.0"), 1.0),
        Int(round(_parse_number(pick("قطبية", "polarity"; default="1"), 1.0))),
    )
    return register_template!(space, template)
end

function _parse_process_step(value::AbstractString)
    raw = strip(String(value))
    pieces = split(raw, "->"; limit=2)
    action_part = strip(pieces[1])
    result = length(pieces) == 2 ? strip(pieces[2]) : ""
    words = split(action_part)
    if length(words) >= 3
        return ProcessStep(String(words[1]), String(words[2]), String(words[3]), String(result))
    elseif length(words) == 2
        return ProcessStep("source", String(words[1]), String(words[2]), String(result))
    elseif length(words) == 1
        return ProcessStep("source", String(words[1]), "target", String(result))
    end
    return ProcessStep("source", "حدث", "target", String(result))
end

function _parse_process!(space::SimulationSpace, name::AbstractString,
                         parent_expr::Union{Nothing,AbstractString}, body::AbstractString)
    domain = ""
    intensity = 1.0
    steps = ProcessStep[]
    attrs = Dict{String,Any}()

    for (raw_key, raw_val) in _split_fields(String(body))
        key = strip(String(raw_key))
        if key in ("مجال", "domain")
            domain = strip(String(raw_val))
        elseif key in ("شدة", "intensity")
            intensity = _parse_number(raw_val, 1.0)
        elseif key in ("خطوة", "step")
            push!(steps, _parse_process_step(raw_val))
        else
            attrs[property_key(key)] = _parse_value(raw_val)
        end
    end

    parent = parent_expr === nothing ? "" : String(strip(String(parent_expr)))
    process = ProcessConcept(String(strip(String(name))), parent, domain, intensity, steps, attrs)
    return register_process!(space, process)
end

function _parse_chain_step(value::AbstractString)
    raw = strip(String(value))
    pieces = split(raw, "->"; limit=2)
    action_part = strip(pieces[1])
    result = length(pieces) == 2 ? strip(pieces[2]) : ""
    words = split(action_part)
    if length(words) >= 3
        return ChainStep(String(words[1]), String(words[2]), String(words[3]), String(result))
    elseif length(words) == 2
        return ChainStep("source", String(words[1]), String(words[2]), String(result))
    elseif length(words) == 1
        return ChainStep("source", String(words[1]), "target", String(result))
    end
    return ChainStep("source", "حدث", "target", String(result))
end

function _parse_event_chain!(space::SimulationSpace, name::AbstractString, body::AbstractString)
    source_class = ""
    action = ""
    target_class = ""
    confidence = 1.0
    steps = ChainStep[]

    for (raw_key, raw_val) in _split_fields(String(body))
        key = strip(String(raw_key))
        if key in ("مصدر", "فاعل", "source")
            source_class = strip(String(raw_val))
        elseif key in ("فعل", "حدث", "action")
            action = strip(String(raw_val))
        elseif key in ("هدف", "مفعول", "target")
            target_class = strip(String(raw_val))
        elseif key in ("ثقة", "احتمال", "confidence")
            confidence = _parse_number(raw_val, 1.0)
        elseif key in ("خطوة", "step")
            push!(steps, _parse_chain_step(raw_val))
        end
    end

    chain = EventChain(String(strip(String(name))), source_class, action, target_class, steps, confidence)
    return register_event_chain!(space, chain)
end

function _parse_circumstance!(space::SimulationSpace, name::AbstractString, body::AbstractString)
    entity_name = String(strip(String(name)))
    _ensure_thing!(space, entity_name)
    location = ""
    time = ""
    for (raw_key, raw_val) in _split_fields(String(body))
        key = strip(String(raw_key))
        if key in ("مكان", "موقع", "بلد", "دار", "location", "place")
            location = strip(String(raw_val))
        elseif key in ("زمان", "وقت", "يوم", "ساعة", "time")
            time = strip(String(raw_val))
        end
    end
    return set_circumstance!(space, entity_name; location=location, time=time)
end

function _parse_event_order!(space::SimulationSpace, body::AbstractString)
    text = strip(String(body))
    m = match(r"^(.+?)\s+(قبل|بعد|أثناء|مع)\s+(.+?)$", text)
    m === nothing && return false
    return relate_events!(space, strip(m.captures[1]), strip(m.captures[2]), strip(m.captures[3]))
end

function _parse_relation!(space::SimulationSpace, body::AbstractString)
    text = strip(String(body))
    confidence = 1.0
    evidence = "adl_relation"

    fields = _split_fields(text)
    if !isempty(fields)
        values = Dict{String,String}()
        for (raw_key, raw_val) in fields
            values[strip(String(raw_key))] = strip(String(raw_val))
        end
        function pick(names...; default="")
            for name in names
                haskey(values, name) && return values[name]
            end
            return default
        end
        source = pick("مصدر", "طرف1", "source")
        relation = pick("علاقة", "نوع", "relation")
        target = pick("هدف", "طرف2", "target")
        confidence = _parse_number(pick("ثقة", "confidence"; default="1.0"), 1.0)
        evidence = pick("دليل", "evidence"; default=evidence)
        (isempty(source) || isempty(relation) || isempty(target)) && return false
        assert_relation!(space, source, relation, target; confidence=confidence, evidence=evidence)
        record_frame!(space, CausalFrame([source, target], "علاقة", "relation=$relation | confidence=$confidence"))
        return true
    end

    words = split(text)
    length(words) >= 3 || return false
    source = String(words[1])
    relation = String(words[2])
    target = join(words[3:end], " ")
    assert_relation!(space, source, relation, target; confidence=confidence, evidence=evidence)
    record_frame!(space, CausalFrame([source, target], "علاقة", "relation=$relation | confidence=$confidence"))
    return true
end

function _parse_quantified_fact!(space::SimulationSpace, body::AbstractString)
    text = strip(String(body))
    quantifier = ""
    subject = ""
    predicate = ""
    object = ""
    polarity = 1
    confidence = 1.0
    evidence = "adl_quantifier"

    fields = _split_fields(text)
    if !isempty(fields)
        values = Dict{String,String}()
        for (raw_key, raw_val) in fields
            values[strip(String(raw_key))] = strip(String(raw_val))
        end
        function pick(names...; default="")
            for name in names
                haskey(values, name) && return values[name]
            end
            return default
        end
        quantifier = pick("مقدار", "كم", "quantifier")
        subject = pick("موضوع", "نطاق", "subject")
        predicate = pick("حكم", "علاقة", "predicate")
        object = pick("هدف", "محمول", "object")
        confidence = _parse_number(pick("ثقة", "confidence"; default="1.0"), 1.0)
        polarity = Int(round(_parse_number(pick("قطبية", "polarity"; default="1"), 1.0)))
    else
        words = split(text)
        length(words) >= 3 || return false
        quantifier = String(words[1])
        subject = String(words[2])
        if words[3] in ("لا", "ليس", "ليست", "لن", "لم")
            polarity = -1
            length(words) >= 4 || return false
            predicate = String(words[4])
            object = length(words) >= 5 ? join(words[5:end], " ") : ""
        else
            predicate = String(words[3])
            object = length(words) >= 4 ? join(words[4:end], " ") : ""
        end
    end

    (isempty(quantifier) || isempty(subject) || isempty(predicate)) && return false
    fact = assert_quantified_fact!(space, quantifier, subject, predicate, object;
                                   polarity=polarity, confidence=confidence, evidence=evidence)
    record_frame!(space, CausalFrame([subject], "كم", "quantifier=$(fact.quantifier) | predicate=$(fact.predicate) | object=$(fact.object) | polarity=$(fact.polarity) | confidence=$(fact.confidence)"))
    return true
end

function _parse_comparison!(space::SimulationSpace, body::AbstractString)
    text = strip(String(body))
    left = ""
    right = ""
    property = "value"
    comparator = ""
    confidence = 1.0
    computed = false

    fields = _split_fields(text)
    if !isempty(fields)
        values = Dict{String,String}()
        for (raw_key, raw_val) in fields
            values[strip(String(raw_key))] = strip(String(raw_val))
        end
        function pick(names...; default="")
            for name in names
                haskey(values, name) && return values[name]
            end
            return default
        end
        left = pick("طرف1", "أيسر", "left")
        right = pick("طرف2", "أيمن", "right")
        property = property_key(pick("خاصية", "property"; default=property))
        comparator = pick("نوع", "مقارن", "comparator")
        confidence = _parse_number(pick("ثقة", "confidence"; default="1.0"), 1.0)
        computed = Bool(_parse_value(pick("احسب", "computed"; default="false")))
    else
        m = match(r"^(.+?)\s+(أكبر|أصغر|أكثر|أقل|يساوي|مثل)\s+من\s+(.+?)(?:\s+في\s+(.+?))?$", text)
        if m === nothing
            m = match(r"^(.+?)\s+(greater|larger|smaller|less|more|equal|equals)\s+than\s+(.+?)(?:\s+in\s+(.+?))?$", text)
        end
        m === nothing && return false
        left = strip(m.captures[1])
        comparator = strip(m.captures[2])
        right = strip(m.captures[3])
        property = m.captures[4] === nothing ? property : property_key(strip(m.captures[4]))
    end

    (isempty(left) || isempty(right)) && return false
    fact = computed ? compare_entities!(space, left, right, property; confidence=confidence) :
                      assert_comparison!(space, left, property, comparator, right;
                                         confidence=confidence, evidence="adl_comparison")
    fact === nothing && return false
    record_frame!(space, CausalFrame([fact.left, fact.right], "مقارنة", "property=$(fact.property) | comparator=$(fact.comparator) | confidence=$(fact.confidence)"))
    return true
end

function _parse_metaphor!(space::SimulationSpace, body::AbstractString)
    text = strip(String(body))
    values = Dict{String,String}()
    for (raw_key, raw_val) in _split_fields(text)
        values[strip(String(raw_key))] = strip(String(raw_val))
    end
    function pick(names...; default="")
        for name in names
            haskey(values, name) && return values[name]
        end
        return default
    end

    if isempty(values)
        words = split(text)
        length(words) >= 3 || return false
        expression = join(words, " ")
        fact = assert_metaphor!(space, expression;
                                literal_subject=String(words[2]),
                                borrowed_actor=String(words[1]),
                                action=String(words[3]),
                                evidence="adl_metaphor")
        record_frame!(space, CausalFrame([fact.literal_subject, fact.borrowed_actor], "مجاز", "expression=$(fact.expression) | action=$(fact.action) | confidence=$(fact.confidence)"))
        return true
    end

    expression = pick("تعبير", "expression")
    literal_subject = pick("موضوع", "حامل", "literal_subject")
    borrowed_actor = pick("مستعار", "فاعل_مستعار", "borrowed_actor")
    action = pick("فعل", "action")
    fact = assert_metaphor!(space, expression;
                            source_domain=pick("مجال_المصدر", "source_domain"),
                            target_domain=pick("مجال_الهدف", "target_domain"),
                            literal_subject=literal_subject,
                            borrowed_actor=borrowed_actor,
                            action=action,
                            transferred_property=pick("خاصية_منقولة", "transferred_property"),
                            confidence=_parse_number(pick("ثقة", "confidence"; default="1.0"), 1.0),
                            evidence=pick("دليل", "evidence"; default="adl_metaphor"))
    record_frame!(space, CausalFrame(filter(!isempty, [fact.literal_subject, fact.borrowed_actor]), "مجاز", "expression=$(fact.expression) | action=$(fact.action) | confidence=$(fact.confidence)"))
    return true
end

function _parse_intent!(space::SimulationSpace, body::AbstractString)
    text = strip(String(body))
    values = Dict{String,String}()
    for (raw_key, raw_val) in _split_fields(text)
        values[strip(String(raw_key))] = strip(String(raw_val))
    end
    if isempty(values)
        m = match(r"^(.+?)\s+(.+?)\s+(.+?)\s+ل(?:ـ|كي|)\s*(.+?)$", text)
        m === nothing && return false
        fact = assert_intent!(space, strip(m.captures[1]), strip(m.captures[2]);
                              target=strip(m.captures[3]), intent=strip(m.captures[4]),
                              goal=strip(m.captures[4]), evidence="adl_intent")
        record_frame!(space, CausalFrame([fact.actor, fact.target], "نية", "intent=$(fact.intent) | goal=$(fact.goal) | actual=$(fact.actual_result)"))
        return true
    end
    function pick(names...; default="")
        for name in names
            haskey(values, name) && return values[name]
        end
        return default
    end
    fact = assert_intent!(space, pick("فاعل", "actor"), pick("فعل", "action");
                          target=pick("هدف_الفعل", "target"),
                          intent=pick("نية", "intent"),
                          goal=pick("غاية", "هدف", "goal"),
                          actual_result=pick("نتيجة_واقعة", "نتيجة", "actual_result"),
                          confidence=_parse_number(pick("ثقة", "confidence"; default="1.0"), 1.0),
                          evidence=pick("دليل", "evidence"; default="adl_intent"))
    record_frame!(space, CausalFrame(filter(!isempty, [fact.actor, fact.target]), "نية", "intent=$(fact.intent) | goal=$(fact.goal) | actual=$(fact.actual_result)"))
    return true
end

function _parse_speech_act!(space::SimulationSpace, body::AbstractString)
    text = strip(String(body))
    values = Dict{String,String}()
    for (raw_key, raw_val) in _split_fields(text)
        values[strip(String(raw_key))] = strip(String(raw_val))
    end

    if isempty(values)
        m = match(r"^(.+?)\s+(.+?)\s+\"(.+?)\"\s*->\s*(.+?)\s+(.+?)\s+\"(.+?)\"\s*$", text)
        m === nothing && return false
        fact = assert_speech_act!(
            space,
            strip(m.captures[1]),
            strip(m.captures[2]),
            strip(m.captures[3]);
            responder=strip(m.captures[4]),
            response_act=strip(m.captures[5]),
            response_content=strip(m.captures[6]),
            confidence=0.86,
            evidence="adl_speech_act",
        )
        record_frame!(space, CausalFrame(
            filter(!isempty, [fact.speaker, fact.responder]),
            fact.act_type,
            "speech_content=$(fact.content) | response_act=$(fact.response_act) | response_content=$(fact.response_content)",
        ))
        return true
    end

    function pick(names...; default="")
        for name in names
            haskey(values, name) && return values[name]
        end
        return default
    end

    fact = assert_speech_act!(
        space,
        pick("متكلم", "فاعل", "speaker", "actor"; default="س"),
        pick("نوع", "فعل_قولي", "نوع_الفعل", "act_type"; default="قول"),
        pick("محتوى", "قول", "content"; default="");
        responder=pick("مجيب", "متلقي", "responder"; default="ج"),
        response_act=pick("نوع_الرد", "رد_قولي", "response_act"; default="رد_قولي"),
        response_content=pick("محتوى_الرد", "رد", "response_content"; default=""),
        confidence=_parse_number(pick("ثقة", "confidence"; default="1.0"), 1.0),
        evidence=pick("دليل", "evidence"; default="adl_speech_act"),
    )
    record_frame!(space, CausalFrame(
        filter(!isempty, [fact.speaker, fact.responder]),
        fact.act_type,
        "speech_content=$(fact.content) | response_act=$(fact.response_act) | response_content=$(fact.response_content)",
    ))
    return true
end

function _parse_exception_rule!(space::SimulationSpace, body::AbstractString)
    text = strip(String(body))
    values = Dict{String,String}()
    for (raw_key, raw_val) in _split_fields(text)
        values[strip(String(raw_key))] = strip(String(raw_val))
    end
    if isempty(values)
        m = match(r"^(.+?)\s+إلا\s+إذا\s+(.+?)$", text)
        m === nothing && return false
        fact = add_exception_rule!(space, strip(m.captures[1]), "", strip(m.captures[2]);
                                   evidence="adl_exception")
        record_frame!(space, CausalFrame([fact.rule_name], "استثناء", "exception=$(fact.exception) | priority=$(fact.priority) | confidence=$(fact.confidence)"))
        return true
    end
    function pick(names...; default="")
        for name in names
            haskey(values, name) && return values[name]
        end
        return default
    end
    rule_name = pick("قاعدة", "rule")
    condition = pick("شرط", "condition")
    exception = pick("استثناء", "exception")
    fact = add_exception_rule!(space, rule_name, condition, exception;
                               priority=Int(round(_parse_number(pick("أولوية", "priority"; default="1"), 1.0))),
                               confidence=_parse_number(pick("ثقة", "confidence"; default="1.0"), 1.0),
                               evidence=pick("دليل", "evidence"; default="adl_exception"))
    record_frame!(space, CausalFrame([fact.rule_name], "استثناء", "exception=$(fact.exception) | priority=$(fact.priority) | confidence=$(fact.confidence)"))
    return true
end

function _adl_fields(body::AbstractString)
    values = Dict{String,String}()
    for (raw_key, raw_val) in _split_fields(String(body))
        values[strip(String(raw_key))] = strip(String(raw_val))
    end
    return values
end

function _adl_pick(values::Dict{String,String}, names...; default="")
    for name in names
        haskey(values, name) && return values[name]
    end
    return default
end

function _parse_governance_rule!(space::SimulationSpace, body::AbstractString,
                                 mode::Symbol)
    values = _adl_fields(body)
    if isempty(values)
        words = split(strip(String(body)))
        length(words) >= 3 || return false
        subject = String(words[1])
        predicate = String(words[2])
        object = join(words[3:end], " ")
        if mode == :curated
            curate_rule!(space, subject, predicate, object; source="adl_curated")
        else
            propose_rule!(space, subject, predicate, object; source="adl_proposed")
        end
        return true
    end

    subject = _adl_pick(values, "subject", "source", "فاعل", "مصدر", "موضوع")
    predicate = _adl_pick(values, "predicate", "relation", "action", "حكم", "علاقة", "فعل")
    object = _adl_pick(values, "object", "target", "نتيجة", "هدف", "محمول")
    condition = _adl_pick(values, "condition", "شرط")
    tags = _split_list(_adl_pick(values, "tags", "وسوم"; default=""))
    id = _adl_pick(values, "id", "معرف"; default="")
    polarity = Int(round(_parse_number(_adl_pick(values, "polarity", "قطبية"; default="1"), 1.0)))
    priority = Int(round(_parse_number(_adl_pick(values, "priority", "أولوية"; default=mode == :curated ? "100" : "10"), 10.0)))
    confidence = _parse_number(_adl_pick(values, "confidence", "ثقة"; default=mode == :curated ? "1.0" : "0.5"), 1.0)
    source = _adl_pick(values, "source_label", "دليل", "evidence"; default=mode == :curated ? "adl_curated" : "adl_proposed")
    rationale = _adl_pick(values, "rationale", "تعليل"; default="")
    (isempty(subject) || isempty(predicate) || isempty(object)) && return false
    if mode == :curated
        curate_rule!(space, subject, predicate, object;
                     condition=condition, polarity=polarity, priority=priority,
                     confidence=confidence, source=source, rationale=rationale,
                     tags=tags, id=id)
    else
        propose_rule!(space, subject, predicate, object;
                      condition=condition, polarity=polarity, priority=priority,
                      confidence=confidence, source=source, rationale=rationale,
                      tags=tags, id=id)
    end
    return true
end

function _parse_approve_rule!(space::SimulationSpace, body::AbstractString)
    values = _adl_fields(body)
    id = isempty(values) ? strip(String(body)) : _adl_pick(values, "id", "معرف")
    isempty(id) && return false
    approve_rule!(space, id)
    return true
end

function _parse_reject_rule!(space::SimulationSpace, body::AbstractString)
    values = _adl_fields(body)
    if isempty(values)
        id = strip(String(body))
        isempty(id) && return false
        reject_rule!(space, id; reason="manual_rejection")
        return true
    end
    id = _adl_pick(values, "id", "معرف")
    reason = _adl_pick(values, "reason", "سبب"; default="manual_rejection")
    pattern = _adl_pick(values, "pattern", "نمط"; default="")
    tags = _split_list(_adl_pick(values, "tags", "وسوم"; default=""))
    confidence = _parse_number(_adl_pick(values, "confidence", "ثقة"; default="0.8"), 0.8)
    isempty(id) && return false
    reject_rule!(space, id; reason=reason, lesson_pattern=pattern,
                 penalty_tags=tags, confidence=confidence)
    return true
end

function _parse_rejection_lesson!(space::SimulationSpace, body::AbstractString)
    values = _adl_fields(body)
    reason = _adl_pick(values, "reason", "سبب")
    pattern = _adl_pick(values, "pattern", "نمط"; default="")
    tags = _split_list(_adl_pick(values, "tags", "وسوم"; default=""))
    confidence = _parse_number(_adl_pick(values, "confidence", "ثقة"; default="0.8"), 0.8)
    evidence = _adl_pick(values, "evidence", "دليل"; default="adl_lesson")
    isempty(reason) && return false
    learn_rejection_lesson!(space, reason; pattern=pattern, penalty_tags=tags,
                            confidence=confidence, evidence=evidence)
    return true
end

function _parse_corpus_annotation!(space::SimulationSpace, body::AbstractString)
    values = _adl_fields(body)
    sentence_id = _adl_pick(values, "sentence_id", "id", "معرف")
    span = _adl_pick(values, "span", "text", "نص")
    tag = _adl_pick(values, "tag", "وسم")
    confidence = _parse_number(_adl_pick(values, "confidence", "ثقة"; default="1.0"), 1.0)
    reason = _adl_pick(values, "reason", "سبب"; default="adl_annotation")
    effect_sem = _parse_number(_adl_pick(values, "effect_on_k_sem", "اثر_دلالي"; default="1.0"), 1.0)
    effect_causal = _parse_number(_adl_pick(values, "effect_on_k_causal", "اثر_سببي"; default="1.0"), 1.0)
    effect_aql = _adl_pick(values, "effect_on_al_aql", "اثر_العقل"; default="tag")
    (isempty(sentence_id) || isempty(span) || isempty(tag)) && return false
    annotate_corpus_sentence!(space, sentence_id, span, tag;
                              confidence=confidence, reason=reason,
                              effect_on_k_sem=effect_sem,
                              effect_on_k_causal=effect_causal,
                              effect_on_al_aql=effect_aql)
    return true
end

function _logical_adl_lines(text::AbstractString)
    lines = String[]
    buffer = ""
    depth = 0
    for raw_line in split(String(text), '\n')
        line = strip(raw_line)
        isempty(line) && continue
        startswith(line, "#") && continue
        startswith(line, "//") && continue
        buffer = isempty(buffer) ? line : buffer * " " * line
        depth += count(==('{'), line) - count(==('}'), line)
        if depth <= 0
            push!(lines, strip(buffer))
            buffer = ""
            depth = 0
        end
    end
    !isempty(strip(buffer)) && push!(lines, strip(buffer))
    return lines
end

function compile_adl!(space::SimulationSpace, text::AbstractString)
    for raw_line in _logical_adl_lines(text)
        line = strip(raw_line)
        isempty(line) && continue
        startswith(line, "#") && continue
        startswith(line, "//") && continue

        if startswith(line, "curated {") || startswith(line, "constitutional {")
            m = match(r"^(?:curated|constitutional)\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_governance_rule!(space, m.captures[1], :curated); continue)
        end

        if startswith(line, "curated :")
            _parse_governance_rule!(space, replace(line, r"^curated\s*:\s*" => ""), :curated)
            continue
        end

        if startswith(line, "curated ")
            _parse_governance_rule!(space, replace(line, r"^curated\s+" => ""), :curated)
            continue
        end

        if startswith(line, "proposed {") || startswith(line, "proposal {")
            m = match(r"^(?:proposed|proposal)\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_governance_rule!(space, m.captures[1], :proposed); continue)
        end

        if startswith(line, "proposed :") || startswith(line, "proposal :")
            _parse_governance_rule!(space, replace(line, r"^(?:proposed|proposal)\s*:\s*" => ""), :proposed)
            continue
        end

        if startswith(line, "proposed ") || startswith(line, "proposal ")
            _parse_governance_rule!(space, replace(line, r"^(?:proposed|proposal)\s+" => ""), :proposed)
            continue
        end

        if startswith(line, "approve ")
            _parse_approve_rule!(space, replace(line, r"^approve\s+" => ""))
            continue
        end

        if startswith(line, "reject {")
            m = match(r"^reject\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_reject_rule!(space, m.captures[1]); continue)
        end

        if startswith(line, "reject ")
            _parse_reject_rule!(space, replace(line, r"^reject\s+" => ""))
            continue
        end

        if startswith(line, "lesson {") || startswith(line, "rejection_lesson {")
            m = match(r"^(?:lesson|rejection_lesson)\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_rejection_lesson!(space, m.captures[1]); continue)
        end

        if startswith(line, "annotation {") || startswith(line, "corpus_annotation {")
            m = match(r"^(?:annotation|corpus_annotation)\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_corpus_annotation!(space, m.captures[1]); continue)
        end

        if startswith(line, "critical_corpus ")
            critical_corpus_pass!(space, replace(line, r"^critical_corpus\s+" => ""))
            continue
        end

        if startswith(line, "دستور {") || startswith(line, "قاعدة_موثوقة {")
            m = match(r"^(?:دستور|قاعدة_موثوقة)\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_governance_rule!(space, m.captures[1], :curated); continue)
        end

        if startswith(line, "اقتراح {") || startswith(line, "قاعدة_مرشحة {")
            m = match(r"^(?:اقتراح|قاعدة_مرشحة)\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_governance_rule!(space, m.captures[1], :proposed); continue)
        end

        if startswith(line, "اعتماد ")
            _parse_approve_rule!(space, replace(line, r"^اعتماد\s+" => ""))
            continue
        end

        if startswith(line, "رفض {")
            m = match(r"^رفض\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_reject_rule!(space, m.captures[1]); continue)
        end

        if startswith(line, "درس_رفض {")
            m = match(r"^درس_رفض\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_rejection_lesson!(space, m.captures[1]); continue)
        end

        if startswith(line, "وسم_ذاكرة {")
            m = match(r"^وسم_ذاكرة\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_corpus_annotation!(space, m.captures[1]); continue)
        end

        if startswith(line, "class ")
            m = match(r"^class\s+(.+?)(?:\s*<\s*(.+?))?\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_class!(space, strip(m.captures[1]), m.captures[2], m.captures[3]); continue)
        end

        if startswith(line, "classify ")
            m = match(r"^classify\s+(.+?)\s*:\s*(.+?)\s*$", line)
            m !== nothing && (_parse_classification!(space, m.captures[1], m.captures[2]); continue)
        end

        if startswith(line, "opposite ")
            m = match(r"^opposite\s+(.+?)\s*:\s*(.+?)\s*$", line)
            m !== nothing && (register_opposite!(space, strip(m.captures[1]), strip(m.captures[2])); continue)
        end

        if startswith(line, "circumstance ")
            m = match(r"^circumstance\s+(.+?)\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_circumstance!(space, m.captures[1], m.captures[2]); continue)
        end

        if startswith(line, "order ")
            _parse_event_order!(space, replace(line, r"^order\s+" => ""))
            continue
        end

        if startswith(line, "relation {")
            m = match(r"^relation\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_relation!(space, m.captures[1]); continue)
        end

        if startswith(line, "relation :")
            _parse_relation!(space, replace(line, r"^relation\s*:\s*" => ""))
            continue
        end

        if startswith(line, "relation ")
            _parse_relation!(space, replace(line, r"^relation\s+" => ""))
            continue
        end

        if startswith(line, "quantifier {") || startswith(line, "quantified {")
            m = match(r"^(?:quantifier|quantified)\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_quantified_fact!(space, m.captures[1]); continue)
        end

        if startswith(line, "quantifier :") || startswith(line, "quantified :")
            _parse_quantified_fact!(space, replace(line, r"^(?:quantifier|quantified)\s*:\s*" => ""))
            continue
        end

        if startswith(line, "quantifier ") || startswith(line, "quantified ")
            _parse_quantified_fact!(space, replace(line, r"^(?:quantifier|quantified)\s+" => ""))
            continue
        end

        if startswith(line, "comparison {") || startswith(line, "compare {")
            m = match(r"^(?:comparison|compare)\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_comparison!(space, m.captures[1]); continue)
        end

        if startswith(line, "comparison :") || startswith(line, "compare :")
            _parse_comparison!(space, replace(line, r"^(?:comparison|compare)\s*:\s*" => ""))
            continue
        end

        if startswith(line, "comparison ") || startswith(line, "compare ")
            _parse_comparison!(space, replace(line, r"^(?:comparison|compare)\s+" => ""))
            continue
        end

        if startswith(line, "metaphor {")
            m = match(r"^metaphor\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_metaphor!(space, m.captures[1]); continue)
        end

        if startswith(line, "metaphor :")
            _parse_metaphor!(space, replace(line, r"^metaphor\s*:\s*" => ""))
            continue
        end

        if startswith(line, "metaphor ")
            _parse_metaphor!(space, replace(line, r"^metaphor\s+" => ""))
            continue
        end

        if startswith(line, "intent {")
            m = match(r"^intent\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_intent!(space, m.captures[1]); continue)
        end

        if startswith(line, "intent :")
            _parse_intent!(space, replace(line, r"^intent\s*:\s*" => ""))
            continue
        end

        if startswith(line, "intent ")
            _parse_intent!(space, replace(line, r"^intent\s+" => ""))
            continue
        end

        if startswith(line, "speech_act {")
            m = match(r"^speech_act\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_speech_act!(space, m.captures[1]); continue)
        end

        if startswith(line, "speech_act :")
            _parse_speech_act!(space, replace(line, r"^speech_act\s*:\s*" => ""))
            continue
        end

        if startswith(line, "speech_act ")
            _parse_speech_act!(space, replace(line, r"^speech_act\s+" => ""))
            continue
        end

        if startswith(line, "exception {")
            m = match(r"^exception\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_exception_rule!(space, m.captures[1]); continue)
        end

        if startswith(line, "exception :")
            _parse_exception_rule!(space, replace(line, r"^exception\s*:\s*" => ""))
            continue
        end

        if startswith(line, "exception ")
            _parse_exception_rule!(space, replace(line, r"^exception\s+" => ""))
            continue
        end

        if startswith(line, "template conditional ")
            m = match(r"^template\s+conditional\s+(.+?)\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_template!(space, strip(m.captures[1]), m.captures[2]); continue)
        end

        if startswith(line, "template ")
            m = match(r"^template\s+(.+?)\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_template!(space, strip(m.captures[1]), m.captures[2]); continue)
        end

        if startswith(line, "process ")
            m = match(r"^process\s+(.+?)(?:\s*<\s*(.+?))?\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_process!(space, strip(m.captures[1]), m.captures[2], m.captures[3]); continue)
        end

        if startswith(line, "chain ")
            m = match(r"^chain\s+(.+?)\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_event_chain!(space, strip(m.captures[1]), m.captures[2]); continue)
        end

        if startswith(line, "thing ")
            m = match(r"^thing\s+(.+?)\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_entity!(space, strip(m.captures[1]), m.captures[2]); continue)
        end

        if startswith(line, "verb ")
            m = match(r"^verb\s+(.+?)\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_verb!(space, strip(m.captures[1]), m.captures[2]); continue)
        end

        if startswith(line, "rule")
            body = strip(replace(line, r"^rule\s*:?" => ""))
            _parse_rule!(space, body)
            continue
        end

        if startswith(line, "idea")
            body = strip(replace(line, r"^idea\s*:?" => ""))
            _parse_idea!(space, body)
            continue
        end

        if startswith(line, "property ")
            _parse_property!(space, replace(line, r"^property\s+" => ""))
            continue
        end

        if startswith(line, "مفتاح ")
            m = match(r"^مفتاح\s+(.+?)(?:\s*<\s*(.+?))?\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_class!(space, strip(m.captures[1]), m.captures[2], m.captures[3]); continue)
        end

        if startswith(line, "تصنيف ")
            m = match(r"^تصنيف\s+(.+?)\s*:\s*(.+?)\s*$", line)
            m !== nothing && (_parse_classification!(space, m.captures[1], m.captures[2]); continue)
        end

        if startswith(line, "ضد ")
            m = match(r"^ضد\s+(.+?)\s*:\s*(.+?)\s*$", line)
            m !== nothing && (register_opposite!(space, strip(m.captures[1]), strip(m.captures[2])); continue)
        end

        if startswith(line, "ظرف ")
            m = match(r"^ظرف\s+(.+?)\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_circumstance!(space, m.captures[1], m.captures[2]); continue)
        end

        if startswith(line, "ترتيب ")
            _parse_event_order!(space, replace(line, r"^ترتيب\s+" => ""))
            continue
        end

        if startswith(line, "علاقة {")
            m = match(r"^علاقة\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_relation!(space, m.captures[1]); continue)
        end

        if startswith(line, "علاقة :")
            _parse_relation!(space, replace(line, r"^علاقة\s*:\s*" => ""))
            continue
        end

        if startswith(line, "علاقة ")
            _parse_relation!(space, replace(line, r"^علاقة\s+" => ""))
            continue
        end

        if startswith(line, "كم {")
            m = match(r"^كم\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_quantified_fact!(space, m.captures[1]); continue)
        end

        if startswith(line, "كم :")
            _parse_quantified_fact!(space, replace(line, r"^كم\s*:\s*" => ""))
            continue
        end

        if startswith(line, "كم ")
            _parse_quantified_fact!(space, replace(line, r"^كم\s+" => ""))
            continue
        end

        if startswith(line, "مقارنة {")
            m = match(r"^مقارنة\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_comparison!(space, m.captures[1]); continue)
        end

        if startswith(line, "مقارنة :")
            _parse_comparison!(space, replace(line, r"^مقارنة\s*:\s*" => ""))
            continue
        end

        if startswith(line, "مقارنة ")
            _parse_comparison!(space, replace(line, r"^مقارنة\s+" => ""))
            continue
        end

        if startswith(line, "مجاز {")
            m = match(r"^مجاز\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_metaphor!(space, m.captures[1]); continue)
        end

        if startswith(line, "مجاز :")
            _parse_metaphor!(space, replace(line, r"^مجاز\s*:\s*" => ""))
            continue
        end

        if startswith(line, "مجاز ")
            _parse_metaphor!(space, replace(line, r"^مجاز\s+" => ""))
            continue
        end

        if startswith(line, "نية {")
            m = match(r"^نية\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_intent!(space, m.captures[1]); continue)
        end

        if startswith(line, "نية :")
            _parse_intent!(space, replace(line, r"^نية\s*:\s*" => ""))
            continue
        end

        if startswith(line, "نية ")
            _parse_intent!(space, replace(line, r"^نية\s+" => ""))
            continue
        end

        if startswith(line, "فعل_قولي {")
            m = match(r"^فعل_قولي\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_speech_act!(space, m.captures[1]); continue)
        end

        if startswith(line, "فعل_قولي :")
            _parse_speech_act!(space, replace(line, r"^فعل_قولي\s*:\s*" => ""))
            continue
        end

        if startswith(line, "فعل_قولي ")
            _parse_speech_act!(space, replace(line, r"^فعل_قولي\s+" => ""))
            continue
        end

        if startswith(line, "استثناء {")
            m = match(r"^استثناء\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_exception_rule!(space, m.captures[1]); continue)
        end

        if startswith(line, "استثناء :")
            _parse_exception_rule!(space, replace(line, r"^استثناء\s*:\s*" => ""))
            continue
        end

        if startswith(line, "استثناء ")
            _parse_exception_rule!(space, replace(line, r"^استثناء\s+" => ""))
            continue
        end

        if startswith(line, "نمط مشروط ")
            m = match(r"^نمط\s+مشروط\s+(.+?)\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_template!(space, strip(m.captures[1]), m.captures[2]); continue)
        end

        if startswith(line, "نمط ")
            m = match(r"^نمط\s+(.+?)\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_template!(space, strip(m.captures[1]), m.captures[2]); continue)
        end

        if startswith(line, "عملية ")
            m = match(r"^عملية\s+(.+?)(?:\s*<\s*(.+?))?\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_process!(space, strip(m.captures[1]), m.captures[2], m.captures[3]); continue)
        end

        if startswith(line, "سلسلة ")
            m = match(r"^سلسلة\s+(.+?)\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_event_chain!(space, strip(m.captures[1]), m.captures[2]); continue)
        end

        if startswith(line, "شيء ")
            m = match(r"^شيء\s+(.+?)\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_entity!(space, strip(m.captures[1]), m.captures[2]); continue)
        end

        m_entity = match(r"^(.+?)\s*:\s*كائن\s*\{(.*)\}\s*$", line)
        if m_entity !== nothing
            _parse_entity!(space, strip(m_entity.captures[1]), m_entity.captures[2])
            continue
        end

        if startswith(line, "فعل ")
            m = match(r"^فعل\s+(.+?)\s*\{(.*)\}\s*$", line)
            m !== nothing && (_parse_verb!(space, strip(m.captures[1]), m.captures[2]); continue)
        end

        m_verb = match(r"^(.+?)\s*:\s*فعل\s*\{(.*)\}\s*$", line)
        if m_verb !== nothing
            _parse_verb!(space, strip(m_verb.captures[1]), m_verb.captures[2])
            continue
        end

        m_typed_entity = match(r"^(.+?)\s*:\s*(.+?)\s*\{(.*)\}\s*$", line)
        if m_typed_entity !== nothing
            _parse_typed_entity!(space, strip(m_typed_entity.captures[1]),
                                 strip(m_typed_entity.captures[2]), m_typed_entity.captures[3])
            continue
        end

        if startswith(line, "قاعدة")
            body = strip(replace(line, r"^قاعدة\s*:?" => ""))
            _parse_rule!(space, body)
            continue
        end

        if startswith(line, "فكرة")
            body = strip(replace(line, r"^فكرة\s*:?" => ""))
            _parse_idea!(space, body)
            continue
        end

        if startswith(line, "خاصية ")
            _parse_property!(space, replace(line, r"^خاصية\s+" => ""))
            continue
        end
    end
    return space
end

function _clean_free_entity(token::AbstractString)
    clean = strip(String(token), ['،', ',', '.', '؛', ';', ':', '؟', '?', '!', '"', '\''])
    clean = replace(clean, r"^(?:ال|بال|كال|وال|فال|لل|ل|ب|ف|و)" => "")
    isempty(clean) && return String(strip(String(token)))
    return clean
end

function _causal_property_from_phrase(phrase::AbstractString)
    text = strip(String(phrase))
    occursin("غلي", text) && return "boiling_point", "درجة_الغليان"
    occursin("حرار", text) && return "temperature", "درجة_الحرارة"
    occursin("ضغط", text) && return "pressure", "الضغط"
    occursin("كثاف", text) && return "density", "الكثافة"
    occursin("سرع", text) && return "motion", "السرعة"
    occursin("خوف", text) && return "fear", "الخوف"
    (occursin("استقرار", text) || occursin("ثبات", text)) && return "stability", "الاستقرار"
    return property_key(text), replace(text, r"\s+" => "_")
end

function _register_increase_template!(space::SimulationSpace, source::AbstractString,
                                      action::AbstractString, target::AbstractString,
                                      property_phrase::AbstractString;
                                      confidence::Real=0.72)
    src = _clean_free_entity(source)
    tgt = _clean_free_entity(target)
    act = strip(String(action))
    prop_key, prop_name = _causal_property_from_phrase(property_phrase)
    result_action = "رفع_$(prop_name)"
    result_state = "زيادة_$(prop_name)"
    haskey(space.entities, src) || register_entity!(space, Thing(src))
    haskey(space.entities, tgt) || register_entity!(space, Thing(tgt))
    haskey(space.dynamic_verbs, result_action) ||
        register_verb!(space, DynamicVerb(result_action, [PropertyEffect(prop_key, 2.0, "self")]))
    template_name = "$(act)_$(src)_يرفع_$(prop_name)"
    template = CausalTemplate(template_name, "auto_text", "", act, "", "",
        "الهدف", result_action, "الهدف", "تأثير_خاصية", result_state, "",
        Float64(confidence), 1)
    register_template!(space, template)
    record_frame!(space, CausalFrame([src, tgt], act,
        "result=$(result_state) | property=$(prop_key) | confidence=$(Float64(confidence))"))
    return true
end

function _clean_dialogue_marker(line::AbstractString)
    text = strip(String(line))
    text = replace(text, r"^\s*(?:سؤال|س|جواب|ج)\s*[:：]\s*" => "")
    return strip(text)
end

function _dialogue_marker_kind(line::AbstractString)
    text = strip(String(line))
    occursin(r"^\s*(?:سؤال|س)\s*[:：]", text) && return :question
    occursin(r"^\s*(?:جواب|ج)\s*[:：]", text) && return :answer
    return :none
end

function _speech_act_type(text::AbstractString)
    value = lowercase(strip(String(text)))
    isempty(value) && return "قول"
    (startswith(value, "السلام عليكم") || startswith(value, "مرحبا") ||
     startswith(value, "أهلا") || startswith(value, "اهلا")) && return "تحية"
    startswith(value, "كيف حالك") && return "سؤال_حال"
    (endswith(value, "؟") || endswith(value, "?") ||
     any(startswith(value, q) for q in ("ما ", "ماذا ", "كيف ", "لماذا ", "هل ", "أين ", "اين ", "متى "))) && return "سؤال"
    return "قول"
end

function _response_act_type(source_type::AbstractString, response::AbstractString)
    src = String(source_type)
    value = lowercase(strip(String(response)))
    src == "تحية" && return "رد_تحية"
    src == "سؤال_حال" && return "جواب_حال"
    src == "سؤال" && return "جواب"
    startswith(value, "نعم") && return "إقرار"
    startswith(value, "لا") && return "نفي"
    return "رد_قولي"
end

function _record_speech_pair!(space::SimulationSpace, source::AbstractString, response::AbstractString;
                              evidence::AbstractString="dialogue_training")
    left = _clean_dialogue_marker(source)
    right = _clean_dialogue_marker(response)
    (isempty(left) || isempty(right)) && return false
    (length(left) > 300 || length(right) > 300) && return false
    act = _speech_act_type(left)
    response_act = _response_act_type(act, right)
    assert_speech_act!(
        space,
        "س",
        act,
        left;
        responder="ج",
        response_act=response_act,
        response_content=right,
        confidence=0.78,
        evidence=evidence,
    )
    record_frame!(space, CausalFrame(
        ["س", "ج"],
        act,
        "speech_content=$(left) | response_act=$(response_act) | response_content=$(right)",
    ))
    return true
end

function _learn_dialogue_pairs!(space::SimulationSpace, text::AbstractString)
    learned = 0
    inline_text = replace(String(text), r"\s+" => " ")
    inline_pair_re = r"(?:سؤال|س)\s*[:：]\s*(.+?)\s*(?:جواب|ج)\s*[:：]\s*(.+?)(?=\s+(?:سؤال|س)\s*[:：]|$)"
    for m in eachmatch(inline_pair_re, inline_text)
        if length(m.captures) == 2 && _record_speech_pair!(space, m.captures[1], m.captures[2]; evidence="dialogue_inline_marker")
            learned += 1
        end
    end

    pending_question = ""
    previous_dialogue_line = ""
    for raw_line in split(String(text), '\n')
        line = strip(raw_line)
        isempty(line) && continue
        length(line) > 400 && !occursin('\t', line) && _dialogue_marker_kind(line) == :none && continue

        if occursin('\t', line)
            parts = split(line, '\t'; limit=2)
            if length(parts) == 2 && _record_speech_pair!(space, parts[1], parts[2]; evidence="dialogue_tab")
                learned += 1
            end
            pending_question = ""
            previous_dialogue_line = ""
            continue
        end

        kind = _dialogue_marker_kind(line)
        if kind == :question
            pending_question = line
            previous_dialogue_line = line
            continue
        elseif kind == :answer
            if !isempty(pending_question) && _record_speech_pair!(space, pending_question, line; evidence="dialogue_marker")
                learned += 1
            end
            pending_question = ""
            previous_dialogue_line = ""
            continue
        end

        if !isempty(previous_dialogue_line) && (_speech_act_type(previous_dialogue_line) in ("تحية", "سؤال_حال", "سؤال"))
            if _record_speech_pair!(space, previous_dialogue_line, line; evidence="dialogue_adjacent")
                learned += 1
            end
            pending_question = ""
            previous_dialogue_line = ""
            continue
        end

        if _speech_act_type(line) in ("تحية", "سؤال_حال", "سؤال")
            previous_dialogue_line = line
        else
            previous_dialogue_line = ""
        end
    end
    return learned
end

function train_from_text!(space::SimulationSpace, text::AbstractString)
    _learn_dialogue_pairs!(space, text)

    for raw_sentence in split(String(text), r"[\n؟?؛;]+|(?<!\d)[.!]+|[.!]+(?!\d)")
        sentence = strip(raw_sentence)
        isempty(sentence) && continue

        m_add_effect = match(r"^(?:إن\s+|ان\s+)?(?:إضافة|اضافة|إضافه|اضافه|أضاف|اضاف)\s+(?:كمية\s+من\s+|كميه\s+من\s+|شيء\s+من\s+)?(\S+)\s+(?:إلى|الى|الي|على|في)\s+(\S+).*?(?:تزيد|يزيد|زاد|يزداد|ترفع|يرفع|ارتفع|ترتفع)\s+(?:من\s+)?(.+?)$", sentence)
        if m_add_effect !== nothing
            src = strip(m_add_effect.captures[1])
            tgt = strip(m_add_effect.captures[2])
            prop = strip(m_add_effect.captures[3])
            _register_increase_template!(space, src, "إضافة", tgt, prop; confidence=0.78)
            _register_increase_template!(space, src, "اضافة", tgt, prop; confidence=0.78)
            continue
        end

        if occursin("إذا", sentence) && occursin("زاد", sentence) && occursin("عن", sentence)
            words = split(sentence)
            idx_zad = findfirst(==("زاد"), words)
            idx_an = findfirst(==("عن"), words)
            if idx_zad !== nothing && idx_an !== nothing && idx_zad + 1 < idx_an && idx_an + 2 <= length(words)
                prop = property_key(words[idx_zad + 1])
                threshold = _parse_number(words[idx_an + 1], 0.5)
                action = String(words[idx_an + 2])
                add_rule!(space, CausalRule(prop, :>, threshold, action))
                continue
            end
        end

        m_effect = match(r"^(\S+)\s+(\S+)\s+على\s+(\S+)\s+يرفع\s+(\S+)$", sentence)
        if m_effect !== nothing
            action = String(m_effect.captures[1])
            prop = property_key(m_effect.captures[4])
            register_verb!(space, DynamicVerb(action, [PropertyEffect(prop, 5.0, "self")]))
            continue
        end

        m_event = match(r"^(\S+)\s+(\S+)\s+على\s+(\S+)$", sentence)
        if m_event !== nothing
            action = String(m_event.captures[1])
            source = String(m_event.captures[2])
            target = String(m_event.captures[3])
            for name in (source, target)
                haskey(space.entities, name) || register_entity!(space, Thing(name))
            end
            evaluate_idea!(space, source, action, target)
            continue
        end

        _infer_implicit_state!(space, sentence) && continue

        words = split(sentence)
        if length(words) >= 2
            attrs = Dict{String,Any}()
            occursin("قوي", sentence) && (attrs["energy"] = 0.8)
            occursin("طاقة", sentence) && (attrs["energy"] = 0.9)
            occursin("خائف", sentence) && (attrs["fear"] = 0.4)
            occursin("سريع", sentence) && (attrs["motion"] = 0.7)
            classes = _sentence_classes(space, words[2:end])
            _register_or_classify_entity!(space, words[1], attrs, classes)
        end
    end
    return space
end

end # module DSL
