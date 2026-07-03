module CausalSpace

using ..Entities: AqlEntity, Thing, Lion, Gazelle, Earth, Furniture,
                  get_property, set_property!, get_attribute, set_attribute!, get_action
using ..Operators: roar!, shake!, run!, fall!
using ..Taxonomy: ConceptTaxonomy, default_aql_taxonomy, apply_taxonomy!
import ..Taxonomy: register_class!, add_subclass!, assign_class!, assigned_classes,
                   is_a, known_classes

export SimulationSpace, CausalFrame, PropertyEffect, DynamicVerb, CausalRule,
       CausalTemplate, ProcessStep, ProcessConcept, Circumstance, TemporalRelation,
       ChainStep, EventChain, RelationFact, SemanticRelation, QuantifiedFact, ComparisonFact,
       MetaphorFact, IntentFact, SpeechActFact, ExceptionRule,
       AqlGovernanceRule, RejectionLesson, CorpusAnnotation,
       register_entity!, register_verb!, add_rule!, apply_verb!, evaluate_idea!,
       interact!, record_frame!, step_simulation!, check_triggers!,
       register_class!, add_subclass!, assign_class!, assigned_classes, is_a, known_classes,
       register_template!, matching_templates, infer_event!,
       register_process!, get_process, instantiate_process!,
       register_opposite!, opposite_of, apply_signed_action!,
       set_circumstance!, get_circumstance, set_location!, set_time!,
       relate_events!, register_event_chain!, run_event_chain!,
       assert_relation!, relations_from, relations_to, relations_between,
       assert_quantified_fact!, quantified_facts_for,
       assert_comparison!, comparisons_for, compare_entities!,
       assert_metaphor!, metaphors_for,
       assert_intent!, intents_for,
       assert_speech_act!, speech_acts_for, speech_responses_for,
       add_exception_rule!, exceptions_for, active_exception,
       curate_rule!, propose_rule!, approve_rule!, reject_rule!,
       learn_rejection_lesson!, score_rule_proposal,
       annotate_corpus_sentence!, critical_corpus_pass!,
       aql_memory_audit, annotation_weight,
       aql_inhibition_score, aql_inhibition_reason

struct CausalFrame
    things::Vector{String}
    event::String
    result::String
end

struct PropertyEffect
    target_property::String
    scale_factor::Float64
    source_property::String
end

struct DynamicVerb
    name::String
    effects::Vector{PropertyEffect}
end

struct CausalRule
    property::String
    operator::Symbol
    threshold::Float64
    action_name::String
end

struct CausalTemplate
    name::String
    domain::String
    source_class::String
    action::String
    target_class::String
    action_kind::String
    result_actor::String
    result_action::String
    result_target::String
    result_kind::String
    result_state::String
    condition::String
    confidence::Float64
    polarity::Int
end

struct ProcessStep
    actor_role::String
    action::String
    target_role::String
    result::String
end

struct ProcessConcept
    name::String
    parent::String
    domain::String
    intensity::Float64
    steps::Vector{ProcessStep}
    attributes::Dict{String,Any}
end

struct Circumstance
    location::String
    time::String
end

struct TemporalRelation
    first::String
    relation::String
    second::String
end

struct ChainStep
    actor_role::String
    action::String
    target_role::String
    result::String
end

struct EventChain
    name::String
    source_class::String
    action::String
    target_class::String
    steps::Vector{ChainStep}
    confidence::Float64
end

struct RelationFact
    source::String
    relation::String
    target::String
    confidence::Float64
    evidence::String
end

const SemanticRelation = RelationFact

struct QuantifiedFact
    quantifier::String
    subject::String
    predicate::String
    object::String
    polarity::Int
    confidence::Float64
    evidence::String
end

struct ComparisonFact
    left::String
    property::String
    comparator::String
    right::String
    confidence::Float64
    evidence::String
end

struct MetaphorFact
    expression::String
    source_domain::String
    target_domain::String
    literal_subject::String
    borrowed_actor::String
    action::String
    transferred_property::String
    confidence::Float64
    evidence::String
end

struct IntentFact
    actor::String
    action::String
    target::String
    intent::String
    goal::String
    actual_result::String
    confidence::Float64
    evidence::String
end

struct SpeechActFact
    speaker::String
    act_type::String
    content::String
    responder::String
    response_act::String
    response_content::String
    confidence::Float64
    evidence::String
end

struct ExceptionRule
    rule_name::String
    condition::String
    exception::String
    priority::Int
    confidence::Float64
    evidence::String
end

struct AqlGovernanceRule
    id::String
    subject::String
    predicate::String
    object::String
    condition::String
    polarity::Int
    priority::Int
    confidence::Float64
    source::String
    status::String
    rationale::String
    tags::Vector{String}
end

struct RejectionLesson
    id::String
    reason::String
    pattern::String
    penalty_tags::Vector{String}
    confidence::Float64
    evidence::String
end

struct CorpusAnnotation
    sentence_id::String
    span::String
    tag::String
    confidence::Float64
    reason::String
    effect_on_k_sem::Float64
    effect_on_k_causal::Float64
    effect_on_al_aql::String
end

mutable struct SimulationSpace
    entities::Dict{String,AqlEntity}
    dynamic_verbs::Dict{String,DynamicVerb}
    rules::Vector{CausalRule}
    log::Vector{String}
    in_simulation::Bool
    taxonomy::ConceptTaxonomy
    templates::Vector{CausalTemplate}
    processes::Dict{String,ProcessConcept}
    opposites::Dict{String,String}
    circumstances::Dict{String,Circumstance}
    temporal_relations::Vector{TemporalRelation}
    event_chains::Dict{String,EventChain}
    semantic_relations::Vector{RelationFact}
    quantified_facts::Vector{QuantifiedFact}
    comparisons::Vector{ComparisonFact}
    metaphors::Vector{MetaphorFact}
    intents::Vector{IntentFact}
    speech_acts::Vector{SpeechActFact}
    exception_rules::Vector{ExceptionRule}
    curated_rules::Dict{String,AqlGovernanceRule}
    proposed_rules::Dict{String,AqlGovernanceRule}
    rejected_rules::Dict{String,AqlGovernanceRule}
    rejection_lessons::Vector{RejectionLesson}
    corpus_annotations::Vector{CorpusAnnotation}

    SimulationSpace() = new(Dict{String,AqlEntity}(), Dict{String,DynamicVerb}(),
                             CausalRule[], String[], false, default_aql_taxonomy(),
                             CausalTemplate[], Dict{String,ProcessConcept}(),
                             Dict{String,String}(), Dict{String,Circumstance}(),
                             TemporalRelation[], Dict{String,EventChain}(),
                             RelationFact[], QuantifiedFact[], ComparisonFact[],
                             MetaphorFact[], IntentFact[], SpeechActFact[], ExceptionRule[],
                             Dict{String,AqlGovernanceRule}(),
                             Dict{String,AqlGovernanceRule}(),
                             Dict{String,AqlGovernanceRule}(),
                             RejectionLesson[], CorpusAnnotation[])
end

function record_frame!(space::SimulationSpace, frame::CausalFrame)
    push!(space.log, "فكرة: الأشياء=$(join(frame.things, ", ")) | الحدث=$(frame.event) | النتيجة=$(frame.result)")
    return frame
end

function register_entity!(space::SimulationSpace, e::AqlEntity; classes=String[])
    for class_name in classes
        assign_class!(space.taxonomy, e, class_name; apply=false)
    end
    apply_taxonomy!(space.taxonomy, e)
    space.entities[e.name] = e
    haskey(space.circumstances, e.name) || (space.circumstances[e.name] = Circumstance("غير_محدد", "غير_محدد"))
    push!(space.log, "تسجيل كيان: [$(e.name)]")
    return e
end

function set_circumstance!(space::SimulationSpace, entity_name::AbstractString;
                           location::AbstractString="",
                           time::AbstractString="")
    name = String(entity_name)
    current = get(space.circumstances, name, Circumstance("غير_محدد", "غير_محدد"))
    new_location = isempty(String(location)) ? current.location : String(location)
    new_time = isempty(String(time)) ? current.time : String(time)
    space.circumstances[name] = Circumstance(new_location, new_time)
    if haskey(space.entities, name) && space.entities[name] isa Thing
        set_attribute!(space.entities[name], "location", new_location)
        set_attribute!(space.entities[name], "time", new_time)
    end
    push!(space.log, "circumstance: [$name] location=[$new_location] time=[$new_time]")
    return space.circumstances[name]
end

get_circumstance(space::SimulationSpace, entity_name::AbstractString) =
    get(space.circumstances, String(entity_name), Circumstance("غير_محدد", "غير_محدد"))

set_location!(space::SimulationSpace, entity_name::AbstractString, location::AbstractString) =
    set_circumstance!(space, entity_name; location=location)

set_time!(space::SimulationSpace, entity_name::AbstractString, time::AbstractString) =
    set_circumstance!(space, entity_name; time=time)

function relate_events!(space::SimulationSpace, first::AbstractString,
                        relation::AbstractString, second::AbstractString)
    rel = TemporalRelation(String(first), String(relation), String(second))
    push!(space.temporal_relations, rel)
    push!(space.log, "temporal relation: [$(rel.first)] $(rel.relation) [$(rel.second)]")
    return rel
end

function assert_relation!(space::SimulationSpace, source::AbstractString,
                          relation::AbstractString, target::AbstractString;
                          confidence::Real=1.0, evidence::AbstractString="")
    src = String(source)
    relname = String(relation)
    tgt = String(target)
    haskey(space.entities, src) || register_entity!(space, Thing(src))
    haskey(space.entities, tgt) || register_entity!(space, Thing(tgt))
    rel = RelationFact(src, relname, tgt, Float64(confidence), String(evidence))
    push!(space.semantic_relations, rel)
    push!(space.log, "semantic relation: [$src] --$relname--> [$tgt] confidence=$(rel.confidence)")
    return rel
end

relations_from(space::SimulationSpace, source::AbstractString) =
    [rel for rel in space.semantic_relations if rel.source == String(source)]

relations_to(space::SimulationSpace, target::AbstractString) =
    [rel for rel in space.semantic_relations if rel.target == String(target)]

function relations_between(space::SimulationSpace, a::AbstractString, b::AbstractString)
    left, right = String(a), String(b)
    return [rel for rel in space.semantic_relations
            if (rel.source == left && rel.target == right) ||
               (rel.source == right && rel.target == left)]
end

function assert_quantified_fact!(space::SimulationSpace, quantifier::AbstractString,
                                 subject::AbstractString, predicate::AbstractString,
                                 object::AbstractString="";
                                 polarity::Integer=1, confidence::Real=1.0,
                                 evidence::AbstractString="")
    fact = QuantifiedFact(String(quantifier), String(subject), String(predicate), String(object),
                          Int(polarity) < 0 ? -1 : 1, Float64(confidence), String(evidence))
    push!(space.quantified_facts, fact)
    push!(space.log, "quantified fact: $(fact.quantifier) [$(fact.subject)] $(fact.predicate) [$(fact.object)] polarity=$(fact.polarity)")
    return fact
end

quantified_facts_for(space::SimulationSpace, subject::AbstractString) =
    [fact for fact in space.quantified_facts if fact.subject == String(subject)]

function assert_comparison!(space::SimulationSpace, left::AbstractString,
                            property::AbstractString, comparator::AbstractString,
                            right::AbstractString; confidence::Real=1.0,
                            evidence::AbstractString="")
    l = String(left)
    r = String(right)
    haskey(space.entities, l) || register_entity!(space, Thing(l))
    haskey(space.entities, r) || register_entity!(space, Thing(r))
    fact = ComparisonFact(l, String(property), String(comparator), r,
                          Float64(confidence), String(evidence))
    push!(space.comparisons, fact)
    push!(space.log, "comparison: [$l] $(fact.property) $(fact.comparator) [$r] confidence=$(fact.confidence)")
    return fact
end

function comparisons_for(space::SimulationSpace, entity_name::AbstractString)
    name = String(entity_name)
    return [fact for fact in space.comparisons if fact.left == name || fact.right == name]
end

function _numeric_value(entity::AqlEntity, property::AbstractString)
    prop = String(property)
    prop in ("mass", "كتلة") && hasproperty(entity, :mass) && return Float64(getfield(entity, :mass))
    amp, _ = get_property(entity, prop)
    amp > 0.0 && return amp
    if entity isa Thing
        value = get_attribute(entity, prop, nothing)
        value isa Number && return Float64(value)
    end
    return nothing
end

function compare_entities!(space::SimulationSpace, left::AbstractString,
                           right::AbstractString, property::AbstractString;
                           confidence::Real=1.0)
    l, r = String(left), String(right)
    haskey(space.entities, l) || return nothing
    haskey(space.entities, r) || return nothing
    left_value = _numeric_value(space.entities[l], property)
    right_value = _numeric_value(space.entities[r], property)
    (left_value === nothing || right_value === nothing) && return nothing
    comparator = left_value > right_value ? "أكبر" : left_value < right_value ? "أصغر" : "يساوي"
    return assert_comparison!(space, l, property, comparator, r;
                              confidence=confidence, evidence="computed")
end

function assert_metaphor!(space::SimulationSpace, expression::AbstractString;
                          source_domain::AbstractString="",
                          target_domain::AbstractString="",
                          literal_subject::AbstractString="",
                          borrowed_actor::AbstractString="",
                          action::AbstractString="",
                          transferred_property::AbstractString="",
                          confidence::Real=1.0,
                          evidence::AbstractString="")
    fact = MetaphorFact(String(expression), String(source_domain), String(target_domain),
                        String(literal_subject), String(borrowed_actor), String(action),
                        String(transferred_property), Float64(confidence), String(evidence))
    push!(space.metaphors, fact)
    push!(space.log, "metaphor: [$(fact.expression)] source_domain=$(fact.source_domain) target_domain=$(fact.target_domain)")
    return fact
end

function metaphors_for(space::SimulationSpace, name::AbstractString)
    key = String(name)
    return [fact for fact in space.metaphors
            if fact.literal_subject == key || fact.borrowed_actor == key ||
               occursin(key, fact.expression)]
end

function assert_intent!(space::SimulationSpace, actor::AbstractString, action::AbstractString;
                        target::AbstractString="", intent::AbstractString="",
                        goal::AbstractString="", actual_result::AbstractString="",
                        confidence::Real=1.0, evidence::AbstractString="")
    a = String(actor)
    t = String(target)
    haskey(space.entities, a) || register_entity!(space, Thing(a))
    !isempty(t) && !haskey(space.entities, t) && register_entity!(space, Thing(t))
    fact = IntentFact(a, String(action), t, String(intent), String(goal),
                      String(actual_result), Float64(confidence), String(evidence))
    push!(space.intents, fact)
    push!(space.log, "intent: [$a] action=$(fact.action) goal=$(fact.goal) actual=$(fact.actual_result)")
    return fact
end

intents_for(space::SimulationSpace, actor::AbstractString) =
    [fact for fact in space.intents if fact.actor == String(actor)]

function assert_speech_act!(space::SimulationSpace, speaker::AbstractString,
                            act_type::AbstractString, content::AbstractString;
                            responder::AbstractString="",
                            response_act::AbstractString="",
                            response_content::AbstractString="",
                            confidence::Real=1.0,
                            evidence::AbstractString="")
    s = String(strip(String(speaker)))
    r = String(strip(String(responder)))
    isempty(s) && (s = "speaker")
    haskey(space.entities, s) || register_entity!(space, Thing(s))
    !isempty(r) && !haskey(space.entities, r) && register_entity!(space, Thing(r))
    fact = SpeechActFact(
        s,
        String(strip(String(act_type))),
        String(strip(String(content))),
        r,
        String(strip(String(response_act))),
        String(strip(String(response_content))),
        Float64(confidence),
        String(evidence),
    )
    push!(space.speech_acts, fact)
    push!(space.log, "speech_act: [$s] act=$(fact.act_type) content=$(fact.content) responder=$(fact.responder) response=$(fact.response_act)")
    return fact
end

speech_acts_for(space::SimulationSpace, speaker::AbstractString) =
    [fact for fact in space.speech_acts if fact.speaker == String(speaker)]

function speech_responses_for(space::SimulationSpace, content::AbstractString)
    key = _aql_norm_text(content)
    isempty(key) && return SpeechActFact[]
    return [fact for fact in space.speech_acts
            if _aql_text_has(fact.content, key) || _aql_text_has(key, fact.content)]
end

function add_exception_rule!(space::SimulationSpace, rule_name::AbstractString,
                             condition::AbstractString, exception::AbstractString;
                             priority::Integer=1, confidence::Real=1.0,
                             evidence::AbstractString="")
    fact = ExceptionRule(String(rule_name), String(condition), String(exception),
                         Int(priority), Float64(confidence), String(evidence))
    push!(space.exception_rules, fact)
    sort!(space.exception_rules; by=r -> r.priority, rev=true)
    push!(space.log, "exception rule: [$(fact.rule_name)] except=$(fact.exception) priority=$(fact.priority)")
    return fact
end

exceptions_for(space::SimulationSpace, rule_name::AbstractString) =
    [rule for rule in space.exception_rules if rule.rule_name == String(rule_name)]

function active_exception(space::SimulationSpace, rule_name::AbstractString)
    matches = exceptions_for(space, rule_name)
    isempty(matches) && return nothing
    return first(matches)
end

function _aql_norm_text(text::AbstractString)
    s = lowercase(strip(String(text)))
    s = replace(s, r"[\t\r\n,.;:!?()\[\]{}\"']+" => " ")
    return strip(replace(s, r"\s+" => " "))
end

function _aql_text_has(text::AbstractString, needle::AbstractString)
    value = _aql_norm_text(needle)
    isempty(value) && return false
    hay = _aql_norm_text(text)
    occursin(value, hay) && return true
    parts = split(value)
    isempty(parts) && return false
    hay_parts = Set(split(hay))
    return all(p -> p in hay_parts, parts)
end

function _next_governance_id(space::SimulationSpace, prefix::AbstractString)
    n = length(space.curated_rules) + length(space.proposed_rules) +
        length(space.rejected_rules) + length(space.rejection_lessons) +
        length(space.corpus_annotations) + 1
    base = String(prefix)
    id = "$(base)$(n)"
    while haskey(space.curated_rules, id) || haskey(space.proposed_rules, id) ||
          haskey(space.rejected_rules, id)
        n += 1
        id = "$(base)$(n)"
    end
    return id
end

function _governance_rule(id::AbstractString, subject::AbstractString,
                          predicate::AbstractString, object::AbstractString;
                          condition::AbstractString="", polarity::Integer=1,
                          priority::Integer=0, confidence::Real=1.0,
                          source::AbstractString="", status::AbstractString="proposed",
                          rationale::AbstractString="", tags::Vector{String}=String[])
    return AqlGovernanceRule(String(id), String(subject), String(predicate), String(object),
                             String(condition), Int(polarity) < 0 ? -1 : 1,
                             Int(priority), clamp(Float64(confidence), 0.0, 1.0),
                             String(source), String(status), String(rationale), copy(tags))
end

function _rule_text(subject::AbstractString, predicate::AbstractString, object::AbstractString,
                    condition::AbstractString="", tags::Vector{String}=String[])
    return join(filter(!isempty, String[String(subject), String(predicate), String(object),
                                      String(condition), tags...]), " ")
end

function score_rule_proposal(space::SimulationSpace, subject::AbstractString,
                             predicate::AbstractString, object::AbstractString;
                             condition::AbstractString="", tags::Vector{String}=String[])
    text = _rule_text(subject, predicate, object, condition, tags)
    tagset = Set(_aql_norm_text.(tags))
    score = 1.0
    for lesson in space.rejection_lessons
        pattern_hit = !isempty(lesson.pattern) && _aql_text_has(text, lesson.pattern)
        tag_hit = any(t -> (_aql_norm_text(t) in tagset) || _aql_text_has(text, t),
                      lesson.penalty_tags)
        reason_hit = !isempty(lesson.reason) && _aql_text_has(text, lesson.reason)
        if pattern_hit || tag_hit || reason_hit
            score *= clamp(1.0 - 0.75 * lesson.confidence, 0.05, 1.0)
        end
    end
    return clamp(score, 0.0, 1.0)
end

function curate_rule!(space::SimulationSpace, subject::AbstractString,
                      predicate::AbstractString, object::AbstractString;
                      condition::AbstractString="", polarity::Integer=1,
                      priority::Integer=100, confidence::Real=1.0,
                      source::AbstractString="curated", rationale::AbstractString="",
                      tags::Vector{String}=String[], id::AbstractString="")
    rid = isempty(String(id)) ? _next_governance_id(space, "curated_") : String(id)
    rule = _governance_rule(rid, subject, predicate, object;
                            condition=condition, polarity=polarity, priority=priority,
                            confidence=confidence, source=source, status="curated",
                            rationale=rationale, tags=tags)
    space.curated_rules[rid] = rule
    delete!(space.proposed_rules, rid)
    delete!(space.rejected_rules, rid)
    if !isempty(rule.subject) && !isempty(rule.predicate) && !isempty(rule.object)
        assert_relation!(space, rule.subject, rule.predicate, rule.object;
                         confidence=rule.confidence, evidence="curated_rule")
    end
    push!(space.log, "aql curated rule: [$(rule.id)] $(rule.subject) $(rule.predicate) $(rule.object)")
    return rule
end

function propose_rule!(space::SimulationSpace, subject::AbstractString,
                       predicate::AbstractString, object::AbstractString;
                       condition::AbstractString="", polarity::Integer=1,
                       priority::Integer=10, confidence::Real=0.5,
                       source::AbstractString="proposed", rationale::AbstractString="",
                       tags::Vector{String}=String[], id::AbstractString="")
    rid = isempty(String(id)) ? _next_governance_id(space, "proposed_") : String(id)
    gate = score_rule_proposal(space, subject, predicate, object;
                               condition=condition, tags=tags)
    adjusted = clamp(Float64(confidence) * gate, 0.0, 1.0)
    if gate < 0.35
        rule = _governance_rule(rid, subject, predicate, object;
                                condition=condition, polarity=polarity, priority=priority,
                                confidence=adjusted, source=source, status="rejected",
                                rationale=isempty(rationale) ? "auto_rejected_by_rejection_lessons" :
                                          "auto_rejected_by_rejection_lessons | $(rationale)",
                                tags=tags)
        space.rejected_rules[rid] = rule
        push!(space.log, "aql proposed rule auto-rejected: [$(rule.id)] confidence=$(rule.confidence)")
        return rule
    end
    rule = _governance_rule(rid, subject, predicate, object;
                            condition=condition, polarity=polarity, priority=priority,
                            confidence=adjusted, source=source, status="proposed",
                            rationale=rationale, tags=tags)
    space.proposed_rules[rid] = rule
    push!(space.log, "aql proposed rule: [$(rule.id)] confidence=$(rule.confidence)")
    return rule
end

function approve_rule!(space::SimulationSpace, id::AbstractString; priority::Union{Nothing,Integer}=nothing)
    rid = String(id)
    haskey(space.proposed_rules, rid) || return nothing
    rule = space.proposed_rules[rid]
    delete!(space.proposed_rules, rid)
    return curate_rule!(space, rule.subject, rule.predicate, rule.object;
                        condition=rule.condition, polarity=rule.polarity,
                        priority=priority === nothing ? max(rule.priority, 100) : Int(priority),
                        confidence=rule.confidence, source=rule.source,
                        rationale=rule.rationale, tags=rule.tags, id=rule.id)
end

function learn_rejection_lesson!(space::SimulationSpace, reason::AbstractString;
                                 pattern::AbstractString="",
                                 penalty_tags::Vector{String}=String[],
                                 confidence::Real=0.8,
                                 evidence::AbstractString="")
    id = _next_governance_id(space, "lesson_")
    lesson = RejectionLesson(id, String(reason), String(pattern), copy(penalty_tags),
                             clamp(Float64(confidence), 0.0, 1.0), String(evidence))
    push!(space.rejection_lessons, lesson)
    push!(space.log, "aql rejection lesson: [$(lesson.id)] reason=$(lesson.reason)")
    return lesson
end

function reject_rule!(space::SimulationSpace, id::AbstractString;
                      reason::AbstractString="", lesson_pattern::AbstractString="",
                      penalty_tags::Vector{String}=String[],
                      confidence::Real=0.8)
    rid = String(id)
    rule = get(space.proposed_rules, rid, get(space.curated_rules, rid, nothing))
    rule === nothing && return nothing
    delete!(space.proposed_rules, rid)
    delete!(space.curated_rules, rid)
    rationale = isempty(String(reason)) ? rule.rationale : String(reason)
    rejected = _governance_rule(rule.id, rule.subject, rule.predicate, rule.object;
                                condition=rule.condition, polarity=rule.polarity,
                                priority=rule.priority, confidence=rule.confidence,
                                source=rule.source, status="rejected",
                                rationale=rationale, tags=rule.tags)
    space.rejected_rules[rid] = rejected
    if !isempty(String(reason))
        learn_rejection_lesson!(space, reason;
                                pattern=isempty(String(lesson_pattern)) ? rule.predicate : lesson_pattern,
                                penalty_tags=isempty(penalty_tags) ? rule.tags : penalty_tags,
                                confidence=confidence,
                                evidence="rejected_rule:$(rule.id)")
    end
    push!(space.log, "aql rejected rule: [$(rule.id)] reason=$(rationale)")
    return rejected
end

function annotate_corpus_sentence!(space::SimulationSpace, sentence_id::AbstractString,
                                   span::AbstractString, tag::AbstractString;
                                   confidence::Real=1.0, reason::AbstractString="",
                                   effect_on_k_sem::Real=1.0,
                                   effect_on_k_causal::Real=1.0,
                                   effect_on_al_aql::AbstractString="none")
    annotation = CorpusAnnotation(String(sentence_id), String(span), String(tag),
                                  clamp(Float64(confidence), 0.0, 1.0),
                                  String(reason), Float64(effect_on_k_sem),
                                  Float64(effect_on_k_causal), String(effect_on_al_aql))
    push!(space.corpus_annotations, annotation)
    push!(space.log, "aql corpus annotation: [$(annotation.sentence_id)] tag=$(annotation.tag)")
    return annotation
end

const _CRITICAL_CAUSAL_MARKERS = ("because", "therefore", "causes", "cause", "leads to",
                                  "results in", "if", "then", "لان", "لأن", "فإن", "فان",
                                  "اذا", "إذا", "يسبب", "تسبب", "يؤدي", "تؤدي", "ينتج",
                                  "تزيد", "يزيد", "ترفع", "يرفع")
const _CRITICAL_CORRELATION_MARKERS = ("correlates", "associated", "linked", "يرتبط",
                                       "مرتبطة", "مصاحب", "يتزامن")
const _CRITICAL_METAPHOR_MARKERS = ("metaphor", "as if", "كأن", "كانه", "كأنه",
                                    "ابتسم الصباح", "جيش الليل", "زحف الليل")
const _CRITICAL_CONDITION_MARKERS = ("unless", "except", "requires", "الا اذا", "إلا إذا",
                                     "بشرط", "شرط", "يتطلب")
const _CRITICAL_UNCERTAIN_MARKERS = ("may", "might", "perhaps", "probably", "قد",
                                     "ربما", "لعل", "يحتمل")

function _critical_tags_for_sentence(sentence::AbstractString)
    text = String(sentence)
    tags = String[]
    any(m -> _aql_text_has(text, m), _CRITICAL_CAUSAL_MARKERS) && push!(tags, "causal")
    any(m -> _aql_text_has(text, m), _CRITICAL_CORRELATION_MARKERS) && push!(tags, "correlation")
    any(m -> _aql_text_has(text, m), _CRITICAL_METAPHOR_MARKERS) && push!(tags, "metaphoric")
    any(m -> _aql_text_has(text, m), _CRITICAL_CONDITION_MARKERS) && push!(tags, "requires_condition")
    any(m -> _aql_text_has(text, m), _CRITICAL_UNCERTAIN_MARKERS) && push!(tags, "uncertain")
    return tags
end

function critical_corpus_pass!(space::SimulationSpace, sentences;
                               source::AbstractString="runtime",
                               max_annotations::Int=10_000)
    count = 0
    for (idx, raw) in enumerate(sentences)
        count >= max_annotations && break
        sentence = strip(String(raw))
        isempty(sentence) && continue
        tags = _critical_tags_for_sentence(sentence)
        for lesson in space.rejection_lessons
            if !isempty(lesson.pattern) && _aql_text_has(sentence, lesson.pattern)
                push!(tags, "lesson:$(lesson.reason)")
            end
        end
        unique!(tags)
        for tag in tags
            count >= max_annotations && break
            effect_sem = tag == "metaphoric" ? 0.65 : tag == "correlation" ? 0.80 : 1.0
            effect_causal = tag == "correlation" ? 0.45 :
                            tag == "requires_condition" ? 0.75 : 1.0
            effect_aql = tag in ("causal", "requires_condition") ? "review" :
                         startswith(tag, "lesson:") ? "penalize" : "tag"
            annotate_corpus_sentence!(space, "$(source):$(idx)", sentence, tag;
                                      confidence=0.70, reason="critical_corpus_pass",
                                      effect_on_k_sem=effect_sem,
                                      effect_on_k_causal=effect_causal,
                                      effect_on_al_aql=effect_aql)
            count += 1
        end
    end
    return count
end

function critical_corpus_pass!(space::SimulationSpace, text::AbstractString; kwargs...)
    sentences = split(String(text), r"[\n.!?؛;]+")
    return critical_corpus_pass!(space, sentences; kwargs...)
end

function annotation_weight(space::SimulationSpace, sentence_id::AbstractString;
                           matrix::Symbol=:causal)
    id = String(sentence_id)
    weight = 1.0
    matched = false
    for item in space.corpus_annotations
        item.sentence_id == id || continue
        matched = true
        factor = matrix in (:sem, :semantic, :K_sem) ?
                 item.effect_on_k_sem : item.effect_on_k_causal
        weight *= factor
    end
    matched || return 1.0
    return clamp(weight, 0.0, 4.0)
end

function aql_memory_audit(space::SimulationSpace)
    tag_counts = Dict{String,Int}()
    action_counts = Dict{String,Int}()
    sem_log_weight = 0.0
    causal_log_weight = 0.0
    for item in space.corpus_annotations
        tag_counts[item.tag] = get(tag_counts, item.tag, 0) + 1
        action_counts[item.effect_on_al_aql] = get(action_counts, item.effect_on_al_aql, 0) + 1
        sem_log_weight += log(max(item.effect_on_k_sem, eps(Float64)))
        causal_log_weight += log(max(item.effect_on_k_causal, eps(Float64)))
    end
    n = length(space.corpus_annotations)
    return Dict{String,Any}(
        "annotations" => n,
        "tags" => tag_counts,
        "al_aql_actions" => action_counts,
        "curated_rules" => length(space.curated_rules),
        "proposed_rules" => length(space.proposed_rules),
        "rejected_rules" => length(space.rejected_rules),
        "rejection_lessons" => length(space.rejection_lessons),
        "suggested_k_sem_multiplier" => n == 0 ? 1.0 : clamp(exp(sem_log_weight / n), 0.0, 4.0),
        "suggested_k_causal_multiplier" => n == 0 ? 1.0 : clamp(exp(causal_log_weight / n), 0.0, 4.0),
    )
end

function _rule_matches_prompt(rule::AqlGovernanceRule, prompt::AbstractString)
    return _aql_text_has(prompt, rule.subject) ||
           _aql_text_has(prompt, rule.predicate) ||
           _aql_text_has(prompt, rule.object) ||
           _aql_text_has(prompt, rule.condition)
end

function _rule_matches_candidate(rule::AqlGovernanceRule, candidate::AbstractString)
    return _aql_text_has(candidate, rule.subject) ||
           _aql_text_has(candidate, rule.predicate) ||
           _aql_text_has(candidate, rule.object) ||
           any(t -> _aql_text_has(candidate, t), rule.tags)
end

function aql_inhibition_score(space::SimulationSpace, prompt::AbstractString,
                              candidate::AbstractString)
    score = 0.0
    for rule in values(space.curated_rules)
        if rule.polarity < 0 && _rule_matches_prompt(rule, prompt) &&
           _rule_matches_candidate(rule, candidate)
            score = max(score, clamp(0.65 + 0.35 * rule.confidence, 0.0, 1.0))
        end
    end
    for rule in values(space.rejected_rules)
        if _rule_matches_prompt(rule, prompt) && _rule_matches_candidate(rule, candidate)
            score = max(score, clamp(0.55 + 0.40 * max(rule.confidence, 0.5), 0.0, 1.0))
        end
    end
    for lesson in space.rejection_lessons
        prompt_hit = !isempty(lesson.pattern) && _aql_text_has(prompt, lesson.pattern)
        cand_hit = any(t -> _aql_text_has(candidate, t), lesson.penalty_tags)
        if prompt_hit && cand_hit
            score = max(score, clamp(0.45 + 0.40 * lesson.confidence, 0.0, 0.95))
        end
    end
    return clamp(score, 0.0, 1.0)
end

function aql_inhibition_reason(space::SimulationSpace, prompt::AbstractString,
                               candidate::AbstractString)
    for rule in values(space.curated_rules)
        if rule.polarity < 0 && _rule_matches_prompt(rule, prompt) &&
           _rule_matches_candidate(rule, candidate)
            return "curated_negative_rule:$(rule.id)"
        end
    end
    for rule in values(space.rejected_rules)
        if _rule_matches_prompt(rule, prompt) && _rule_matches_candidate(rule, candidate)
            return "rejected_rule:$(rule.id)"
        end
    end
    for lesson in space.rejection_lessons
        prompt_hit = !isempty(lesson.pattern) && _aql_text_has(prompt, lesson.pattern)
        cand_hit = any(t -> _aql_text_has(candidate, t), lesson.penalty_tags)
        if prompt_hit && cand_hit
            return "rejection_lesson:$(lesson.id)"
        end
    end
    return ""
end

function register_class!(space::SimulationSpace, name::AbstractString; kwargs...)
    cls = register_class!(space.taxonomy, name; kwargs...)
    push!(space.log, "register abstract key: [$(cls.name)]")
    return cls
end

function add_subclass!(space::SimulationSpace, child::AbstractString, parent::AbstractString; kwargs...)
    cls = add_subclass!(space.taxonomy, child, parent; kwargs...)
    push!(space.log, "register abstract subkey: [$(cls.name)] < [$(parent)]")
    return cls
end

function assign_class!(space::SimulationSpace, entity_name::AbstractString, class_name::AbstractString; apply::Bool=true)
    key = String(entity_name)
    haskey(space.entities, key) || error("Unknown entity: $key")
    entity = space.entities[key]
    classes = assign_class!(space.taxonomy, entity, class_name; apply=apply)
    push!(space.log, "classify entity: [$(entity.name)] -> [$(class_name)]")
    return classes
end

assigned_classes(space::SimulationSpace, entity_name::AbstractString) =
    assigned_classes(space.taxonomy, space.entities[String(entity_name)])

is_a(space::SimulationSpace, entity_name::AbstractString, class_name::AbstractString) =
    is_a(space.taxonomy, space.entities[String(entity_name)], class_name)

known_classes(space::SimulationSpace) = known_classes(space.taxonomy)

function register_verb!(space::SimulationSpace, verb::DynamicVerb)
    space.dynamic_verbs[verb.name] = verb
    push!(space.log, "تسجيل فعل: [$(verb.name)]")
    return verb
end

function add_rule!(space::SimulationSpace, rule::CausalRule)
    push!(space.rules, rule)
    push!(space.log, "تسجيل قاعدة: [$(rule.property)] $(rule.operator) $(rule.threshold) -> [$(rule.action_name)]")
    return rule
end

function register_opposite!(space::SimulationSpace, a::AbstractString, b::AbstractString)
    left, right = String(a), String(b)
    space.opposites[left] = right
    space.opposites[right] = left
    push!(space.log, "register opposite: [$left] = -[$right]")
    return left => right
end

opposite_of(space::SimulationSpace, name::AbstractString) =
    get(space.opposites, String(name), nothing)

function register_template!(space::SimulationSpace, template::CausalTemplate)
    push!(space.templates, template)
    push!(space.log, "register causal template: [$(template.name)]")
    return template
end

function _condition_key(name::AbstractString)
    key = String(name)
    key == "مسافة" && return "distance"
    key == "بعد" && return "distance"
    key == "خوف" && return "fear"
    key == "طاقة" && return "energy"
    key == "حركة" && return "motion"
    return key
end

function _condition_value(space::SimulationSpace, entity_name::AbstractString, key::AbstractString)
    haskey(space.entities, String(entity_name)) || return nothing
    entity = space.entities[String(entity_name)]
    prop = _condition_key(key)
    amp, _ = get_property(entity, prop)
    amp > 0.0 && return amp
    if entity isa Thing
        value = get_attribute(entity, prop, nothing)
        value isa Number && return Float64(value)
    end
    return nothing
end

function _condition_holds(space::SimulationSpace, condition::AbstractString,
                          source_name::AbstractString, target_name::AbstractString)
    text = strip(String(condition))
    isempty(text) && return true
    m = match(r"^(?:(source|target|المصدر|الهدف)\.)?([^\s<>!=]+)\s*(<=|>=|<|>|==|=)\s*(-?[0-9.]+)$", text)
    m === nothing && return true

    role = m.captures[1] === nothing ? "source" : String(m.captures[1])
    key = String(m.captures[2])
    op = String(m.captures[3])
    threshold = try
        parse(Float64, String(m.captures[4]))
    catch
        return false
    end
    entity_name = role in ("target", "الهدف") ? target_name : source_name
    value = _condition_value(space, entity_name, key)
    value === nothing && return false
    op == "<" && return value < threshold
    op == ">" && return value > threshold
    op == "<=" && return value <= threshold
    op == ">=" && return value >= threshold
    return isapprox(value, threshold; atol=1e-9)
end

function _template_matches(space::SimulationSpace, template::CausalTemplate,
                           source_name::AbstractString, action::AbstractString,
                           target_name::AbstractString; kind::AbstractString="")
    src, tgt = String(source_name), String(target_name)
    haskey(space.entities, src) || return false
    haskey(space.entities, tgt) || return false
    template.action == String(action) || return false
    isempty(template.action_kind) || template.action_kind == String(kind) || return false
    isempty(template.source_class) || is_a(space, src, template.source_class) || return false
    isempty(template.target_class) || is_a(space, tgt, template.target_class) || return false
    _condition_holds(space, template.condition, source_name, target_name) || return false
    return true
end

function matching_templates(space::SimulationSpace, source_name::AbstractString,
                            action::AbstractString, target_name::AbstractString; kind::AbstractString="")
    return [template for template in space.templates
            if _template_matches(space, template, source_name, action, target_name; kind=kind)]
end

function _resolve_role(role::AbstractString, source_name::AbstractString, target_name::AbstractString)
    r = String(role)
    r in ("source", "المصدر", "الفاعل") && return String(source_name)
    r in ("target", "الهدف", "المفعول") && return String(target_name)
    r in ("both", "كلاهما") && return "$(source_name)+$(target_name)"
    return isempty(r) ? "" : r
end

function _apply_template_result_action!(space::SimulationSpace, template::CausalTemplate,
                                        source_name::AbstractString, target_name::AbstractString)
    isempty(template.result_action) && return false
    haskey(space.dynamic_verbs, template.result_action) || return false

    actor = _resolve_role(template.result_actor, source_name, target_name)
    target = _resolve_role(template.result_target, source_name, target_name)
    isempty(actor) && (actor = String(source_name))
    isempty(target) && (target = String(target_name))
    actor == "$(source_name)+$(target_name)" && (actor = String(source_name))

    haskey(space.entities, actor) || register_entity!(space, Thing(actor))
    haskey(space.entities, target) || register_entity!(space, Thing(target))
    applied = interact!(space, actor, template.result_action, target; strength=template.confidence)
    applied && push!(space.log, "template result applied: [$(template.name)] -> [$(template.result_action)] actor=[$actor] target=[$target]")
    return applied
end

function infer_event!(space::SimulationSpace, source_name::AbstractString,
                      action::AbstractString, target_name::AbstractString; kind::AbstractString="")
    frames = CausalFrame[]
    for template in matching_templates(space, source_name, action, target_name; kind=kind)
        actor = _resolve_role(template.result_actor, source_name, target_name)
        target = _resolve_role(template.result_target, source_name, target_name)
        parts = String[]
        !isempty(actor) && push!(parts, "actor=$actor")
        !isempty(template.result_action) && push!(parts, "action=$(template.result_action)")
        !isempty(template.result_kind) && push!(parts, "kind=$(template.result_kind)")
        !isempty(target) && push!(parts, "target=$target")
        !isempty(template.result_state) && push!(parts, "state=$(template.result_state)")
        !isempty(template.condition) && push!(parts, "condition=$(template.condition)")
        push!(parts, "confidence=$(template.confidence)")
        push!(parts, "polarity=$(template.polarity)")
        frame = CausalFrame([String(source_name), String(target_name)], String(action), join(parts, " | "))
        record_frame!(space, frame)
        _apply_template_result_action!(space, template, source_name, target_name)
        push!(frames, frame)
    end
    return frames
end

function apply_signed_action!(space::SimulationSpace, source_name::AbstractString,
                              action_name::AbstractString, target_name::AbstractString=source_name;
                              polarity::Int=1, strength::Float64=1.0)
    action = String(action_name)
    if polarity >= 0
        return evaluate_idea!(space, String(source_name), action, String(target_name); strength=strength)
    end

    opposite = opposite_of(space, action)
    if opposite !== nothing
        record_frame!(space, CausalFrame([String(source_name), String(target_name)], action,
            "polarity=negative | opposite=$opposite"))
        return evaluate_idea!(space, String(source_name), opposite, String(source_name); strength=strength)
    end

    record_frame!(space, CausalFrame([String(source_name), String(target_name)], action,
        "polarity=negative | no_opposite_registered"))
    return false
end

function register_event_chain!(space::SimulationSpace, chain::EventChain)
    space.event_chains[chain.name] = chain
    push!(space.log, "register event chain: [$(chain.name)]")
    return chain
end

function _chain_matches(space::SimulationSpace, chain::EventChain,
                        source_name::AbstractString, action::AbstractString,
                        target_name::AbstractString)
    chain.action == String(action) || return false
    haskey(space.entities, String(source_name)) || return false
    haskey(space.entities, String(target_name)) || return false
    isempty(chain.source_class) || is_a(space, source_name, chain.source_class) || return false
    isempty(chain.target_class) || is_a(space, target_name, chain.target_class) || return false
    return true
end

function _resolve_chain_role(space::SimulationSpace, role::AbstractString,
                             source_name::AbstractString, target_name::AbstractString)
    r = String(role)
    r in ("source", "المصدر", "الفاعل") && return String(source_name)
    r in ("target", "الهدف", "المفعول") && return String(target_name)
    isempty(r) && return ""
    haskey(space.entities, r) || register_entity!(space, Thing(r))
    return r
end

function run_event_chain!(space::SimulationSpace, chain_name::AbstractString,
                          source_name::AbstractString, target_name::AbstractString)
    chain = get(space.event_chains, String(chain_name), nothing)
    chain === nothing && return CausalFrame[]
    return run_event_chain!(space, chain, source_name, target_name)
end

function run_event_chain!(space::SimulationSpace, chain::EventChain,
                          source_name::AbstractString, target_name::AbstractString)
    _chain_matches(space, chain, source_name, chain.action, target_name) || return CausalFrame[]
    frames = CausalFrame[]
    previous_event = "$(source_name)_$(chain.action)_$(target_name)"
    for (idx, step) in enumerate(chain.steps)
        actor = _resolve_chain_role(space, step.actor_role, source_name, target_name)
        target = _resolve_chain_role(space, step.target_role, source_name, target_name)
        things = filter(!isempty, [actor, target])
        result = "chain=$(chain.name) | step=$idx | confidence=$(chain.confidence)"
        !isempty(step.result) && (result *= " | result=$(step.result)")
        frame = CausalFrame(things, step.action, result)
        record_frame!(space, frame)
        current_event = "$(actor)_$(step.action)_$(target)"
        relate_events!(space, previous_event, "بعد", current_event)
        previous_event = current_event
        push!(frames, frame)
    end
    return frames
end

function register_process!(space::SimulationSpace, process::ProcessConcept)
    steps = copy(process.steps)
    attrs = Dict{String,Any}(process.attributes)
    intensity = process.intensity

    if !isempty(process.parent) && haskey(space.processes, process.parent)
        parent = space.processes[process.parent]
        isempty(steps) && (steps = copy(parent.steps))
        inherited = Dict{String,Any}(parent.attributes)
        merge!(inherited, attrs)
        attrs = inherited
        intensity *= parent.intensity
    end

    resolved = ProcessConcept(process.name, process.parent, process.domain,
                              intensity, steps, attrs)
    space.processes[resolved.name] = resolved
    push!(space.log, "register process concept: [$(resolved.name)]")
    return resolved
end

get_process(space::SimulationSpace, name::AbstractString) =
    get(space.processes, String(name), nothing)

function _resolve_process_role(role::AbstractString, source_name::AbstractString, target_name::AbstractString)
    r = String(role)
    r in ("source", "المصدر", "الفاعل", "المعطي", "الحاكم") && return String(source_name)
    r in ("target", "الهدف", "المفعول", "المعطى_له", "المحكوم_له") && return String(target_name)
    return isempty(r) ? "" : r
end

function instantiate_process!(space::SimulationSpace, process_name::AbstractString,
                              source_name::AbstractString, target_name::AbstractString)
    process = get_process(space, process_name)
    process === nothing && return CausalFrame[]
    haskey(space.entities, String(source_name)) || register_entity!(space, Thing(String(source_name)))
    haskey(space.entities, String(target_name)) || register_entity!(space, Thing(String(target_name)))

    frames = CausalFrame[]
    for step in process.steps
        actor = _resolve_process_role(step.actor_role, source_name, target_name)
        target = _resolve_process_role(step.target_role, source_name, target_name)
        things = filter(!isempty, [actor, target])
        result = "process=$(process.name) | intensity=$(process.intensity)"
        !isempty(target) && (result *= " | target=$target")
        !isempty(step.result) && (result *= " | result=$(step.result)")
        frame = CausalFrame(things, step.action, result)
        record_frame!(space, frame)
        push!(frames, frame)
    end
    return frames
end

function _property_strength(e::AqlEntity, prop::String)
    prop == "self" && return 1.0
    amp, _ = get_property(e, prop)
    if amp > 0.0
        return amp
    end
    if e isa Thing
        val = get_attribute(e, prop, 0.0)
        val isa Number && return Float64(val)
    end
    return 0.0
end

function _apply_property_delta!(target::AqlEntity, prop::String, delta::Float64)
    amp, phase = get_property(target, prop)
    if set_property!(target, prop, clamp(amp + delta, 0.0, 1.0), phase)
        return true
    end
    if target isa Thing
        old = get_attribute(target, prop, 0.0)
        old_num = old isa Number ? Float64(old) : 0.0
        set_attribute!(target, prop, old_num + delta)
        return true
    end
    return false
end

function apply_verb!(space::SimulationSpace, verb::DynamicVerb,
                     source::AqlEntity, target::AqlEntity=source; strength::Float64=1.0)
    source_power = max(0.1, source.mass)
    target_inertia = max(0.1, target.mass)
    for effect in verb.effects
        carrier = _property_strength(source, effect.source_property)
        carrier = effect.source_property == "self" ? 1.0 : max(0.1, carrier)
        delta = 0.1 * effect.scale_factor * carrier * source_power * strength / target_inertia
        _apply_property_delta!(target, effect.target_property, delta)
    end
    push!(space.log, "حدث: [$(source.name)] قام بفعل [$(verb.name)] على [$(target.name)]")
    triggers = step_simulation!(space)
    record_frame!(space, CausalFrame([source.name, target.name], verb.name,
        triggers > 0 ? "نتيجة سببية تلقائية" : "تغيرت حالة الهدف"))
    return target
end

function evaluate_idea!(space::SimulationSpace, source_name::String,
                        action_name::String, target_name::String=source_name; strength::Float64=1.0)
    return interact!(space, source_name, action_name, target_name; strength=strength)
end

function interact!(space::SimulationSpace, source_name::String, verb_fn::Function, target_name::String)
    if !haskey(space.entities, source_name)
        push!(space.log, "فشل الحدث: المصدر [$source_name] غير مسجل")
        return false
    end
    if !haskey(space.entities, target_name)
        push!(space.log, "فشل الحدث: الهدف [$target_name] غير مسجل")
        return false
    end

    src = space.entities[source_name]
    tgt = space.entities[target_name]
    push!(space.log, "حدث: [$source_name] قام بفعل [$(nameof(verb_fn))] على [$target_name]")
    verb_fn(src, tgt)
    step_simulation!(space)
    return true
end

function interact!(space::SimulationSpace, source_name::String, action_name::String,
                   target_name::String; strength::Float64=1.0)
    if !haskey(space.entities, source_name)
        push!(space.log, "فشل الحدث: المصدر [$source_name] غير مسجل")
        return false
    end
    if !haskey(space.entities, target_name)
        push!(space.log, "فشل الحدث: الهدف [$target_name] غير مسجل")
        return false
    end

    src = space.entities[source_name]
    tgt = space.entities[target_name]
    action = src isa Thing ? get_action(src, action_name) : nothing
    dyn_verb = get(space.dynamic_verbs, action_name, nothing)

    if dyn_verb !== nothing
        apply_verb!(space, dyn_verb, src, tgt; strength=strength)
        return true
    end

    if action === nothing
        push!(space.log, "فشل الحدث: [$source_name] لا يملك الفعل [$action_name]")
        return false
    end

    push!(space.log, "حدث: [$source_name] قام بفعل [$action_name] على [$target_name] بقوة $(round(strength; digits=2))")
    try
        action(src, tgt; strength=strength)
    catch err
        if err isa MethodError
            action(src, tgt)
        else
            rethrow()
        end
    end
    triggers = step_simulation!(space)
    record_frame!(space, CausalFrame([source_name, target_name], action_name,
        triggers > 0 ? "نتيجة سببية تلقائية" : "تغيرت حالة الهدف"))
    return true
end

check_triggers!(e::AqlEntity, space::SimulationSpace) = false

function _check_dynamic_rules!(e::AqlEntity, space::SimulationSpace)
    triggered = false
    for rule in space.rules
        amp, _ = get_property(e, rule.property)
        if amp == 0.0 && e isa Thing
            val = get_attribute(e, rule.property, 0.0)
            amp = val isa Number ? Float64(val) : 0.0
        end
        passed = rule.operator == :> ? amp > rule.threshold :
                 rule.operator == :< ? amp < rule.threshold : false
        if passed && haskey(space.dynamic_verbs, rule.action_name)
            push!(space.log, "نتيجة سببية: [$(e.name)] حقق [$(rule.property) $(rule.operator) $(rule.threshold)] -> [$(rule.action_name)]")
            apply_verb!(space, space.dynamic_verbs[rule.action_name], e, e)
            triggered = true
        end
    end
    return triggered
end

function check_triggers!(e::Gazelle, space::SimulationSpace)
    fear_amp, _ = get_property(e, "fear")
    motion_amp, _ = get_property(e, "motion")
    if fear_amp > 0.5 && motion_amp < 0.5
        push!(space.log, "نتيجة سببية: [$(e.name)] خاف جدا -> جرى وهرب")
        run!(e)
        return true
    end
    return false
end

function check_triggers!(e::Furniture, space::SimulationSpace)
    stability_amp, _ = get_property(e, "stability")
    motion_amp, _ = get_property(e, "motion")
    integrity_amp, _ = get_property(e, "integrity")
    if stability_amp < 0.3 && motion_amp < 0.3 && integrity_amp > 0.15
        push!(space.log, "نتيجة سببية: [$(e.name)] فقد الاستقرار -> سقط")
        fall!(e)
        return true
    end
    return false
end

function step_simulation!(space::SimulationSpace; max_steps::Int=5)
    space.in_simulation && return 0
    space.in_simulation = true
    triggered = true
    loop_count = 0
    trigger_count = 0

    try
        while triggered && loop_count < max_steps
            triggered = false
            loop_count += 1
            for e in collect(values(space.entities))
                if check_triggers!(e, space)
                    triggered = true
                    trigger_count += 1
                end
                if _check_dynamic_rules!(e, space)
                    triggered = true
                    trigger_count += 1
                end
            end
        end
        if triggered
            push!(space.log, "توقفت حلقة السببية بعد بلوغ الحد الأعلى ($max_steps)")
        end
    finally
        space.in_simulation = false
    end

    return trigger_count
end

end # module CausalSpace
