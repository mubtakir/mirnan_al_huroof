module AlAql

include("entities.jl")
include("operators.jl")
include("taxonomy.jl")
include("causal_space.jl")
include("dsl.jl")
include("arabic_analysis.jl")

using .Entities
using .Operators
using .Taxonomy
using .CausalSpace
using .DSL
using .ArabicAnalysis

# إعادة تصدير رموز الكيانات والأنواع
export AqlEntity, Thing, Lion, Gazelle, Earth, Furniture,
       get_property, set_property!, get_attribute, set_attribute!,
       register_action!, get_action, has_action, print_entity, PROPERTY_MAP

export ConceptClass, ConceptTaxonomy, default_aql_taxonomy,
       register_class!, add_subclass!, assign_class!, assigned_classes,
       class_lineage, inherited_attributes, apply_taxonomy!, is_a,
       explain_classification, known_classes, load_taxonomy, save_taxonomy,
       DEFAULT_TAXONOMY_FILE

# إعادة تصدير الأفعال (الدوال)
export roar!, shake!, run!, fall!, become!, emit_scent!, attract_beneficial_insects!

# إعادة تصدير رموز فضاء السببية
export SimulationSpace, CausalFrame, PropertyEffect, DynamicVerb, CausalRule,
       CausalTemplate, ProcessStep, ProcessConcept, Circumstance, TemporalRelation,
       ChainStep, EventChain, RelationFact, SemanticRelation, QuantifiedFact, ComparisonFact,
       MetaphorFact, IntentFact, SpeechActFact, ExceptionRule,
       AqlGovernanceRule, RejectionLesson, CorpusAnnotation,
       register_entity!, register_verb!, add_rule!, apply_verb!, evaluate_idea!,
       interact!, record_frame!, step_simulation!, check_triggers!,
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

export property_key, compile_adl!, train_from_text!

export MorphAnalysis, TemporalInfo, BinaryRelation, InferredEvent, EntityKnowledge,
       AdvancedKnowledgeBase, strip_diacritics, strip_al, strip_prep_prefix,
       has_tanwin_nasb, tokenize_arabic, analyze_arabic_word, analyze_arabic_sentence,
       segment_sentences, deep_understand, get_or_create_entity!,
       add_relation!, add_role!, add_temporal_info!, get_entity_relations,
       get_entities_with_attribute, find_relations_between

end # module AlAql
