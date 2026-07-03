"""
مرنان الجديد (Mirnan New) — نموذج لغوي فيزيائي ديناميكي.

إعادة بناء من جديد مع تحسينات:
- حروف عربية بمعاملات فريدة (9 معاملات × 3 حدود)
- عامل أسي لتكبير التباعد بين الحروف
- جبر كليفورد 22D لتجميع الحروف
- جاذبية دلالية (تقارب المفاهيم)
- جاذبية نحوية (تماسك الجملة)
"""

module MirnanNew

# ═══════════════════════════════════════════════════════
# المعالجة المسبقة
# ═══════════════════════════════════════════════════════
include("preprocessing/Preprocessing.jl")

# ═══════════════════════════════════════════════════════
# الدلالة العربية
# ═══════════════════════════════════════════════════════
include("semantics/Semantics.jl")

# ═══════════════════════════════════════════════════════
# النحو العربي
# ═══════════════════════════════════════════════════════
include("grammar/Grammar.jl")

# ═══════════════════════════════════════════════════════
# المحركات الفيزيائية
# ═══════════════════════════════════════════════════════
include("physics/Physics.jl")

# ═══════════════════════════════════════════════════════
# الذكاء التوليفي (SIO)
# ═══════════════════════════════════════════════════════
include("sio/SIO.jl")

# ═══════════════════════════════════════════════════════
# خادم API
# ═══════════════════════════════════════════════════════
include("api/server.jl")

# ═══════════════════════════════════════════════════════
# الأدوات المساعدة
# ═══════════════════════════════════════════════════════
include("utils/DataLoader.jl")

# ═══════════════════════════════════════════════════════
# الربط المتكامل
# ═══════════════════════════════════════════════════════
include("integration/Integration.jl")

# إعادة تصدير الوحدات
using .Preprocessing
using .Semantics
using .Grammar
using .Physics
using .SIO
using .APIServer
using .DataLoader
using .Integration

# إعادة تصدير الرموز الرئيسية من Physics
using .Physics: MirnanGenerator, generate!, get_physics_report, pattern_memory_summary,
       gpu_init!, reset!, learn_from_feedback!,
       save_runtime_learning!, load_runtime_learning!,
       PRNNStrategy,
       RAPGKnowledgeBase, init_rapg_db!, store_passage!, load_rapg_kb, retrieve,
       Chromosome, evolve_weights!, evaluate_fitness!,
       MirnanCfg, load_config, get_val, get_section,
       compute_word_phase_vector, compute_word_mass, compute_word_frequency,
       compute_word_frequency_with_irab, phase_similarity, compute_extended_phase_vector,
       get_irab_omega_bias, extract_irab,
       TwistorEngine, TwistorOperator, MorphPattern,
       compute_twistor, apply_twistor, twistor_similarity,
       learn_patterns!, predict_derivation, find_morphological_pairs, score_twistor_candidate,
       BeamReasoner, ReasonStep, ReasonChain, reason!, reason_steps!, explain_reasoning,
       CodePhaseEngine, CodeConcept, CodePattern,
       encode_concept, decode_to_code, learn_code_patterns!, generate_code_physics,
       WordPhysicsProfile, LexicalNeighbor, CoinedWord,
       word_physics_profile, phonosemantic_quality,
       nearest_phase_words, analyze_unknown_word,
       coin_word_for_concept,
       SymbolicMathEngine, MathOperation, MathPattern,
       encode_number, encode_operation, solve_arithmetic, learn_math_pattern!,
       ResponsePlanner, ResponseArchitect, TrajectoryPlanner, TrajectoryMilestone,
       plan_response!, response_fidelity, architect_score, trajectory_score, plan!,
       PhysicsQualityReport, compute_quality_report, k_density, transition_entropy,
       density_purity, density_spectral_entropy, overall_quality_report,
       MirnanCerebellum, CerebellumObservation, CerebellumPolicy,
       observe_prompt, choose_policy!, apply_cerebellum_policy!,
       learn_from_outcome!, policy_summary, reset_cerebellum!,
       SenseCandidate, SenseSuperposition, SenseMeasurement,
       has_sense_inventory, sense_inventory, build_superposition,
       measure_senses, explain_measurement, top_sense,
       SelfReviewEngine, GenerationReview, ReviewMemoryPrediction,
       ReviewTreatmentPrediction, review_generation!, review_summary,
       predict_review_repair, review_memory_summary,
       learn_review_treatment!, predict_review_treatment,
       treatment_memory_summary, self_review_state_dict,
       restore_self_review_state!, reset_self_review!,
       BayanLogicClaim, BayanLogicAudit, BayanLogicKernel, extract_bayan_claims,
       audit_bayan_logic, bayan_logic_summary,
       WordBalanceConfig, build_word_balance_weights, balance_summary, pair_balance_weight,
       LinguisticPatternRecord, LinguisticPatternMemory,
       learn_lisan_from_text!, train_lisan_from_texts!,
        select_lisan_pattern, token_role_phase, ROLE_SYNTAX_PHASE,
        save_lisan, load_lisan, lisan_to_dict,
        has_lisan_patterns, pattern_has_verb, detect_lisan_language,
       load_lisan_markers, default_lisan_marker_path,
       CodePatternRecord, CodePatternMemory,
       learn_code_patterns_from_text!, train_code_patterns_from_texts!,
       select_code_pattern, preferred_code_slot_values,
       generate_code_from_pattern,
       save_al_code, load_al_code, al_code_to_dict,
       has_code_patterns, detect_code_language,
       TadbirPatternRecord, TadbirMemory,
       learn_tadbir_from_text!, train_tadbir_from_texts!,
       select_tadbir_pattern, preferred_tadbir_slot_values,
       render_tadbir_plan,
       save_tadbir, load_tadbir, tadbir_to_dict,
       has_tadbir_patterns,
       HisabPatternRecord, HisabMemory, HisabSolution,
       learn_hisab_from_text!, train_hisab_from_texts!,
       solve_hisab, render_hisab_solution, render_hisab_solution_ar,
       save_hisab, load_hisab, hisab_to_dict,
       has_hisab_patterns,
       Ta3rifRecord, Ta3rifMemory,
       learn_ta3rif_from_text!, train_ta3rif_from_texts!,
       answer_ta3rif, save_ta3rif, load_ta3rif,
       ta3rif_to_dict, has_ta3rif_records,
       SemanticCalculusRecord, SemanticCalculusMemory,
       sentence_semantic_signature, semantic_transform_signature,
       learn_semantic_calculus_from_pair!, learn_semantic_calculus_from_text!,
       train_semantic_calculus_from_texts!,
       select_semantic_transform, semantic_relation_movement,
       semantic_guidance, semantic_answer_plan, semantic_guidance_terms,
       save_semantic_calculus, load_semantic_calculus,
       semantic_calculus_to_dict, has_semantic_calculus,
       QAProjectorMemory, learn_qa_shift!, project_question, retrieve_answer_facts,
       CliffordRotor, construct_rotor, apply_rotor,
       CognitiveTrajectoryTracker, reset_tracker!, absorb_input!, absorb_word!, trajectory_alignment_score,
       SemanticScene, SemanticSceneMemory, SemanticSceneComparison,
       SemanticSceneAnswerComparison,
       extract_semantic_scene, scene_effect_terms,
       learn_semantic_scene_from_text!, train_semantic_scenes_from_texts!,
       has_semantic_scenes, select_semantic_scene, semantic_scene_diagnostic,
       semantic_scenes_to_dict, save_semantic_scenes, load_semantic_scenes,
       compare_semantic_scene_with_calculus, semantic_scene_comparison_diagnostic,
       semantic_scene_answer, compare_semantic_scene_strategies,
       NisbaRelationRecord, NisbaMemory,
       learn_nisba_from_text!, train_nisba_from_texts!,
       select_nisba_relation, nisba_guidance_terms,
       save_nisba, load_nisba, nisba_to_dict,
       has_nisba_relations,
       complete_pattern!, prnn_noise_sample,
       prnn_generate_standalone, prnn_complete_pattern_standalone, prnn_noise_sample_standalone,
       Idea, compute_idea_vector, match_sentence_to_idea, score_candidate_for_idea,
       AqlEntity, Thing, Lion, Gazelle, Earth, Furniture,
       get_property, set_property!, get_attribute, set_attribute!,
       register_action!, get_action, has_action, print_entity, PROPERTY_MAP,
       ConceptClass, ConceptTaxonomy, default_aql_taxonomy,
       register_class!, add_subclass!, assign_class!, assigned_classes,
       class_lineage, inherited_attributes, apply_taxonomy!, is_a,
       explain_classification, known_classes, load_taxonomy, save_taxonomy,
       DEFAULT_TAXONOMY_FILE,
       roar!, shake!, run!, fall!, become!, emit_scent!, attract_beneficial_insects!,
       SimulationSpace, CausalFrame, PropertyEffect, DynamicVerb, CausalRule,
       CausalTemplate, ProcessStep, ProcessConcept, Circumstance, TemporalRelation,
       ChainStep, EventChain, RelationFact, SemanticRelation, QuantifiedFact, ComparisonFact,
       MetaphorFact, IntentFact, SpeechActFact, ExceptionRule,
       AqlGovernanceRule, RejectionLesson, CorpusAnnotation,
       register_entity!, register_verb!, add_rule!, apply_verb!, evaluate_idea!,
       interact!, record_frame!, step_simulation!, register_template!, matching_templates,
       infer_event!, register_process!, get_process, instantiate_process!,
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
       aql_inhibition_score, aql_inhibition_reason,
       property_key, compile_adl!, train_from_text!,
       MorphAnalysis, TemporalInfo, BinaryRelation, InferredEvent, EntityKnowledge,
       AdvancedKnowledgeBase, strip_diacritics, strip_al, strip_prep_prefix,
       has_tanwin_nasb, tokenize_arabic, analyze_arabic_word, analyze_arabic_sentence,
       segment_sentences, deep_understand, get_or_create_entity!,
       add_relation!, add_role!, add_temporal_info!, get_entity_relations,
       get_entities_with_attribute, find_relations_between

# إعادة تصدير الرموز الرئيسية من SIO
using .SIO: SIOOrchestrator, GoalParser, PhasePlanner, PhaseExecutor, SelfMonitor, Integrator, synthesize!

# إعادة تصدير الرموز الرئيسية من المعالجة المسبقة (متوفرة بالفعل من using .Preprocessing)

# إعادة تصدير الرموز الرئيسية من الدلالة
using .Semantics: LetterEmbedding, WordEmbedding, SemanticVector,
       embed_letter, embed_word, compute_semantic_similarity,
       extract_root, compute_word_semantics, SemanticAnalyzer

# إعادة تصدير الرموز الرئيسية من النحو
using .Grammar: Sentence, Phrase, Word, Morpheme,
       SyntacticRole, PhraseType, SentenceType,
       analyze_sentence, parse_sentence, get_syntactic_roles,
       GrammarAnalyzer

# إعادة تصدير الرموز الرئيسية من خادم API (متوفرة بالفعل من using .APIServer)

# إعادة تصدير الرموز الرئيسية من DataLoader
using .DataLoader: load_root_db, load_dictionary, save_analysis, load_analysis,
       get_root_info, search_by_category, get_all_roots, get_statistics

# إعادة تصدير الرموز الرئيسية من Integration
using .Integration: MirnanPipeline, PipelineResult, analyze_text, analyze_word,
       analyze_sentence_full, generate_response, get_analysis_report

# التصدير النهائي
export MirnanGenerator, generate!, get_physics_report, pattern_memory_summary,
       gpu_init!, reset!, learn_from_feedback!,
       save_runtime_learning!, load_runtime_learning!,
       RAPGKnowledgeBase, init_rapg_db!, store_passage!, load_rapg_kb, retrieve,
       PRNNStrategy
export compute_word_phase_vector, compute_word_mass, compute_word_frequency,
       compute_word_frequency_with_irab, phase_similarity, compute_extended_phase_vector
export WordPhysicsProfile, LexicalNeighbor, CoinedWord,
       word_physics_profile, phonosemantic_quality,
       nearest_phase_words, analyze_unknown_word,
       coin_word_for_concept
export WordBalanceConfig, build_word_balance_weights, balance_summary, pair_balance_weight
export LinguisticPatternRecord, LinguisticPatternMemory,
       learn_lisan_from_text!, train_lisan_from_texts!,
        select_lisan_pattern, token_role_phase, ROLE_SYNTAX_PHASE,
        save_lisan, load_lisan, lisan_to_dict,
        has_lisan_patterns, pattern_has_verb, detect_lisan_language,
       load_lisan_markers, default_lisan_marker_path
export CodePatternRecord, CodePatternMemory,
       learn_code_patterns_from_text!, train_code_patterns_from_texts!,
       select_code_pattern, preferred_code_slot_values,
       generate_code_from_pattern,
       save_al_code, load_al_code, al_code_to_dict,
       has_code_patterns, detect_code_language
export TadbirPatternRecord, TadbirMemory,
       learn_tadbir_from_text!, train_tadbir_from_texts!,
       select_tadbir_pattern, preferred_tadbir_slot_values,
       render_tadbir_plan,
       save_tadbir, load_tadbir, tadbir_to_dict,
       has_tadbir_patterns
export HisabPatternRecord, HisabMemory, HisabSolution,
       learn_hisab_from_text!, train_hisab_from_texts!,
       solve_hisab, render_hisab_solution, render_hisab_solution_ar,
       save_hisab, load_hisab, hisab_to_dict,
       has_hisab_patterns
export Ta3rifRecord, Ta3rifMemory,
       learn_ta3rif_from_text!, train_ta3rif_from_texts!,
       answer_ta3rif, save_ta3rif, load_ta3rif,
       ta3rif_to_dict, has_ta3rif_records
export SemanticCalculusRecord, SemanticCalculusMemory,
       sentence_semantic_signature, semantic_transform_signature,
       learn_semantic_calculus_from_pair!, learn_semantic_calculus_from_text!,
       train_semantic_calculus_from_texts!,
       select_semantic_transform, semantic_relation_movement,
       semantic_guidance, semantic_answer_plan, semantic_guidance_terms,
       save_semantic_calculus, load_semantic_calculus,
       semantic_calculus_to_dict, has_semantic_calculus
export QAProjectorMemory, learn_qa_shift!, project_question, retrieve_answer_facts,
       CliffordRotor, construct_rotor, apply_rotor
export CognitiveTrajectoryTracker, reset_tracker!, absorb_input!, absorb_word!, trajectory_alignment_score
export SemanticScene, SemanticSceneMemory, SemanticSceneComparison,
       SemanticSceneAnswerComparison,
       extract_semantic_scene, scene_effect_terms,
       learn_semantic_scene_from_text!, train_semantic_scenes_from_texts!,
       has_semantic_scenes, select_semantic_scene, semantic_scene_diagnostic,
       semantic_scenes_to_dict, save_semantic_scenes, load_semantic_scenes,
       compare_semantic_scene_with_calculus, semantic_scene_comparison_diagnostic,
       semantic_scene_answer, compare_semantic_scene_strategies
export NisbaRelationRecord, NisbaMemory,
       learn_nisba_from_text!, train_nisba_from_texts!,
       select_nisba_relation, nisba_guidance_terms,
       save_nisba, load_nisba, nisba_to_dict,
       has_nisba_relations
export learn_opposition_from_text!, learn_direct_negation_from_text!,
       terms_are_opposed, terms_are_negated
export MirnanCerebellum, CerebellumObservation, CerebellumPolicy,
       observe_prompt, choose_policy!, apply_cerebellum_policy!,
       learn_from_outcome!, policy_summary, reset_cerebellum!
export SenseCandidate, SenseSuperposition, SenseMeasurement,
       has_sense_inventory, sense_inventory, build_superposition,
       measure_senses, explain_measurement, top_sense
export SelfReviewEngine, GenerationReview, ReviewMemoryPrediction,
       ReviewTreatmentPrediction, review_generation!, review_summary,
       predict_review_repair, review_memory_summary,
       learn_review_treatment!, predict_review_treatment,
       treatment_memory_summary, self_review_state_dict,
       restore_self_review_state!, reset_self_review!
export BayanLogicClaim, BayanLogicAudit, BayanLogicKernel, extract_bayan_claims,
       audit_bayan_logic, bayan_logic_summary
export SIOOrchestrator, synthesize!
export normalize_text, extract_sentences, extract_words,
       normalize_arabic, remove_diacritics, normalize_whitespace,
       tokenize_text, preprocess_text, TextPreprocessor
export LetterEmbedding, WordEmbedding, SemanticVector,
       embed_letter, embed_word, compute_semantic_similarity,
       extract_root, compute_word_semantics, SemanticAnalyzer
export Sentence, Phrase, Word, Morpheme,
       SyntacticRole, PhraseType, SentenceType,
       analyze_sentence, parse_sentence, get_syntactic_roles,
       GrammarAnalyzer
export start_server, stop_server, handle_request, set_generator!
export load_root_db, load_dictionary, save_analysis, load_analysis,
       get_root_info, search_by_category, get_all_roots, get_statistics
export MirnanPipeline, PipelineResult, analyze_text, analyze_word,
       analyze_sentence_full, generate_response, get_analysis_report
export complete_pattern!, prnn_noise_sample,
       prnn_generate_standalone, prnn_complete_pattern_standalone, prnn_noise_sample_standalone
export Idea, compute_idea_vector, match_sentence_to_idea, score_candidate_for_idea
export AqlEntity, Thing, Lion, Gazelle, Earth, Furniture,
       get_property, set_property!, get_attribute, set_attribute!,
       register_action!, get_action, has_action, print_entity, PROPERTY_MAP
export ConceptClass, ConceptTaxonomy, default_aql_taxonomy,
       register_class!, add_subclass!, assign_class!, assigned_classes,
       class_lineage, inherited_attributes, apply_taxonomy!, is_a,
       explain_classification, known_classes, load_taxonomy, save_taxonomy,
       DEFAULT_TAXONOMY_FILE
export roar!, shake!, run!, fall!, become!, emit_scent!, attract_beneficial_insects!
export SimulationSpace, CausalFrame, PropertyEffect, DynamicVerb, CausalRule,
       CausalTemplate, ProcessStep, ProcessConcept, Circumstance, TemporalRelation,
       ChainStep, EventChain, RelationFact, SemanticRelation, QuantifiedFact, ComparisonFact,
       MetaphorFact, IntentFact, SpeechActFact, ExceptionRule,
       AqlGovernanceRule, RejectionLesson, CorpusAnnotation,
       register_entity!, register_verb!, add_rule!, apply_verb!, evaluate_idea!,
       interact!, record_frame!, step_simulation!, register_template!, matching_templates,
       infer_event!, register_process!, get_process, instantiate_process!,
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
       aql_inhibition_score, aql_inhibition_reason,
       property_key, compile_adl!, train_from_text!
export MorphAnalysis, TemporalInfo, BinaryRelation, InferredEvent, EntityKnowledge,
       AdvancedKnowledgeBase, strip_diacritics, strip_al, strip_prep_prefix,
       has_tanwin_nasb, tokenize_arabic, analyze_arabic_word, analyze_arabic_sentence,
       segment_sentences, deep_understand, get_or_create_entity!,
       add_relation!, add_role!, add_temporal_info!, get_entity_relations,
       get_entities_with_attribute, find_relations_between
export Chromosome, evolve_weights!, evaluate_fitness!

end # module MirnanNew
