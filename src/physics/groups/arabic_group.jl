# Arabic Linguistics Group
# Part of Phase 4 — Modularity Refactoring

include("../engines/al_lisan.jl")
include("../engines/al_tadbir.jl")
include("../engines/al_hisab.jl")
include("../engines/al_ta3rif.jl")
include("../engines/al_hisban_al_dalali.jl")
include("../engines/clifford_qa_projector.jl")
include("../engines/clifford_trajectory_tracker.jl")
include("../engines/semantic_imagination.jl")
include("../engines/al_nisba.jl")
include("../engines/al_muradif.jl")
include("../engines/al_istinbat.jl")
include("../engines/al_intibah.jl")
include("../engines/marker_discovery.jl")

using .AlLisan: LinguisticPatternRecord, LinguisticPatternMemory,
                learn_lisan_from_text!, train_lisan_from_texts!,
                select_lisan_pattern, token_role_phase, ROLE_SYNTAX_PHASE,
                save_lisan, load_lisan, lisan_to_dict,
                has_lisan_patterns, pattern_has_verb, detect_lisan_language,
                load_lisan_markers, default_lisan_marker_path
using .AlTadbir: TadbirPatternRecord, TadbirMemory,
                  learn_tadbir_from_text!, train_tadbir_from_texts!,
                  select_tadbir_pattern, preferred_tadbir_slot_values,
                  render_tadbir_plan,
                  save_tadbir, load_tadbir, tadbir_to_dict,
                  has_tadbir_patterns
using .AlHisab: HisabPatternRecord, HisabMemory, HisabSolution,
                learn_hisab_from_text!, train_hisab_from_texts!,
                solve_hisab, render_hisab_solution, render_hisab_solution_ar,
                save_hisab, load_hisab, hisab_to_dict,
                has_hisab_patterns
using .AlTa3rif: Ta3rifRecord, Ta3rifMemory,
                 learn_ta3rif_from_text!, train_ta3rif_from_texts!,
                 answer_ta3rif, save_ta3rif, load_ta3rif,
                 merge_ta3rif!,
                 ta3rif_to_dict, has_ta3rif_records
using .AlHisbanAlDalali: SemanticCalculusRecord, SemanticCalculusMemory,
                         sentence_semantic_signature, semantic_transform_signature,
                         learn_semantic_calculus_from_pair!,
                         learn_semantic_calculus_from_text!,
                         train_semantic_calculus_from_texts!,
                         select_semantic_transform, semantic_relation_movement,
                         semantic_guidance, semantic_answer_plan,
                         semantic_guidance_terms,
                         save_semantic_calculus, load_semantic_calculus,
                         semantic_calculus_to_dict, has_semantic_calculus
using .CliffordQAProjectorModule: QAProjectorMemory, learn_qa_shift!, project_question, retrieve_answer_facts,
                                   CliffordRotor, construct_rotor, apply_rotor
using .CliffordTrajectoryTrackerModule: CognitiveTrajectoryTracker, reset_tracker!, absorb_input!, absorb_word!, trajectory_alignment_score
using .SemanticImagination: SemanticScene, SemanticSceneMemory, SemanticSceneComparison,
                             SemanticSceneAnswerComparison,
                             extract_semantic_scene, scene_effect_terms,
                             learn_semantic_scene_from_text!, train_semantic_scenes_from_texts!,
                             has_semantic_scenes, select_semantic_scene, semantic_scene_diagnostic,
                             semantic_scenes_to_dict, save_semantic_scenes, load_semantic_scenes,
                             compare_semantic_scene_with_calculus, semantic_scene_comparison_diagnostic,
                             semantic_scene_answer, compare_semantic_scene_strategies
using .AlNisba: NisbaRelationRecord, NisbaMemory,
                learn_nisba_fact!, learn_nisba_from_text!, train_nisba_from_texts!,
                select_nisba_relation, nisba_guidance_terms,
                save_nisba, load_nisba, nisba_to_dict,
                has_nisba_relations
using .AlMuradif: MuradifCandidate, MuradifMemory,
                  build_muradif_memory, muradif_terms,
                  merge_muradif!,
                  save_muradif, load_muradif, muradif_to_dict,
                  has_muradif_records
using .AlIstinbat: IstinbatAttentionRecord, IstinbatAttentionMemory, RelationFrame, QuantityFrame,
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
using .AlIntibah: SemanticAttentionField, build_semantic_attention,
                  has_semantic_attention, attention_bias_terms
using .MarkerDiscoveryModule: DiscoveredMarker, discover_markers, save_discovered_markers, load_discovered_markers

export SemanticAttentionField, build_semantic_attention,
       has_semantic_attention, attention_bias_terms
export LinguisticPatternRecord, LinguisticPatternMemory,
       learn_lisan_from_text!, train_lisan_from_texts!,
       select_lisan_pattern, token_role_phase, ROLE_SYNTAX_PHASE,
       save_lisan, load_lisan, lisan_to_dict,
       has_lisan_patterns, pattern_has_verb, detect_lisan_language,
       load_lisan_markers, default_lisan_marker_path
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
       merge_ta3rif!,
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
       learn_nisba_fact!, learn_nisba_from_text!, train_nisba_from_texts!,
       select_nisba_relation, nisba_guidance_terms,
       save_nisba, load_nisba, nisba_to_dict,
       has_nisba_relations
export MuradifCandidate, MuradifMemory,
       build_muradif_memory, muradif_terms,
       merge_muradif!,
       save_muradif, load_muradif, muradif_to_dict,
       has_muradif_records
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
export DiscoveredMarker, discover_markers, save_discovered_markers, load_discovered_markers
