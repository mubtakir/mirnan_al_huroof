# Infrastructure & Operators Group
# Part of Phase 4 — Modularity Refactoring

include("../engines/synchronize.jl")
include("../engines/model_bundle.jl")
include("../engines/model_io.jl")
include("../engines/scaling_analysis.jl")
include("../engines/training_balance.jl")
include("../engines/gpu_accelerator.jl")
include("../engines/language_feedback.jl")
include("../engines/contextual_learning.jl")
include("../engines/phase_evolution.jl")
include("../engines/semantic_comprehension.jl")
include("../engines/idea_engine.jl")
include("../engines/bayan_logic_kernel.jl")
include("../engines/physics_metrics.jl")
include("../engines/mirnan_cerebellum.jl")
include("../engines/sense_superposition.jl")
include("../engines/self_review.jl")
include("../engines/root_field.jl")
include("../engines/code_phase_engine.jl")
include("../engines/symbolic_math_engine.jl")
include("../engines/operators.jl")
include("../engines/dialogue_engine.jl")

using .ContextualLearning: ContextualLearningState, CompoundExpressionMemory,
                           QuestionIntentLearner, MetaphorField, EffectMemory,
                           contextual_answer
using .BayanLogicKernelModule: BayanLogicClaim, BayanLogicAudit,
                               BayanLogicKernel, extract_bayan_claims,
                               audit_bayan_logic, bayan_logic_summary
using .PhysicsMetrics: PhysicsQualityReport, compute_quality_report,
                       k_density, transition_entropy, density_purity,
                       density_spectral_entropy, overall_quality_report
using .MirnanCerebellumModule: MirnanCerebellum, CerebellumObservation,
                              CerebellumPolicy, observe_prompt, choose_policy!,
                              apply_cerebellum_policy!, learn_from_outcome!,
                              policy_summary, cerebellum_state_dict,
                              restore_cerebellum_state!, reset_cerebellum!
using .SenseSuperpositionModule: SenseCandidate, SenseSuperposition,
                                 SenseMeasurement, has_sense_inventory,
                                 sense_inventory, build_superposition,
                                 measure_senses, explain_measurement,
                                 top_sense
using .SelfReviewModule: SelfReviewEngine, GenerationReview,
                         ReviewMemoryPrediction, ReviewTreatmentPrediction,
                         review_generation!, review_summary,
                         predict_review_repair, review_memory_summary,
                         learn_review_treatment!, predict_review_treatment,
                         treatment_memory_summary,
                         self_review_state_dict, restore_self_review_state!,
                         reset_self_review!
using .PhaseEvolutionModule: PhaseEvolution, observe_cooccurrence!, evolve!, PhaseEvolutionModule
using .TrainingBalanceModule: WordBalanceConfig, build_word_balance_weights,
                              balance_summary, pair_balance_weight
using .IdeaEngine: Idea, compute_idea_vector, match_sentence_to_idea, score_candidate_for_idea
using .RootFieldModule: find_root_family, root_field_report
using .CodePhaseEngineModule: CodePhaseEngine, CodeConcept, CodePattern,
                              encode_concept, decode_to_code, learn_code_patterns!,
                              generate_code_physics
using .SymbolicMathEngineModule: SymbolicMathEngine, MathOperation, MathPattern,
                                   encode_number, encode_operation, solve_arithmetic,
                                   learn_math_pattern!

export ContextualLearningState, CompoundExpressionMemory, QuestionIntentLearner,
       MetaphorField, EffectMemory, contextual_answer
export CodePhaseEngine, CodeConcept, CodePattern, encode_concept, decode_to_code,
       learn_code_patterns!, generate_code_physics
export SymbolicMathEngine, MathOperation, MathPattern, encode_number, encode_operation,
       solve_arithmetic, learn_math_pattern!
export PhysicsQualityReport, compute_quality_report, k_density, transition_entropy,
       density_purity, density_spectral_entropy, overall_quality_report
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
export PhaseEvolution, observe_cooccurrence!, evolve!, PhaseEvolutionModule
export WordBalanceConfig, build_word_balance_weights, balance_summary, pair_balance_weight
export Idea, compute_idea_vector, match_sentence_to_idea, score_candidate_for_idea
