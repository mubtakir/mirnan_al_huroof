# Generation Group
# Part of Phase 4 — Modularity Refactoring

include("../engines/code_engine.jl")
include("../engines/al_code.jl")
include("../engines/math_bridge.jl")
include("../engines/response_planning.jl")
include("../engines/intent_response_planner.jl")
include("../engines/text_preprocessor.jl")
include("../engines/generator.jl")
include("../engines/al_tawweer.jl")
include("../engines/orchestrator.jl")
include("../engines/multi_pass_generator.jl")

using .Generator: MirnanGenerator, generate!, get_physics_report, pattern_memory_summary,
                  gpu_init!, reset!, learn_from_feedback!,
                  save_runtime_learning!, load_runtime_learning!,
                  PRNNStrategy
using .AlTawweerModule: Chromosome, evolve_weights!, evaluate_fitness!
using .AlCode: CodePatternRecord, CodePatternMemory,
               learn_code_patterns_from_text!, train_code_patterns_from_texts!,
               select_code_pattern, preferred_code_slot_values,
               generate_code_from_pattern,
               save_al_code, load_al_code, al_code_to_dict,
               has_code_patterns, detect_code_language
using .ResponsePlanning: ResponsePlanner, ResponseArchitect, TrajectoryPlanner,
                         TrajectoryMilestone, plan_response!, response_fidelity,
                         architect_score, trajectory_score, plan!
using .IntentResponsePlanner: ResponseIntentPlan, detect_response_intent,
                              render_planned_response, has_plannable_response

export MirnanGenerator, generate!, get_physics_report, pattern_memory_summary,
       gpu_init!, reset!, learn_from_feedback!,
       save_runtime_learning!, load_runtime_learning!,
       PRNNStrategy
export Chromosome, evolve_weights!, evaluate_fitness!
export CodePatternRecord, CodePatternMemory,
       learn_code_patterns_from_text!, train_code_patterns_from_texts!,
       select_code_pattern, preferred_code_slot_values,
       generate_code_from_pattern,
       save_al_code, load_al_code, al_code_to_dict,
       has_code_patterns, detect_code_language
export ResponsePlanner, ResponseArchitect, TrajectoryPlanner, TrajectoryMilestone,
       plan_response!, response_fidelity, architect_score, trajectory_score, plan!
export ResponseIntentPlan, detect_response_intent, render_planned_response,
       has_plannable_response
