test_files = [
    "runtests.jl",
    "test_phase1.jl",
    "test_phase2.jl",
    "test_enhanced.jl",
    "test_enhanced_final.jl",
    "test_euclidean.jl",
    "test_exponential.jl",
    "test_exponential_v2.jl",
    "test_centered_signal.jl",
    "test_raw_params.jl",
    "test_clifford_enhanced.jl",
    "test_clifford_words.jl",
    "test_advanced_physics.jl",
    "test_new_model.jl",
    "test_generator_fallback.jl",
    "test_generation_quality_guard.jl",
    "test_training_preserves_tashkeel.jl",
    "test_train_knowledge_sentences.jl",
    "test_tashkeel_projection_matching.jl",
    "test_dialogue_salam_and_reported_speech.jl",
    "test_social_reply_memory_boundary.jl",
    "test_question_type_matrix.jl",
    "test_relation_frame.jl",
    "test_intent_response_planner.jl",
    "test_context_tension.jl",
    "test_mirnan_cerebellum.jl",
    "test_sense_superposition.jl",
    "test_self_review.jl",
    "test_semantic_gravity.jl",
    "test_semantic_imagination.jl",
    "test_syntactic_gravity.jl",
    "test_contextual_learning.jl",
    "test_contextual_selection_layer.jl",
    "test_al_lisan.jl",
    "test_al_code.jl",
    "test_al_tadbir.jl",
    "test_al_hisab.jl",
    "test_al_ta3rif.jl",
    "test_al_nisba.jl",
    "test_al_muradif.jl",
    "test_al_istinbat.jl",
    "test_al_intibah.jl",
    "test_al_aql_dialogue.jl",
    "test_al_aql_speech_matching.jl",
    "test_al_hisban_al_dalali.jl",
    "test_pattern_memory_integration.jl",
]

project_root = abspath(joinpath(@__DIR__, ".."))
failures = String[]

for f in test_files
    println("\n", "="^60)
    println("Running: $f")
    println("="^60)

    test_path = joinpath(@__DIR__, f)
    cmd = `$(Base.julia_cmd()) --project=$project_root $test_path`

    if success(cmd)
        println("PASSED: $f")
    else
        println("FAILED: $f")
        push!(failures, f)
    end
end

println("\n", "="^60)
println("Summary")
println("="^60)
println("Passed: $(length(test_files) - length(failures)) / $(length(test_files))")

if isempty(failures)
    println("All tests passed.")
else
    println("Failed tests:")
    foreach(f -> println("  - $f"), failures)
    exit(1)
end
