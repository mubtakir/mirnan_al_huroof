using LinearAlgebra
include("src/MirnanNew.jl")
using .MirnanNew
using .MirnanNew.Physics

println("=" ^ 60)
println("PHASE 5 TEST: Learning & Feedback")
println("=" ^ 60)

# Test 1: Phase Evolution
println("\n[1] Phase Evolution...")
try
    using .Physics.PhaseEvolutionModule
    pe = PhaseEvolution(lr_cooc=0.03, dim=27)
    println("  OK PhaseEvolution created")
    w1_pv = randn(27)
    w2_pv = randn(27)
    observe_cooccurrence!(pe, w1_pv, w2_pv; w1_id=1, w2_id=2, distance=1)
    println("  OK observe_cooccurrence! completed")
    new_pvs, info = evolve!(pe, randn(10, 27))
    println("  OK evolve! completed: words_shifted=$(info["words_shifted"])")
catch e
    println("  FAIL: $e")
end

# Test 2: Carrier Wave
println("\n[2] Carrier Wave...")
try
    using .Physics.CarrierWave
    cwe = CarrierWaveEngine()
    println("  OK CarrierWaveEngine created")
    v = randn(27)
    v_rot = rotate_phase(v, 0.5)
    println("  OK rotate_phase: norm=$(round(norm(v_rot); digits=4))")
catch e
    println("  FAIL: $e")
end

# Test 3: Language Feedback
println("\n[3] Language Feedback...")
try
    using .Physics.LanguageFeedback
    lfe = LanguageFeedbackEngine()
    println("  OK LanguageFeedbackEngine created")
    eval_dict = evaluate_sentence!(lfe, ["مرحبا", "بالعالم", "كيف حالك"],
                                    w -> randn(27), w -> 1.0)
    println("  OK evaluate_sentence!: quality=$(eval_dict["quality"]), is_good=$(eval_dict["is_good"])")
    report = get_feedback_report(lfe)
    println("  OK get_feedback_report: total=$(report["total_generations"])")
catch e
    println("  FAIL: $e")
end

println("\n" * "=" ^ 60)
println("PHASE 5 COMPLETE")
println("=" ^ 60)
