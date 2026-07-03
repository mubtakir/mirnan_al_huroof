include("../src/MirnanNew.jl")
using .MirnanNew
using Test

const Physics = MirnanNew.Physics

println("=" ^ 60)
println("SELF REVIEW TEST")
println("=" ^ 60)

@testset "Self review" begin
    sr = Physics.SelfReviewEngine()
    pv_map = Dict(
        "water" => [1.0, 0.0],
        "flows" => [0.95, 0.05],
        "river" => [0.9, 0.1],
        "through" => [0.8, 0.2],
        "the" => [0.5, 0.5],
    )
    pv_fn = w -> get(pv_map, String(w), [0.1, 0.9])

    good = Physics.review_generation!(
        sr,
        "what does water do",
        "water flows through the river.";
        prompt_tokens=split("what does water do"),
        pv_fn=pv_fn,
    )
    bad = Physics.review_generation!(
        sr,
        "what does water do",
        "water water water water water";
        prompt_tokens=split("what does water do"),
        pv_fn=pv_fn,
    )

    @test good.score > bad.score
    @test good.accepted
    @test bad.repetition_score < good.repetition_score
    @test "dominant_repetition" in bad.issues
    @test bad.primary_issue == "repetition"
    @test bad.repair_target == "diversity"

    list_like = Physics.review_generation!(
        sr,
        "العلم نور",
        "ظلام كتاب قراءه طويله مسافات";
        prompt_tokens=["العلم", "نور"],
        pv_fn=pv_fn,
    )
    @test !list_like.accepted
    @test "list_like_output" in list_like.issues
    @test "missing_prompt_anchor" in list_like.issues
    @test list_like.repair_target == "prompt_alignment"

    mem = Physics.SelfReviewEngine()
    Physics.review_generation!(
        mem,
        "what does water do",
        "water water water water water";
        prompt_tokens=split("what does water do"),
        pv_fn=pv_fn,
    )
    prediction = Physics.predict_review_repair(
        mem,
        split("what does fire do");
        prompt="what does fire do",
    )
    @test prediction.primary_issue == "repetition"
    @test prediction.repair_target == "diversity"
    @test prediction.confidence > 0.0
    @test Physics.predict_review_repair(Physics.SelfReviewEngine(), split("what now")).repair_target == "none"

    @test Physics.learn_review_treatment!(
        mem,
        split("what does fire do");
        prompt="what does fire do",
        repair_target="diversity",
        before_score=0.20,
        after_score=0.72,
        chosen=true,
        source="test",
    )
    treatment = Physics.predict_review_treatment(
        mem,
        split("what does stone do");
        prompt="what does stone do",
        default_target="prompt_alignment",
    )
    @test treatment.repair_target == "diversity"
    @test treatment.expected_delta > 0.0
    @test treatment.confidence > 0.0
    treatment_summary = Physics.treatment_memory_summary(mem)
    @test treatment_summary["signature_count"] >= 1

    math_bad = Physics.review_generation!(
        sr,
        "calculate 2 + 2",
        "2 + 2 = 5";
        prompt_tokens=split("calculate 2 + 2"),
        pv_fn=pv_fn,
    )
    @test math_bad.logic_score < 1.0
    @test "numeric_inconsistency" in math_bad.issues
    @test math_bad.primary_issue == "logic"
    @test math_bad.repair_target == "logic"

    kernel = Physics.BayanLogicKernel()
    audit = Physics.audit_bayan_logic(kernel, "the sun is hot", "the sun is not hot.")
    @test !audit.consistent
    @test "bayan_logic_contradiction" in audit.issues

    logic_bad = Physics.review_generation!(
        sr,
        "the sun is hot",
        "the sun is not hot.";
        prompt_tokens=split("the sun is hot"),
        pv_fn=pv_fn,
    )
    @test "bayan_logic_contradiction" in logic_bad.issues
    @test logic_bad.repair_target == "logic"

    summary = Physics.review_summary(sr)
    @test summary["history_count"] == 5
    @test haskey(summary, "last_review")
    @test haskey(summary["last_review"], "repair_target")
    @test haskey(summary, "bayan_logic")
    @test haskey(summary, "review_memory")
    @test haskey(summary, "treatment_memory")

    guidance = Dict{String,Any}(
        "active" => true,
        "movement" => "method",
        "confidence" => 0.9,
        "target_terms" => ["practice", "review"],
    )
    semantic_good = Physics.review_generation!(
        sr,
        "how does learning grow",
        "learning grows through practice and review";
        prompt_tokens=split("how does learning grow"),
        pv_fn=pv_fn,
        semantic_guidance=guidance,
    )
    semantic_bad = Physics.review_generation!(
        sr,
        "how does learning grow",
        "learning is knowledge";
        prompt_tokens=split("how does learning grow"),
        pv_fn=pv_fn,
        semantic_guidance=guidance,
    )
    @test !("semantic_movement_mismatch" in semantic_good.issues)
    @test "semantic_movement_mismatch" in semantic_bad.issues
    @test semantic_bad.primary_issue == "semantic"
    @test semantic_bad.repair_target == "none"

    restored = Physics.SelfReviewEngine()
    Physics.learn_review_treatment!(
        sr,
        split("what does water do");
        prompt="what does water do",
        repair_target="diversity",
        before_score=0.2,
        after_score=0.6,
        chosen=true,
        source="restore_test",
    )
    @test Physics.restore_self_review_state!(restored, Physics.self_review_state_dict(sr))
    @test length(restored.history) == length(sr.history)
    @test !isempty(restored.diagnostic_memory)
    @test !isempty(restored.treatment_memory)

    gen = Physics.MirnanGenerator(Dict("water" => 1, "flows" => 2, "river" => 3))
    @test gen.self_review isa Physics.SelfReviewEngine
    Physics.generate!(gen, "water"; max_words=3)
    @test gen.self_review.last_review !== nothing
    @test haskey(Physics.get_physics_report(gen), "self_review")
end

println("PASSED self review")
