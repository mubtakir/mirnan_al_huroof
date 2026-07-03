include("../src/MirnanNew.jl")
using .MirnanNew
using Test

const Physics = MirnanNew.Physics

@testset "al_hisban_al_dalali Clifford semantic calculus" begin
    dir = mktempdir()
    mem = Physics.SemanticCalculusMemory()

    sig = Physics.sentence_semantic_signature("Knowledge grows through review")
    @test length(sig["vector"]) == 22
    @test sig["scalar"] isa Float64
    @test sig["bivector_norm"] >= 0.0

    trans = Physics.semantic_transform_signature(
        "How does learning grow?",
        "Learning grows through practice and review.",
    )
    @test length(trans["transform"]) == 22
    @test trans["scalar_shift"] isa Float64

    learned = Physics.learn_semantic_calculus_from_pair!(
        mem,
        "How does learning grow?",
        "Learning grows through practice and review.",
    )
    @test learned
    @test Physics.has_semantic_calculus(mem)

    rec = Physics.select_semantic_transform(mem, "How does knowledge grow?")
    @test rec !== nothing
    @test rec.relation == "method_answer"
    @test rec.count == 1

    terms = Physics.semantic_guidance_terms(mem, "How does knowledge grow?"; limit=6)
    @test "practice" in terms
    @test "review" in terms

    guidance = Physics.semantic_guidance(mem, "How does knowledge grow?"; limit=6)
    @test guidance["active"]
    @test guidance["relation"] == "method_answer"
    @test guidance["movement"] == "method"
    @test guidance["answer_frame"] == "mechanism_or_steps"
    @test guidance["confidence"] > 0.0
    @test "practice" in guidance["target_terms"]
    @test length(guidance["projected_signature"]) == 22
    @test Physics.semantic_relation_movement("definition_answer") == "definition"
    @test guidance["answer_plan"]["active"]
    @test guidance["answer_plan"]["plan_signature"] == "method:mechanism_or_steps:mechanism>means>result"
    @test [s["role"] for s in guidance["answer_plan"]["steps"]] == ["mechanism", "means", "result"]

    plan = Physics.semantic_answer_plan(mem, "How does knowledge grow?"; limit=6)
    @test plan["active"]
    @test plan["answer_frame"] == "mechanism_or_steps"
    @test any(step -> "practice" in step["terms"], plan["steps"])

    text_learned = Physics.learn_semantic_calculus_from_text!(
        mem,
        "What is wisdom?\nWisdom is clear judgment.\nKnowledge opens thought. Thought guides action.",
    )
    @test text_learned >= 2
    @test haskey(mem.records, "definition_answer")
    @test haskey(mem.records, "semantic_continuation")

    saved = Physics.save_semantic_calculus(mem, joinpath(dir, "al_hisban_al_dalali.json"))
    @test isfile(saved)

    loaded = Physics.load_semantic_calculus(saved)
    @test Physics.has_semantic_calculus(loaded)
    @test length(loaded.records) == length(mem.records)
    @test "practice" in Physics.semantic_guidance_terms(loaded, "How does knowledge grow?"; limit=8)

    empty_path = joinpath(dir, "empty_hisban.json")
    touch(empty_path)
    @test !Physics.has_semantic_calculus(Physics.load_semantic_calculus(empty_path))

    dirty = Physics.SemanticCalculusMemory()
    dirty.records["dirty"] = Physics.SemanticCalculusRecord(
        "semantic_continuation",
        fill(NaN, 22),
        fill(Inf, 22),
        fill(-Inf, 22),
        NaN,
        Inf,
        1,
        [Dict("source" => "a", "target" => "b")],
        Dict("b" => 1),
    )
    dirty_path = Physics.save_semantic_calculus(dirty, joinpath(dir, "dirty_hisban.json"))
    @test isfile(dirty_path)
    dirty_loaded = Physics.load_semantic_calculus(dirty_path)
    @test Physics.has_semantic_calculus(dirty_loaded)
    @test all(isfinite, dirty_loaded.records["semantic_continuation"].source_signature)
    @test dirty_loaded.records["semantic_continuation"].scalar_shift == 0.0

    gen = Physics.MirnanGenerator(
        Dict("Knowledge" => 1, "grows" => 2, "practice" => 3, "review" => 4);
        model_dir=dir,
    )
    summary = Physics.pattern_memory_summary(gen)
    @test summary["al_hisban_al_dalali"]["records"] >= 1
    @test summary["al_hisban_al_dalali"]["relations"]["method_answer"] >= 1
    @test "review" in Physics.semantic_guidance_terms(gen.hisban, "How does knowledge grow?"; limit=8)
end
