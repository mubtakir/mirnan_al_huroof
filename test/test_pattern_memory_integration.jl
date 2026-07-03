include("../src/MirnanNew.jl")
using .MirnanNew
using Test

const Physics = MirnanNew.Physics

@testset "pattern memory integration report" begin
    dir = mktempdir()

    lisan = Physics.LinguisticPatternMemory()
    Physics.train_lisan_from_texts!(lisan, ["Knowledge raises understanding."])
    Physics.save_lisan(lisan, joinpath(dir, "al_lisan.json"))

    code = Physics.CodePatternMemory()
    Physics.train_code_patterns_from_texts!(
        code,
        ["def add(a, b):\n    result = a + b\n    return result\n"],
    )
    Physics.save_al_code(code, joinpath(dir, "al_code.json"))

    tadbir = Physics.TadbirMemory()
    Physics.train_tadbir_from_texts!(
        tadbir,
        ["inspect failing output -> diagnose root cause -> patch the code -> run focused test -> report result"],
    )
    Physics.save_tadbir(tadbir, joinpath(dir, "al_tadbir.json"))

    hisab = Physics.HisabMemory()
    Physics.train_hisab_from_texts!(hisab, ["2 + 3", "2*x + 3 = 7"])
    Physics.save_hisab(hisab, joinpath(dir, "al_hisab.json"))

    ta3rif = Physics.Ta3rifMemory()
    Physics.train_ta3rif_from_texts!(ta3rif, ["Knowledge is organized understanding."])
    Physics.save_ta3rif(ta3rif, joinpath(dir, "al_ta3rif.json"))

    hisban = Physics.SemanticCalculusMemory()
    Physics.learn_semantic_calculus_from_pair!(
        hisban,
        "How does knowledge grow?",
        "Knowledge raises understanding through practice and review.",
    )
    Physics.save_semantic_calculus(hisban, joinpath(dir, "al_hisban_al_dalali.json"))

    nisba = Physics.NisbaMemory()
    Physics.train_nisba_from_texts!(nisba, ["Knowledge raises understanding because practice strengthens review."])
    Physics.save_nisba(nisba, joinpath(dir, "al_nisba.json"))

    vocab = Dict(
        "Knowledge" => 1,
        "knowledge" => 2,
        "raises" => 3,
        "understanding" => 4,
        "practice" => 5,
        "review" => 6,
        "code" => 7,
        "plan" => 8,
        "test" => 9,
        "math" => 10,
    )
    gen = Physics.MirnanGenerator(vocab; model_dir=dir)
    gen.self_review.enabled = false

    summary = Physics.pattern_memory_summary(gen)
    @test summary["al_lisan"]["patterns"] >= 1
    @test summary["al_lisan"]["languages"]["en"] >= 1
    @test summary["al_code"]["patterns"] >= 1
    @test summary["al_code"]["languages"]["python"] >= 1
    @test summary["al_tadbir"]["patterns"] >= 1
    @test summary["al_tadbir"]["domains"]["engineering"] >= 1
    @test summary["al_hisab"]["patterns"] >= 2
    @test summary["al_hisab"]["problem_types"]["linear_equation"] >= 1
    @test summary["al_ta3rif"]["subjects"] >= 1
    @test summary["al_ta3rif"]["definitions"] >= 1
    @test summary["al_hisban_al_dalali"]["records"] >= 1
    @test summary["al_hisban_al_dalali"]["relations"]["method_answer"] >= 1
    @test summary["al_hisban_al_dalali"]["movements"]["method"] >= 1
    @test summary["al_nisba"]["records"] >= 1
    @test summary["al_nisba"]["relation_types"]["causal"] >= 1
    @test haskey(summary, "al_aql")

    report = Physics.get_physics_report(gen, ["Knowledge", "raises"])
    @test haskey(report, "pattern_memories")
    @test report["pattern_memories"]["al_code"]["patterns"] == summary["al_code"]["patterns"]
    @test report["pattern_memories"]["al_ta3rif"]["subjects"] == summary["al_ta3rif"]["subjects"]
    @test report["pattern_memories"]["al_hisban_al_dalali"]["records"] == summary["al_hisban_al_dalali"]["records"]
    @test report["pattern_memories"]["al_nisba"]["records"] == summary["al_nisba"]["records"]

    guidance = Physics.semantic_guidance(gen.hisban, "How does knowledge grow?")
    @test guidance["movement"] == "method"
    @test "practice" in guidance["target_terms"]
    @test guidance["answer_plan"]["plan_signature"] == "method:mechanism_or_steps:mechanism>means>result"

    language_result = Physics.generate!(
        gen,
        "Knowledge";
        mode="standard",
        max_words=5,
    )
    language_lower = lowercase(language_result)
    @test occursin("knowledge", language_lower)
    @test any(w -> occursin(w, language_lower), ["raises", "understanding", "practice", "review"])

    code_result = Physics.generate!(
        gen,
        "python function named sum_values";
        mode="code",
        max_words=8,
    )
    @test occursin("def sum_values", code_result)
    @test occursin("return", code_result)

    tadbir_result = Physics.generate!(
        gen,
        "plan to debug code and run tests";
        mode="standard",
        max_words=8,
    )
    @test occursin("1. inspect", tadbir_result)
    @test occursin("run focused test", tadbir_result)

    hisab_result = Physics.generate!(
        gen,
        "2*x + 3 = 7";
        mode="math",
        max_words=8,
    )
    @test occursin("x = 2", hisab_result)
    @test occursin("verified: true", hisab_result)
end
