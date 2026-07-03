include("../src/MirnanNew.jl")
using .MirnanNew
using Test
using JSON

const Physics = MirnanNew.Physics
const Gen = Physics.Generator

@testset "strict no-template generation gates" begin
    old_strict = get(ENV, "MIRNAN_STRICT_NO_TEMPLATES", nothing)
    old_relation_cache = Gen._SEMANTIC_RELATION_KNOWLEDGE[]
    old_istinbat = Gen._LEARNED_ISTINBAT_MEMORY[]
    try
        ENV["MIRNAN_STRICT_NO_TEMPLATES"] = "1"
        Gen._SEMANTIC_RELATION_KNOWLEDGE[] = nothing
        Gen._LEARNED_ISTINBAT_MEMORY[] = nothing

        gen = Physics.MirnanGenerator(Dict("العلم" => 1, "نور" => 2); model_dir=mktempdir())

        @test isempty(Gen._yesno_declarative_field_answer(gen,
            "هل العلم نور؟",
            ["هل", "العلم", "نور"]))

        @test isempty(Gen._semantic_contradiction_yesno_answer(
            "هل العدل شر؟",
            ["هل", "العدل", "شر"]))

        @test isempty(Gen._semantic_contradiction_yesno_answer(
            "هل الجهل نور؟",
            ["هل", "الجهل", "نور"]))

        @test isempty(Gen._semantic_relation_memory_answer(
            ["ما", "الذي", "يجعل", "العلم", "كالمصباح"]))

        @test isempty(Gen._semantic_relation_gate_answer(
            "ما الذي يجعل العلم كالمصباح؟",
            ["ما", "الذي", "يجعل", "العلم", "كالمصباح"]))

        @test isempty(Gen._negated_how_relation_answer(
            ["كيف", "لا", "تهذب", "الرحمة", "القوة"]))

        @test isempty(Gen._canonical_yesno_relation_answer(
            ["هل", "العلم", "يزيد", "الفهم"]))

        @test isempty(Gen._canonical_explanatory_relation_answer(
            ["لماذا", "يزيد", "العلم", "الفهم"]))

        @test isempty(Gen._difference_concept_gloss("العلم"))

        facts_path = joinpath(@__DIR__, "..", "knowledge", "semantic_relation_facts.json")
        answers_path = joinpath(@__DIR__, "..", "knowledge", "semantic_relation_answers.json")
        @test isfile(facts_path)
        @test !isfile(answers_path)
        facts = JSON.parsefile(facts_path)
        @test all(item -> !haskey(item, "answer"), get(facts, "records", Any[]))
        @test all(item -> !haskey(item, "evidence"), get(facts, "records", Any[]))
        @test all(item -> haskey(item, "relation_type") && haskey(item, "terms"), get(facts, "records", Any[]))

        generator_source = read(joinpath(@__DIR__, "..", "src", "physics", "engines", "generator.jl"), String)
        @test !occursin("Legacy literal fallback", generator_source)
        @test !occursin("\\u064a\\u0634\\u0628\\u0647 \\u0627\\u0644\\u0639\\u0644\\u0645", generator_source)
        @test !occursin("\\u062a\\u062a\\u062d\\u0648\\u0644 \\u0627\\u0644\\u0642\\u0648\\u0629", generator_source)
        @test !occursin("\\u064a\\u0645\\u0646\\u0639 \\u0627\\u0644\\u0639\\u0642\\u0644", generator_source)

        nisba_mem = Physics.NisbaMemory()
        Physics.train_nisba_from_texts!(nisba_mem, ["العلم يفتح الفهم، والفهم يهدي الى الرحمه."])
        gen.nisba = nisba_mem
        @test isempty(Gen._nisba_relation_answer(gen,
            "ما العلاقة بين العلم والفهم؟",
            ["ما", "العلاقة", "بين", "العلم", "والفهم"],
            Gen.detect_response_intent("ما العلاقة بين العلم والفهم؟")))
    finally
        if old_strict === nothing
            delete!(ENV, "MIRNAN_STRICT_NO_TEMPLATES")
        else
            ENV["MIRNAN_STRICT_NO_TEMPLATES"] = old_strict
        end
        Gen._SEMANTIC_RELATION_KNOWLEDGE[] = old_relation_cache
        Gen._LEARNED_ISTINBAT_MEMORY[] = old_istinbat
    end
end
