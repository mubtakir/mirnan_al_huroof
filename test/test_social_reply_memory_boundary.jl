include("../src/MirnanNew.jl")
using .MirnanNew
using Test

const Physics = MirnanNew.Physics
const IRP = Physics.IntentResponsePlanner
const Gen = Physics.Generator

function _tiny_social_generator()
    vocab = Dict{String,Int}(
        "السلام" => 1,
        "عليكم" => 2,
        "كيف" => 3,
        "حالك" => 4,
        "شكرا" => 5,
        "اسمك" => 6,
        "مرنان" => 7,
        "العدل" => 8,
        "يحفظ" => 9,
    )
    gen = Gen.MirnanGenerator(vocab; model_dir=mktempdir())
    Gen._load_persistent_dialogue_facts!(gen, normpath(joinpath(@__DIR__, "..", "knowledge", "dialogue_facts.json")))
    return gen
end

@testset "social reply memory is the only ready-answer boundary" begin
    old_strict = get(ENV, "MIRNAN_STRICT_NO_TEMPLATES", nothing)
    try
        ENV["MIRNAN_STRICT_NO_TEMPLATES"] = "1"
        gen = _tiny_social_generator()

        social_cases = [
            ("السلام عليكم", "greeting", "وعليكم"),
            ("كيف حالك؟", "wellbeing", "الحمد"),
            ("شكرا", "thanks", "عفوا"),
            ("ما اسمك؟", "identity", "مرنان"),
        ]

        for (prompt, subject, expected) in social_cases
            plan = IRP.detect_response_intent(prompt)
            @test plan.intent == "dialogue"
            @test plan.subject == subject
            answer = Gen._aql_answer!(gen, prompt)
            @test occursin(expected, answer)
        end

        non_social_cases = [
            "ما معنى السلام؟",
            "هل العدل يحفظ السلام؟",
            "لماذا يحفظ العدل السلام؟",
        ]

        for prompt in non_social_cases
            plan = IRP.detect_response_intent(prompt)
            @test !(plan.intent == "dialogue" && plan.subject == "greeting")
            @test isempty(Gen._dialogue_greeting_memory_answer(gen, prompt))
            @test isempty(Gen._aql_answer!(gen, prompt))
        end
    finally
        if old_strict === nothing
            delete!(ENV, "MIRNAN_STRICT_NO_TEMPLATES")
        else
            ENV["MIRNAN_STRICT_NO_TEMPLATES"] = old_strict
        end
    end
end
