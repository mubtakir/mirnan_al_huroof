include("../src/MirnanNew.jl")
using .MirnanNew
using Test
using JSON

const Physics = MirnanNew.Physics
const IRP = Physics.IntentResponsePlanner
const Gen = Physics.Generator

@testset "dialogue salam and reported speech cleanup" begin
    salam = IRP.detect_response_intent("سلام")
    @test salam.intent == "dialogue"
    @test salam.subject == "greeting"
    @test Gen._dialogue_answer_incompatible("greeting", "الي اللقاء! اتمني لك يوما سعيدا.")
    @test !Gen._dialogue_answer_incompatible("greeting", "وعليكم السلام ورحمة الله.")

    trimmed = Gen._trim_reported_speech_prefix(
        ["قالت", "لها", "العلم", "نور", "والجهل", "ظلام"],
        ["العلم", "نور"],
    )
    @test trimmed == ["العلم", "نور", "والجهل", "ظلام"]

    facts_path = normpath(joinpath(@__DIR__, "..", "knowledge", "dialogue_facts.json"))
    facts = JSON.parsefile(facts_path)
    @test length(get(facts, "speech_acts", Any[])) >= 8

    gen = Gen.MirnanGenerator(Dict{String,Int}(
        "السلام" => 1,
        "عليكم" => 2,
        "سلام" => 3,
        "كيف" => 4,
        "الصحة" => 5,
        "الحمد" => 6,
        "بخير" => 7,
    ); model_dir=normpath(joinpath(@__DIR__, "..", "model")))
    @test any(f -> f.evidence == "persistent_dialogue_convention", gen.aql_space.speech_acts)
    @test occursin("وعليكم", Gen._dialogue_greeting_memory_answer(gen, "السلام عليكم"))
    @test occursin("الحمد", Gen._aql_speech_act_answer(gen, "كيف الصحة؟"))
    @test isempty(Gen._dialogue_greeting_memory_answer(gen, "ما معنى السلام؟"))
end
