include("../src/MirnanNew.jl")
using .MirnanNew
using Test

const Physics = MirnanNew.Physics
const CL = MirnanNew.Physics.ContextualLearning

println("=" ^ 60)
println("CONTEXTUAL LEARNING TEST")
println("=" ^ 60)

@testset "Contextual learning" begin
    vocab = Dict(
        "جيش" => 1,
        "الليل" => 2,
        "زحف" => 3,
        "غطى" => 4,
        "الظلام" => 5,
        "السكون" => 6,
        "قبيلة" => 7,
        "روايتي" => 8,
    )

    gen = Physics.MirnanGenerator(vocab)

    intent = CL.detect_question_intent(gen.contextual_learning.question_intents, split("ماذا فعل جيش الليل"))
    @test intent.kind == "action_query"
    @test intent.subject_tokens == ["جيش", "الليل"]

    compounds = CL.detect_compounds(gen.contextual_learning.compounds, split("ماذا فعل جيش الليل"))
    @test !isempty(compounds)
    @test compounds[1].text == "جيش الليل"
    @test length(compounds[1].vector) == Physics.Constants.TOTAL_DIM

    answer = Physics.generate!(gen, "ماذا فعل جيش الليل"; max_words=6)
    @test occursin("جيش الليل", answer)
    @test occursin("زحف", answer)
    @test occursin("غطى", answer)

    Physics.learn_from_feedback!(
        gen,
        "ماذا فعل جيش الليل",
        "زحف جيش الليل فغطى المكان بالسكون";
        rating=1.0,
        note="جيش الليل صورة مجازية",
    )
    @test !isempty(gen.contextual_learning.feedback_log)
    @test haskey(gen.contextual_learning.effects.effects, "زحف")

    Physics.learn_from_feedback!(
        gen,
        "ماذا فعل جيش الليل",
        "جيش الليل قبيلة في الرواية";
        rating=1.0,
        note="جيش الليل اسم قبيلة في روايتي",
    )
    @test get(gen.contextual_learning.entity_kinds, "جيش الليل", "") == "اسم"

    entity_answer = Physics.generate!(gen, "ماذا فعل جيش الليل"; max_words=6)
    @test occursin("كيان", entity_answer)
    @test occursin("سياق", entity_answer)
end

println("PASSED contextual learning")
