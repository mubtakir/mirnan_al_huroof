using Test

include("../src/MirnanNew.jl")
using .MirnanNew

const Aql = MirnanNew.Physics.AlAql
const Gen = MirnanNew.Physics.Generator

@testset "al_aql dialogue speech acts" begin
    space = Aql.SimulationSpace()
    Aql.train_from_text!(space, """
    سؤال: السلام عليكم
    جواب: وعليكم السلام ورحمة الله وبركاته.

    كيف حالك؟
    أنا بخير والحمد لله.
    """)

    @test length(space.speech_acts) >= 2
    greeting = first(filter(f -> occursin("السلام", f.content), space.speech_acts))
    @test greeting.act_type == "تحية"
    @test greeting.response_act == "رد_تحية"
    @test occursin("وعليكم", greeting.response_content)

    wellbeing = first(filter(f -> occursin("كيف حالك", f.content), space.speech_acts))
    @test wellbeing.act_type == "سؤال_حال"
    @test wellbeing.response_act == "جواب_حال"

    matches = Aql.speech_responses_for(space, "السلام عليكم")
    @test !isempty(matches)
    @test any(f -> occursin("وعليكم", f.response_content), matches)

    saved = Gen._aql_runtime_dict(space)
    restored = Aql.SimulationSpace()
    Gen._restore_aql_runtime!(restored, saved)
    @test length(restored.speech_acts) == length(space.speech_acts)
    @test any(f -> f.response_act == "رد_تحية", restored.speech_acts)

    adl_space = Aql.SimulationSpace()
    Aql.compile_adl!(adl_space, """
    فعل_قولي { متكلم: س; نوع: تحية; محتوى: السلام عليكم; مجيب: ج; نوع_الرد: رد_تحية; محتوى_الرد: وعليكم السلام ورحمة الله وبركاته; ثقة: 0.9 }
    speech_act { speaker: user; act_type: question; content: كيف حالك; responder: mirnan; response_act: answer; response_content: بخير والحمد لله; confidence: 0.8 }
    """)

    @test length(adl_space.speech_acts) == 2
    @test any(f -> f.speaker == "س" && f.response_act == "رد_تحية", adl_space.speech_acts)
    @test any(f -> f.speaker == "user" && f.response_act == "answer", adl_space.speech_acts)

    noisy_space = Aql.SimulationSpace()
    noisy_text = repeat("هذه فقرة تعليمية طويلة عن العلوم والمعرفة وفي وسطها كلمة مرحبا لكنها ليست حوارا منظما. ", 12)
    Aql.train_from_text!(noisy_space, noisy_text * "\nهذا سطر تال لا ينبغي أن يصبح ردا.")
    @test isempty(noisy_space.speech_acts)
end
