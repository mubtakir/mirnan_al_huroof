include("../src/MirnanNew.jl")
using .MirnanNew
using Test

const Physics = MirnanNew.Physics

println("=" ^ 60)
println("SENSE SUPERPOSITION TEST")
println("=" ^ 60)

@testset "Sense superposition" begin
    @test Physics.has_sense_inventory("عين")
    @test Physics.has_sense_inventory("جذر")

    no_context = Physics.measure_senses("عين", String[])
    @test !no_context.collapsed
    @test no_context.reason == "mixed_state"

    spring = Physics.measure_senses("عين", ["شرب", "ماء", "الجبل", "نبع"])
    @test spring.selected_id == "spring"
    @test spring.collapsed
    @test spring.confidence > 0.55

    spy = Physics.measure_senses("عين", ["جاسوس", "مراقبة", "العدو", "سر"])
    @test spy.selected_id == "spy"
    @test spy.collapsed

    math_root = Physics.measure_senses("جذر", ["حل", "معادلة", "تربيعي", "عدد"])
    @test math_root.selected_id == "math"
    @test math_root.collapsed

    explanation = Physics.explain_measurement(spring)
    @test occursin("انهارت", explanation)
    @test occursin("نبع", explanation)
end

println("PASSED sense superposition")
