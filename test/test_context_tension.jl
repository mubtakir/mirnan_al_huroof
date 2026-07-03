include("../src/MirnanNew.jl")
using .MirnanNew
using Test

const Physics = MirnanNew.Physics
const GeneratorModule = MirnanNew.Physics.Generator

println("=" ^ 60)
println("CONTEXT TENSION TEST")
println("=" ^ 60)

@testset "Context tension scoring" begin
    vocab = Dict(
        "العلم" => 1,
        "نور" => 2,
        "تعلم" => 3,
        "بحر" => 4,
    )

    gen = Physics.MirnanGenerator(vocab)
    for key in collect(keys(gen.scoring_weights))
        gen.scoring_weights[key] = 0.0
    end
    gen.scoring_weights["context_tension"] = 8.0

    context = ["العلم", "نور"]
    prompt_pv = [GeneratorModule._pv(gen, w) for w in context]
    used = Set(context)

    related_score, _ = GeneratorModule._score(
        gen, "تعلم", used, prompt_pv, prompt_pv;
        prev_word="نور", context_words=context,
    )
    unrelated_score, _ = GeneratorModule._score(
        gen, "بحر", used, prompt_pv, prompt_pv;
        prev_word="نور", context_words=context,
    )

    @test isfinite(related_score)
    @test isfinite(unrelated_score)
    @test related_score > unrelated_score
end

println("PASSED context tension")
