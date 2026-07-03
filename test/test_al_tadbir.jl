include("../src/MirnanNew.jl")
using .MirnanNew
using Test

const Physics = MirnanNew.Physics

@testset "al_tadbir procedural plan memory" begin
    dir = mktempdir()
    mem = Physics.TadbirMemory()

    engineering_plan = "inspect failing output -> diagnose root cause -> patch the code -> run focused test -> report result"
    research_plan = "read sources -> compare claims -> summarize evidence -> report conclusion"

    learned = Physics.train_tadbir_from_texts!(mem, [engineering_plan, research_plan])

    @test learned == 2
    @test Physics.has_tadbir_patterns(mem)

    rec = Physics.select_tadbir_pattern(mem, "plan to debug code and run tests")
    @test rec !== nothing
    @test rec.domain == "engineering"
    @test rec.roles == ["observe", "diagnose", "execute", "verify", "report"]
    @test !isempty(Physics.preferred_tadbir_slot_values(rec, "diagnose"))

    steps = Physics.render_tadbir_plan(mem, "plan to debug code and run tests")
    @test length(steps) == 5
    @test occursin("inspect", steps[1])
    @test occursin("test", steps[4])

    saved = Physics.save_tadbir(mem, joinpath(dir, "al_tadbir.json"))
    @test isfile(saved)

    loaded = Physics.load_tadbir(saved)
    @test Physics.has_tadbir_patterns(loaded)

    gen = Physics.MirnanGenerator(Dict("plan" => 1, "code" => 2, "test" => 3); model_dir=dir)
    gen.self_review.enabled = false
    result = Physics.generate!(
        gen,
        "plan to debug code and run tests";
        mode="standard",
        max_words=8,
    )

    @test occursin("1. inspect", result)
    @test occursin("diagnose", result)
    @test occursin("run focused test", result)
end
