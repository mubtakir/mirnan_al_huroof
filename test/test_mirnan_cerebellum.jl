include("../src/MirnanNew.jl")
using .MirnanNew
using Test

const Physics = MirnanNew.Physics

println("=" ^ 60)
println("MIRNAN CEREBELLUM TEST")
println("=" ^ 60)

@testset "Mirnan cerebellum" begin
    cb = Physics.MirnanCerebellum()

    obs = Physics.observe_prompt(cb, ["ما", "معنى", "عين"];
                                 prompt="ما معنى عين",
                                 vocab_size=1000,
                                 k_density=0.0)
    @test "ambiguity" in obs.tags
    @test "meaning_query" in obs.tags

    policy = Physics.choose_policy!(cb, obs)
    @test policy.mode == "standard"
    @test policy.sense_mode == "measure"
    @test policy.confidence > 0.5
    @test haskey(policy.weight_multipliers, "density_resonance")

    weights = Dict(
        "density_resonance" => 2.0,
        "prompt_align" => 6.0,
        "context_tension" => 2.0,
        "root_affinity" => 4.0,
    )
    Physics.apply_cerebellum_policy!(weights, policy)
    @test weights["density_resonance"] > 2.0
    @test weights["prompt_align"] > 6.0

    reward = Physics.learn_from_outcome!(cb, obs, policy, "عين الماء نبع في الجبل")
    @test 0.0 <= reward <= 1.0
    @test !isempty(cb.history)

    state = Physics.MirnanCerebellum()
    restored = Physics.MirnanCerebellumModule.restore_cerebellum_state!(
        state,
        Physics.MirnanCerebellumModule.cerebellum_state_dict(cb),
    )
    @test restored
    @test !isempty(state.history)

    vocab = Dict("عين" => 1, "ماء" => 2, "نبع" => 3, "جاسوس" => 4, "بصر" => 5)
    gen = Physics.MirnanGenerator(vocab)
    @test gen.cerebellum isa Physics.MirnanCerebellum
    report = Physics.get_physics_report(gen, ["عين"])
    @test haskey(report, "cerebellum")
end

println("PASSED mirnan cerebellum")
