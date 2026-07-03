include("../src/MirnanNew.jl")
using .MirnanNew
using Test

const Physics = MirnanNew.Physics

@testset "al_hisab verified math memory" begin
    dir = mktempdir()
    mem = Physics.HisabMemory()

    learned = Physics.train_hisab_from_texts!(mem, ["2 + 3", "2*x + 3 = 7"])

    @test learned == 2
    @test Physics.has_hisab_patterns(mem)
    @test haskey(mem.patterns, "binary_arithmetic")
    @test haskey(mem.patterns, "linear_equation")

    sum_sol = Physics.solve_hisab(mem, "2 + 3")
    @test sum_sol !== nothing
    @test sum_sol.problem_type == "binary_arithmetic"
    @test sum_sol.result == "5"
    @test sum_sol.verified
    @test occursin("2 + 3 = 5", sum_sol.check)

    arabic_digit_sol = Physics.solve_hisab(mem, "\u0665\u0669\u0669 + \u0661")
    @test arabic_digit_sol !== nothing
    @test arabic_digit_sol.problem_type == "binary_arithmetic"
    @test arabic_digit_sol.result == "600"
    @test arabic_digit_sol.verified

    area_sol = Physics.solve_hisab(mem, "\u0627\u062d\u0633\u0628 \u0645\u0633\u0627\u062d\u0629 \u0645\u0631\u0628\u0639 \u0637\u0648\u0644 \u0636\u0644\u0639\u0647 5.")
    @test area_sol !== nothing
    @test area_sol.problem_type == "square_area"
    @test area_sol.result == "25"
    @test area_sol.verified

    root_sol = Physics.solve_hisab(mem, "\u0645\u0627 \u0647\u0648 \u062c\u0630\u0631 16\u061f")
    @test root_sol !== nothing
    @test root_sol.problem_type == "square_root"
    @test root_sol.result == "4"
    @test root_sol.verified

    remainder_sol = Physics.solve_hisab(mem, "\u0625\u0630\u0627 \u0643\u0627\u0646 \u0644\u062f\u064a\u0643 10 \u062a\u0641\u0627\u062d\u0627\u062a \u0648\u0623\u0643\u0644\u062a 3\u060c \u0643\u0645 \u062a\u0628\u0642\u0649\u061f")
    @test remainder_sol !== nothing
    @test remainder_sol.problem_type == "remainder_word_problem"
    @test remainder_sol.result == "7"
    @test remainder_sol.verified

    huge_mem = Physics.HisabMemory()
    @test Physics.train_hisab_from_texts!(huge_mem, ["78978030007213340000 + 1"]) == 1
    huge_saved = Physics.save_hisab(huge_mem, joinpath(dir, "al_hisab_huge.json"))
    @test isfile(huge_saved)
    @test Physics.has_hisab_patterns(Physics.load_hisab(huge_saved))

    linear_sol = Physics.solve_hisab(mem, "2*x + 3 = 7")
    @test linear_sol !== nothing
    @test linear_sol.problem_type == "linear_equation"
    @test linear_sol.result == "x = 2"
    @test linear_sol.verified
    @test occursin("2 * 2 + 3 = 7", linear_sol.check)

    rendered = Physics.render_hisab_solution(linear_sol)
    @test occursin("verify by substitution", rendered)
    @test occursin("verified: true", rendered)

    rendered_ar = Physics.render_hisab_solution_ar(sum_sol)
    @test occursin("\u0645\u062c\u0645\u0648\u0639 2 + 3 \u0647\u0648 5", rendered_ar)
    @test occursin("\u0627\u0644\u062a\u062d\u0642\u0642: 2 + 3 = 5", rendered_ar)
    @test occursin("\u0645\u0624\u0643\u062f: \u0646\u0639\u0645", rendered_ar)

    saved = Physics.save_hisab(mem, joinpath(dir, "al_hisab.json"))
    @test isfile(saved)

    loaded = Physics.load_hisab(saved)
    @test Physics.has_hisab_patterns(loaded)

    gen = Physics.MirnanGenerator(Dict("math" => 1); model_dir=dir)
    gen.self_review.enabled = false
    result = Physics.generate!(
        gen,
        "2*x + 3 = 7";
        mode="math",
        max_words=8,
    )

    @test occursin("x = 2", result)
    @test occursin("verified: true", result)

    arabic_result = Physics.generate!(
        gen,
        "\u0645\u0627 \u0645\u062c\u0645\u0648\u0639 1 + 6";
        mode="math",
        max_words=8,
    )

    @test occursin("\u0645\u062c\u0645\u0648\u0639 1 + 6 \u0647\u0648 7", arabic_result)
    @test occursin("\u0645\u0624\u0643\u062f: \u0646\u0639\u0645", arabic_result)

    remainder_result = Physics.generate!(
        gen,
        "\u0625\u0630\u0627 \u0643\u0627\u0646 \u0644\u062f\u064a\u0643 10 \u062a\u0641\u0627\u062d\u0627\u062a \u0648\u0623\u0643\u0644\u062a 3\u060c \u0643\u0645 \u062a\u0628\u0642\u0649\u061f";
        mode="auto",
        max_words=12,
    )
    @test occursin("\u0627\u0644\u0628\u0627\u0642\u064a \u0647\u0648 7", remainder_result)
end
