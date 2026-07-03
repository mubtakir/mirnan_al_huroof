include("../src/MirnanNew.jl")
using .MirnanNew
using Test

const Physics = MirnanNew.Physics

@testset "al_code code pattern memory" begin
    dir = mktempdir()
    mem = Physics.CodePatternMemory()

    py_code = """
def add(a, b):
    result = a + b
    return result
"""

    jl_code = """
function add(a, b)
    result = a + b
    return result
end
"""

    jl_module = """
module Tools
struct Item
end
end
"""

    jl_struct = """
struct Item
end
"""

    jl_testset = """
@testset "addition" begin
    @test add(1, 2) == 3
end
"""

    jl_try = """
try
    value = risky()
catch e
    value = nothing
end
"""

    bayan_code = "\u062f\u0627\u0644\u0629 \u0627\u0628\u062f\u0623\n" *
                 "    \u0627\u0637\u0628\u0639 \"ok\"\n" *
                 "\u0646\u0647\u0627\u064a\u0629\n"

    learned = Physics.train_code_patterns_from_texts!(
        mem,
        [py_code, jl_code, jl_module, jl_struct, jl_testset, jl_try, bayan_code],
    )

    @test learned == 7
    @test Physics.has_code_patterns(mem)

    py_rec = Physics.select_code_pattern(mem, "python function named sum_values"; language="python")
    @test py_rec !== nothing
    @test py_rec.language == "python"
    @test "function_def" in py_rec.roles
    @test "return" in py_rec.roles
    @test "add" in Physics.preferred_code_slot_values(py_rec, "name")

    jl_rec = Physics.select_code_pattern(mem, "julia function"; language="julia")
    @test jl_rec !== nothing
    @test jl_rec.language == "julia"
    @test "end" in jl_rec.roles

    module_rec = Physics.select_code_pattern(mem, "julia module"; language="julia")
    @test module_rec !== nothing
    @test "module_def" in module_rec.roles

    struct_rec = Physics.select_code_pattern(mem, "julia struct"; language="julia")
    @test struct_rec !== nothing
    @test "struct_def" in struct_rec.roles

    testset_rec = Physics.select_code_pattern(mem, "julia testset"; language="julia")
    @test testset_rec !== nothing
    @test "testset" in testset_rec.roles

    try_rec = Physics.select_code_pattern(mem, "julia try catch"; language="julia")
    @test try_rec !== nothing
    @test "try" in try_rec.roles
    @test "catch" in try_rec.roles

    bayan_rec = Physics.select_code_pattern(mem, "bayan function"; language="bayan")
    @test bayan_rec !== nothing
    @test bayan_rec.language == "bayan"
    @test "function_def" in bayan_rec.roles

    saved = Physics.save_al_code(mem, joinpath(dir, "al_code.json"))
    @test isfile(saved)

    loaded = Physics.load_al_code(saved)
    @test Physics.has_code_patterns(loaded)

    generated = Physics.generate_code_from_pattern(
        loaded,
        "python function named sum_values";
        language="python",
    )
    @test occursin("def sum_values", generated)
    @test occursin("return", generated)

    generated_add = Physics.generate_code_from_pattern(
        loaded,
        "write python function add two numbers";
        language="python",
    )
    @test occursin("def add(a, b):", generated_add)
    @test occursin("return a + b", generated_add)
    @test !occursin("pass", generated_add)

    generated_add_without_memory = Physics.generate_code_from_pattern(
        Physics.CodePatternMemory(),
        "write python function add two numbers";
        language="python",
    )
    @test occursin("def add(a, b):", generated_add_without_memory)
    @test occursin("return a + b", generated_add_without_memory)
    @test !occursin("pass", generated_add_without_memory)

    generated_struct = Physics.generate_code_from_pattern(
        loaded,
        "julia struct named Box";
        language="julia",
    )
    @test occursin("struct Box", generated_struct)

    generated_try = Physics.generate_code_from_pattern(
        loaded,
        "julia try catch";
        language="julia",
    )
    @test occursin("try", generated_try)
    @test occursin("catch", generated_try)

    generated_loop = Physics.generate_code_from_pattern(
        loaded,
        "\u0627\u0643\u062a\u0628 \u062d\u0644\u0642\u0629 \u062a\u0643\u0631\u0627\u0631 \u062a\u0637\u0628\u0639 \u0627\u0644\u0623\u0639\u062f\u0627\u062f \u0645\u0646 1 \u0625\u0644\u0649 10";
        language="python",
    )
    @test occursin("for i in range(1, 11):", generated_loop)
    @test occursin("print(i)", generated_loop)

    generated_loop_to_five = Physics.generate_code_from_pattern(
        loaded,
        "\u0627\u0643\u062a\u0628 \u062d\u0644\u0642\u0629 \u0628\u0627\u064a\u062b\u0648\u0646 \u062a\u0637\u0628\u0639 \u0627\u0644\u0623\u0639\u062f\u0627\u062f \u0645\u0646 1 \u0625\u0644\u0649 5";
        language="python",
    )
    @test occursin("for i in range(1, 6):", generated_loop_to_five)
    @test !occursin("range(1, 11)", generated_loop_to_five)

    generated_condition = Physics.generate_code_from_pattern(
        loaded,
        "\u0627\u0643\u062a\u0628 \u0634\u0631\u0637\u0627 \u064a\u062a\u062d\u0642\u0642 \u0625\u0630\u0627 \u0643\u0627\u0646 \u0627\u0644\u0631\u0642\u0645 \u0623\u0643\u0628\u0631 \u0645\u0646 10";
        language="python",
    )
    @test occursin("if number > 10:", generated_condition)

    generated_julia_variable = Physics.generate_code_from_pattern(
        loaded,
        "\u0643\u064a\u0641 \u062a\u0639\u0631\u0641 \u0645\u062a\u063a\u064a\u0631\u0627 \u0641\u064a \u062c\u0648\u0644\u064a\u0627\u061f";
        language="julia",
    )
    @test strip(generated_julia_variable) == "x = 1"

    generated_max = Physics.generate_code_from_pattern(
        loaded,
        "\u0627\u0643\u062a\u0628 \u062f\u0627\u0644\u0629 \u062a\u0631\u062c\u0639 \u0623\u0643\u0628\u0631 \u0639\u062f\u062f\u064a\u0646.";
        language="python",
    )
    @test occursin("def max_two(a, b):", generated_max)
    @test occursin("return a if a > b else b", generated_max)

    gen = Physics.MirnanGenerator(Dict("code" => 1); model_dir=dir)
    code_result = Physics.generate!(
        gen,
        "python function named sum_values";
        mode="code",
        max_words=8,
    )
    @test occursin("def sum_values", code_result)
    @test occursin("return", code_result)

    code_add = Physics.generate!(
        gen,
        "write python function add two numbers";
        mode="code",
        max_words=12,
    )
    @test occursin("def add(a, b):", code_add)
    @test occursin("return a + b", code_add)
    @test !occursin("pass", code_add)

    code_auto_julia = Physics.generate!(
        gen,
        "\u0643\u064a\u0641 \u062a\u0639\u0631\u0641 \u0645\u062a\u063a\u064a\u0631\u0627 \u0641\u064a \u062c\u0648\u0644\u064a\u0627\u061f";
        mode="auto",
        max_words=8,
    )
    @test strip(code_auto_julia) == "x = 1"
end
