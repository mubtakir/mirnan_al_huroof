println("=== Testing SIO Math & Code ===")
using Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "src", "MirnanNew.jl"))
using .MirnanNew
using .MirnanNew.Physics

include(joinpath(@__DIR__, "train.jl"))

data = load_model()
vocab = data["vocab"]
K_sem = data["K_sem"]
K_syn = data["K_syn"]
println("Vocab: $(length(vocab)) words")

gen = MirnanNew.Physics.Generator.MirnanGenerator(vocab, K_sem; K_syn=K_syn)

sio = MirnanNew.SIO.SIOOrchestrator(gen)

safe(s::String, n::Int=80) = length(s) <= n ? s : s[1:nextind(s, 1, n)] * "..."

println("\n=== Math Tests ===")
math_tests = [
    "٢ + ٣",
    "٥ × ١٠",
    "١٠٠ مقسوم ٥",
    "جذر ٩",
    "مطلق -٥",
    "جذر (١٦) + ٢",
]
for goal in math_tests
    result = MirnanNew.SIO.synthesize!(sio, goal)
    mode = MirnanNew.Physics.MathBridgeModule.detect_mode(goal)
    println("  '$goal' (mode=$mode) -> '$(safe(result["deliverable"]))'")
end

println("\n=== Code Tests ===")
code_tests = [
    "def fibonacci(n)",
    "اكتب دالة تحسب مضروب عدد",
    "اكتب خوارزمية ترتيب في بايثون",
    "دالة للتحقق من عدد أولي",
]
for goal in code_tests
    result = MirnanNew.SIO.synthesize!(sio, goal)
    mode = MirnanNew.Physics.MathBridgeModule.detect_mode(goal)
    println("  '$goal' (mode=$mode) ->")
    println("    $(safe(result["deliverable"], 120))")
end

println("\n=== Text Tests ===")
text_tests = ["العلم نور", "الحمد لله"]
for goal in text_tests
    result = MirnanNew.SIO.synthesize!(sio, goal)
    mode = MirnanNew.Physics.MathBridgeModule.detect_mode(goal)
    println("  '$goal' (mode=$mode) -> '$(safe(result["deliverable"]))'")
end
