println("=== Wave-Based Generation Test ===")
using Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "src", "MirnanNew.jl"))
using .MirnanNew
using .MirnanNew.Physics
using .MirnanNew.Physics.WaveField: WaveContribution, wave_superposition, born_rule
using LinearAlgebra

include(joinpath(@__DIR__, "train.jl"))

println("\n=== Wave Superposition Test ===")
w1 = WaveContribution(5.0, 0.0)
w2 = WaveContribution(6.0, 0.1)
w3 = WaveContribution(5.0, π)
total = wave_superposition([w1, w2, w3])
println("  align(5, 0) + prompt(6, 0.1) + k_sem(5, π)")
println("  Amplitude: $(round(total.amplitude; digits=4))")
println("  Born P: $(round(born_rule(total); digits=4))")

w4 = WaveContribution(5.0, 0.0)
w5 = WaveContribution(6.0, 0.0)
w6 = WaveContribution(5.0, 0.0)
total2 = wave_superposition([w4, w5, w6])
println("\n  All same phase:")
println("  Amplitude: $(round(total2.amplitude; digits=4))")
println("  Born P: $(round(born_rule(total2); digits=4))")

println("\n=== Loading Model ===")
data = load_model()
vocab = data["vocab"]
K_sem = data["K_sem"]
K_syn = data["K_syn"]
println("Vocab size: $(length(vocab))")

gen = MirnanNew.Physics.Generator.MirnanGenerator(vocab, K_sem; K_syn=K_syn)

println("\n=== Wave-Based Generation ===")
for prompt in ["العلم نور", "مرحبا", "الحمد لله"]
    result = MirnanNew.generate!(gen, prompt; mode="standard", max_words=5)
    println("  '$prompt' -> '$result'")
end
