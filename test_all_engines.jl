println("=== Testing All Engines Wired ===")
using Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "src", "MirnanNew.jl"))
using .MirnanNew
using .MirnanNew.Physics
using .MirnanNew.Physics.WaveField: WaveContribution, wave_superposition, born_rule
using LinearAlgebra

include(joinpath(@__DIR__, "train.jl"))

data = load_model()
vocab = data["vocab"]
K_sem = data["K_sem"]
K_syn = data["K_syn"]
println("Vocab: $(length(vocab)) words")

gen = MirnanNew.Physics.Generator.MirnanGenerator(vocab, K_sem; K_syn=K_syn)

println("\n=== Wave-Based Generation (All Engines) ===")
for prompt in ["العلم نور", "مرحبا", "الحمد لله", "الماء حياة", "الكتاب مفتاح"]
    result = MirnanNew.generate!(gen, prompt; mode="standard", max_words=5)
    println("  '$prompt' -> '$result'")
end

println("\n=== Wave Superposition Verification ===")
for (label, waves) in [
    ("All constructive (0,0,0)", [WaveContribution(5,0.0), WaveContribution(6,0.0), WaveContribution(5,0.0)]),
    ("Two align, one opposes", [WaveContribution(5,0.0), WaveContribution(6,0.0), WaveContribution(5,π)]),
    ("All random phases", [WaveContribution(5,0.5), WaveContribution(6,1.2), WaveContribution(5,2.8)]),
    ("Two cancel, one survives", [WaveContribution(5,0.0), WaveContribution(5,π), WaveContribution(3,0.0)]),
]
    total = wave_superposition(waves)
    sum_amp = sum(w.amplitude for w in waves)
    println("  $label:")
    println("    sum_amp=$sum_amp -> total_amp=$(round(total.amplitude; digits=4)), P=$(round(born_rule(total); digits=4))")
end
