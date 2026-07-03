println("=== Testing SIO Orchestrator ===")
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

println("\n=== SIO Multi-Phase Generation ===")
sio = MirnanNew.SIO.SIOOrchestrator(gen)

# Test 1: Simple goal
println("\n--- Goal: 'العلم نور' ---")
result1 = MirnanNew.SIO.synthesize!(sio, "العلم نور")
println("  Phases completed: $(result1["phases_completed"])")
println("  Phases failed: $(result1["phases_failed"])")
println("  Total time: $(result1["time_elapsed"])s")
if !isempty(result1["deliverable"])
    println("  Deliverable: $(result1["deliverable"][1:min(100, end)])...")
end

# Test 2: Report goal (multi-phase)
println("\n--- Goal: 'تقرير عن الفيزياء' ---")
result2 = MirnanNew.SIO.synthesize!(sio, "تقرير عن الفيزياء")
println("  Phases: $(result2["phases_completed"])")
println("  Time: $(result2["time_elapsed"])s")
for detail in result2["phase_details"]
    println("    $(detail["name"]): coherence=$(round(detail["coherence"]; digits=2)), iterations=$(detail["iterations"])")
end

println("\n=== Done ===")
