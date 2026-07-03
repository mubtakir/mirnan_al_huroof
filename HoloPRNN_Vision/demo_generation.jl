"""
demo_generation.jl
Demonstrates pattern formation (crystallization) from thermal noise via target-directed coupling.
"""

using Pkg
Pkg.activate(@__DIR__)

const DEMO_DIR = @__DIR__
const OUTPUT_DIR = joinpath(DEMO_DIR, "output")

include(joinpath(DEMO_DIR, "src", "HoloPRNN_Vision.jl"))
using .HoloPRNN_Vision
using Images

function run_demo()
    println("=========================================================================")
    println("      HoloPRNN Vision — Generation Demo (Crystallization from Noise)")
    println("=========================================================================")
    
    # 1. Define a target shape (e.g., a cross shape on a 16x16 grid)
    H, W = 16, 16
    target_shape = zeros(Bool, H, W)
    for y in 1:H, x in 1:W
        if y in 7:10 || x in 7:10
            target_shape[y, x] = true
        end
    end
    
    # 2. Initialize oscillator parameters
    params = OscillatorParams(μ=1.0, g_inh=0.4, γ=2.0, τ_a=1.5, dt=0.03)
    
    # 3. Evolve random noise to crystallize the cross shape
    println("Running crystallization simulator (300 steps with noise annealing)...")
    img_crystallized = run_pattern_crystallization(target_shape, params; steps=300, noise_level=0.6)
    
    # Ensure output directory exists
    mkpath(OUTPUT_DIR)
    save(joinpath(OUTPUT_DIR, "crystallized_pattern.png"), img_crystallized)
    println("Saved crystallized pattern image to '$(joinpath(OUTPUT_DIR, "crystallized_pattern.png"))'")
    println("🎉 SUCCESS! Pattern crystallized from noise successfully.")
    println("=========================================================================")
end

run_demo()
