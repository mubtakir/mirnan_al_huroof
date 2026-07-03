"""
demo_inpainting.jl
Demonstrates wave-based image completion (inpainting) on a corrupted geometric shape.
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
    println("      HoloPRNN Vision — Inpainting Demo (Wave Image Completion)")
    println("=========================================================================")
    
    # 1. Create a simple 16x16 image: Red square on a Green background
    H, W = 16, 16
    img_orig = [y in 5:12 && x in 5:12 ? RGB(1.0, 0.0, 0.0) : RGB(0.0, 0.8, 0.0) for y in 1:H, x in 1:W]
    
    # 2. Corrupt it: draw a black 4x4 hole in the middle (pixels 7:10, 7:10)
    img_corrupted = copy(img_orig)
    mask = zeros(Bool, H, W)
    for y in 7:10, x in 7:10
        img_corrupted[y, x] = RGB(0.0, 0.0, 0.0)
        mask[y, x] = true
    end
    
    # Ensure output directory exists
    mkpath(OUTPUT_DIR)
    save(joinpath(OUTPUT_DIR, "inpainting_before.png"), img_corrupted)
    println("Saved corrupted image to '$(joinpath(OUTPUT_DIR, "inpainting_before.png"))'")
    
    # 3. Initialize oscillator parameters
    params = OscillatorParams(μ=1.0, g_inh=0.4, γ=2.0, τ_a=1.5, dt=0.03)
    
    # 4. Run wave-based inpainting
    println("Running wave propagation inpainting (350 steps)...")
    img_restored = run_inpainting(img_corrupted, mask, params; D_spatial=0.3, steps=350)
    
    save(joinpath(OUTPUT_DIR, "inpainting_after.png"), img_restored)
    println("Saved restored image to '$(joinpath(OUTPUT_DIR, "inpainting_after.png"))'")
    
    # Calculate recovery quality: check difference in the corrupted region
    diff = 0.0
    for y in 7:10, x in 7:10
        c1, c2 = img_orig[y, x], img_restored[y, x]
        diff += abs(Float64(c1.r) - Float64(c2.r)) + abs(Float64(c1.g) - Float64(c2.g)) + abs(Float64(c1.b) - Float64(c2.b))
    end
    mean_diff = diff / 48.0 # 16 pixels * 3 channels
    println("Mean reconstruction error in the corrupted region: $(round(mean_diff, digits=4))")
    
    if mean_diff < 0.25
        println("🎉 SUCCESS! Wave propagation completed the image successfully.")
    else
        println("⚠️ Reconstruction quality is low. Adjust coupling parameters.")
    end
    println("=========================================================================")
end

run_demo()
