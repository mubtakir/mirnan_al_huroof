"""
demo_style_transfer.jl
Demonstrates wave-based style transfer by marrying content structure with style phase coupling.
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
    println("      HoloPRNN Vision — Style Transfer Demo (Wave Texture Marriage)")
    println("=========================================================================")
    
    # 1. Create a content image (e.g., a simple square in the center)
    H, W = 16, 16
    img_content = [y in 6:11 && x in 6:11 ? RGB(1.0, 1.0, 1.0) : RGB(0.0, 0.0, 0.0) for y in 1:H, x in 1:W]
    
    # 2. Create a style image (e.g., alternating blue and yellow horizontal stripes)
    img_style = [y % 2 == 0 ? RGB(0.0, 0.0, 1.0) : RGB(1.0, 0.8, 0.0) for y in 1:H, x in 1:W]
    
    # Ensure output directory exists
    mkpath(OUTPUT_DIR)
    save(joinpath(OUTPUT_DIR, "style_content.png"), img_content)
    save(joinpath(OUTPUT_DIR, "style_style.png"), img_style)
    println("Saved content and style reference images to '$(OUTPUT_DIR)'")
    
    # 3. Initialize oscillator parameters
    params = OscillatorParams(μ=1.0, g_inh=0.3, γ=2.0, τ_a=1.5, dt=0.03)
    
    # 4. Run style transfer
    println("Running wave style transfer (250 steps)...")
    img_result = run_style_transfer(img_content, img_style, params; D_spatial=0.25, steps=250)
    
    save(joinpath(OUTPUT_DIR, "style_result.png"), img_result)
    println("Saved style transfer result to '$(joinpath(OUTPUT_DIR, "style_result.png"))'")
    println("🎉 SUCCESS! Style transfer demo completed successfully.")
    println("=========================================================================")
end

run_demo()
