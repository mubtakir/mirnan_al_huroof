"""
demo_semantic_scene.jl
Renders small visual wave scenes from semantic event records.
"""

using Pkg
Pkg.activate(@__DIR__)

const DEMO_DIR = @__DIR__
const OUTPUT_DIR = joinpath(DEMO_DIR, "output")

include(joinpath(DEMO_DIR, "src", "HoloPRNN_Vision.jl"))
using .HoloPRNN_Vision
using Images

function _scene_record(actor, action, patient, effects; confidence=1.0)
    return (
        actor = actor,
        action = action,
        patient = patient,
        effect_candidates = effects,
        confidence = confidence,
    )
end

function run_demo()
    println("=========================================================================")
    println("      HoloPRNN Vision - Semantic Scene Demo")
    println("=========================================================================")

    mkpath(OUTPUT_DIR)

    scenes = [
        ("hit_ball", _scene_record("Khalid", "hit", "ball", ["moved", "away", "changed"])),
        ("broke_cup", _scene_record("child", "broke", "cup", ["pieces", "separated", "lost shape"])),
        ("lit_room", _scene_record("lamp", "illuminated", "room", ["clear", "visible", "appearance"])),
    ]

    for (name, semantic_scene) in scenes
        visual_scene = semantic_scene_to_visual_scene(semantic_scene; width=64, height=48)
        img = render_visual_scene(visual_scene)
        path = joinpath(OUTPUT_DIR, "semantic_scene_$(name).png")
        save(path, img)
        println("Saved $(path)")
    end

    println("=========================================================================")
end

run_demo()
