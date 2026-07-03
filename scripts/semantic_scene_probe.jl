#!/usr/bin/env julia
# Developer diagnostic probe for Mirnan semantic imagination.
# This script does not call generate! and does not affect runtime behavior.

include(joinpath(@__DIR__, "..", "src", "MirnanNew.jl"))

using .MirnanNew

const Physics = MirnanNew.Physics

const DEFAULT_PAIRS = [
    ("Khalid hit the ball", "The ball moved away and changed position."),
    ("The player pushed the stone", "The stone moved from its place."),
    ("The child broke the cup", "The cup lost its shape and separated into pieces."),
    ("The lamp illuminated the room", "The room became visible and clear."),
    ("\u0636\u0631\u0628 \u062e\u0627\u0644\u062f \u0627\u0644\u0643\u0631\u0629",
     "\u0627\u0628\u062a\u0639\u062f\u062a \u0627\u0644\u0643\u0631\u0629 \u0648\u062a\u063a\u064a\u0631 \u0645\u0648\u0636\u0639\u0647\u0627."),
]

const DEFAULT_PROMPTS = [
    "Khalid hit the ball",
    "What happened when the player pushed the stone?",
    "The child broke the cup",
    "The lamp illuminated the room",
    "\u0645\u0627\u0630\u0627 \u064a\u062d\u062f\u062b \u0639\u0646\u062f\u0645\u0627 \u064a\u0636\u0631\u0628 \u062e\u0627\u0644\u062f \u0627\u0644\u0643\u0631\u0629\u061f",
]

function _print_help()
    println("""
Semantic scene diagnostic probe.

Usage:
  julia semantic_scene_probe.jl [text-file]

If text-file is provided, each non-empty line is used as a probe prompt and the
whole file is used to build a small temporary scene/calculus memory.

This tool is diagnostic only. It does not call generate!.
""")
end

function _read_lines(path::AbstractString)
    isfile(path) || error("file not found: $path")
    return String[strip(line) for line in eachline(path) if !isempty(strip(line))]
end

function _build_default_memories()
    calculus = Physics.SemanticCalculusMemory()
    scene_mem = Physics.SemanticSceneMemory()
    for (source, target) in DEFAULT_PAIRS
        pair_calculus = Physics.SemanticCalculusMemory()
        Physics.learn_semantic_calculus_from_pair!(pair_calculus, source, target)
        Physics.learn_semantic_calculus_from_pair!(calculus, source, target)
        Physics.learn_semantic_scene_from_text!(scene_mem, pair_calculus, source)
    end
    return scene_mem, calculus, copy(DEFAULT_PROMPTS)
end

function _build_file_memories(path::AbstractString)
    lines = _read_lines(path)
    text = join(lines, "\n")
    calculus = Physics.SemanticCalculusMemory()
    scene_mem = Physics.SemanticSceneMemory()
    Physics.learn_semantic_calculus_from_text!(calculus, text)
    Physics.learn_semantic_scene_from_text!(scene_mem, calculus, text)
    return scene_mem, calculus, lines
end

function _scene_summary(scene)
    scene === nothing && return "none"
    return "actor=$(scene.actor) | action=$(scene.action) | patient=$(scene.patient)"
end

function _print_comparison(i::Int, cmp)
    println("="^72)
    println("PROBE #$i")
    println("PROMPT: $(cmp.prompt)")
    println("AGREEMENT: $(cmp.agreement)")
    println("OVERLAP: $(round(cmp.overlap_score; digits=3))")
    println("SCENE_CONFIDENCE: $(round(cmp.scene_confidence; digits=3))")
    println("GUIDANCE_CONFIDENCE: $(round(cmp.guidance_confidence; digits=3))")
    println("SCENE: $(_scene_summary(cmp.scene))")
    println("SCENE_EFFECTS: $(join(cmp.scene_effect_terms, ", "))")
    println("GUIDANCE_TERMS_FILTERED: $(join(cmp.guidance_terms, ", "))")
    println("GUIDANCE_TERMS_RAW: $(join(cmp.raw_guidance_terms, ", "))")
end

function main()
    if any(arg -> arg in ("-h", "--help"), ARGS)
        _print_help()
        return
    end

    scene_mem, calculus, prompts = isempty(ARGS) ? _build_default_memories() : _build_file_memories(ARGS[1])
    println("Semantic scene probe")
    println("scenes: $(length(scene_mem.scenes))")
    println("prompts: $(length(prompts))")
    for (i, prompt) in enumerate(prompts)
        cmp = Physics.compare_semantic_scene_with_calculus(scene_mem, calculus, prompt)
        _print_comparison(i, cmp)
        answer = Physics.semantic_scene_answer(scene_mem, calculus, prompt)
        println("SCENE_ANSWER: $(isempty(answer) ? "(empty)" : answer)")
    end
end

main()
