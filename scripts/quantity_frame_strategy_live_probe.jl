#!/usr/bin/env julia
# Live gated probe for QuantityFrameStrategy.

const MIRNAN_DIR = dirname(@__DIR__)

include(joinpath(MIRNAN_DIR, "src", "MirnanNew.jl"))

using .MirnanNew

const Physics = MirnanNew.Physics
const Gen = MirnanNew.Physics.Generator

function _small_generator()
    vocab = Dict{String,Int}(
        "\u0643\u0645" => 1,
        "\u0639\u062f\u062f" => 2,
        "\u0627\u0644\u0637\u0644\u0627\u0628" => 3,
        "30" => 4,
        "\u0647\u0644" => 5,
        "\u0644\u0645\u0627\u0630\u0627" => 6,
        "\u064a\u062f\u0631\u0633" => 7,
    )
    return Physics.MirnanGenerator(vocab, nothing; model_dir=mktempdir())
end

function _set_or_delete!(name::String, value)
    if value === nothing
        delete!(ENV, name)
    else
        ENV[name] = value
    end
end

function _ask(gen, prompt::AbstractString; gate::Bool)
    old_quantity = get(ENV, "MIRNAN_ENABLE_QUANTITY_FRAME_STRATEGY", nothing)
    old_state = get(ENV, "MIRNAN_ENABLE_STATE_FRAME_STRATEGY", nothing)
    old_spatial = get(ENV, "MIRNAN_ENABLE_SPATIAL_FRAME_STRATEGY", nothing)
    old_temporal = get(ENV, "MIRNAN_ENABLE_TEMPORAL_FRAME_STRATEGY", nothing)
    old_cond = get(ENV, "MIRNAN_ENABLE_CONDITIONAL_FRAME_STRATEGY", nothing)
    old_scene_purpose = get(ENV, "MIRNAN_ENABLE_SCENE_PURPOSE_STRATEGY", nothing)
    old_scene = get(ENV, "MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY", nothing)
    old_relation = get(ENV, "MIRNAN_ENABLE_RELATION_FRAME_STRATEGY", nothing)
    try
        ENV["MIRNAN_ENABLE_QUANTITY_FRAME_STRATEGY"] = gate ? "1" : "0"
        ENV["MIRNAN_ENABLE_STATE_FRAME_STRATEGY"] = "0"
        ENV["MIRNAN_ENABLE_SPATIAL_FRAME_STRATEGY"] = "0"
        ENV["MIRNAN_ENABLE_TEMPORAL_FRAME_STRATEGY"] = "0"
        ENV["MIRNAN_ENABLE_CONDITIONAL_FRAME_STRATEGY"] = "0"
        ENV["MIRNAN_ENABLE_SCENE_PURPOSE_STRATEGY"] = "0"
        ENV["MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY"] = "0"
        ENV["MIRNAN_ENABLE_RELATION_FRAME_STRATEGY"] = "0"
        return Physics.generate!(gen, String(prompt); mode="auto", max_words=40)
    finally
        _set_or_delete!("MIRNAN_ENABLE_QUANTITY_FRAME_STRATEGY", old_quantity)
        _set_or_delete!("MIRNAN_ENABLE_STATE_FRAME_STRATEGY", old_state)
        _set_or_delete!("MIRNAN_ENABLE_SPATIAL_FRAME_STRATEGY", old_spatial)
        _set_or_delete!("MIRNAN_ENABLE_TEMPORAL_FRAME_STRATEGY", old_temporal)
        _set_or_delete!("MIRNAN_ENABLE_CONDITIONAL_FRAME_STRATEGY", old_cond)
        _set_or_delete!("MIRNAN_ENABLE_SCENE_PURPOSE_STRATEGY", old_scene_purpose)
        _set_or_delete!("MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY", old_scene)
        _set_or_delete!("MIRNAN_ENABLE_RELATION_FRAME_STRATEGY", old_relation)
    end
end

function _controlled_memory()
    mem = Physics.QuantityFrameMemory()
    Physics.learn_quantity_frames_from_text!(
        mem,
        "\u0627\u0644\u0637\u0644\u0627\u0628 \u0639\u062f\u062f 30.",
    )
    return mem
end

function _print_case(gen, label::AbstractString, prompt::AbstractString)
    println("="^72)
    println("CASE: $(label)")
    println("PROMPT: $(prompt)")
    println("-- independent quantity answer --")
    mem = Gen._LEARNED_QUANTITY_MEMORY[]
    independent = mem === nothing ? "" : Physics.quantity_answer(mem, String(prompt))
    println(isempty(independent) ? "(empty)" : independent)
    println("-- gate off --")
    println(_ask(gen, prompt; gate=false))
    println("-- gate on --")
    println(_ask(gen, prompt; gate=true))
end

function main()
    gen = _small_generator()
    saved = Gen._LEARNED_QUANTITY_MEMORY[]
    controlled = _controlled_memory()

    println("QuantityFrameStrategy live probe")
    println("generator: small controlled")
    println("gate variable: MIRNAN_ENABLE_QUANTITY_FRAME_STRATEGY")
    try
        Gen._LEARNED_QUANTITY_MEMORY[] = controlled
        _print_case(gen, "controlled quantity", "\u0643\u0645 \u0639\u062f\u062f \u0627\u0644\u0637\u0644\u0627\u0628\u061f")
        _print_case(gen, "yes/no guard", "\u0647\u0644 \u0639\u062f\u062f \u0627\u0644\u0637\u0644\u0627\u0628 30\u061f")
        _print_case(gen, "non-quantity guard", "\u0644\u0645\u0627\u0630\u0627 \u064a\u062f\u0631\u0633 \u0627\u0644\u0637\u0644\u0627\u0628\u061f")
    finally
        Gen._LEARNED_QUANTITY_MEMORY[] = saved
    end
end

main()
