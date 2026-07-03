#!/usr/bin/env julia
# Live gated probe for TemporalFrameStrategy.

const MIRNAN_DIR = dirname(@__DIR__)

include(joinpath(MIRNAN_DIR, "src", "MirnanNew.jl"))

using .MirnanNew

const Physics = MirnanNew.Physics
const Gen = MirnanNew.Physics.Generator

function _small_generator()
    vocab = Dict{String,Int}(
        "\u0645\u062a\u0649" => 1,
        "\u0633\u0627\u0641\u0631" => 2,
        "\u0627\u0644\u0637\u0627\u0644\u0628" => 3,
        "\u0642\u0628\u0644" => 4,
        "\u0627\u0644\u0641\u062c\u0631" => 5,
        "\u0647\u0644" => 6,
        "\u0645\u0627" => 7,
        "\u0645\u0639\u0646\u0649" => 8,
        "\u0627\u0644\u0633\u0644\u0627\u0645" => 9,
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
    old_temporal = get(ENV, "MIRNAN_ENABLE_TEMPORAL_FRAME_STRATEGY", nothing)
    old_cond = get(ENV, "MIRNAN_ENABLE_CONDITIONAL_FRAME_STRATEGY", nothing)
    old_scene_purpose = get(ENV, "MIRNAN_ENABLE_SCENE_PURPOSE_STRATEGY", nothing)
    old_scene = get(ENV, "MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY", nothing)
    old_relation = get(ENV, "MIRNAN_ENABLE_RELATION_FRAME_STRATEGY", nothing)
    try
        ENV["MIRNAN_ENABLE_TEMPORAL_FRAME_STRATEGY"] = gate ? "1" : "0"
        ENV["MIRNAN_ENABLE_CONDITIONAL_FRAME_STRATEGY"] = "0"
        ENV["MIRNAN_ENABLE_SCENE_PURPOSE_STRATEGY"] = "0"
        ENV["MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY"] = "0"
        ENV["MIRNAN_ENABLE_RELATION_FRAME_STRATEGY"] = "0"
        return Physics.generate!(gen, String(prompt); mode="auto", max_words=40)
    finally
        _set_or_delete!("MIRNAN_ENABLE_TEMPORAL_FRAME_STRATEGY", old_temporal)
        _set_or_delete!("MIRNAN_ENABLE_CONDITIONAL_FRAME_STRATEGY", old_cond)
        _set_or_delete!("MIRNAN_ENABLE_SCENE_PURPOSE_STRATEGY", old_scene_purpose)
        _set_or_delete!("MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY", old_scene)
        _set_or_delete!("MIRNAN_ENABLE_RELATION_FRAME_STRATEGY", old_relation)
    end
end

function _controlled_memory()
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(
        mem,
        "\u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628 \u0642\u0628\u0644 \u0627\u0644\u0641\u062c\u0631.",
    )
    return mem
end

function _print_case(gen, label::AbstractString, prompt::AbstractString)
    println("="^72)
    println("CASE: $(label)")
    println("PROMPT: $(prompt)")
    println("-- independent temporal answer --")
    mem = Gen._LEARNED_ISTINBAT_MEMORY[]
    independent = mem === nothing ? "" : Physics.temporal_answer(mem, String(prompt))
    println(isempty(independent) ? "(empty)" : independent)
    println("-- gate off --")
    println(_ask(gen, prompt; gate=false))
    println("-- gate on --")
    println(_ask(gen, prompt; gate=true))
end

function main()
    gen = _small_generator()
    saved = Gen._LEARNED_ISTINBAT_MEMORY[]
    controlled = _controlled_memory()

    println("TemporalFrameStrategy live probe")
    println("generator: small controlled")
    println("gate variable: MIRNAN_ENABLE_TEMPORAL_FRAME_STRATEGY")
    try
        Gen._LEARNED_ISTINBAT_MEMORY[] = controlled
        _print_case(gen, "controlled temporal", "\u0645\u062a\u0649 \u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628\u061f")
        _print_case(gen, "yes/no guard", "\u0647\u0644 \u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628\u061f")
        _print_case(gen, "non-temporal guard", "\u0645\u0627 \u0645\u0639\u0646\u0649 \u0627\u0644\u0633\u0644\u0627\u0645\u061f")
    finally
        Gen._LEARNED_ISTINBAT_MEMORY[] = saved
    end
end

main()
