#!/usr/bin/env julia
# Live gated probe for QuantityFrameStrategy on the trained Mirnan model.

using JSON
using SparseArrays

const MIRNAN_DIR = dirname(@__DIR__)
const MODEL_DIR = joinpath(MIRNAN_DIR, "model")

include(joinpath(MIRNAN_DIR, "src", "MirnanNew.jl"))

using .MirnanNew

const Physics = MirnanNew.Physics
const Gen = MirnanNew.Physics.Generator

function _load_sparse_dat(path::String, vocab_size::Int)
    isfile(path) || return spzeros(vocab_size, vocab_size)
    open(path, "r") do io
        header = readline(io)
        header == "SPARSE_CSC" || return spzeros(vocab_size, vocab_size)
        m = read(io, Int32)
        n = read(io, Int32)
        nnz = read(io, Int32)
        colptr = Vector{Int32}(undef, Int(n) + 1)
        rowval = Vector{Int32}(undef, Int(nnz))
        nzval = Vector{Float32}(undef, Int(nnz))
        read!(io, colptr)
        read!(io, rowval)
        read!(io, nzval)
        return SparseMatrixCSC(Int(m), Int(n), Int.(colptr), Int.(rowval), nzval)
    end
end

function _load_trained_generator()
    vocab_path = joinpath(MODEL_DIR, "vocab.json")
    isfile(vocab_path) || error("Missing trained model. Run models/mirnan/train.jl first.")
    raw_vocab = JSON.parsefile(vocab_path)
    vocab = Dict{String,Int}(String(k) => Int(v) for (k, v) in raw_vocab)
    v = length(vocab)
    k_sem = _load_sparse_dat(joinpath(MODEL_DIR, "K_sem.dat"), v)
    k_syn = _load_sparse_dat(joinpath(MODEL_DIR, "K_syn.dat"), v)
    k_causal = _load_sparse_dat(joinpath(MODEL_DIR, "K_causal.dat"), v)
    return Physics.MirnanGenerator(vocab, k_sem; K_syn=k_syn, K_causal=k_causal, model_dir=MODEL_DIR)
end

function _set_or_delete!(name::String, value)
    if value === nothing
        delete!(ENV, name)
    else
        ENV[name] = value
    end
end

function _short(s; limit::Int=220)
    text = replace(String(s), r"\s+" => " ") |> strip
    ncodeunits(text) <= limit && return text
    cut = prevind(text, min(ncodeunits(text), limit))
    return text[1:cut] * "..."
end

function _env_on(name::String, default::String="0")
    raw = lowercase(strip(get(ENV, name, default)))
    return raw in ("1", "true", "yes", "on")
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
    old_strict = get(ENV, "MIRNAN_STRICT_NO_TEMPLATES", nothing)
    try
        ENV["MIRNAN_ENABLE_QUANTITY_FRAME_STRATEGY"] = gate ? "1" : "0"
        ENV["MIRNAN_ENABLE_STATE_FRAME_STRATEGY"] = "0"
        ENV["MIRNAN_ENABLE_SPATIAL_FRAME_STRATEGY"] = "0"
        ENV["MIRNAN_ENABLE_TEMPORAL_FRAME_STRATEGY"] = "0"
        ENV["MIRNAN_ENABLE_CONDITIONAL_FRAME_STRATEGY"] = "0"
        ENV["MIRNAN_ENABLE_SCENE_PURPOSE_STRATEGY"] = "0"
        ENV["MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY"] = "0"
        ENV["MIRNAN_ENABLE_RELATION_FRAME_STRATEGY"] = "0"
        ENV["MIRNAN_STRICT_NO_TEMPLATES"] = "0"
        return Physics.generate!(gen, String(prompt); mode="auto", max_words=24)
    finally
        _set_or_delete!("MIRNAN_ENABLE_QUANTITY_FRAME_STRATEGY", old_quantity)
        _set_or_delete!("MIRNAN_ENABLE_STATE_FRAME_STRATEGY", old_state)
        _set_or_delete!("MIRNAN_ENABLE_SPATIAL_FRAME_STRATEGY", old_spatial)
        _set_or_delete!("MIRNAN_ENABLE_TEMPORAL_FRAME_STRATEGY", old_temporal)
        _set_or_delete!("MIRNAN_ENABLE_CONDITIONAL_FRAME_STRATEGY", old_cond)
        _set_or_delete!("MIRNAN_ENABLE_SCENE_PURPOSE_STRATEGY", old_scene_purpose)
        _set_or_delete!("MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY", old_scene)
        _set_or_delete!("MIRNAN_ENABLE_RELATION_FRAME_STRATEGY", old_relation)
        _set_or_delete!("MIRNAN_STRICT_NO_TEMPLATES", old_strict)
    end
end

function _print_case(gen, mem, label::AbstractString, prompt::AbstractString; show_gate_off::Bool=false)
    println("="^72)
    println("CASE: $(label)")
    println("PROMPT: $(prompt)")
    println("-- independent quantity answer --")
    independent = Physics.quantity_answer(mem, String(prompt))
    println(isempty(independent) ? "(empty)" : _short(independent))
    if show_gate_off
        println("-- gate off --")
        println(_short(_ask(gen, prompt; gate=false)))
    end
    println("-- gate on --")
    println(_short(_ask(gen, prompt; gate=true)))
    flush(stdout)
end

function _frame_quality(frame)
    target = strip(frame.target)
    value = strip(frame.value)
    isempty(target) && return false
    lowered = lowercase(target)
    startswith(lowered, "\u0646\u0639\u0645") && return false
    startswith(lowered, "\u0644\u0627") && return false
    frame.quantity_type in ("count", "measure", "comparison") && isempty(value) && return false
    length(split(target)) <= 10 || return false
    length(split(value)) <= 14 || return false
    return true
end

function _prompt_for_frame(frame)
    target = strip(frame.target)
    qtype = frame.quantity_type
    if qtype == "count"
        return "\u0643\u0645 \u0639\u062f\u062f $(target)\u061f"
    elseif qtype == "measure"
        return "\u0645\u0627 \u0645\u0642\u062f\u0627\u0631 $(target)\u061f"
    elseif qtype == "comparison"
        return "\u0623\u064a\u0647\u0645\u0627 \u0623\u0643\u062b\u0631 $(target)\u061f"
    elseif qtype == "quantifier_scope"
        return "\u0645\u0627 \u0646\u0637\u0627\u0642 $(target)\u061f"
    elseif qtype == "vague_quantity"
        return "\u0645\u0627 \u0643\u0645\u064a\u0629 $(target)\u061f"
    end
    return "\u0643\u0645 $(target)\u061f"
end

function _find_answerable_prompt(mem, qtype::AbstractString)
    for frame in mem.frames
        frame.quantity_type == qtype || continue
        _frame_quality(frame) || continue
        prompt = _prompt_for_frame(frame)
        isempty(Physics.quantity_answer([frame], prompt)) && continue
        return prompt
    end
    return ""
end

function main()
    quantity_path = joinpath(MODEL_DIR, "quantity_memory.json")
    isfile(quantity_path) || error("Missing quantity memory: $(quantity_path)")

    mem = Gen._LEARNED_QUANTITY_MEMORY[]
    mem === nothing && (mem = Physics.load_quantity_memory(quantity_path))
    Gen._LEARNED_QUANTITY_MEMORY[] = Physics.has_quantity_records(mem) ? mem : nothing

    println("QuantityFrameStrategy trained live probe")
    println("quantity_frames: $(length(mem.frames))")
    println("gate variable: MIRNAN_ENABLE_QUANTITY_FRAME_STRATEGY")
    show_gate_off = _env_on("MIRNAN_QUANTITY_TRAINED_PROBE_GATE_OFF", "0")
    println("gate_off_comparison: $(show_gate_off ? "enabled" : "disabled")")
    println("loading trained generator...")
    flush(stdout)

    gen = _load_trained_generator()
    println("trained generator loaded")
    flush(stdout)

    _print_case(gen, mem, "trained count", "كم عدد جمع القلة يدل على؟"; show_gate_off=show_gate_off)
    comparison_prompt = _find_answerable_prompt(mem, "comparison")
    if !isempty(comparison_prompt)
        _print_case(gen, mem, "trained comparison", comparison_prompt; show_gate_off=show_gate_off)
    else
        println("="^72)
        println("CASE: trained comparison")
        println("No answerable trained comparison frame found.")
    end
    scope_prompt = _find_answerable_prompt(mem, "quantifier_scope")
    if !isempty(scope_prompt)
        _print_case(gen, mem, "trained quantifier scope", scope_prompt; show_gate_off=show_gate_off)
    else
        println("="^72)
        println("CASE: trained quantifier scope")
        println("No answerable trained quantifier-scope frame found.")
    end
    _print_case(gen, mem, "definition guard", "ما معنى جمع القلة؟"; show_gate_off=show_gate_off)
    _print_case(gen, mem, "yes/no guard", "هل عدد الطلاب 30؟"; show_gate_off=show_gate_off)

    if _env_on("MIRNAN_QUANTITY_TRAINED_PROBE_FULL", "0")
        _print_case(gen, mem, "why guard", "لماذا يزيد العلم الفهم؟"; show_gate_off=show_gate_off)
    end
end

main()
