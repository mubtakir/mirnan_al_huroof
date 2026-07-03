#!/usr/bin/env julia
# Probe trained quantity_memory.json and the gated QuantityFrameStrategy.

const MIRNAN_DIR = dirname(@__DIR__)
const MODEL_DIR = joinpath(MIRNAN_DIR, "model")

include(joinpath(MIRNAN_DIR, "src", "MirnanNew.jl"))

using .MirnanNew
using JSON
using SparseArrays

const Physics = MirnanNew.Physics
const Gen = MirnanNew.Physics.Generator

function _env_int(name::String, default::Int)
    raw = get(ENV, name, "")
    isempty(strip(raw)) && return default
    try
        return parse(Int, raw)
    catch
        return default
    end
end

function _env_on(name::String, default::String="0")
    raw = lowercase(strip(get(ENV, name, default)))
    return raw in ("1", "true", "yes", "on")
end

function _short(s; limit::Int=120)
    text = replace(String(s), r"\s+" => " ") |> strip
    ncodeunits(text) <= limit && return text
    cut = prevind(text, min(ncodeunits(text), limit))
    return text[1:cut] * "..."
end

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

function _frame_quality(frame)
    target = strip(frame.target)
    value = strip(frame.value)
    isempty(target) && return false
    frame.quantity_type in ("count", "measure", "comparison") && isempty(value) && return false
    length(split(target)) <= 12 || return false
    length(split(value)) <= 18 || return false
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

function _frame_statement(frame)
    target = strip(frame.target)
    value = strip(frame.value)
    marker = strip(frame.marker)
    qtype = frame.quantity_type
    if qtype == "count"
        isempty(value) && return "\u0627\u0644\u0639\u062f\u062f \u064a\u062a\u0639\u0644\u0642 \u0628\u0640 $(target)."
        return "\u0627\u0644\u0625\u0637\u0627\u0631 \u0627\u0644\u0639\u062f\u062f\u064a: $(target) \u2192 $(value)."
    elseif qtype == "measure"
        isempty(value) && return "\u0627\u0644\u0645\u0642\u062f\u0627\u0631 \u064a\u062a\u0639\u0644\u0642 \u0628\u0640 $(target)."
        return "\u0627\u0644\u0645\u0642\u062f\u0627\u0631: $(target) \u2192 $(value)."
    elseif qtype == "comparison"
        return "\u0627\u0644\u0645\u0642\u0627\u0631\u0646\u0629: $(target) $(marker) $(value)."
    elseif qtype == "quantifier_scope"
        isempty(value) && return "\u0627\u0644\u0646\u0637\u0627\u0642: $(marker) $(target)."
        return "\u0627\u0644\u0646\u0637\u0627\u0642: $(marker) $(target) $(value)."
    elseif qtype == "vague_quantity"
        isempty(value) && return "\u0627\u0644\u0643\u0645\u064a\u0629 \u0627\u0644\u062a\u0642\u0631\u064a\u0628\u064a\u0629: $(target) $(marker)."
        return "\u0627\u0644\u0643\u0645\u064a\u0629 \u0627\u0644\u062a\u0642\u0631\u064a\u0628\u064a\u0629: $(target) $(marker) $(value)."
    end
    return ""
end

function _type_counts(frames)
    counts = Dict{String,Int}()
    for frame in frames
        counts[frame.quantity_type] = get(counts, frame.quantity_type, 0) + 1
    end
    return counts
end

function _pick_examples(mem; limit::Int=4)
    chosen = Physics.QuantityFrame[]
    seen_types = Set{String}()
    for frame in mem.frames
        _frame_quality(frame) || continue
        isempty(Physics.quantity_answer([frame], _prompt_for_frame(frame))) && continue
        if !(frame.quantity_type in seen_types)
            push!(chosen, frame)
            push!(seen_types, frame.quantity_type)
        end
        length(chosen) >= limit && break
    end
    if length(chosen) < limit
        for frame in mem.frames
            _frame_quality(frame) || continue
            isempty(Physics.quantity_answer([frame], _prompt_for_frame(frame))) && continue
            frame in chosen && continue
            push!(chosen, frame)
            length(chosen) >= limit && break
        end
    end
    return chosen
end

function _print_frame(frame, idx::Int)
    println("$(idx). type=$(frame.quantity_type) marker=$(frame.marker) confidence=$(round(frame.confidence; digits=3))")
    println("   target=$(_short(frame.target))")
    println("   value=$(_short(frame.value))")
end

function main()
    path = joinpath(MODEL_DIR, "quantity_memory.json")
    isfile(path) || error("Missing quantity memory: $(path)")

    mem = Physics.load_quantity_memory(path)
    Gen._LEARNED_QUANTITY_MEMORY[] = Physics.has_quantity_records(mem) ? mem : nothing
    use_generation = _env_on("MIRNAN_QUANTITY_PROBE_GENERATE", "0")
    gen = use_generation ? _load_trained_generator() : nothing

    examples = _pick_examples(mem; limit=_env_int("MIRNAN_QUANTITY_PROBE_LIMIT", 4))
    counts = _type_counts(mem.frames)

    println("Trained QuantityFrame memory probe")
    println("quantity_memory: $(path)")
    println("frames: $(length(mem.frames))")
    println("types: " * join(["$(k)=$(v)" for (k, v) in sort(collect(counts); by=x->x[1])], ", "))
    println("examples: $(length(examples))")
    println("generation_probe: $(use_generation ? "enabled" : "disabled")")

    isempty(examples) && begin
        println("No usable trained quantity examples found.")
        return
    end

    println("="^72)
    println("SELECTED FRAMES")
    for (i, frame) in enumerate(examples)
        _print_frame(frame, i)
    end

    for (i, frame) in enumerate(examples)
        prompt = _prompt_for_frame(frame)
        independent = Physics.quantity_answer(mem, prompt)
        println("="^72)
        println("CASE #$(i)")
        println("MEMORY_STATEMENT: $(_short(_frame_statement(frame); limit=180))")
        println("DIAGNOSTIC_PROMPT: $(_short(prompt))")
        println("FRAME: type=$(frame.quantity_type) marker=$(frame.marker)")
        println("-- independent quantity answer --")
        local_answer = Physics.quantity_answer([frame], prompt)
        println(isempty(local_answer) ? "(empty)" : _short(local_answer; limit=180))
        if gen !== nothing
            println("-- gate off --")
            println(_short(_ask(gen, prompt; gate=false); limit=180))
            println("-- gate on --")
            println(_short(_ask(gen, prompt; gate=true); limit=180))
        end
    end
end

main()
