#!/usr/bin/env julia
# Live gated probe for ScenePurposeStrategy on the trained Mirnan model.

using JSON
using SparseArrays

const MIRNAN_DIR = dirname(@__DIR__)

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
        colptr = read!(io, Vector{Int32}(undef, n + 1))
        rows = read!(io, Vector{Int32}(undef, nnz))
        vals = read!(io, Vector{Float64}(undef, nnz))
        return SparseMatrixCSC(Int(m), Int(n), Vector{Int}(colptr), Vector{Int}(rows), vals)
    end
end

function _load_trained_generator()
    model_dir = joinpath(MIRNAN_DIR, "model")
    vocab_path = joinpath(model_dir, "vocab.json")
    isfile(vocab_path) || error("Missing trained model. Run models/mirnan/train.jl first.")
    raw_vocab = JSON.parsefile(vocab_path)
    vocab = Dict{String,Int}(String(k) => Int(v) for (k, v) in raw_vocab)
    v = length(vocab)
    k_sem = _load_sparse_dat(joinpath(model_dir, "K_sem.dat"), v)
    k_syn = _load_sparse_dat(joinpath(model_dir, "K_syn.dat"), v)
    k_causal = _load_sparse_dat(joinpath(model_dir, "K_causal.dat"), v)
    return Physics.MirnanGenerator(vocab, k_sem; K_syn=k_syn, K_causal=k_causal, model_dir=model_dir)
end

function _set_or_delete!(name::String, value)
    if value === nothing
        delete!(ENV, name)
    else
        ENV[name] = value
    end
end

function _ask(gen, prompt::AbstractString; gate::Bool)
    old_scene_purpose = get(ENV, "MIRNAN_ENABLE_SCENE_PURPOSE_STRATEGY", nothing)
    old_scene = get(ENV, "MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY", nothing)
    old_relation = get(ENV, "MIRNAN_ENABLE_RELATION_FRAME_STRATEGY", nothing)
    try
        ENV["MIRNAN_ENABLE_SCENE_PURPOSE_STRATEGY"] = gate ? "1" : "0"
        ENV["MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY"] = "0"
        ENV["MIRNAN_ENABLE_RELATION_FRAME_STRATEGY"] = "0"
        return Physics.generate!(gen, String(prompt); mode="auto", max_words=72)
    finally
        _set_or_delete!("MIRNAN_ENABLE_SCENE_PURPOSE_STRATEGY", old_scene_purpose)
        _set_or_delete!("MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY", old_scene)
        _set_or_delete!("MIRNAN_ENABLE_RELATION_FRAME_STRATEGY", old_relation)
    end
end

function _independent_answer(gen, prompt::AbstractString)
    scene_mem = Gen._LEARNED_SEMANTIC_SCENE_MEMORY[]
    (scene_mem === nothing || !Physics.has_semantic_scenes(scene_mem)) && (scene_mem = gen.semantic_scenes)
    istinbat_mem = Gen._LEARNED_ISTINBAT_MEMORY[]
    istinbat_mem === nothing && return ""
    return Physics.scene_purpose_answer(scene_mem, gen.hisban, istinbat_mem, String(prompt))
end

function _print_case(gen, label::AbstractString, prompt::AbstractString)
    println("="^72)
    println("CASE: $(label)")
    println("PROMPT: $(prompt)")
    println("-- independent scene-purpose answer --")
    independent = _independent_answer(gen, prompt)
    println(isempty(independent) ? "(empty)" : independent)
    println("-- gate off --")
    println(_ask(gen, prompt; gate=false))
    println("-- gate on --")
    println(_ask(gen, prompt; gate=true))
end

function main()
    gen = _load_trained_generator()
    summary = Physics.pattern_memory_summary(gen)
    scene_info = get(summary, "semantic_scenes", Dict{String,Any}())
    scene_count = get(scene_info, "scenes", 0)
    istinbat_info = get(summary, "al_istinbat", Dict{String,Any}())
    istinbat_count = get(istinbat_info, "records", 0)

    println("ScenePurposeStrategy live probe")
    println("semantic_scenes: $(scene_count)")
    println("istinbat_records: $(istinbat_count)")
    println("gate variable: MIRNAN_ENABLE_SCENE_PURPOSE_STRATEGY")

    _print_case(gen, "trained cooperative", "\u0644\u0645\u0627\u0630\u0627 \u062f\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062d\u062c\u0631\u061f")
    _print_case(gen, "known event/purpose probe", "\u0644\u0645\u0627\u0630\u0627 hit ball\u061f")
    _print_case(gen, "purpose only guard", "\u0644\u0645\u0627\u0630\u0627 \u0641\u062a\u062d \u0639\u0644\u0628\u0647 \u0627\u0646\u0642\u0644 \u0645\u062d\u062a\u0648\u064a \u0645\u062a\u0628\u0642\u064a \u0641\u0648\u0631\u0627\u064b \u0648\u0639\u0627\u0621 \u0632\u062c\u0627\u062c\u064a \u0628\u0644\u0627\u0633\u062a\u064a\u061f")
    _print_case(gen, "yes/no guard", "\u0647\u0644 \u062f\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062d\u062c\u0631\u061f")
    _print_case(gen, "definition guard", "\u0645\u0627 \u0645\u0639\u0646\u0649 \u0627\u0644\u0633\u0644\u0627\u0645\u061f")
end

main()
