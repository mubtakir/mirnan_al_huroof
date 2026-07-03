#!/usr/bin/env julia
# Live gated probe for SemanticSceneStrategy on the trained Mirnan model.

using JSON
using SparseArrays

const MIRNAN_DIR = dirname(@__DIR__)

include(joinpath(MIRNAN_DIR, "src", "MirnanNew.jl"))

using .MirnanNew

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
    return MirnanNew.Physics.MirnanGenerator(vocab, k_sem; K_syn=k_syn, K_causal=k_causal, model_dir=model_dir)
end

function _ask(gen, prompt::AbstractString; gate::Bool)
    old_gate = get(ENV, "MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY", nothing)
    try
        ENV["MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY"] = gate ? "1" : "0"
        return MirnanNew.Physics.generate!(gen, String(prompt); mode="auto", max_words=56)
    finally
        if old_gate === nothing
            delete!(ENV, "MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY")
        else
            ENV["MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY"] = old_gate
        end
    end
end

function _print_case(gen, label::AbstractString, prompt::AbstractString)
    println("="^72)
    println("CASE: $(label)")
    println("PROMPT: $(prompt)")
    independent = MirnanNew.Physics.semantic_scene_answer(gen.semantic_scenes, gen.hisban, prompt)
    println("-- semantic scene answer --")
    println(isempty(independent) ? "(empty)" : independent)
    println("-- gate off --")
    println(_ask(gen, prompt; gate=false))
    println("-- gate on --")
    println(_ask(gen, prompt; gate=true))
end

function _memory_event_prompt(gen)
    for scene in gen.semantic_scenes.scenes
        isempty(strip(scene.action)) && continue
        isempty(strip(scene.patient)) && continue
        action = lowercase(String(scene.action))
        patient = lowercase(String(scene.patient))
        preferred = occursin("hit", action) || occursin("\u0636\u0631\u0628", action) ||
                    occursin("push", action) || occursin("\u062f\u0641\u0639", action) ||
                    occursin("broke", action) || occursin("break", action) || occursin("\u0643\u0633\u0631", action) ||
                    occursin("illumin", action) || occursin("\u0627\u0636\u0627", action)
        concrete = occursin("ball", patient) || occursin("\u0643\u0631\u0629", patient) ||
                   occursin("stone", patient) || occursin("\u062d\u062c\u0631", patient) ||
                   occursin("cup", patient) || occursin("\u0643\u0623\u0633", patient) ||
                   occursin("room", patient) || occursin("\u063a\u0631\u0641\u0629", patient)
        (preferred && concrete) || continue
        answer = MirnanNew.Physics.semantic_scene_answer(
            gen.semantic_scenes, gen.hisban,
            "What happens when $(scene.action) $(scene.patient)?")
        isempty(answer) && continue
        return "What happens when $(scene.action) $(scene.patient)?"
    end
    return "What happens when Khalid hit the ball?"
end

function main()
    gen = _load_trained_generator()
    summary = MirnanNew.Physics.pattern_memory_summary(gen)
    scene_info = get(summary, "semantic_scenes", Dict{String,Any}())
    scene_count = get(scene_info, "scenes", 0)
    println("SemanticSceneStrategy live probe")
    println("semantic_scenes: $(scene_count)")
    println("gate variable: MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY")

    _print_case(gen, "event/effect from memory", _memory_event_prompt(gen))
    _print_case(gen, "event/effect fixed", "What happens when Khalid hit the ball?")
    _print_case(gen, "yes/no guard", "Does Khalid hit the ball?")
    _print_case(gen, "definition guard", "\u0645\u0627 \u0645\u0639\u0646\u0649 \u0627\u0644\u0633\u0644\u0627\u0645\u061f")
end

main()
