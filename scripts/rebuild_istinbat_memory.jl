#!/usr/bin/env julia
# Rebuild/refresh the trained al_istinbat memory with structured relation seeds.
#
# This is intentionally lighter than full training: it keeps the current trained
# memory and injects line-level RelationFrame examples from data/uqra.

const MIRNAN_DIR = dirname(@__DIR__)

include(joinpath(MIRNAN_DIR, "src", "MirnanNew.jl"))

using .MirnanNew

const Physics = MirnanNew.Physics

function _uqra_lines()
    dir = joinpath(MIRNAN_DIR, "data", "uqra")
    texts = String[]
    metadata = Dict{String,Any}[]
    isdir(dir) || return texts, metadata
    for path in sort(filter(p -> lowercase(splitext(p)[2]) == ".txt",
                            readdir(dir; join=true)))
        idx = 1
        for line in eachline(path)
            s = strip(line)
            isempty(s) && continue
            startswith(s, "[") && continue
            push!(texts, s)
            push!(metadata, Dict{String,Any}(
                "file_name" => basename(path),
                "paragraph_index" => idx,
                "source_dir" => "uqra",
                "knowledge_type" => "structured_relation_seed",
            ))
            idx += 1
        end
    end
    return texts, metadata
end

function main()
    model_dir = joinpath(MIRNAN_DIR, "model")
    path = joinpath(model_dir, "al_istinbat.json")
    isfile(path) || error("Missing al_istinbat.json. Run training first.")

    mem = Physics.load_istinbat(path)
    before = length(mem.records)
    texts, metadata = _uqra_lines()
    learned = Physics.train_istinbat_from_texts!(mem, texts, metadata; max_items=20_000)
    saved = Physics.save_istinbat(mem, path)

    println("Mirnan al_istinbat focused rebuild")
    println("uqra_lines: ", length(texts))
    println("learned_observations: ", learned)
    println("records_before: ", before)
    println("records_after: ", length(mem.records))
    println("saved: ", saved)
    println()
    println("Smoke:")
    println("  متى سافر الطالب؟ => ", Physics.temporal_answer(mem, "متى سافر الطالب؟"))
    println("  أين جلس الطفل؟ => ", Physics.spatial_answer(mem, "أين جلس الطفل؟"))
    println("  كيف دخل الطفل؟ => ", Physics.state_answer(mem, "كيف دخل الطفل؟"))
end

main()
