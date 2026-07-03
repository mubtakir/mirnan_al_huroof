#!/usr/bin/env julia
# Rebuild only model/quantity_memory.json using the current QuantityFrame extractor.

const MIRNAN_DIR = dirname(@__DIR__)

include(joinpath(MIRNAN_DIR, "train.jl"))

function main()
    granularity = _parse_segment_level(get(ENV, "MIRNAN_SEGMENT_LEVEL", "paragraph"), :paragraph)
    max_items = try
        parse(Int, get(ENV, "MIRNAN_QUANTITY_MAX_ITEMS", "50000"))
    catch
        50_000
    end

    println("Quantity memory rebuild")
    println("segment_level: $(granularity)")
    println("max_items: $(max_items)")

    code_blocks = String[]
    raw_texts, raw_metadata = load_all_corpus(granularity=granularity, code_blocks=code_blocks)

    texts = String[]
    metadata = Dict{String,Any}[]
    for (idx, text) in enumerate(raw_texts)
        cleaned = MirnanNew.Physics.MathBridgeModule.strip_code_blocks(text)
        cleaned = MirnanNew.Physics.CodeEngineModule.strip_inline_code(cleaned)
        isempty(strip(cleaned)) && continue
        push!(texts, cleaned)
        push!(metadata, raw_metadata[idx])
    end

    mem = MirnanNew.Physics.QuantityFrameMemory()
    learned = MirnanNew.Physics.train_quantity_frames_from_texts!(
        mem, texts, metadata; max_items=max_items)
    path = MirnanNew.Physics.save_quantity_memory(mem, joinpath(MODEL_DIR, "quantity_memory.json"))

    counts = Dict{String,Int}()
    for frame in mem.frames
        counts[frame.quantity_type] = get(counts, frame.quantity_type, 0) + 1
    end

    println("texts: $(length(texts))")
    println("learned_quantity_frames: $(learned)")
    println("types: " * join(["$(k)=$(v)" for (k, v) in sort(collect(counts); by=x->x[1])], ", "))
    println("saved: $(path)")
end

main()
