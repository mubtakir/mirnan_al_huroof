#!/usr/bin/env julia
# Build only semantic_scenes.json for an already trained Mirnan model.

const MIRNAN_DIR = dirname(@__DIR__)
const MAJNON_ROOT = dirname(dirname(MIRNAN_DIR))
const DATA_DIR = joinpath(MIRNAN_DIR, "data")
const MODEL_DIR = joinpath(MIRNAN_DIR, "model")

const SEED_EVENT_PAIRS = [
    ("Khalid hit the ball", "The ball moved away and changed position."),
    ("The player pushed the stone", "The stone moved from its place."),
    ("The child broke the cup", "The cup lost its shape and separated into pieces."),
    ("The lamp illuminated the room", "The room became visible and clear."),
    ("\u0636\u0631\u0628 \u062e\u0627\u0644\u062f \u0627\u0644\u0643\u0631\u0629",
     "\u0627\u0628\u062a\u0639\u062f\u062a \u0627\u0644\u0643\u0631\u0629 \u0648\u062a\u063a\u064a\u0631 \u0645\u0648\u0636\u0639\u0647\u0627."),
    ("\u062f\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062d\u062c\u0631",
     "\u062a\u062d\u0631\u0643 \u0627\u0644\u062d\u062c\u0631 \u0645\u0646 \u0645\u0643\u0627\u0646\u0647."),
    ("\u0643\u0633\u0631 \u0627\u0644\u0637\u0641\u0644 \u0627\u0644\u0643\u0623\u0633",
     "\u0627\u0646\u0641\u0635\u0644 \u0627\u0644\u0643\u0623\u0633 \u0648\u062a\u0644\u0641\u062a \u0647\u064a\u0626\u062a\u0647."),
    ("\u0623\u0636\u0627\u0621 \u0627\u0644\u0645\u0635\u0628\u0627\u062d \u0627\u0644\u063a\u0631\u0641\u0629",
     "\u0638\u0647\u0631\u062a \u0627\u0644\u063a\u0631\u0641\u0629 \u0648\u0627\u062a\u0636\u062d\u062a."),
]

include(joinpath(MIRNAN_DIR, "src", "MirnanNew.jl"))

using .MirnanNew

function _env_int(name::String, default::Int)
    raw = get(ENV, name, "")
    isempty(strip(raw)) && return default
    try
        return parse(Int, raw)
    catch
        return default
    end
end

function _read_text_file(path::AbstractString; max_chars::Int)
    try
        text = read(path, String)
        return length(text) > max_chars ? text[1:max_chars] : text
    catch
        return ""
    end
end

function _collect_texts_from_dir(root::AbstractString; limit::Int, max_chars::Int)
    texts = String[]
    isdir(root) || return texts
    for (dir, _, files) in walkdir(root)
        lower_dir = lowercase(dir)
        if occursin("data_quarantine", lower_dir) || occursin("code", lower_dir)
            continue
        end
        for file in sort(files)
            length(texts) >= limit && return texts
            lower = lowercase(file)
            if startswith(lower, "format_rules") || occursin("powershell", lower) || occursin("api_server", lower)
                continue
            end
            if endswith(lower, ".txt") || endswith(lower, ".md")
                text = strip(_read_text_file(joinpath(dir, file); max_chars=max_chars))
                isempty(text) || push!(texts, text)
                if length(texts) > 0 && length(texts) % 250 == 0
                    println("  collected texts: $(length(texts))")
                    flush(stdout)
                end
            end
        end
    end
    return texts
end

function _maybe_push_file!(texts::Vector{String}, path::AbstractString, limit::Int, max_chars::Int)
    length(texts) >= limit && return
    isfile(path) || return
    text = strip(_read_text_file(path; max_chars=max_chars))
    isempty(text) || push!(texts, text)
end

function main()
    text_limit = _env_int("MIRNAN_SEMANTIC_SCENE_TEXT_LIMIT", 3_000)
    scene_limit = _env_int("MIRNAN_SEMANTIC_SCENE_LIMIT", 20_000)
    max_chars = _env_int("MIRNAN_SEMANTIC_SCENE_FILE_CHARS", 200_000)

    println("Semantic scene memory build")
    println("text_limit: $(text_limit)")
    println("scene_limit: $(scene_limit)")
    println("file_char_limit: $(max_chars)")
    flush(stdout)

    hisban_path = joinpath(MODEL_DIR, "al_hisban_al_dalali.json")
    isfile(hisban_path) || error("Missing al_hisban_al_dalali.json. Train Mirnan first.")
    println("loading al_hisban_al_dalali...")
    flush(stdout)
    hisban = MirnanNew.Physics.load_semantic_calculus(hisban_path)

    println("collecting texts from data...")
    flush(stdout)
    texts = _collect_texts_from_dir(DATA_DIR; limit=text_limit, max_chars=max_chars)
    if get(ENV, "MIRNAN_SEMANTIC_SCENE_INCLUDE_AGENT", "0") in ("1", "true", "yes", "on")
        _maybe_push_file!(texts, joinpath(MAJNON_ROOT, ".agent_workspace", ".agent", "mirnan", "training_corpus.txt"), text_limit, max_chars)
        _maybe_push_file!(texts, joinpath(MAJNON_ROOT, ".agent", "mirnan", "training_corpus.txt"), text_limit, max_chars)
    end

    scene_mem = MirnanNew.Physics.SemanticSceneMemory(max_scenes=scene_limit)
    seed_learned = 0
    for (source, target) in SEED_EVENT_PAIRS
        MirnanNew.Physics.learn_semantic_calculus_from_pair!(hisban, source, target)
        seed_calc = MirnanNew.Physics.SemanticCalculusMemory()
        MirnanNew.Physics.learn_semantic_calculus_from_pair!(seed_calc, source, target)
        seed_learned += MirnanNew.Physics.learn_semantic_scene_from_text!(scene_mem, seed_calc, source)
    end
    println("seed_scenes: $(seed_learned)")
    println("training semantic scenes from $(length(texts)) texts...")
    flush(stdout)
    learned = 0
    for (i, text) in enumerate(texts)
        learned >= scene_limit && break
        learned += MirnanNew.Physics.learn_semantic_scene_from_text!(scene_mem, hisban, text)
        if i == 1 || i % 100 == 0
            println("  processed: $(i)/$(length(texts)) | scenes: $(learned)")
            flush(stdout)
        end
    end
    out_path = MirnanNew.Physics.save_semantic_scenes(
        scene_mem, joinpath(MODEL_DIR, "semantic_scenes.json"))

    println("texts: $(length(texts))")
    println("learned_scenes: $(learned)")
    println("saved: $(out_path)")
end

main()
