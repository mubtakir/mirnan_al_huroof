#!/usr/bin/env julia
# Probe the response polisher on trained Mirnan memories.
#
# This uses trained memories directly instead of running full generation twice,
# which keeps the probe fast while still exercising post-training answers.

const MIRNAN_DIR = dirname(@__DIR__)

include(joinpath(MIRNAN_DIR, "src", "MirnanNew.jl"))

using .MirnanNew

const Physics = MirnanNew.Physics
const Gen = MirnanNew.Physics.Generator

function _load_trained_memories()
    model_dir = joinpath(MIRNAN_DIR, "model")
    isdir(model_dir) || error("Missing trained model directory. Run models/mirnan/train.jl first.")
    return (
        istinbat = Physics.load_istinbat(joinpath(model_dir, "al_istinbat.json")),
        quantity = Physics.load_quantity_memory(joinpath(model_dir, "quantity_memory.json")),
        scenes = Physics.load_semantic_scenes(joinpath(model_dir, "semantic_scenes.json")),
        hisban = Physics.load_semantic_calculus(joinpath(model_dir, "al_hisban_al_dalali.json")),
    )
end

function _polish(prompt::AbstractString, answer::AbstractString; polisher::Bool)
    return Gen.polish_response(String(prompt), String(answer); enabled=polisher)
end

function _direct_answer(memories, label::AbstractString, prompt::AbstractString)
    s = String(prompt)
    istinbat = memories.istinbat
    quantity = memories.quantity
    scene_mem = memories.scenes

    label == "quantity" && return Physics.quantity_answer(quantity, s)
    label == "conditional" && return Physics.conditional_answer(istinbat, s)
    label == "temporal" && return Physics.temporal_answer(istinbat, s)
    label == "spatial" && return Physics.spatial_answer(istinbat, s)
    label == "state" && return Physics.state_answer(istinbat, s)
    label == "scene-purpose" && return Physics.scene_purpose_answer(scene_mem, memories.hisban, istinbat, s)
    return ""
end

function _has_arabic(s::AbstractString)
    return occursin(r"[\u0600-\u06FF]", String(s))
end

function _auto_prompt_for_record(label::AbstractString, rec)
    subject = strip(join(rec.before_terms, " "))
    isempty(subject) && return ""
    length(split(subject)) >= 2 || return ""
    arabic = _has_arabic(subject)
    if label == "temporal"
        return arabic ? "\u0645\u062a\u0649 $(subject)\u061f" : "when $(subject)?"
    elseif label == "spatial"
        return arabic ? "\u0623\u064a\u0646 $(subject)\u061f" : "where $(subject)?"
    elseif label == "state"
        return arabic ? "\u0643\u064a\u0641 $(subject)\u061f" : "how $(subject)?"
    elseif label == "conditional"
        return arabic ? "\u0645\u0627\u0630\u0627 \u064a\u062d\u062f\u062b \u0625\u0630\u0627 $(subject)\u061f" : "what happens if $(subject)?"
    end
    return ""
end

function _trained_prompt_fallback(memories, label::AbstractString)
    label in ("conditional", "temporal", "spatial", "state") || return "", ""
    for rec in values(memories.istinbat.records)
        rec.relation_type == label || continue
        prompt = _auto_prompt_for_record(label, rec)
        isempty(prompt) && continue
        direct = _direct_answer(memories, label, prompt)
        isempty(strip(direct)) && continue
        return prompt, direct
    end
    return "", ""
end

function _print_case(memories, label::AbstractString, prompt::AbstractString)
    println("="^72)
    println("CASE: $(label)")
    println("PROMPT: $(prompt)")
    direct = _direct_answer(memories, label, prompt)
    fallback_prompt, fallback_answer = "", ""
    if isempty(strip(direct))
        fallback_prompt, fallback_answer = _trained_prompt_fallback(memories, label)
        if !isempty(strip(fallback_answer))
            println("fixed prompt had no direct answer; using trained-memory prompt:")
            println("PROMPT: $(fallback_prompt)")
            prompt = fallback_prompt
            direct = fallback_answer
        end
    end
    if label == "conditional"
        rec = Physics.select_relation_frame_attention(memories.istinbat, String(prompt))
        if rec !== nothing
            println("selected: relation=$(rec.relation_type) marker=$(rec.marker) before=$(join(rec.before_terms, ",")) after=$(join(rec.after_terms, ",")) examples=$(length(rec.examples))")
            !isempty(rec.examples) && println("selected_example: ", first(rec.examples))
        else
            println("selected: none")
        end
    end
    if isempty(strip(direct))
        println("No direct trained answer found for this layer/prompt.")
        return
    end
    off = _polish(prompt, direct; polisher=false)
    on = _polish(prompt, direct; polisher=true)
    profile = Gen.response_polish_profile(String(prompt), off)
    println("profile: kind=$(profile.kind) language=$(profile.language) repetition=$(profile.has_repetition) run_on=$(profile.run_on) preserved=$(profile.preserved)")
    println("-- polisher off --")
    println(off)
    println("-- polisher on --")
    println(on)
    println("-- changed --")
    println(off == on ? "no" : "yes")
end

function main()
    memories = _load_trained_memories()
    limit = tryparse(Int, get(ENV, "MIRNAN_TRAINED_POLISHER_PROBE_LIMIT", "3"))
    limit === nothing && (limit = 3)

    println("Trained ResponsePolisher probe")
    println("case_limit: ", limit)
    println("source: trained memories, direct layer answers")

    cases = Pair{String,String}[
        "quantity" => "\u0643\u0645 \u0639\u062F\u062F \u062C\u0645\u0639 \u0627\u0644\u0642\u0644\u0629 \u064A\u062F\u0644 \u0639\u0644\u0649\u061f",
        "conditional" => "\u0645\u0627\u0630\u0627 \u064A\u062D\u062F\u062B \u0625\u0630\u0627 \u0632\u0627\u062F \u0627\u0644\u0639\u0644\u0645\u061f",
        "scene-purpose" => "\u0644\u0645\u0627\u0630\u0627 \u062F\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062D\u062C\u0631\u061f",
        "temporal" => "\u0645\u062A\u0649 \u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628\u061F",
        "spatial" => "\u0623\u064A\u0646 \u062C\u0644\u0633 \u0627\u0644\u0637\u0641\u0644\u061F",
        "state" => "\u0643\u064A\u0641 \u062F\u062E\u0644 \u0627\u0644\u0637\u0641\u0644\u061F",
    ]

    for (i, (label, prompt)) in enumerate(cases)
        i > limit && break
        _print_case(memories, label, prompt)
    end
end

main()
