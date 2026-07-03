#!/usr/bin/env julia
# Diagnostic probe for RelationFrame conditional memory.

const MIRNAN_DIR = dirname(@__DIR__)
const MODEL_DIR = joinpath(MIRNAN_DIR, "model")

include(joinpath(MIRNAN_DIR, "src", "MirnanNew.jl"))

using .MirnanNew

const Physics = MirnanNew.Physics

function _env_int(name::String, default::Int)
    raw = get(ENV, name, "")
    isempty(strip(raw)) && return default
    try
        return parse(Int, raw)
    catch
        return default
    end
end

function _term_text(terms::Vector{String})
    return strip(join(terms, " "))
end

function _conditional_prompt(rec)
    left = _term_text(rec.before_terms)
    right = _term_text(rec.after_terms)
    core = isempty(left) ? right : left
    isempty(core) && return ""
    return "\u0645\u0627\u0630\u0627 \u064a\u062d\u062f\u062b \u0625\u0630\u0627 $(core)\u061f"
end

function _mock_generate(prompt::AbstractString)
    return "\u062c\u0648\u0627\u0628 \u0639\u0627\u0645: $(prompt)"
end

function _print_result(label::String, mem, prompt::String)
    result = Physics.compare_conditional_strategies(mem, _mock_generate, prompt)
    println("="^72)
    println("CASE: $(label)")
    println("PROMPT: $(prompt)")
    println("MEMORY_HAS_CONDITIONAL: $(result.memory_has_conditional)")
    println("CONDITIONAL_CONFIDENCE: $(round(result.conditional_confidence; digits=3))")
    println("OVERLAP: $(round(result.overlap_score; digits=3))")
    println("RELATION_TYPE: $(result.relation_type)")
    println("-- conditional answer --")
    println(isempty(result.conditional_answer) ? "(empty)" : result.conditional_answer)
    println("-- generate answer --")
    println(isempty(result.generate_answer) ? "(empty)" : result.generate_answer)
end

function _controlled_memory()
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(
        mem,
        "\u0625\u0630\u0627 \u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645 \u0632\u0627\u062f \u0627\u0644\u0641\u0647\u0645.",
    )
    return mem
end

function _find_trained_conditional(mem, limit::Int)
    seen = 0
    for rec in values(mem.records)
        rec.relation_type == "conditional" || continue
        seen += 1
        left = _term_text(rec.before_terms)
        right = _term_text(rec.after_terms)
        if !isempty(left) && !isempty(right)
            prompt = _conditional_prompt(rec)
            if !isempty(prompt)
                result = Physics.compare_conditional_strategies(mem, _mock_generate, prompt)
                result.memory_has_conditional && return prompt
            end
        end
        seen >= limit && break
    end
    return ""
end

function main()
    limit = _env_int("MIRNAN_CONDITIONAL_PROBE_LIMIT", 300)
    istinbat_path = joinpath(MODEL_DIR, "al_istinbat.json")
    mem = Physics.load_istinbat(istinbat_path)
    conditional_count = count(rec -> rec.relation_type == "conditional", values(mem.records))

    println("Conditional RelationFrame probe")
    println("istinbat_records: $(length(mem.records))")
    println("conditional_records: $(conditional_count)")
    println("scan_limit: $(limit)")

    controlled = _controlled_memory()
    _print_result(
        "controlled conditional sanity",
        controlled,
        "\u0645\u0627\u0630\u0627 \u064a\u062d\u062f\u062b \u0625\u0630\u0627 \u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645\u061f",
    )

    prompt = _find_trained_conditional(mem, limit)
    if isempty(prompt)
        println("="^72)
        println("CASE: trained conditional")
        println("No trained conditional example found in scan window.")
    else
        _print_result("trained conditional", mem, prompt)
    end

    _print_result(
        "yes/no guard",
        mem,
        "\u0647\u0644 \u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645\u061f",
    )
end

main()
