#!/usr/bin/env julia
# Wider diagnostic matrix for Mirnan semantic imagination.
# This script does not call generate! and does not affect runtime behavior.

include(joinpath(@__DIR__, "..", "src", "MirnanNew.jl"))

using .MirnanNew

const Physics = MirnanNew.Physics

const DEFAULT_MATRIX = joinpath(@__DIR__, "..", "test", "fixtures", "semantic_scene_probe_matrix.tsv")

struct ProbeCase
    source::String
    target::String
    prompt::String
    expected::String
end

function _decode_unicode_escapes(s::AbstractString)
    return replace(String(s), r"\\u[0-9A-Fa-f]{4}" => m -> string(Char(parse(Int, String(m)[3:end]; base=16))))
end

function _print_help()
    println("""
Semantic scene probe matrix.

Usage:
  julia semantic_scene_probe_matrix.jl [matrix.tsv]

TSV columns:
  source<TAB>target<TAB>prompt<TAB>expected

Expected labels are diagnostic expectations such as:
  aligned, partial, scene_only, calculus_only, divergent, none

This tool is diagnostic only. It does not call generate!.
""")
end

function _read_cases(path::AbstractString)
    isfile(path) || error("matrix file not found: $path")
    cases = ProbeCase[]
    for (lineno, raw) in enumerate(eachline(path))
        line = strip(raw)
        isempty(line) && continue
        startswith(line, "#") && continue
        cols = split(line, '\t')
        length(cols) == 4 || error("bad TSV line $lineno: expected 4 columns")
        push!(cases, ProbeCase(_decode_unicode_escapes(strip(cols[1])),
                               _decode_unicode_escapes(strip(cols[2])),
                               _decode_unicode_escapes(strip(cols[3])),
                               strip(cols[4])))
    end
    return cases
end

function _build_shared_calculus(cases)
    calculus = Physics.SemanticCalculusMemory()
    for c in cases
        Physics.learn_semantic_calculus_from_pair!(calculus, c.source, c.target)
    end
    return calculus
end

function _build_case_scene(c::ProbeCase)
    pair_calculus = Physics.SemanticCalculusMemory()
    scene_mem = Physics.SemanticSceneMemory()
    Physics.learn_semantic_calculus_from_pair!(pair_calculus, c.source, c.target)
    Physics.learn_semantic_scene_from_text!(scene_mem, pair_calculus, c.source)
    return scene_mem
end

function _actual_ok(expected::String, actual::String)
    expected == actual && return true
    expected == "aligned" && actual == "partial" && return true
    expected == "partial" && actual == "aligned" && return true
    return false
end

function _needs_review(expected::String, actual::String, cmp)
    _actual_ok(expected, actual) && return false
    actual == "aligned" && expected in ("scene_only", "divergent", "none") && return true
    actual == "partial" && expected in ("none",) && return true
    return cmp.overlap_score >= 0.80 && expected != actual
end

function _scene_summary(scene)
    scene === nothing && return "none"
    return "$(scene.action)/$(scene.patient)"
end

function _print_row(i::Int, c::ProbeCase, cmp)
    actual = cmp.agreement
    ok = _actual_ok(c.expected, actual)
    review = _needs_review(c.expected, actual, cmp)
    status = ok ? "OK" : (review ? "REVIEW" : "MISS")
    println(join([
        string(i),
        status,
        "expected=$(c.expected)",
        "actual=$(actual)",
        "overlap=$(round(cmp.overlap_score; digits=3))",
        "scene=$(_scene_summary(cmp.scene))",
        "prompt=$(c.prompt)",
        "filtered=$(join(cmp.guidance_terms, ","))",
    ], " | "))
    return status
end

function main()
    if any(arg -> arg in ("-h", "--help"), ARGS)
        _print_help()
        return
    end

    path = isempty(ARGS) ? DEFAULT_MATRIX : ARGS[1]
    cases = _read_cases(path)
    calculus = _build_shared_calculus(cases)

    counts = Dict("OK" => 0, "MISS" => 0, "REVIEW" => 0)
    println("Semantic scene probe matrix")
    println("cases: $(length(cases))")
    println("matrix: $path")
    println("-"^96)
    for (i, c) in enumerate(cases)
        scene_mem = _build_case_scene(c)
        cmp = Physics.compare_semantic_scene_with_calculus(scene_mem, calculus, c.prompt)
        status = _print_row(i, c, cmp)
        counts[status] = get(counts, status, 0) + 1
    end
    println("-"^96)
    println("SUMMARY: OK=$(counts["OK"]) REVIEW=$(counts["REVIEW"]) MISS=$(counts["MISS"]) TOTAL=$(length(cases))")
end

main()
