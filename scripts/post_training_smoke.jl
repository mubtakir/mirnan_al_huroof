#!/usr/bin/env julia

# Lightweight post-training probes. This is intentionally separate from the
# regular test suite because it loads trained model artifacts and can take time.

const MIRNAN_DIR = dirname(@__DIR__)
const SCRIPT_DIR = @__DIR__

function _set_default_env!(name::AbstractString, value::AbstractString)
    if !haskey(ENV, name) || isempty(strip(get(ENV, name, "")))
        ENV[name] = value
    end
    return ENV[name]
end

function _run_probe(label::AbstractString, script_name::AbstractString)
    script_path = joinpath(SCRIPT_DIR, script_name)
    println()
    println("=" ^ 72)
    println("POST-TRAINING SMOKE: ", label)
    println("=" ^ 72)
    if !isfile(script_path)
        println("FAILED: missing script: ", script_path)
        return false
    end

    ok = success(`$(Base.julia_cmd()) --project=$MIRNAN_DIR $script_path`)
    println(ok ? "PASSED: $label" : "FAILED: $label")
    return ok
end

function main()
    println("Mirnan post-training smoke checks")
    println("project: ", MIRNAN_DIR)

    _set_default_env!("MIRNAN_BRIDGE_PROBE_LIMIT", "200")
    _set_default_env!("MIRNAN_BRIDGE_SCENE_ONLY_LIMIT", "60")
    _set_default_env!("MIRNAN_BRIDGE_PROBE_SECONDS", "45")
    _set_default_env!("MIRNAN_QUANTITY_TRAINED_PROBE_GATE_OFF", "0")

    probes = Pair{String,String}[
        "response polisher" => "response_polisher_live_probe.jl",
        "semantic scene / purpose bridge" => "semantic_scene_purpose_bridge_probe.jl",
        "quantity memory" => "trained_quantity_memory_probe.jl",
    ]
    heavy_gates = lowercase(strip(get(ENV, "MIRNAN_POST_SMOKE_HEAVY_GATES", "0"))) in ("1", "true", "yes", "on")
    if heavy_gates
        push!(probes, "quantity trained gate" => "quantity_trained_question_probe.jl")
        push!(probes, "trained response polisher" => "trained_response_polisher_probe.jl")
    else
        println("heavy gate probes: skipped (set MIRNAN_POST_SMOKE_HEAVY_GATES=1 to enable)")
    end

    results = Pair{String,Bool}[]
    for (label, script_name) in probes
        push!(results, label => _run_probe(label, script_name))
    end

    println()
    println("=" ^ 72)
    println("POST-TRAINING SMOKE SUMMARY")
    println("=" ^ 72)
    for (label, ok) in results
        println(rpad(label, 36), ok ? "PASS" : "FAIL")
    end

    failed = count(!last(result) for result in results)
    if failed > 0
        println("failed: ", failed, " / ", length(results))
        exit(1)
    end
    println("all post-training smoke checks passed")
end

main()
