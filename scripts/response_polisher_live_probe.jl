#!/usr/bin/env julia
# Live probe for the conservative response polisher.
#
# This script compares the same answer with MIRNAN_ENABLE_RESPONSE_POLISHER off
# and on. It does not generate new answers; it only shows the final wording pass.

const MIRNAN_DIR = dirname(@__DIR__)

include(joinpath(MIRNAN_DIR, "src", "MirnanNew.jl"))

using .MirnanNew

const Gen = MirnanNew.Physics.Generator

function _set_or_delete!(name::String, value)
    if value === nothing
        delete!(ENV, name)
    else
        ENV[name] = value
    end
end

function _polish(prompt::AbstractString, answer::AbstractString; gate::Bool)
    old = get(ENV, "MIRNAN_ENABLE_RESPONSE_POLISHER", nothing)
    try
        ENV["MIRNAN_ENABLE_RESPONSE_POLISHER"] = gate ? "1" : "0"
        enabled = get(ENV, "MIRNAN_ENABLE_RESPONSE_POLISHER", "0") == "1"
        return Gen.polish_response(String(prompt), String(answer); enabled=enabled)
    finally
        _set_or_delete!("MIRNAN_ENABLE_RESPONSE_POLISHER", old)
    end
end

function _print_profile(prompt::AbstractString, answer::AbstractString)
    profile = Gen.response_polish_profile(String(prompt), String(answer))
    println("profile: kind=$(profile.kind) language=$(profile.language) repetition=$(profile.has_repetition) run_on=$(profile.run_on) preserved=$(profile.preserved)")
end

function _print_case(label::AbstractString, prompt::AbstractString, answer::AbstractString)
    println("="^72)
    println("CASE: $(label)")
    println("PROMPT: $(prompt)")
    _print_profile(prompt, answer)
    println("-- gate off --")
    println(_polish(prompt, answer; gate=false))
    println("-- gate on --")
    println(_polish(prompt, answer; gate=true))
end

function main()
    println("ResponsePolisher live probe")
    println("gate variable: MIRNAN_ENABLE_RESPONSE_POLISHER")

    _print_case(
        "repeated words",
        "\u0644\u0645\u0627\u0630\u0627\u061F",
        "\u0627\u0644\u0639\u0644\u0645   \u0627\u0644\u0639\u0644\u0645   \u064A\u0632\u064A\u062F   \u0627\u0644\u0641\u0647\u0645",
    )

    _print_case(
        "conditional surface split",
        "\u0645\u0627\u0630\u0627 \u064A\u062D\u062F\u062B \u0625\u0630\u0627 \u0632\u0627\u062F \u0627\u0644\u0639\u0644\u0645\u061F",
        "\u0625\u0630\u0627 \u0632\u0627\u062F \u0627\u0644\u0639\u0644\u0645, \u064A\u062A\u0631\u062A\u0628 \u0639\u0644\u0649 \u0630\u0644\u0643 \u0632\u0627\u062F \u0627\u0644\u0641\u0647\u0645",
    )

    _print_case(
        "scene-purpose surface split",
        "\u0644\u0645\u0627\u0630\u0627 \u062F\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062D\u062C\u0631\u061F",
        "\u0639\u0646\u062F \u062F\u0641\u0639 \u0627\u0644\u062D\u062C\u0631 \u064A\u0646\u062A\u0642\u0644 \u0627\u0644\u0645\u062A\u0623\u062B\u0631 \u0645\u0646 \u0633\u0643\u0648\u0646 \u0625\u0644\u0649 \u062D\u0631\u0643\u0629, \u0648\u0645\u0646 \u062C\u0647\u0629 \u0627\u0644\u063A\u0627\u064A\u0629: \u0625\u0628\u0639\u0627\u062F \u0627\u0644\u062D\u062C\u0631 \u0639\u0646 \u0627\u0644\u0637\u0631\u064A\u0642",
    )

    _print_case(
        "state surface split",
        "\u0643\u064A\u0641 \u062F\u062E\u0644 \u0627\u0644\u0637\u0641\u0644\u061F",
        "\u062F\u062E\u0644 \u0627\u0644\u0637\u0641\u0644, \u0648\u0643\u0627\u0646 \u0639\u0644\u0649 \u062D\u0627\u0644 \u062E\u0627\u0626\u0641",
    )

    _print_case(
        "English quantity repetition",
        "how many students?",
        "The number of students is is 30",
    )

    _print_case(
        "preserved code-like memory",
        "\u0627\u0634\u0631\u062D",
        "\u2500\u2500\u2500 Majnon code memory\nsha1:abc\nprintln(1)",
    )
end

main()
