include("../src/MirnanNew.jl")
using .MirnanNew
using Test

const Physics = MirnanNew.Physics

@testset "al_lisan linguistic pattern memory" begin
    dir = mktempdir()
    mem = Physics.LinguisticPatternMemory()

    custom_marker = "\u062d\u064a\u0646\u0626\u0630\u0645\u0627"
    custom_marker_path = joinpath(dir, "custom_lisan_markers.json")
    write(custom_marker_path, "{\"arabic_fixed_markers\":[\"\\u062d\\u064a\\u0646\\u0626\\u0630\\u0645\\u0627\"]}")
    custom_mem = Physics.LinguisticPatternMemory(marker_path=custom_marker_path)
    custom_learned = Physics.learn_lisan_from_text!(
        custom_mem,
        custom_marker * " " * "\u0627\u0644\u0639\u0644\u0645 \u0646\u0627\u0641\u0639.",
    )
    @test custom_learned == 1
    custom_rec = Physics.select_lisan_pattern(
        custom_mem,
        [custom_marker];
        prefer_verbal=false,
    )
    @test custom_rec !== nothing
    @test "marker" in custom_rec.roles

    categorized_marker = "\u0631\u064a\u062b\u0645\u0627"
    categorized_marker_path = joinpath(dir, "categorized_lisan_markers.json")
    write(categorized_marker_path,
          "{\"arabic_fixed_markers\":{\"temporal\":[\"\\u0631\\u064a\\u062b\\u0645\\u0627\"]}}")
    categorized_mem = Physics.LinguisticPatternMemory(marker_path=categorized_marker_path)
    @test categorized_marker in categorized_mem.fixed_markers

    nominal_text = "\u0627\u0644\u0639\u0644\u0645 \u0646\u0648\u0631. " *
                   "\u0628\u0627\u0628 \u0627\u0644\u0628\u064a\u062a \u0645\u0641\u062a\u0648\u062d. " *
                   "\u0637\u0627\u0644\u0645\u0627 \u0627\u0644\u0645\u0637\u0631 \u0646\u0627\u0641\u0639."

    learned = Physics.learn_lisan_from_text!(mem, nominal_text)

    @test learned >= 3

    english_text = "Knowledge raises understanding. " *
                   "The student is curious. " *
                   "If language moves, meaning follows."

    learned_en = Physics.learn_lisan_from_text!(mem, english_text)

    @test learned_en >= 3

    expanded_english_text = "Does language shape meaning? " *
                            "There is a pattern. " *
                            "Meaning is shaped by language. " *
                            "Can knowledge raise wisdom? " *
                            "However language guides thought."

    learned_en_more = Physics.learn_lisan_from_text!(mem, expanded_english_text)

    @test learned_en_more >= 5
    @test Physics.has_lisan_patterns(mem)

    saved = Physics.save_lisan(mem, joinpath(dir, "al_lisan.json"))
    @test isfile(saved)

    loaded = Physics.load_lisan(saved)
    @test Physics.has_lisan_patterns(loaded)

    nominal_rec = Physics.select_lisan_pattern(
        loaded,
        ["\u0627\u0644\u0639\u0644\u0645"];
        prefer_verbal=false,
    )
    @test nominal_rec !== nothing
    @test nominal_rec.language == "ar"

    english_rec = Physics.select_lisan_pattern(
        loaded,
        ["Knowledge"];
        prefer_verbal=false,
    )
    @test english_rec !== nothing
    @test english_rec.language == "en"

    # Test pattern-based word listing (no K_sem needed)
    ar_words = Physics.lisan_to_dict(loaded)["patterns"]
    eng_patterns = filter(p -> p["language"] == "en", ar_words)
    ar_patterns = filter(p -> p["language"] == "ar", ar_words)

    @test length(ar_patterns) >= 1
    @test length(eng_patterns) >= 1

    # Test token_role_phase mapping
    @test Physics.token_role_phase("verb") ≈ π/2
    @test Physics.token_role_phase("noun") ≈ π/4
    @test Physics.token_role_phase("adjective") ≈ 3π/4
    @test Physics.ROLE_SYNTAX_PHASE["verb"] ≈ π/2
    @test Physics.ROLE_SYNTAX_PHASE["subject"] ≈ π/4

    # Verify loaded patterns contain examples
    for p in ar_patterns
        @test !isempty(p["roles"])
        @test length(p["examples"]) >= 1
    end
end
