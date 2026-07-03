using Test

include(joinpath(@__DIR__, "..", "src", "MirnanNew.jl"))

const Gen = MirnanNew.Physics.Generator

@testset "response polisher conservative final wording" begin
    raw = "  العلم   العلم   يزيد   الفهم  "
    @test Gen.polish_response("لماذا؟", raw; enabled=false) == strip(raw)

    polished = Gen.polish_response("لماذا؟", raw; enabled=true)
    @test polished == "العلم يزيد الفهم."

    english = Gen.polish_response("why?", "science science increases understanding"; enabled=true)
    @test english == "science increases understanding."

    arabic_punct = Gen.polish_response("لماذا؟", "العلم يزيد الفهم, والعدل يحفظ السلام"; enabled=true)
    @test occursin("\u060C", arabic_punct)
    @test endswith(arabic_punct, ".")

    run_on = Gen.polish_response(
        "اشرح",
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen",
        enabled=true,
    )
    @test occursin(",", run_on)
    @test endswith(run_on, ".")

    purpose_profile = Gen.response_polish_profile(
        "\u0644\u0645\u0627\u0630\u0627\u061F",
        "\u0627\u0644\u063A\u0627\u064A\u0629 \u0645\u0646 \u0627\u0644\u062F\u0631\u0627\u0633\u0629 \u0647\u064A \u0627\u0644\u0646\u062C\u0627\u062D.",
    )
    @test purpose_profile.kind == "purpose"
    @test purpose_profile.language == "arabic"
    @test purpose_profile.has_repetition == false

    conditional_profile = Gen.response_polish_profile(
        "what happens if student studies?",
        "If student studies happens, the result is student succeeds.",
    )
    @test conditional_profile.kind == "conditional"
    @test conditional_profile.language == "latin"

    conditional_polished = Gen.polish_response(
        "\u0645\u0627\u0630\u0627 \u064A\u062D\u062F\u062B \u0625\u0630\u0627 \u0632\u0627\u062F \u0627\u0644\u0639\u0644\u0645\u061F",
        "\u0625\u0630\u0627 \u0632\u0627\u062F \u0627\u0644\u0639\u0644\u0645, \u064A\u062A\u0631\u062A\u0628 \u0639\u0644\u0649 \u0630\u0644\u0643 \u0632\u0627\u062F \u0627\u0644\u0641\u0647\u0645",
        enabled=true,
    )
    @test occursin("\u061B \u064A\u062A\u0631\u062A\u0628 \u0639\u0644\u0649 \u0630\u0644\u0643", conditional_polished)

    quantity_profile = Gen.response_polish_profile(
        "\u0643\u0645 \u0639\u062F\u062F \u0627\u0644\u0637\u0644\u0627\u0628\u061F",
        "\u0639\u062F\u062F \u0627\u0644\u0637\u0644\u0627\u0628 \u0647\u0648 30.",
    )
    @test quantity_profile.kind == "quantity"

    trained_quantity_profile = Gen.response_polish_profile(
        "\u0643\u0645 \u0639\u062F\u062F \u062C\u0645\u0639 \u0627\u0644\u0642\u0644\u0629 \u064A\u062F\u0644 \u0639\u0644\u0649\u061F",
        "\u064A\u062F\u0644 \u062C\u0645\u0639 \u0627\u0644\u0642\u0644\u0629 \u0639\u0644\u0649 \u0645\u0646 \u062B\u0644\u0627\u062B\u0629 \u0625\u0644\u0649 \u0639\u0634\u0631\u0629.",
    )
    @test trained_quantity_profile.kind == "quantity"

    scene_profile = Gen.response_polish_profile(
        "what happens when hit affects ball?",
        "When hit affects ball, the semantic effect includes movement.",
    )
    @test scene_profile.kind == "scene"

    temporal_profile = Gen.response_polish_profile(
        "\u0645\u062A\u0649 \u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628\u061F",
        "\u0643\u0627\u0646 \u0633\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628 \u0642\u0628\u0644 \u0627\u0644\u0641\u062C\u0631.",
    )
    @test temporal_profile.kind == "temporal"

    spatial_profile = Gen.response_polish_profile(
        "\u0623\u064A\u0646 \u062C\u0644\u0633 \u0627\u0644\u0637\u0641\u0644\u061F",
        "\u0643\u0627\u0646 \u0645\u0643\u0627\u0646 \u062C\u0644\u0648\u0633 \u0627\u0644\u0637\u0641\u0644 \u062D\u064A\u062B \u0627\u0644\u062D\u062F\u064A\u0642\u0629.",
    )
    @test spatial_profile.kind == "spatial"

    composite = Gen.polish_response(
        "\u0644\u0645\u0627\u0630\u0627\u061F",
        "\u0639\u0646\u062F \u062F\u0641\u0639 \u0627\u0644\u062D\u062C\u0631 \u064A\u0646\u062A\u0642\u0644 \u0627\u0644\u0645\u062A\u0623\u062B\u0631 \u0645\u0646 \u0633\u0643\u0648\u0646 \u0625\u0644\u0649 \u062D\u0631\u0643\u0629, \u0648\u0645\u0646 \u062C\u0647\u0629 \u0627\u0644\u063A\u0627\u064A\u0629: \u0625\u0628\u0639\u0627\u062F \u0627\u0644\u062D\u062C\u0631",
        enabled=true,
    )
    @test occursin("\u061B \u0648\u0645\u0646 \u062C\u0647\u0629 \u0627\u0644\u063A\u0627\u064A\u0629:", composite)
    @test Gen.response_polish_profile("\u0644\u0645\u0627\u0630\u0627\u061F", composite).kind == "scene"

    trained_scene_profile = Gen.response_polish_profile(
        "\u0644\u0645\u0627\u0630\u0627 \u062F\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062D\u062C\u0631\u061F",
        "\u062F\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062D\u062C\u0631\u061B \u0641\u0627\u0646\u062A\u0642\u0644 \u0627\u0644\u062D\u062C\u0631 \u0645\u0646 \u0633\u0643\u0648\u0646 \u0625\u0644\u0649 \u062D\u0631\u0643\u0629\u061B \u0648\u0638\u0647\u0631\u062A \u0622\u062B\u0627\u0631 \u0645\u062B\u0644 \u062D\u0631\u0643\u0629\u061B \u0648\u0643\u0627\u0646\u062A \u0627\u0644\u063A\u0627\u064A\u0629 \u0625\u0628\u0639\u0627\u062F \u0627\u0644\u062D\u062C\u0631.",
    )
    @test trained_scene_profile.kind == "scene"

    state_polished = Gen.polish_response(
        "\u0643\u064A\u0641 \u062F\u062E\u0644 \u0627\u0644\u0637\u0641\u0644\u061F",
        "\u062F\u062E\u0644 \u0627\u0644\u0637\u0641\u0644, \u0648\u0643\u0627\u0646 \u0639\u0644\u0649 \u062D\u0627\u0644 \u062E\u0627\u0626\u0641",
        enabled=true,
    )
    @test occursin("\u061B \u0648\u0643\u0627\u0646 \u0639\u0644\u0649 \u062D\u0627\u0644", state_polished)
    @test Gen.response_polish_profile("\u0643\u064A\u0641\u061F", state_polished).kind == "state"

    quantity_polished = Gen.polish_response(
        "how many students?",
        "The number of students is is 30",
        enabled=true,
    )
    @test quantity_polished == "The number of students is 30."

    codeish = "\u2500\u2500\u2500 Majnon code memory\nsha1:abc\nprintln(1)"
    @test Gen.response_polish_profile("\u0627\u0634\u0631\u062D", codeish).preserved == true
    @test Gen.polish_response("\u0627\u0634\u0631\u062D", codeish; enabled=true) == strip(codeish)
end
