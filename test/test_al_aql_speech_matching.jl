using Test

include("../src/MirnanNew.jl")
using .MirnanNew

const Aql = MirnanNew.Physics.AlAql
const Gen = MirnanNew.Physics.Generator

@testset "al_aql speech act fuzzy matching" begin
    prompt = "\u0643\u064a\u0641 \u062a\u0645 \u0643\u0634\u0641 \u0645\u0624\u0627\u0645\u0631\u0629 \u0627\u0644\u0642\u0627\u0626\u062f \u0627\u0644\u062e\u0627\u0626\u0646\u061f"
    prompt_terms = Gen._clean_prompt_terms(prompt)
    prompt_norm = lowercase(join(prompt_terms, " "))
    prompt_keys = Gen._aql_speech_match_keys(prompt_terms)
    fact = Aql.SpeechActFact(
        "\u0633",
        "\u0633\u0624\u0627\u0644",
        "\u0643\u064a\u0641 \u062a\u0645 \u0643\u0634\u0641 \u0645\u0648\u0627\u0645\u0631\u0647 \u0627\u0644\u0642\u0627\u064a\u062f \u0627\u0644\u062e\u0627\u064a\u0646",
        "\u062c",
        "\u062c\u0648\u0627\u0628",
        "\u0643\u0634\u0641\u0647\u0627 \u0627\u062d\u062f \u0627\u0644\u062c\u0646\u0648\u062f.",
        0.78,
        "test",
    )
    @test Gen._aql_speech_act_score(prompt_terms, prompt_norm, prompt_keys, fact) >= 0.62

    opinion_prompt = "\u0645\u0627 \u0631\u0623\u064a\u0643 \u0641\u064a \u0627\u0644\u0642\u0631\u0627\u0621\u0629\u061f"
    opinion_terms = Gen._clean_prompt_terms(opinion_prompt)
    opinion_norm = lowercase(join(opinion_terms, " "))
    opinion_keys = Gen._aql_speech_match_keys(opinion_terms)
    friendship_fact = Aql.SpeechActFact(
        "\u0633",
        "\u0633\u0624\u0627\u0644",
        "\u0645\u0627 \u0631\u0623\u064a\u0643 \u0641\u064a \u0627\u0644\u0635\u062f\u0627\u0642\u0629\u061f",
        "\u062c",
        "\u062c\u0648\u0627\u0628",
        "\u0627\u0644\u0635\u062f\u0627\u0642\u0629 \u0639\u0644\u0627\u0642\u0629 \u062c\u0645\u064a\u0644\u0629.",
        0.8,
        "test",
    )
    @test Gen._aql_speech_act_score(opinion_terms, opinion_norm, opinion_keys, friendship_fact) == 0.0
end
