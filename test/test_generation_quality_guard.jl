include("../src/MirnanNew.jl")
using .MirnanNew
using Test
using SparseArrays

const Physics = MirnanNew.Physics
const Gen = Physics.Generator

@testset "generation quality guard" begin
    science_prompt = ["\u0627\u0644\u0639\u0644\u0645", "\u0646\u0648\u0631"]
    bad_science = "\u0641\u0631\u0648\u0639 \u0627\u0637\u0644\u0628\u0648\u0627 \u0627\u0644\u062f\u0631\u0633 \u0627\u0633\u062a\u0627\u0630\u064a \u0627\u0644\u0631\u064a\u0627\u0636\u064a\u0627\u062a"

    @test Gen._needs_simple_declarative_template(science_prompt, bad_science)
    @test isempty(Gen._simple_text_template(science_prompt))
    metadata_science = "\u0627\u0644\u0639\u0644\u0645\u064f \u0631\u0627\u064a\u0643 \u0627\u0644\u062a\u0639\u0644\u064a\u0645 \u0645\u0633\u0648\u0648\u0644\u064a\u0647 \u062a\u0642\u0639 \u0627\u0644\u0635\u062f\u064a\u0642\u0647 \u062a\u0631\u0628\u0637\u0647\u0627 \u062a\u0635\u0646\u064a\u0641:\u0627\u0644\u0639\u0627\u0628"
    @test Gen._needs_simple_declarative_template(science_prompt, metadata_science)

    sky_prompt = ["\u0627\u0644\u0633\u0645\u0627\u0621", "\u0635\u0627\u0641\u064a\u0629"]
    bad_sky = "\u0644\u0645\u0627\u0630\u0627 \u0632\u0631\u0642\u0627\u0621 \u0627\u0644\u0634\u0645\u0633 \u0636\u0648\u0621 \u0627\u0644\u0627\u0631\u0636"

    @test Gen._needs_simple_declarative_template(sky_prompt, bad_sky)
    @test isempty(Gen._simple_text_template(sky_prompt))

    good_sky = "\u0627\u0644\u0633\u0645\u0627\u0621 \u0635\u0627\u0641\u064a\u0629 \u0648\u0627\u0633\u0639\u0629\u060c \u064a\u0645\u0644\u0624\u0647\u0627 \u0627\u0644\u0636\u0648\u0621."
    @test !Gen._needs_simple_declarative_template(sky_prompt, good_sky)

    wisdom_prompt = ["\u0627\u0644\u062d\u0643\u0645\u0629", "\u0637\u0631\u064a\u0642"]
    bad_wisdom = "\u0627\u0644\u062a\u0627\u0645\u0644 \u064a\u0636\u064a\u0621 \u0648\u062d\u0631\u0627\u0631\u062a\u0647 \u0627\u0644\u0628\u0634\u0631\u064a\u0647 \u0627\u0644\u0639\u0644\u0645 \u0627\u0644\u062d\u0636\u0627\u0631\u0627\u062a \u0627\u0644\u0642\u062f\u064a\u0645\u0647 \u0648\u0636\u0639\u062a"
    @test Gen._needs_simple_declarative_template(wisdom_prompt, bad_wisdom)
    @test isempty(Gen._simple_text_template(wisdom_prompt))

    learning_prompt = ["\u0643\u064a\u0641", "\u064a\u062a\u0639\u0644\u0645", "\u0627\u0644\u0625\u0646\u0633\u0627\u0646"]
    bad_learning = "\u0633\u0627\u0645\u0631 \u0628\u0627\u0641\u0644\u0648\u0641 \u0627\u0644\u0645\u0639\u0632\u0632 \u062e\u0644\u0627\u0644 \u0627\u0644\u062f\u0648\u0644 \u0627\u0646\u062a\u0627\u062c \u0627\u0644\u0645\u0648\u0627\u062f \u0645\u0627\u062f\u0647"
    @test Gen._needs_simple_question_template(learning_prompt, bad_learning)
    @test isempty(Gen._simple_question_template(learning_prompt))

    cause_prompt = ["\u0625\u0630\u0627", "\u0632\u0627\u062f", "\u0627\u0644\u0639\u0644\u0645", "\u0632\u0627\u062f", "\u0627\u0644\u0641\u0647\u0645"]
    bad_cause = "\u062a\u0639\u0632\u0632 \u0627\u0644\u062a\u0641\u0627\u0639\u0644\u064a\u0647 \u0627\u0644\u062a\u0648\u0627\u0632\u0646 \u064a\u0633\u0627\u0648\u064a \u0648\u0644\u0627\u062f\u0647 \u064a\u0641\u0633\u0631 \u0646\u0645\u0648 \u062d\u0636\u0627\u0631\u0647"
    @test Gen._needs_simple_declarative_template(cause_prompt, bad_cause)
    @test isempty(Gen._simple_text_template(cause_prompt))

    meaning_prompt = ["\u0645\u0627", "\u0645\u0639\u0646\u0649", "\u0627\u0644\u062d\u0643\u0645\u0629\u061f"]
    bad_meaning = "\u0641\u064a\u0632\u064a\u0627\u0621 \u0627\u0644\u062d\u0631\u0641 \u0644\u0643\u0644\u0645\u0629 \u0627\u0644\u062c\u0630\u0631\u064a \u062d\u0643\u0645 \u0627\u0644\u0643\u062a\u0644\u0629 \u0627\u0644\u062a\u0646\u0627\u063a\u0645"
    @test Gen._needs_simple_question_template(meaning_prompt, bad_meaning)
    @test isempty(Gen._simple_question_template(meaning_prompt))
    spectral_meaning = "\u0641\u064a\u0632\u064a\u0627\u0621 \u0627\u0644\u062d\u0631\u0641 \u0644\u0643\u0644\u0645\u0629 \u0627\u0644\u062c\u0630\u0631\u064a \u062d\u0643\u0645\u060c \u0627\u0644\u0643\u062a\u0644\u0629 \u0627\u0644\u062a\u0646\u0627\u063a\u0645 \u0627\u0644\u062b\u0628\u0627\u062a."
    @test Gen._needs_simple_question_template(meaning_prompt, spectral_meaning)

    english_prompt = ["Knowledge"]
    english_template = Gen._simple_text_template(english_prompt)
    @test isempty(english_template)

    greeting_prompt = ["\u0627\u0644\u0633\u0644\u0627\u0645", "\u0639\u0644\u064a\u0643\u0645"]
    @test isempty(Gen._simple_text_template(greeting_prompt))

    science_definition_prompt = ["\u0645\u0627", "\u0647\u0648", "\u0627\u0644\u0639\u0644\u0645"]
    raw_aql = "\u0641\u064a \u0641\u0636\u0627\u0621 \u0627\u0644\u0639\u0642\u0644: \u0627\u0644\u0639\u0644\u0645 \u0643\u064a\u0627\u0646 \u0645\u0633\u062c\u0644\u061b \u0639\u0644\u0627\u0642\u0627\u062a\u0647: \u0627\u0644\u0639\u0644\u0645 \u062c\u0632\u0621_\u0645\u0646 \u0646\u0648\u0631\u061b \u0633\u0645\u0627\u062a\u0647: __classes=String[]."
    @test Gen._needs_simple_question_template(science_definition_prompt, raw_aql)
    science_definition = Gen._simple_question_template(science_definition_prompt)
    @test isempty(science_definition)
    @test !occursin("\u0641\u064a \u0641\u0636\u0627\u0621 \u0627\u0644\u0639\u0642\u0644", science_definition)

    name_prompt = ["\u0645\u0627", "\u0627\u0633\u0645\u0643"]
    @test isempty(Gen._simple_question_template(name_prompt))

    mirnan_prompt = ["\u0645\u0627", "\u0647\u0648", "\u0645\u0631\u0646\u0627\u0646"]
    @test isempty(Gen._simple_question_template(mirnan_prompt))

    justice_prompt = ["\u0643\u064a\u0641", "\u064a\u0643\u0648\u0646", "\u0627\u0644\u0639\u062f\u0644"]
    @test isempty(Gen._simple_question_template(justice_prompt))
    @test !occursin("\u062e\u0637\u0648\u0627\u062a \u0645\u062a\u062f\u0631\u062c\u0629", Gen._simple_question_template(justice_prompt))
    mechanistic_phrase = "\u0646\u0648\u0631 \u0627\u0644\u0639\u0644\u0645 \u0639\u0628\u0631 \u0641\u0647\u0645 \u0627\u0644\u0633\u064a\u0627\u0642 \u0648\u062a\u0646\u0638\u064a\u0645 \u0627\u0644\u062e\u0637\u0648\u0627\u062a\u060c \u062b\u0645 \u0645\u0631\u0627\u062c\u0639\u0629 \u0627\u0644\u0646\u062a\u064a\u062c\u0629."
    @test !occursin("\u0641\u0647\u0645 \u0627\u0644\u0633\u064a\u0627\u0642", Gen._simple_question_template(["\u0643\u064a\u0641", "\u0627\u0644\u0639\u0644\u0645", "\u0646\u0648\u0631"]))
    @test Gen._needs_simple_question_template(["\u0643\u064a\u0641", "\u0627\u0644\u0639\u0644\u0645", "\u0646\u0648\u0631"], mechanistic_phrase)

    yesno_prompt = "\u0647\u0644 \u062a\u062d\u0628 \u0627\u0644\u062a\u0639\u0644\u0645\u061f"
    messy_yesno = "\u0646\u0639\u0645 \u0627\u062d\u0628 \u064a\u0641\u062a\u062d \u0627\u0644\u0641\u0647\u0645 \u062a\u0639\u0644\u0645"
    @test Gen._repair_dialogue_yesno_word_order(yesno_prompt, messy_yesno) ==
          "\u0646\u0639\u0645\u060c \u0623\u062d\u0628 \u0627\u0644\u062a\u0639\u0644\u0645 \u0644\u0623\u0646\u0647 \u064a\u0641\u062a\u062d \u0627\u0644\u0641\u0647\u0645."
    unrelated_yesno = "\u0646\u0639\u0645 \u0627\u0644\u0645\u0639\u0631\u0641\u0647 \u0646\u0627\u0641\u0639\u0647 \u0639\u0645\u0644."
    @test Gen._repair_dialogue_yesno_word_order("\u0647\u0644 \u062a\u0624\u0645\u0646 \u0628\u0627\u0644\u0645\u0639\u0631\u0641\u0629\u061f", unrelated_yesno) == unrelated_yesno
    @test Gen._remove_wrong_yesno_for_question(
        "\u0645\u062a\u0649 \u0633\u0627\u0641\u0631 \u0627\u0644\u0631\u062c\u0644\u061f",
        "\u0646\u0639\u0645\u060c \u0633\u0627\u0641\u0631 \u0627\u0644\u0631\u062c\u0644.",
    ) == "\u0644\u0627 \u0623\u062c\u062f \u0632\u0645\u0646\u0627\u064b \u0645\u062d\u062f\u062f\u0627\u064b \u0641\u064a \u0627\u0644\u0630\u0627\u0643\u0631\u0629."
    @test Gen._remove_wrong_yesno_for_question(
        "\u0647\u0644 \u0633\u0627\u0641\u0631 \u0627\u0644\u0631\u062c\u0644\u061f",
        "\u0646\u0639\u0645\u060c \u0633\u0627\u0641\u0631 \u0627\u0644\u0631\u062c\u0644.",
    ) == "\u0646\u0639\u0645\u060c \u0633\u0627\u0641\u0631 \u0627\u0644\u0631\u062c\u0644."

    @test Gen._extract_benefit_subject("\u0645\u0627 \u0641\u0627\u0626\u062f\u0629 \u0627\u0644\u0646\u0648\u0645\u061f") == "\u0627\u0644\u0646\u0648\u0645"
    @test Gen._extract_benefit_subject("\u0644\u0645\u0627\u0630\u0627 \u0627\u0644\u0646\u0648\u0645 \u0636\u0631\u0648\u0631\u064a\u061f") == "\u0627\u0644\u0646\u0648\u0645"

    lexical_words = Gen._lexical_words_from_prompt("\u0627\u0644\u0639\u0644\u0645 \u0646\u0648\u0631.")
    @test lexical_words == ["\u0627\u0644\u0639\u0644\u0645", "\u0646\u0648\u0631"]

    @test Gen._explicit_causal_prompt("\u0644\u0645\u0627\u0630\u0627 \u064a\u0632\u064a\u062f \u0627\u0644\u0639\u0644\u0645 \u0627\u0644\u0641\u0647\u0645\u061f")
    @test !Gen._explicit_causal_prompt("\u0643\u064a\u0641 \u064a\u062a\u0639\u0644\u0645 \u0627\u0644\u0625\u0646\u0633\u0627\u0646\u061f")
    @test Gen._strong_code_request("\u0627\u0643\u062a\u0628 \u062d\u0644\u0642\u0629 \u0628\u0627\u064a\u062b\u0648\u0646 \u062a\u0637\u0628\u0639 \u0627\u0644\u0623\u0639\u062f\u0627\u062f \u0645\u0646 1 \u0625\u0644\u0649 5")

    fear_anchors = Gen._relationship_anchor_keys(["\u0644\u0645\u0627\u0630\u0627", "\u064a\u062e\u0634\u0649", "\u0627\u0644\u062c\u0627\u0647\u0644", "\u0627\u0644\u0639\u0644\u0645\u061f"])
    @test length(fear_anchors) == 2
    @test any(a -> "\u062c\u0627\u0647\u0644" in a || "\u0627\u0644\u062c\u0627\u0647\u0644" in a, fear_anchors)
    @test any(a -> "\u0639\u0644\u0645" in a || "\u0627\u0644\u0639\u0644\u0645" in a, fear_anchors)
    light_anchors = Gen._relationship_anchor_keys(["\u0643\u064a\u0641", "\u064a\u0628\u062f\u062f", "\u0627\u0644\u0646\u0648\u0631", "\u0627\u0644\u0638\u0644\u0645\u0629\u061f"])
    @test length(light_anchors) == 2
    @test "جهل" in Gen._generation_keys("\u0627\u0644\u063a\u0641\u0644\u0629")
    @test "ظلم" in Gen._generation_keys("\u0627\u0644\u062c\u0648\u0631")
    @test "عدل" in Gen._generation_keys("\u0627\u0644\u0625\u0646\u0635\u0627\u0641")
    @test "قوه" in Gen._generation_keys("\u0627\u0644\u0633\u0644\u0637\u0629")
    @test "نور" in Gen._generation_keys("\u0627\u0644\u0628\u0635\u064a\u0631\u0629")
    @test "ظلام" in Gen._generation_keys("\u0627\u0644\u062d\u064a\u0631\u0629")

    yn_gen = Physics.MirnanGenerator(Dict("\u0627\u0644\u0639\u0644\u0645" => 1, "\u0646\u0648\u0631" => 2); model_dir=mktempdir())
    yn = Gen._yesno_declarative_field_answer(yn_gen, "\u0647\u0644 \u0627\u0644\u0639\u0644\u0645 \u0646\u0648\u0631", ["\u0647\u0644", "\u0627\u0644\u0639\u0644\u0645", "\u0646\u0648\u0631"])
    @test isempty(yn)
    @test isempty(Gen._yesno_declarative_field_answer(yn_gen,
        "\u0647\u0644 \u0627\u0644\u0639\u062f\u0644 \u0644\u064a\u0633 \u062e\u064a\u0631\u061f",
        ["\u0647\u0644", "\u0627\u0644\u0639\u062f\u0644", "\u0644\u064a\u0633", "\u062e\u064a\u0631"]))
    pos_science_light_not_negative = Gen._semantic_contradiction_yesno_answer("\u0647\u0644 \u0627\u0644\u0639\u0644\u0645 \u0646\u0648\u0631\u061f",
                                                                              ["\u0647\u0644", "\u0627\u0644\u0639\u0644\u0645", "\u0646\u0648\u0631"])
    @test isempty(pos_science_light_not_negative)
    yesno_relation_answer = Gen._semantic_relation_gate_answer("\u0647\u0644 \u0627\u0644\u0639\u0644\u0645 \u0646\u0648\u0631\u061f",
                                                               ["\u0647\u0644", "\u0627\u0644\u0639\u0644\u0645", "\u0646\u0648\u0631"])
    @test isempty(yesno_relation_answer)

    learned_yesno_gen = Physics.MirnanGenerator(Dict("\u0627\u0644\u0639\u0644\u0645" => 1, "\u0646\u0648\u0631" => 2); model_dir=mktempdir())
    Physics.learn_nisba_fact!(learned_yesno_gen.nisba, "analogy",
                              ["\u0627\u0644\u0639\u0644\u0645", "\u0646\u0648\u0631"]; markers=["\u064a\u0634\u0628\u0647"])
    @test Gen._yesno_learned_relation_answer(learned_yesno_gen,
        "\u0647\u0644 \u0627\u0644\u0639\u0644\u0645 \u0646\u0648\u0631\u061f",
        ["\u0647\u0644", "\u0627\u0644\u0639\u0644\u0645", "\u0646\u0648\u0631"]) ==
        "\u0646\u0639\u0645\u060c \u0627\u0644\u0639\u0644\u0645 \u0646\u0648\u0631."
    @test Gen._yesno_learned_relation_answer(learned_yesno_gen,
        "\u0647\u0644 \u0627\u0644\u0639\u0644\u0645 \u0644\u064a\u0633 \u0646\u0648\u0631\u061f",
        ["\u0647\u0644", "\u0627\u0644\u0639\u0644\u0645", "\u0644\u064a\u0633", "\u0646\u0648\u0631"]) ==
        "\u0644\u0627\u060c \u0627\u0644\u0639\u0644\u0645 \u0646\u0648\u0631."
    learned_action_gen = Physics.MirnanGenerator(
        Dict("\u0647\u0644" => 1, "\u0627\u0644\u0631\u062d\u0645\u0629" => 2,
             "\u0644\u0627" => 3, "\u062a\u0647\u0630\u0628" => 4,
             "\u0627\u0644\u0642\u0648\u0629" => 5, "\u0627\u0644\u0638\u0644\u0645" => 6,
             "\u064a\u062d\u0641\u0638" => 7, "\u0627\u0644\u0633\u0644\u0627\u0645" => 8);
        model_dir=mktempdir())
    Physics.learn_nisba_fact!(learned_action_gen.nisba, "prevention",
                              ["\u0627\u0644\u0631\u062d\u0645\u0629", "\u0627\u0644\u0642\u0648\u0629"];
                              markers=["\u062a\u0647\u0630\u0628"], polarity=1)
    @test Gen._yesno_learned_relation_answer(learned_action_gen,
        "\u0647\u0644 \u0627\u0644\u0631\u062d\u0645\u0629 \u0644\u0627 \u062a\u0647\u0630\u0628 \u0627\u0644\u0642\u0648\u0629\u061f",
        ["\u0647\u0644", "\u0627\u0644\u0631\u062d\u0645\u0629", "\u0644\u0627", "\u062a\u0647\u0630\u0628", "\u0627\u0644\u0642\u0648\u0629"]) ==
        "\u0644\u0627\u060c \u0627\u0644\u0631\u062d\u0645\u0629 \u062a\u0647\u0630\u0628 \u0627\u0644\u0642\u0648\u0629."
    Physics.learn_nisba_fact!(learned_action_gen.nisba, "causal",
                              ["\u0627\u0644\u0638\u0644\u0645", "\u0627\u0644\u0633\u0644\u0627\u0645"];
                              markers=["\u064a\u062d\u0641\u0638"], polarity=-1)
    @test Gen._yesno_learned_relation_answer(learned_action_gen,
        "\u0647\u0644 \u0627\u0644\u0638\u0644\u0645 \u064a\u062d\u0641\u0638 \u0627\u0644\u0633\u0644\u0627\u0645\u061f",
        ["\u0647\u0644", "\u0627\u0644\u0638\u0644\u0645", "\u064a\u062d\u0641\u0638", "\u0627\u0644\u0633\u0644\u0627\u0645"]) ==
        "\u0644\u0627\u060c \u0627\u0644\u0638\u0644\u0645 \u0644\u0627 \u064a\u062d\u0641\u0638 \u0627\u0644\u0633\u0644\u0627\u0645."
    @test Gen._yesno_learned_relation_answer(learned_action_gen,
        "\u0647\u0644 \u0627\u0644\u0638\u0644\u0645 \u0644\u0627 \u064a\u062d\u0641\u0638 \u0627\u0644\u0633\u0644\u0627\u0645\u061f",
        ["\u0647\u0644", "\u0627\u0644\u0638\u0644\u0645", "\u0644\u0627", "\u064a\u062d\u0641\u0638", "\u0627\u0644\u0633\u0644\u0627\u0645"]) ==
        "\u0646\u0639\u0645\u060c \u0627\u0644\u0638\u0644\u0645 \u0644\u0627 \u064a\u062d\u0641\u0638 \u0627\u0644\u0633\u0644\u0627\u0645."
    @test Gen._yesno_negated_relation_statement(
        ["\u062f\u0641\u0639", "\u0627\u0644\u0644\u0627\u0639\u0628", "\u0627\u0644\u062d\u062c\u0631"],
        ["\u062f\u0641\u0639", "\u0627\u0644\u0644\u0627\u0639\u0628", "\u0627\u0644\u062d\u062c\u0631"]) ==
        "\u0644\u0645 \u064a\u062f\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062d\u062c\u0631"
    @test Gen._yesno_negated_relation_statement(
        ["\u064a\u062f\u0631\u0633", "\u0627\u0644\u0637\u0627\u0644\u0628"],
        ["\u064a\u062f\u0631\u0633", "\u0627\u0644\u0637\u0627\u0644\u0628"]) ==
        "\u0644\u0627 \u064a\u062f\u0631\u0633 \u0627\u0644\u0637\u0627\u0644\u0628"
    @test !occursin("\u062f\u0641\u0639 \u0644\u0627",
        Gen._yesno_negated_relation_statement(
            ["\u062f\u0641\u0639", "\u0627\u0644\u0644\u0627\u0639\u0628", "\u0627\u0644\u062d\u062c\u0631"],
            ["\u062f\u0641\u0639", "\u0627\u0644\u0644\u0627\u0639\u0628", "\u0627\u0644\u062d\u062c\u0631"]))
    tmp_relation_route = mktempdir()
    route_nisba = Physics.NisbaMemory()
    Physics.learn_nisba_fact!(route_nisba, "causal", ["\u0627\u0644\u0638\u0644\u0645", "\u0627\u0644\u0633\u0644\u0627\u0645"];
                              markers=["\u064a\u062d\u0641\u0638"], polarity=-1)
    Physics.save_nisba(route_nisba, joinpath(tmp_relation_route, "al_nisba.json"))
    route_istinbat = Physics.IstinbatAttentionMemory()
    Physics.learn_opposition_from_text!(route_istinbat, "\u0627\u0644\u0638\u0644\u0645 \u0636\u062f \u0627\u0644\u0633\u0644\u0627\u0645.")
    Physics.save_istinbat(route_istinbat, joinpath(tmp_relation_route, "al_istinbat.json"))
    route_vocab_words = ["\u0647\u0644", "\u0627\u0644\u0638\u0644\u0645", "\u0644\u0627", "\u064a\u062d\u0641\u0638", "\u0627\u0644\u0633\u0644\u0627\u0645"]
    route_vocab = Dict{String,Int}(w => i for (i, w) in enumerate(route_vocab_words))
    route_gen = Physics.MirnanGenerator(route_vocab, spzeros(length(route_vocab), length(route_vocab)); model_dir=tmp_relation_route)
    route_answer = Physics.generate!(route_gen, "\u0647\u0644 \u0627\u0644\u0638\u0644\u0645 \u064a\u062d\u0641\u0638 \u0627\u0644\u0633\u0644\u0627\u0645\u061f"; max_words=8)
    @test route_answer == "\u0644\u0627\u060c \u0627\u0644\u0638\u0644\u0645 \u0644\u0627 \u064a\u062d\u0641\u0638 \u0627\u0644\u0633\u0644\u0627\u0645."
    route_answer_neg = Physics.generate!(route_gen, "\u0647\u0644 \u0627\u0644\u0638\u0644\u0645 \u0644\u0627 \u064a\u062d\u0641\u0638 \u0627\u0644\u0633\u0644\u0627\u0645\u061f"; max_words=8)
    @test route_answer_neg == "\u0646\u0639\u0645\u060c \u0627\u0644\u0638\u0644\u0645 \u0644\u0627 \u064a\u062d\u0641\u0638 \u0627\u0644\u0633\u0644\u0627\u0645."
    tmp_opposed_relation = mktempdir()
    opposed_istinbat = Physics.IstinbatAttentionMemory()
    Physics.learn_opposition_from_text!(opposed_istinbat, "\u0627\u0644\u0638\u0644\u0645 \u0636\u062f \u0627\u0644\u0633\u0644\u0627\u0645.")
    Physics.save_istinbat(opposed_istinbat, joinpath(tmp_opposed_relation, "al_istinbat.json"))
    opposed_vocab_words = ["\u0647\u0644", "\u0627\u0644\u0638\u0644\u0645", "\u0644\u0627", "\u064a\u062d\u0641\u0638", "\u0627\u0644\u0633\u0644\u0627\u0645"]
    opposed_vocab = Dict{String,Int}(w => i for (i, w) in enumerate(opposed_vocab_words))
    opposed_gen = Physics.MirnanGenerator(opposed_vocab, spzeros(length(opposed_vocab), length(opposed_vocab)); model_dir=tmp_opposed_relation)
    @test Physics.generate!(opposed_gen, "\u0647\u0644 \u0627\u0644\u0638\u0644\u0645 \u064a\u062d\u0641\u0638 \u0627\u0644\u0633\u0644\u0627\u0645\u061f"; max_words=8) ==
          "\u0644\u0627\u060c \u0627\u0644\u0638\u0644\u0645 \u0644\u0627 \u064a\u062d\u0641\u0638 \u0627\u0644\u0633\u0644\u0627\u0645."
    @test Physics.generate!(opposed_gen, "\u0647\u0644 \u0627\u0644\u0638\u0644\u0645 \u0644\u0627 \u064a\u062d\u0641\u0638 \u0627\u0644\u0633\u0644\u0627\u0645\u061f"; max_words=8) ==
          "\u0646\u0639\u0645\u060c \u0627\u0644\u0638\u0644\u0645 \u0644\u0627 \u064a\u062d\u0641\u0638 \u0627\u0644\u0633\u0644\u0627\u0645."
    tmp_negative_context = mktempdir()
    context_istinbat = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(context_istinbat,
        "\u0644\u0627 \u064a\u062a\u062d\u0642\u0642 \u0627\u0644\u0633\u0644\u0627\u0645 \u062d\u064a\u062b \u064a\u063a\u0644\u0628 \u0627\u0644\u0638\u0644\u0645 \u0648\u0627\u0644\u0642\u0633\u0648\u0629.")
    Physics.save_istinbat(context_istinbat, joinpath(tmp_negative_context, "al_istinbat.json"))
    context_vocab_words = ["\u0647\u0644", "\u0627\u0644\u0638\u0644\u0645", "\u0644\u0627", "\u064a\u062d\u0641\u0638", "\u0627\u0644\u0633\u0644\u0627\u0645"]
    context_vocab = Dict{String,Int}(w => i for (i, w) in enumerate(context_vocab_words))
    context_gen = Physics.MirnanGenerator(context_vocab, spzeros(length(context_vocab), length(context_vocab)); model_dir=tmp_negative_context)
    @test Physics.generate!(context_gen, "\u0647\u0644 \u0627\u0644\u0638\u0644\u0645 \u064a\u062d\u0641\u0638 \u0627\u0644\u0633\u0644\u0627\u0645\u061f"; max_words=8) ==
          "\u0644\u0627\u060c \u0627\u0644\u0638\u0644\u0645 \u0644\u0627 \u064a\u062d\u0641\u0638 \u0627\u0644\u0633\u0644\u0627\u0645."
    tmp_negative_operator = mktempdir()
    op_nisba = Physics.NisbaMemory()
    Physics.learn_nisba_fact!(op_nisba, "causal", ["\u0627\u0644\u062c\u0647\u0644", "\u0627\u0644\u0641\u0647\u0645"];
                              markers=["\u064a\u0632\u064a\u062f"], polarity=1)
    Physics.learn_nisba_fact!(op_nisba, "causal", ["\u0627\u0644\u0638\u0644\u0645", "\u0627\u0644\u062b\u0642\u0629"];
                              markers=["\u064a\u0635\u0646\u0639"], polarity=1)
    Physics.learn_nisba_fact!(op_nisba, "causal", ["\u0627\u0644\u0639\u0644\u0645", "\u0627\u0644\u0641\u0647\u0645"];
                              markers=["\u064a\u0632\u064a\u062f"], polarity=1)
    Physics.learn_nisba_fact!(op_nisba, "causal", ["\u0627\u0644\u0639\u062f\u0644", "\u0627\u0644\u0633\u0644\u0627\u0645"];
                              markers=["\u064a\u062d\u0641\u0638"], polarity=1)
    Physics.learn_nisba_fact!(op_nisba, "causal", ["\u0627\u0644\u0631\u062d\u0645\u0629", "\u0627\u0644\u062b\u0642\u0629"];
                              markers=["\u062a\u0628\u0646\u064a"], polarity=1)
    Physics.save_nisba(op_nisba, joinpath(tmp_negative_operator, "al_nisba.json"))
    op_istinbat = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(op_istinbat, "\u0627\u0644\u062c\u0647\u0644 \u064a\u062d\u062c\u0628 \u0627\u0644\u0641\u0647\u0645.")
    Physics.learn_istinbat_from_text!(op_istinbat, "\u0627\u0644\u0638\u0644\u0645 \u064a\u0641\u0633\u062f \u0627\u0644\u062b\u0642\u0629.")
    Physics.save_istinbat(op_istinbat, joinpath(tmp_negative_operator, "al_istinbat.json"))
    op_vocab_words = ["\u0647\u0644", "\u0627\u0644\u062c\u0647\u0644", "\u0644\u0627", "\u064a\u0632\u064a\u062f", "\u0627\u0644\u0641\u0647\u0645",
                      "\u0627\u0644\u0638\u0644\u0645", "\u064a\u0635\u0646\u0639", "\u0627\u0644\u062b\u0642\u0629", "\u0627\u0644\u0639\u0644\u0645",
                      "\u0627\u0644\u0639\u062f\u0644", "\u064a\u062d\u0641\u0638", "\u0627\u0644\u0633\u0644\u0627\u0645",
                      "\u0627\u0644\u0631\u062d\u0645\u0629", "\u062a\u0628\u0646\u064a"]
    op_vocab = Dict{String,Int}(w => i for (i, w) in enumerate(op_vocab_words))
    op_gen = Physics.MirnanGenerator(op_vocab, spzeros(length(op_vocab), length(op_vocab)); model_dir=tmp_negative_operator)
    @test Physics.generate!(op_gen, "\u0647\u0644 \u0627\u0644\u062c\u0647\u0644 \u064a\u0632\u064a\u062f \u0627\u0644\u0641\u0647\u0645\u061f"; max_words=8) ==
          "\u0644\u0627\u060c \u0627\u0644\u062c\u0647\u0644 \u0644\u0627 \u064a\u0632\u064a\u062f \u0627\u0644\u0641\u0647\u0645."
    @test Physics.generate!(op_gen, "\u0647\u0644 \u0627\u0644\u062c\u0647\u0644 \u0644\u0627 \u064a\u0632\u064a\u062f \u0627\u0644\u0641\u0647\u0645\u061f"; max_words=8) ==
          "\u0646\u0639\u0645\u060c \u0627\u0644\u062c\u0647\u0644 \u0644\u0627 \u064a\u0632\u064a\u062f \u0627\u0644\u0641\u0647\u0645."
    @test Physics.generate!(op_gen, "\u0647\u0644 \u0627\u0644\u0638\u0644\u0645 \u064a\u0635\u0646\u0639 \u0627\u0644\u062b\u0642\u0629\u061f"; max_words=8) ==
          "\u0644\u0627\u060c \u0627\u0644\u0638\u0644\u0645 \u0644\u0627 \u064a\u0635\u0646\u0639 \u0627\u0644\u062b\u0642\u0629."
    @test Physics.generate!(op_gen, "\u0647\u0644 \u0627\u0644\u0638\u0644\u0645 \u0644\u0627 \u064a\u0635\u0646\u0639 \u0627\u0644\u062b\u0642\u0629\u061f"; max_words=8) ==
          "\u0646\u0639\u0645\u060c \u0627\u0644\u0638\u0644\u0645 \u0644\u0627 \u064a\u0635\u0646\u0639 \u0627\u0644\u062b\u0642\u0629."
    @test Physics.generate!(op_gen, "\u0647\u0644 \u0627\u0644\u0639\u0644\u0645 \u064a\u0632\u064a\u062f \u0627\u0644\u0641\u0647\u0645\u061f"; max_words=8) ==
          "\u0646\u0639\u0645\u060c \u0627\u0644\u0639\u0644\u0645 \u064a\u0632\u064a\u062f \u0627\u0644\u0641\u0647\u0645."
    @test Physics.generate!(op_gen, "\u0647\u0644 \u0627\u0644\u0639\u0644\u0645 \u0644\u0627 \u064a\u0632\u064a\u062f \u0627\u0644\u0641\u0647\u0645\u061f"; max_words=8) ==
          "\u0644\u0627\u060c \u0627\u0644\u0639\u0644\u0645 \u064a\u0632\u064a\u062f \u0627\u0644\u0641\u0647\u0645."
    @test Physics.generate!(op_gen, "\u0647\u0644 \u0627\u0644\u0639\u062f\u0644 \u064a\u062d\u0641\u0638 \u0627\u0644\u0633\u0644\u0627\u0645\u061f"; max_words=8) ==
          "\u0646\u0639\u0645\u060c \u0627\u0644\u0639\u062f\u0644 \u064a\u062d\u0641\u0638 \u0627\u0644\u0633\u0644\u0627\u0645."
    @test Physics.generate!(op_gen, "\u0647\u0644 \u0627\u0644\u0639\u062f\u0644 \u0644\u0627 \u064a\u062d\u0641\u0638 \u0627\u0644\u0633\u0644\u0627\u0645\u061f"; max_words=8) ==
          "\u0644\u0627\u060c \u0627\u0644\u0639\u062f\u0644 \u064a\u062d\u0641\u0638 \u0627\u0644\u0633\u0644\u0627\u0645."
    @test Physics.generate!(op_gen, "\u0647\u0644 \u0627\u0644\u0631\u062d\u0645\u0629 \u062a\u0628\u0646\u064a \u0627\u0644\u062b\u0642\u0629\u061f"; max_words=8) ==
          "\u0646\u0639\u0645\u060c \u0627\u0644\u0631\u062d\u0645\u0629 \u062a\u0628\u0646\u064a \u0627\u0644\u062b\u0642\u0629."
    @test Physics.generate!(op_gen, "\u0647\u0644 \u0627\u0644\u0631\u062d\u0645\u0629 \u0644\u0627 \u062a\u0628\u0646\u064a \u0627\u0644\u062b\u0642\u0629\u061f"; max_words=8) ==
          "\u0644\u0627\u060c \u0627\u0644\u0631\u062d\u0645\u0629 \u062a\u0628\u0646\u064a \u0627\u0644\u062b\u0642\u0629."
    @test isempty(Gen._yesno_opposed_relation_answer(op_gen,
        ["\u0647\u0644", "\u0627\u0644\u0639\u0644\u0645", "\u064a\u0632\u064a\u062f", "\u0627\u0644\u0641\u0647\u0645"]))
    @test isempty(Gen._yesno_opposed_relation_answer(op_gen,
        ["\u0647\u0644", "\u0627\u0644\u0639\u062f\u0644", "\u064a\u062d\u0641\u0638", "\u0627\u0644\u0633\u0644\u0627\u0645"]))
    @test isempty(Gen._yesno_opposed_relation_answer(op_gen,
        ["\u0647\u0644", "\u0627\u0644\u0631\u062d\u0645\u0629", "\u062a\u0628\u0646\u064a", "\u0627\u0644\u062b\u0642\u0629"]))
    @test isempty(Gen._yesno_learned_relation_answer(Physics.MirnanGenerator(Dict("\u0627\u0644\u0639\u062f\u0644" => 1, "\u062e\u064a\u0631" => 2); model_dir=mktempdir()),
        "\u0647\u0644 \u0627\u0644\u0639\u062f\u0644 \u062e\u064a\u0631\u061f",
        ["\u0647\u0644", "\u0627\u0644\u0639\u062f\u0644", "\u062e\u064a\u0631"]))

    neg_light = Gen._semantic_contradiction_yesno_answer("\u0647\u0644 \u0627\u0644\u062c\u0647\u0644 \u0646\u0648\u0631\u061f",
                                                         ["\u0647\u0644", "\u0627\u0644\u062c\u0647\u0644", "\u0646\u0648\u0631"])
    @test isempty(neg_light)

    neg_peace = Gen._semantic_contradiction_yesno_answer("\u0647\u0644 \u0627\u0644\u0638\u0644\u0645 \u064a\u062d\u0641\u0638 \u0627\u0644\u0633\u0644\u0627\u0645\u061f",
                                                         ["\u0647\u0644", "\u0627\u0644\u0638\u0644\u0645", "\u064a\u062d\u0641\u0638", "\u0627\u0644\u0633\u0644\u0627\u0645"])
    @test isempty(neg_peace)

    neg_force = Gen._semantic_contradiction_yesno_answer("\u0647\u0644 \u0627\u0644\u0642\u0648\u0629 \u0628\u0644\u0627 \u0631\u062d\u0645\u0629 \u062e\u064a\u0631\u061f",
                                                         ["\u0647\u0644", "\u0627\u0644\u0642\u0648\u0629", "\u0628\u0644\u0627", "\u0631\u062d\u0645\u0629", "\u062e\u064a\u0631"])
    @test isempty(neg_force)

    neg_conflict = Gen._semantic_contradiction_yesno_answer("\u0647\u0644 \u0627\u0644\u0639\u0644\u0645 \u064a\u0633\u0628\u0628 \u0627\u0644\u0646\u0632\u0627\u0639\u061f",
                                                            ["\u0647\u0644", "\u0627\u0644\u0639\u0644\u0645", "\u064a\u0633\u0628\u0628", "\u0627\u0644\u0646\u0632\u0627\u0639"])
    @test isempty(neg_conflict)

    neg_useful = Gen._semantic_contradiction_yesno_answer("\u0647\u0644 \u0627\u0644\u062c\u0647\u0644 \u0645\u0641\u064a\u062f\u061f",
                                                          ["\u0647\u0644", "\u0627\u0644\u062c\u0647\u0644", "\u0645\u0641\u064a\u062f"])
    @test isempty(neg_useful)

    neg_good = Gen._semantic_contradiction_yesno_answer("\u0647\u0644 \u0627\u0644\u0638\u0644\u0645 \u062e\u064a\u0631\u061f",
                                                        ["\u0647\u0644", "\u0627\u0644\u0638\u0644\u0645", "\u062e\u064a\u0631"])
    @test isempty(neg_good)
    neg_justice_not_good = Gen._semantic_contradiction_yesno_answer("\u0647\u0644 \u0627\u0644\u0639\u062f\u0644 \u0644\u064a\u0633 \u062e\u064a\u0631\u061f",
                                                                    ["\u0647\u0644", "\u0627\u0644\u0639\u062f\u0644", "\u0644\u064a\u0633", "\u062e\u064a\u0631"])
    @test isempty(neg_justice_not_good)
    neg_justice_evil = Gen._semantic_contradiction_yesno_answer("\u0647\u0644 \u0627\u0644\u0639\u062f\u0644 \u0634\u0631\u061f",
                                                                ["\u0647\u0644", "\u0627\u0644\u0639\u062f\u0644", "\u0634\u0631"])
    @test isempty(neg_justice_evil)
    neg_mercy_force = Gen._semantic_contradiction_yesno_answer("\u0647\u0644 \u0627\u0644\u0631\u062d\u0645\u0629 \u0644\u0627 \u062a\u0647\u0630\u0628 \u0627\u0644\u0642\u0648\u0629\u061f",
                                                               ["\u0647\u0644", "\u0627\u0644\u0631\u062d\u0645\u0629", "\u0644\u0627", "\u062a\u0647\u0630\u0628", "\u0627\u0644\u0642\u0648\u0629"])
    @test isempty(neg_mercy_force)

    neg_reason = Gen._semantic_contradiction_yesno_answer("\u0647\u0644 \u0627\u0644\u0642\u0648\u0629 \u0628\u0644\u0627 \u0639\u0642\u0644 \u0635\u0648\u0627\u0628\u061f",
                                                          ["\u0647\u0644", "\u0627\u0644\u0642\u0648\u0629", "\u0628\u0644\u0627", "\u0639\u0642\u0644", "\u0635\u0648\u0627\u0628"])
    @test isempty(neg_reason)

    neg_enough = Gen._semantic_contradiction_yesno_answer("\u0647\u0644 \u0627\u0644\u0639\u0644\u0645 \u0648\u062d\u062f\u0647 \u064a\u0643\u0641\u064a \u0644\u0644\u0639\u0645\u0644\u061f",
                                                          ["\u0647\u0644", "\u0627\u0644\u0639\u0644\u0645", "\u0648\u062d\u062f\u0647", "\u064a\u0643\u0641\u064a", "\u0644\u0644\u0639\u0645\u0644"])
    @test isempty(neg_enough)

    neg_make_peace = Gen._semantic_contradiction_yesno_answer("\u0647\u0644 \u0627\u0644\u0638\u0644\u0645 \u064a\u0635\u0646\u0639 \u0627\u0644\u0633\u0644\u0627\u0645\u061f",
                                                              ["\u0647\u0644", "\u0627\u0644\u0638\u0644\u0645", "\u064a\u0635\u0646\u0639", "\u0627\u0644\u0633\u0644\u0627\u0645"])
    @test isempty(neg_make_peace)
    neg_gafla = Gen._semantic_contradiction_yesno_answer("\u0647\u0644 \u0627\u0644\u063a\u0641\u0644\u0629 \u0637\u0631\u064a\u0642 \u0644\u0644\u062e\u064a\u0631\u061f",
                                                         ["\u0647\u0644", "\u0627\u0644\u063a\u0641\u0644\u0629", "\u0637\u0631\u064a\u0642", "\u0644\u0644\u062e\u064a\u0631"])
    @test isempty(neg_gafla)
    neg_jawr = Gen._semantic_contradiction_yesno_answer("\u0647\u0644 \u0627\u0644\u062c\u0648\u0631 \u064a\u0635\u0646\u0639 \u0627\u0644\u0633\u0644\u0627\u0645\u061f",
                                                        ["\u0647\u0644", "\u0627\u0644\u062c\u0648\u0631", "\u064a\u0635\u0646\u0639", "\u0627\u0644\u0633\u0644\u0627\u0645"])
    @test isempty(neg_jawr)

    diff_gen = Physics.MirnanGenerator(Dict("\u0627\u0644\u0639\u062f\u0644" => 1, "\u0627\u0644\u0631\u062d\u0645\u0629" => 2); model_dir=mktempdir())
    diff_gen.ta3rif.records["\u0627\u0644\u0639\u062f\u0644"] =
        Physics.Ta3rifRecord("\u0627\u0644\u0639\u062f\u0644", 1,
                             Dict("\u0627\u0639\u0637\u0627\u0621 \u0643\u0644 \u0630\u064a \u062d\u0642 \u062d\u0642\u0647" => 1),
                             Dict{String,Int}(), Dict{String,Int}(), String[])
    diff_gen.ta3rif.records["\u0627\u0644\u0631\u062d\u0645\u0629"] =
        Physics.Ta3rifRecord("\u0627\u0644\u0631\u062d\u0645\u0629", 1,
                             Dict("\u0631\u0642\u0629 \u062a\u062f\u0641\u0639 \u0627\u0644\u0649 \u0627\u0644\u0627\u062d\u0633\u0627\u0646 \u0648\u062a\u062e\u0641\u064a\u0641 \u0627\u0644\u0627\u0630\u0649" => 1),
                             Dict{String,Int}(), Dict{String,Int}(), String[])
    persistent_defs = Physics.load_ta3rif(joinpath(@__DIR__, "..", "knowledge", "definitions.json"))
    Physics.merge_ta3rif!(diff_gen.ta3rif, persistent_defs)
    diff_answer = Gen._difference_answer(diff_gen, "\u0628\u064a\u0651\u0646 \u0627\u0644\u0641\u0631\u0642 \u0628\u064a\u0646 \u0627\u0644\u0639\u062f\u0644 \u0648\u0627\u0644\u0631\u062d\u0645\u0629.")
    @test occursin("\u0627\u0644\u0639\u062f\u0644", diff_answer)
    @test occursin("\u0627\u0644\u0631\u062d\u0645\u0629", diff_answer)
    @test occursin("\u0623\u0645\u0627", diff_answer)
    diff_peace = Gen._difference_answer(diff_gen, "\u0628\u064a\u0651\u0646 \u0627\u0644\u0641\u0631\u0642 \u0628\u064a\u0646 \u0627\u0644\u0633\u0644\u0627\u0645 \u0648\u0627\u0644\u0646\u0632\u0627\u0639.")
    @test occursin("\u0627\u0644\u0633\u0644\u0627\u0645", diff_peace)
    @test occursin("\u0627\u0644\u0646\u0632\u0627\u0639", diff_peace)
    @test !("بيّن الفرق بين السلام والنزاع." == diff_peace)
    diff_force = Gen._difference_answer(diff_gen, "\u0628\u064a\u0651\u0646 \u0627\u0644\u0641\u0631\u0642 \u0628\u064a\u0646 \u0627\u0644\u0642\u0648\u0629 \u0648\u0627\u0644\u0638\u0644\u0645.")
    @test occursin("\u0627\u0644\u0642\u0648\u0629", diff_force)
    @test occursin("\u0627\u0644\u0638\u0644\u0645", diff_force)
    @test !("بيّن الفرق بين القوة والظلم." == diff_force)
    diff_light = Gen._difference_answer(diff_gen, "\u0628\u064a\u0651\u0646 \u0627\u0644\u0641\u0631\u0642 \u0628\u064a\u0646 \u0627\u0644\u0646\u0648\u0631 \u0648\u0627\u0644\u0638\u0644\u0627\u0645.")
    @test occursin("\u0627\u0644\u0646\u0648\u0631", diff_light)
    @test occursin("\u0627\u0644\u0638\u0644\u0627\u0645", diff_light)
    @test !("بيّن الفرق بين النور والظلام." == diff_light)
    diff_insight = Gen._difference_answer(diff_gen, "\u0628\u064a\u0651\u0646 \u0627\u0644\u0641\u0631\u0642 \u0628\u064a\u0646 \u0627\u0644\u0628\u0635\u064a\u0631\u0629 \u0648\u0627\u0644\u062d\u064a\u0631\u0629.")
    @test occursin("\u0627\u0644\u0628\u0635\u064a\u0631\u0629", diff_insight)
    @test occursin("\u0627\u0644\u062d\u064a\u0631\u0629", diff_insight)
    @test !("بيّن الفرق بين البصيرة والحيرة." == diff_insight)
    diff_power = Gen._difference_answer(diff_gen, "\u0628\u064a\u0651\u0646 \u0627\u0644\u0641\u0631\u0642 \u0628\u064a\u0646 \u0627\u0644\u0645\u0639\u0631\u0641\u0629 \u0648\u0627\u0644\u0633\u0644\u0637\u0629.")
    @test occursin("\u0627\u0644\u0645\u0639\u0631\u0641\u0629", diff_power)
    @test occursin("\u0627\u0644\u0633\u0644\u0637\u0629", diff_power)
    @test !("بيّن الفرق بين المعرفة والسلطة." == diff_power)

    nisba_gen = Physics.MirnanGenerator(Dict("\u0627\u0644\u0639\u0644\u0645" => 1, "\u0627\u0644\u0641\u0647\u0645" => 2); model_dir=mktempdir())
    nisba_mem = Physics.NisbaMemory()
    Physics.train_nisba_from_texts!(nisba_mem, ["\u0627\u0644\u0639\u0644\u0645 \u064a\u0641\u062a\u062d \u0627\u0644\u0641\u0647\u0645\u060c \u0648\u0627\u0644\u0641\u0647\u0645 \u064a\u0647\u062f\u064a \u0627\u0644\u0649 \u0627\u0644\u0631\u062d\u0645\u0647."])
    nisba_gen.nisba = nisba_mem
    nisba_plan = Gen.detect_response_intent("\u0645\u0627 \u0627\u0644\u0639\u0644\u0627\u0642\u0629 \u0628\u064a\u0646 \u0627\u0644\u0639\u0644\u0645 \u0648\u0627\u0644\u0641\u0647\u0645\u061f")
    nisba_answer = Gen._nisba_relation_answer(nisba_gen,
        "\u0645\u0627 \u0627\u0644\u0639\u0644\u0627\u0642\u0629 \u0628\u064a\u0646 \u0627\u0644\u0639\u0644\u0645 \u0648\u0627\u0644\u0641\u0647\u0645\u061f",
        ["\u0645\u0627", "\u0627\u0644\u0639\u0644\u0627\u0642\u0629", "\u0628\u064a\u0646", "\u0627\u0644\u0639\u0644\u0645", "\u0648\u0627\u0644\u0641\u0647\u0645"],
        nisba_plan)
    @test isempty(nisba_answer)
    @test isempty(Gen._nisba_relation_answer(nisba_gen,
        "\u0644\u0645\u0627\u0630\u0627 \u064a\u062e\u0634\u0649 \u0627\u0644\u062c\u0627\u0647\u0644 \u0627\u0644\u0639\u0644\u0645\u061f",
        ["\u0644\u0645\u0627\u0630\u0627", "\u064a\u062e\u0634\u0649", "\u0627\u0644\u062c\u0627\u0647\u0644", "\u0627\u0644\u0639\u0644\u0645"],
        Gen.detect_response_intent("\u0644\u0645\u0627\u0630\u0627 \u064a\u062e\u0634\u0649 \u0627\u0644\u062c\u0627\u0647\u0644 \u0627\u0644\u0639\u0644\u0645\u061f")))
    @test isempty(Gen._nisba_relation_answer(nisba_gen,
        "\u0647\u0644 \u0627\u0644\u062c\u0647\u0644 \u0646\u0648\u0631\u061f",
        ["\u0647\u0644", "\u0627\u0644\u062c\u0647\u0644", "\u0646\u0648\u0631"],
        Gen.detect_response_intent("\u0647\u0644 \u0627\u0644\u062c\u0647\u0644 \u0646\u0648\u0631\u061f")))
    @test isempty(Gen._nisba_relation_answer(nisba_gen,
        "\u0628\u064a\u0651\u0646 \u0627\u0644\u0641\u0631\u0642 \u0628\u064a\u0646 \u0627\u0644\u0639\u062f\u0644 \u0648\u0627\u0644\u0631\u062d\u0645\u0629.",
        ["\u0628\u064a\u0651\u0646", "\u0627\u0644\u0641\u0631\u0642", "\u0628\u064a\u0646", "\u0627\u0644\u0639\u062f\u0644", "\u0648\u0627\u0644\u0631\u062d\u0645\u0629"],
        Gen.detect_response_intent("\u0628\u064a\u0651\u0646 \u0627\u0644\u0641\u0631\u0642 \u0628\u064a\u0646 \u0627\u0644\u0639\u062f\u0644 \u0648\u0627\u0644\u0631\u062d\u0645\u0629.")))

    simile_answer = Gen._semantic_relation_gate_answer("\u0645\u0627 \u0627\u0644\u0630\u064a \u064a\u062c\u0639\u0644 \u0627\u0644\u0639\u0644\u0645 \u0634\u0628\u064a\u0647\u0627 \u0628\u0627\u0644\u0646\u0648\u0631\u061f",
                                                       ["\u0645\u0627", "\u0627\u0644\u0630\u064a", "\u064a\u062c\u0639\u0644", "\u0627\u0644\u0639\u0644\u0645", "\u0634\u0628\u064a\u0647\u0627", "\u0628\u0627\u0644\u0646\u0648\u0631"])
    @test isempty(simile_answer)
    lamp_answer = Gen._semantic_relation_gate_answer("\u0645\u0627 \u0627\u0644\u0630\u064a \u064a\u062c\u0639\u0644 \u0627\u0644\u0639\u0644\u0645 \u0643\u0627\u0644\u0645\u0635\u0628\u0627\u062d\u061f",
                                                     ["\u0645\u0627", "\u0627\u0644\u0630\u064a", "\u064a\u062c\u0639\u0644", "\u0627\u0644\u0639\u0644\u0645", "\u0643\u0627\u0644\u0645\u0635\u0628\u0627\u062d"])
    @test isempty(lamp_answer)

    transform_answer = Gen._semantic_relation_gate_answer("\u0645\u062a\u0649 \u062a\u062a\u062d\u0648\u0644 \u0627\u0644\u0642\u0648\u0629 \u0625\u0644\u0649 \u0638\u0644\u0645\u061f",
                                                          ["\u0645\u062a\u0649", "\u062a\u062a\u062d\u0648\u0644", "\u0627\u0644\u0642\u0648\u0629", "\u0625\u0644\u0649", "\u0638\u0644\u0645"])
    @test isempty(transform_answer)
    mercy_force = Gen._semantic_relation_gate_answer("\u0645\u0627 \u0627\u0644\u0639\u0644\u0627\u0642\u0629 \u0628\u064a\u0646 \u0627\u0644\u0631\u062d\u0645\u0629 \u0648\u0627\u0644\u0642\u0648\u0629\u061f",
                                                     ["\u0645\u0627", "\u0627\u0644\u0639\u0644\u0627\u0642\u0629", "\u0628\u064a\u0646", "\u0627\u0644\u0631\u062d\u0645\u0629", "\u0648\u0627\u0644\u0642\u0648\u0629"])
    @test isempty(mercy_force)
    how_not_mercy_force = Gen._semantic_relation_gate_answer("\u0643\u064a\u0641 \u0644\u0627 \u062a\u0647\u0630\u0628 \u0627\u0644\u0631\u062d\u0645\u0629 \u0627\u0644\u0642\u0648\u0629\u061f",
                                                            ["\u0643\u064a\u0641", "\u0644\u0627", "\u062a\u0647\u0630\u0628", "\u0627\u0644\u0631\u062d\u0645\u0629", "\u0627\u0644\u0642\u0648\u0629"])
    @test isempty(how_not_mercy_force)

    knowledge_force = Gen._semantic_relation_gate_answer("\u0645\u062a\u0649 \u062a\u062a\u062d\u0648\u0644 \u0627\u0644\u0645\u0639\u0631\u0641\u0629 \u0625\u0644\u0649 \u0642\u0648\u0629\u061f",
                                                          ["\u0645\u062a\u0649", "\u062a\u062a\u062d\u0648\u0644", "\u0627\u0644\u0645\u0639\u0631\u0641\u0629", "\u0625\u0644\u0649", "\u0642\u0648\u0629"])
    @test isempty(knowledge_force)

    light_dark = Gen._semantic_relation_gate_answer("\u0643\u064a\u0641 \u064a\u0637\u0631\u062f \u0627\u0644\u0636\u0648\u0621 \u0627\u0644\u0638\u0644\u0627\u0645\u061f",
                                                     ["\u0643\u064a\u0641", "\u064a\u0637\u0631\u062f", "\u0627\u0644\u0636\u0648\u0621", "\u0627\u0644\u0638\u0644\u0627\u0645"])
    @test isempty(light_dark)

    experience_ability = Gen._semantic_relation_gate_answer("\u0645\u062a\u0649 \u062a\u0635\u0628\u062d \u0627\u0644\u062e\u0628\u0631\u0629 \u0642\u062f\u0631\u0629\u061f",
                                                            ["\u0645\u062a\u0649", "\u062a\u0635\u0628\u062d", "\u0627\u0644\u062e\u0628\u0631\u0629", "\u0642\u062f\u0631\u0629"])
    @test isempty(experience_ability)
    tajriba_ability = Gen._semantic_relation_gate_answer("\u0645\u062a\u0649 \u062a\u0635\u0628\u062d \u0627\u0644\u062a\u062c\u0631\u0628\u0629 \u0642\u062f\u0631\u0629\u061f",
                                                         ["\u0645\u062a\u0649", "\u062a\u0635\u0628\u062d", "\u0627\u0644\u062a\u062c\u0631\u0628\u0629", "\u0642\u062f\u0631\u0629"])
    @test isempty(tajriba_ability)
    knowledge_effect = Gen._semantic_relation_gate_answer("\u0645\u062a\u0649 \u062a\u0635\u064a\u0631 \u0627\u0644\u0645\u0639\u0631\u0641\u0629 \u0623\u062b\u0631\u0627 \u0641\u064a \u0627\u0644\u0648\u0627\u0642\u0639\u061f",
                                                          ["\u0645\u062a\u0649", "\u062a\u0635\u064a\u0631", "\u0627\u0644\u0645\u0639\u0631\u0641\u0629", "\u0623\u062b\u0631\u0627", "\u0641\u064a", "\u0627\u0644\u0648\u0627\u0642\u0639"])
    @test isempty(knowledge_effect)

    society_justice = Gen._semantic_relation_gate_answer("\u0644\u0645\u0627\u0630\u0627 \u064a\u062d\u062a\u0627\u062c \u0627\u0644\u0645\u062c\u062a\u0645\u0639 \u0625\u0644\u0649 \u0627\u0644\u0639\u062f\u0627\u0644\u0629\u061f",
                                                          ["\u0644\u0645\u0627\u0630\u0627", "\u064a\u062d\u062a\u0627\u062c", "\u0627\u0644\u0645\u062c\u062a\u0645\u0639", "\u0625\u0644\u0649", "\u0627\u0644\u0639\u062f\u0627\u0644\u0629"])
    @test isempty(society_justice)

    force_reason = Gen._semantic_relation_gate_answer("\u0644\u0645\u0627\u0630\u0627 \u062a\u062d\u062a\u0627\u062c \u0627\u0644\u0642\u0648\u0629 \u0625\u0644\u0649 \u0627\u0644\u0639\u0642\u0644\u061f",
                                                       ["\u0644\u0645\u0627\u0630\u0627", "\u062a\u062d\u062a\u0627\u062c", "\u0627\u0644\u0642\u0648\u0629", "\u0625\u0644\u0649", "\u0627\u0644\u0639\u0642\u0644"])
    @test isempty(force_reason)

    how_force_reason = Gen._semantic_relation_gate_answer("\u0643\u064a\u0641 \u064a\u0636\u0628\u0637 \u0627\u0644\u0639\u0642\u0644 \u0627\u0644\u0642\u0648\u0629\u061f",
                                                           ["\u0643\u064a\u0641", "\u064a\u0636\u0628\u0637", "\u0627\u0644\u0639\u0642\u0644", "\u0627\u0644\u0642\u0648\u0629"])
    @test isempty(how_force_reason)
    insight_authority = Gen._semantic_relation_gate_answer("\u0643\u064a\u0641 \u062a\u0636\u0628\u0637 \u0627\u0644\u0628\u0635\u064a\u0631\u0629 \u0627\u0644\u0633\u0644\u0637\u0629\u061f",
                                                           ["\u0643\u064a\u0641", "\u062a\u0636\u0628\u0637", "\u0627\u0644\u0628\u0635\u064a\u0631\u0629", "\u0627\u0644\u0633\u0644\u0637\u0629"])
    @test isempty(insight_authority)

    equity_conflict = Gen._semantic_relation_gate_answer("\u0643\u064a\u0641 \u062a\u0645\u0646\u0639 \u0627\u0644\u0639\u062f\u0627\u0644\u0629 \u0627\u0644\u0646\u0632\u0627\u0639\u061f",
                                                         ["\u0643\u064a\u0641", "\u062a\u0645\u0646\u0639", "\u0627\u0644\u0639\u062f\u0627\u0644\u0629", "\u0627\u0644\u0646\u0632\u0627\u0639"])
    @test isempty(equity_conflict)
    learning_work = Gen._semantic_relation_gate_answer("\u0645\u0627 \u0627\u0644\u0639\u0644\u0627\u0642\u0629 \u0628\u064a\u0646 \u0627\u0644\u062a\u0639\u0644\u0645 \u0648\u0627\u0644\u0639\u0645\u0644\u061f",
                                                       ["\u0645\u0627", "\u0627\u0644\u0639\u0644\u0627\u0642\u0629", "\u0628\u064a\u0646", "\u0627\u0644\u062a\u0639\u0644\u0645", "\u0648\u0627\u0644\u0639\u0645\u0644"])
    @test isempty(learning_work)
    work_understanding = Gen._semantic_relation_gate_answer("\u0644\u0645\u0627\u0630\u0627 \u064a\u062d\u062a\u0627\u062c \u0627\u0644\u0639\u0645\u0644 \u0625\u0644\u0649 \u0641\u0647\u0645\u061f",
                                                           ["\u0644\u0645\u0627\u0630\u0627", "\u064a\u062d\u062a\u0627\u062c", "\u0627\u0644\u0639\u0645\u0644", "\u0625\u0644\u0649", "\u0641\u0647\u0645"])
    @test isempty(work_understanding)

    peace_answer = Gen._semantic_relation_gate_answer("\u0644\u0645\u0627\u0630\u0627 \u064a\u062d\u062a\u0627\u062c \u0627\u0644\u0633\u0644\u0627\u0645 \u0625\u0644\u0649 \u0639\u0644\u0645 \u0644\u0627 \u0625\u0644\u0649 \u0642\u0648\u0629 \u0641\u0642\u0637\u061f",
                                                      ["\u0644\u0645\u0627\u0630\u0627", "\u064a\u062d\u062a\u0627\u062c", "\u0627\u0644\u0633\u0644\u0627\u0645", "\u0625\u0644\u0649", "\u0639\u0644\u0645", "\u0644\u0627", "\u0625\u0644\u0649", "\u0642\u0648\u0629", "\u0641\u0642\u0637"])
    @test isempty(peace_answer)
    enough_peace = Gen._semantic_relation_gate_answer("\u0644\u0645\u0627\u0630\u0627 \u0644\u0627 \u062a\u0643\u0641\u064a \u0627\u0644\u0642\u0648\u0629 \u0644\u062d\u0641\u0638 \u0627\u0644\u0633\u0644\u0627\u0645\u061f",
                                                      ["\u0644\u0645\u0627\u0630\u0627", "\u0644\u0627", "\u062a\u0643\u0641\u064a", "\u0627\u0644\u0642\u0648\u0629", "\u0644\u062d\u0641\u0638", "\u0627\u0644\u0633\u0644\u0627\u0645"])
    @test isempty(enough_peace)

    reason_answer = Gen._semantic_relation_gate_answer("\u0643\u064a\u0641 \u064a\u0645\u0646\u0639 \u0627\u0644\u0639\u0642\u0644 \u0627\u0644\u062a\u0647\u0648\u0631\u061f",
                                                       ["\u0643\u064a\u0641", "\u064a\u0645\u0646\u0639", "\u0627\u0644\u0639\u0642\u0644", "\u0627\u0644\u062a\u0647\u0648\u0631"])
    @test isempty(reason_answer)
end

@testset "learned opposition yes/no route" begin
    tmp = mktempdir()
    nisba = Physics.NisbaMemory()
    Physics.learn_nisba_from_text!(nisba, "يكون العدل خيراً حين يحمي حقوق الناس.")
    Physics.save_nisba(nisba, joinpath(tmp, "al_nisba.json"))

    istinbat = Physics.IstinbatAttentionMemory()
    Physics.learn_opposition_from_text!(istinbat, "العقل يميز بين الخير والشر.")
    Physics.save_istinbat(istinbat, joinpath(tmp, "al_istinbat.json"))

    vocab_words = ["هل", "العدل", "خير", "شر", "ليس", "العقل", "يميز", "بين", "الخير", "والشر"]
    vocab = Dict{String,Int}(w => i for (i, w) in enumerate(vocab_words))
    gen = Physics.MirnanGenerator(vocab, spzeros(length(vocab), length(vocab)); model_dir=tmp)

    answer = Physics.generate!(gen, "هل العدل شر؟"; max_words=8)
    @test startswith(answer, "لا")
    @test occursin("العدل", answer)
    @test occursin("شر", answer)
    answer_neg = Physics.generate!(gen, "هل العدل ليس شر؟"; max_words=8)
    @test startswith(answer_neg, "نعم")
    @test occursin("العدل", answer_neg)
    @test occursin("ليس", answer_neg)
    @test occursin("شر", answer_neg)

    tmp2 = mktempdir()
    istinbat2 = Physics.IstinbatAttentionMemory()
    Physics.learn_direct_negation_from_text!(istinbat2, "الجهل ليس نوراً؛ بل يحجب الرؤية.")
    Physics.save_istinbat(istinbat2, joinpath(tmp2, "al_istinbat.json"))

    vocab2_words = ["هل", "الجهل", "نور", "ليس", "نوراً"]
    vocab2 = Dict{String,Int}(w => i for (i, w) in enumerate(vocab2_words))
    gen2 = Physics.MirnanGenerator(vocab2, spzeros(length(vocab2), length(vocab2)); model_dir=tmp2)
    answer2 = Physics.generate!(gen2, "هل الجهل نور؟"; max_words=8)
    @test startswith(answer2, "لا")
    @test occursin("الجهل", answer2)
    @test occursin("نور", answer2)
    answer2_neg = Physics.generate!(gen2, "هل الجهل ليس نور؟"; max_words=8)
    @test startswith(answer2_neg, "نعم")
    @test occursin("الجهل", answer2_neg)
    @test occursin("ليس", answer2_neg)
    @test occursin("نور", answer2_neg)

    tmp3 = mktempdir()
    nisba3 = Physics.NisbaMemory()
    Physics.learn_nisba_from_text!(nisba3, "العلم يزيد الفهم.")
    Physics.save_nisba(nisba3, joinpath(tmp3, "al_nisba.json"))
    vocab3_words = ["لماذا", "يزيد", "العلم", "الفهم", "لا"]
    vocab3 = Dict{String,Int}(w => i for (i, w) in enumerate(vocab3_words))
    gen3 = Physics.MirnanGenerator(vocab3, spzeros(length(vocab3), length(vocab3)); model_dir=tmp3)
    positive_explanation = Physics.generate!(gen3, "لماذا يزيد العلم الفهم؟"; max_words=8)
    @test !startswith(positive_explanation, "لا")
    reversed_explanation = Physics.generate!(gen3, "لماذا يزيد الفهم العلم؟"; max_words=8)
    @test startswith(reversed_explanation, "لا")
    @test occursin("الفهم", reversed_explanation)
    @test occursin("يزيد", reversed_explanation)
    @test occursin("العلم", reversed_explanation)

    tmp4 = mktempdir()
    nisba4 = Physics.NisbaMemory()
    Physics.learn_nisba_from_text!(nisba4, "العدل يحفظ السلام في النفوس وفي العلاقات بين الناس.")
    Physics.learn_nisba_from_text!(nisba4, "السلام ينتج عن العدل، ويثمر التعاون والازدهار.")
    Physics.save_nisba(nisba4, joinpath(tmp4, "al_nisba.json"))
    vocab4_words = ["لماذا", "يحفظ", "العدل", "السلام", "لا"]
    vocab4 = Dict{String,Int}(w => i for (i, w) in enumerate(vocab4_words))
    gen4 = Physics.MirnanGenerator(vocab4, spzeros(length(vocab4), length(vocab4)); model_dir=tmp4)
    direct_explanation = Physics.generate!(gen4, "لماذا يحفظ العدل السلام؟"; max_words=10)
    @test !startswith(direct_explanation, "لا")
    reversed_peace = Physics.generate!(gen4, "لماذا يحفظ السلام العدل؟"; max_words=10)
    @test startswith(reversed_peace, "لا")
    @test occursin("السلام", reversed_peace)
    @test occursin("يحفظ", reversed_peace)
    @test occursin("العدل", reversed_peace)
end

@testset "relation and difference guarded fallback" begin
    @test Gen._list_like_generation_output("التعلم العقل العلم الفهم القلوب الظلم العدل")
    @test !Gen._list_like_generation_output("الرحمة تهذب القوة، وتجعلها في خدمة الحق.")
    @test Gen._relation_or_difference_prompt("ما العلاقة بين الرحمة والقوة؟")
    @test Gen._relation_or_difference_prompt("ما الفرق بين السلام والخوف؟")

    tmp = mktempdir()
    mkpath(joinpath(tmp, "data", "corpus"))
    mkpath(joinpath(tmp, "model"))
    write(joinpath(tmp, "data", "corpus", "pairs.txt"),
          "الرحمة والقسوة خلقان ينعكسان على صاحبهما قبل أن ينعكسا على غيره.\n\n" *
          "السلام هو الأمان السائد في النفس والمجتمع نتيجة غياب الخوف والظلم والاضطراب.\n")
    vocab_words = ["الرحمة", "القسوة", "السلام", "الخوف", "العلاقة", "الفرق", "بين"]
    vocab = Dict{String,Int}(w => i for (i, w) in enumerate(vocab_words))
    gen = Physics.MirnanGenerator(vocab, spzeros(length(vocab), length(vocab)); model_dir=joinpath(tmp, "model"))

    mercy_cruelty = Gen._learned_pair_evidence_sentence(gen, "الرحمة", "القسوة")
    @test occursin("الرحمة", mercy_cruelty)
    @test occursin("القسوة", mercy_cruelty)
    peace_fear = Gen._difference_answer(gen, "ما الفرق بين السلام والخوف؟")
    @test occursin("السلام", peace_fear)
    @test occursin("الخوف", peace_fear)
    @test !Gen._list_like_generation_output(peace_fear)

    negated_anchors = Gen._relationship_anchor_keys(["\u0644\u0645\u0627\u0630\u0627", "\u0644\u0627", "\u064a\u0632\u064a\u062f", "\u0627\u0644\u062c\u0647\u0644", "\u0627\u0644\u0641\u0647\u0645\u061f"])
    @test length(negated_anchors) == 2
    @test all(a -> !("\u0644\u0627" in a) && !("\u064a\u0632\u064a\u062f" in a), negated_anchors)

    remove_anchors = Gen._relationship_anchor_keys(["\u0643\u064a\u0641", "\u064a\u0632\u064a\u0644", "\u0627\u0644\u0633\u0644\u0627\u0645", "\u0627\u0644\u062e\u0648\u0641\u061f"])
    @test length(remove_anchors) == 2
    @test all(a -> !("\u064a\u0632\u064a\u0644" in a), remove_anchors)

    action_prompt = ["\u0643\u064a\u0641", "\u064a\u0645\u0646\u0639", "\u0627\u0644\u0639\u0644\u0645", "\u0627\u0644\u062c\u0647\u0644\u061f"]
    action_anchors = Gen._relationship_identity_anchor_keys(action_prompt)
    action_keys = Gen._prompt_relation_action_keys(action_prompt)
    prompt_anchor_keys = Gen._expanded_anchor_keys_for_penalty(action_anchors)
    bad_witness = "\u0645\u0646 \u062a\u0639\u0644\u0645 \u0627\u0644\u062a\u0648\u0627\u0636\u0639\u060c \u062a\u0639\u0644\u0645 \u0645\u0646 \u0643\u0644 \u0625\u0646\u0633\u0627\u0646\u061b \u0648\u0645\u0646 \u062a\u0643\u0628\u0631\u060c \u062d\u0631\u0645 \u0646\u0641\u0633\u0647 \u0645\u0646 \u0639\u0644\u0645 \u0643\u062b\u064a\u0631\u060c \u0644\u0623\u0646 \u0627\u0644\u0643\u0628\u0631 \u064a\u0645\u0646\u0639 \u0627\u0644\u0623\u062e\u0630 \u0639\u0646 \u0627\u0644\u063a\u064a\u0631."
    good_witness = "\u064a\u062d\u062a\u0627\u062c \u0627\u0644\u0625\u0646\u0633\u0627\u0646 \u0625\u0644\u0649 \u0627\u0644\u0639\u0644\u0645 \u0644\u0623\u0646\u0647 \u064a\u0628\u062f\u062f \u0638\u0644\u0645\u0629 \u0627\u0644\u062c\u0647\u0644\u060c \u0648\u064a\u0633\u0627\u0639\u062f\u0647 \u0639\u0644\u0649 \u0627\u0644\u062a\u0645\u064a\u064a\u0632 \u0628\u064a\u0646 \u0627\u0644\u062d\u0642 \u0648\u0627\u0644\u0628\u0627\u0637\u0644."
    @test !Gen._action_clause_support(bad_witness, action_anchors, action_keys, prompt_anchor_keys)
    @test Gen._action_clause_support(good_witness, action_anchors, action_keys, prompt_anchor_keys)

    tmp_action = mktempdir()
    istinbat_action = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(istinbat_action,
        "\u064a\u062d\u062a\u0627\u062c \u0627\u0644\u0625\u0646\u0633\u0627\u0646 \u0625\u0644\u0649 \u0627\u0644\u0639\u0644\u0645 \u0644\u0623\u0646\u0647 \u064a\u0628\u062f\u062f \u0638\u0644\u0645\u0629 \u0627\u0644\u062c\u0647\u0644.")
    Physics.learn_istinbat_from_text!(istinbat_action,
        "\u0627\u0644\u0633\u0644\u0627\u0645 \u0647\u0648 \u0634\u0639\u0648\u0631 \u0628\u0627\u0644\u0623\u0645\u0627\u0646 \u0648\u0627\u0644\u0627\u0633\u062a\u0642\u0631\u0627\u0631\u060c \u0648\u063a\u064a\u0627\u0628 \u0627\u0644\u0627\u0639\u062a\u062f\u0627\u0621 \u0648\u0627\u0644\u062e\u0648\u0641.")
    Physics.save_istinbat(istinbat_action, joinpath(tmp_action, "al_istinbat.json"))
    vocab_action_words = ["\u0647\u0644", "\u0627\u0644\u0639\u0644\u0645", "\u064a\u0645\u0646\u0639", "\u0627\u0644\u062c\u0647\u0644",
                          "\u0627\u0644\u0633\u0644\u0627\u0645", "\u064a\u0632\u064a\u0644", "\u0627\u0644\u062e\u0648\u0641", "\u0644\u0627"]
    vocab_action = Dict{String,Int}(w => i for (i, w) in enumerate(vocab_action_words))
    Physics.MirnanGenerator(vocab_action, spzeros(length(vocab_action), length(vocab_action)); model_dir=tmp_action)
    @test startswith(Gen._yesno_direct_witness_answer(["\u0647\u0644", "\u0627\u0644\u0639\u0644\u0645", "\u064a\u0645\u0646\u0639", "\u0627\u0644\u062c\u0647\u0644\u061f"]), "\u0646\u0639\u0645")
    @test startswith(Gen._yesno_direct_witness_answer(["\u0647\u0644", "\u0627\u0644\u0639\u0644\u0645", "\u0644\u0627", "\u064a\u0645\u0646\u0639", "\u0627\u0644\u062c\u0647\u0644\u061f"]), "\u0644\u0627")
    tmp_remove = mktempdir()
    mkpath(joinpath(tmp_remove, "data", "corpus"))
    mkpath(joinpath(tmp_remove, "model"))
    write(joinpath(tmp_remove, "data", "corpus", "peace.txt"),
          "\u0627\u0644\u0633\u0644\u0627\u0645 \u0647\u0648 \u0634\u0639\u0648\u0631 \u0628\u0627\u0644\u0623\u0645\u0627\u0646 \u0648\u0627\u0644\u0627\u0633\u062a\u0642\u0631\u0627\u0631\u060c \u0648\u063a\u064a\u0627\u0628 \u0627\u0644\u0627\u0639\u062a\u062f\u0627\u0621 \u0648\u0627\u0644\u062e\u0648\u0641.")
    nisba_remove = Physics.NisbaMemory()
    Physics.learn_nisba_from_text!(nisba_remove,
        "\u0627\u0644\u0633\u0644\u0627\u0645 \u0647\u0648 \u0634\u0639\u0648\u0631 \u0628\u0627\u0644\u0623\u0645\u0627\u0646 \u0648\u0627\u0644\u0627\u0633\u062a\u0642\u0631\u0627\u0631\u060c \u0648\u063a\u064a\u0627\u0628 \u0627\u0644\u0627\u0639\u062a\u062f\u0627\u0621 \u0648\u0627\u0644\u062e\u0648\u0641.")
    Physics.save_nisba(nisba_remove, joinpath(tmp_remove, "model", "al_nisba.json"))
    vocab_remove = Dict{String,Int}(w => i for (i, w) in enumerate(["\u0647\u0644", "\u0627\u0644\u0633\u0644\u0627\u0645", "\u064a\u0632\u064a\u0644", "\u0627\u0644\u062e\u0648\u0641", "\u0644\u0627"]))
    gen_remove = Physics.MirnanGenerator(vocab_remove, spzeros(length(vocab_remove), length(vocab_remove)); model_dir=joinpath(tmp_remove, "model"))
    @test startswith(Gen._direct_relation_evidence_answer(gen_remove, "\u0647\u0644 \u0627\u0644\u0633\u0644\u0627\u0645 \u064a\u0632\u064a\u0644 \u0627\u0644\u062e\u0648\u0641\u061f",
                                                          ["\u0647\u0644", "\u0627\u0644\u0633\u0644\u0627\u0645", "\u064a\u0632\u064a\u0644", "\u0627\u0644\u062e\u0648\u0641\u061f"]), "\u0646\u0639\u0645")
    @test startswith(Gen._direct_relation_evidence_answer(gen_remove, "\u0647\u0644 \u0627\u0644\u0633\u0644\u0627\u0645 \u0644\u0627 \u064a\u0632\u064a\u0644 \u0627\u0644\u062e\u0648\u0641\u061f",
                                                          ["\u0647\u0644", "\u0627\u0644\u0633\u0644\u0627\u0645", "\u0644\u0627", "\u064a\u0632\u064a\u0644", "\u0627\u0644\u062e\u0648\u0641\u061f"]), "\u0644\u0627")

    tmp_build = mktempdir()
    mkpath(joinpath(tmp_build, "model"))
    nisba_build = Physics.NisbaMemory()
    Physics.learn_nisba_from_text!(nisba_build,
        "\u064a\u0628\u0646\u064a \u0627\u0644\u062b\u0642\u0629\u064e \u0641\u064a \u0627\u0644\u0645\u062c\u062a\u0645\u0639 \u0627\u0644\u0639\u062f\u0644\u064f \u0641\u064a \u0627\u0644\u0623\u062d\u0643\u0627\u0645\u060c \u0648\u0627\u0644\u0635\u062f\u0642 \u0641\u064a \u0627\u0644\u0623\u0642\u0648\u0627\u0644\u060c \u0648\u0627\u0644\u0631\u062d\u0645\u0629 \u0641\u064a \u0627\u0644\u0645\u0639\u0627\u0645\u0644\u0627\u062a.")
    Physics.save_nisba(nisba_build, joinpath(tmp_build, "model", "al_nisba.json"))
    vocab_build = Dict{String,Int}(w => i for (i, w) in enumerate([
        "\u0647\u0644", "\u0644\u0645\u0627\u0630\u0627", "\u0643\u064a\u0641", "\u0644\u0627",
        "\u0627\u0644\u0631\u062d\u0645\u0629", "\u062a\u0628\u0646\u064a", "\u0627\u0644\u062b\u0642\u0629"
    ]))
    gen_build = Physics.MirnanGenerator(vocab_build, spzeros(length(vocab_build), length(vocab_build));
                                        model_dir=joinpath(tmp_build, "model"))
    build_tokens = ["\u0644\u0645\u0627\u0630\u0627", "\u062a\u0628\u0646\u064a", "\u0627\u0644\u0631\u062d\u0645\u0629", "\u0627\u0644\u062b\u0642\u0629\u061f"]
    build_answer = Gen._direct_relation_evidence_answer(gen_build,
        "\u0644\u0645\u0627\u0630\u0627 \u062a\u0628\u0646\u064a \u0627\u0644\u0631\u062d\u0645\u0629 \u0627\u0644\u062b\u0642\u0629\u061f",
        build_tokens)
    @test occursin("\u0627\u0644\u0631\u062d\u0645\u0629", build_answer)
    @test occursin("\u0627\u0644\u062b\u0642\u0629", build_answer)

    reversed_build_tokens = ["\u0644\u0645\u0627\u0630\u0627", "\u0644\u0627", "\u062a\u0628\u0646\u064a", "\u0627\u0644\u062b\u0642\u0629", "\u0627\u0644\u0631\u062d\u0645\u0629\u061f"]
    reversed_build_answer = Gen._explanatory_direction_guard_answer(gen_build,
        "\u0644\u0645\u0627\u0630\u0627 \u0644\u0627 \u062a\u0628\u0646\u064a \u0627\u0644\u062b\u0642\u0629 \u0627\u0644\u0631\u062d\u0645\u0629\u061f",
        reversed_build_tokens,
        Int[])
    @test startswith(reversed_build_answer, "\u0646\u0639\u0645")
    @test occursin("\u0644\u0627", reversed_build_answer)
    @test occursin("\u0627\u0644\u062b\u0642\u0629", reversed_build_answer)
    @test occursin("\u0627\u0644\u0631\u062d\u0645\u0629", reversed_build_answer)
end
