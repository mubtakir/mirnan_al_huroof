include("../src/MirnanNew.jl")
using .MirnanNew
using Test

const Physics = MirnanNew.Physics
const Gen = MirnanNew.Physics.Generator

@testset "semantic imagination scene extraction" begin
    mem = Physics.SemanticCalculusMemory()

    scene = Physics.extract_semantic_scene(mem, "ضرب خالد الكرة")
    @test scene isa Physics.SemanticScene
    @test scene.actor == "خالد"
    @test scene.action == "ضرب"
    @test scene.patient == "الكرة"
    @test "حركة" in scene.effect_candidates
    @test "ابتعاد" in scene.effect_candidates
    @test scene.source == "physical_semantic_prior"
    @test scene.confidence > 0.0

    learned = Physics.SemanticCalculusMemory()
    @test Physics.learn_semantic_calculus_from_pair!(
        learned,
        "Khalid hit the ball",
        "The ball moved away and changed position.",
    )
    learned_scene = Physics.extract_semantic_scene(learned, "Khalid hit the ball")
    @test learned_scene.source == "al_hisban_al_dalali"
    @test learned_scene.guidance_relation == "semantic_continuation"
    @test any(t -> occursin("moved", t) || occursin("away", t) || occursin("position", t),
              learned_scene.effect_candidates)
    @test learned_scene.confidence >= scene.confidence

    quiet = Physics.extract_semantic_scene(mem, "السلام أمن واستقرار")
    @test quiet.action == ""
    @test quiet.patient == ""
    @test isempty(quiet.effect_candidates)
    @test quiet.source == "none"
end

@testset "semantic imagination rich scene fields" begin
    calculus = Physics.SemanticCalculusMemory()

    scene = Physics.extract_semantic_scene(
        calculus,
        "Khalid hit the ball with bat in yard before sunset",
    )
    @test scene.instrument == "bat"
    @test scene.patient == "ball"
    @test scene.place == "yard"
    @test scene.time_marker == "sunset"
    @test scene.state_before == "stable"
    @test scene.state_after == "moved and changed position"
    @test scene.affect_tone == "neutral"

    mem = Physics.SemanticSceneMemory()
    @test Physics.learn_semantic_scene_from_text!(
        mem,
        calculus,
        "Khalid hit the ball with bat in yard before sunset.",
    ) == 1
    path = joinpath(mktempdir(), "rich_semantic_scenes.json")
    @test Physics.save_semantic_scenes(mem, path) == path
    loaded = Physics.load_semantic_scenes(path)
    @test length(loaded.scenes) == 1
    loaded_scene = first(loaded.scenes)
    @test loaded_scene.instrument == "bat"
    @test loaded_scene.place == "yard"
    @test loaded_scene.time_marker == "sunset"
    @test loaded_scene.state_before == "stable"
    @test loaded_scene.state_after == "moved and changed position"
    @test loaded_scene.affect_tone == "neutral"

    diag = Physics.semantic_scene_comparison_diagnostic(
        loaded,
        calculus,
        "What happens when Khalid hit the ball with bat in yard before sunset?",
    )
    @test occursin("scene_instrument: bat", diag)
    @test occursin("scene_place: yard", diag)
    @test occursin("scene_time: sunset", diag)
    @test occursin("scene_state_before: stable", diag)
    @test occursin("scene_state_after: moved and changed position", diag)
end

@testset "semantic imagination purpose boundary cleanup" begin
    calculus = Physics.SemanticCalculusMemory()
    @test Physics.learn_semantic_calculus_from_pair!(
        calculus,
        "\u062f\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062d\u062c\u0631",
        "\u0639\u0644\u0649 \u0644\u0627 \u062c\u0648\u0627\u0628 \u0648\u062a\u062d\u0631\u0643 \u0627\u0644\u062d\u062c\u0631 \u0645\u0646 \u0645\u0643\u0627\u0646\u0647.",
    )
    scene = Physics.extract_semantic_scene(
        calculus,
        "\u062f\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062d\u062c\u0631 \u0644\u0643\u064a \u064a\u0628\u0639\u062f \u0627\u0644\u062d\u062c\u0631 \u0639\u0646 \u0627\u0644\u0637\u0631\u064a\u0642",
    )
    @test scene.action == "\u062f\u0641\u0639"
    @test scene.patient == "\u0627\u0644\u062d\u062c\u0631"
    @test !occursin("\u0644\u0643\u064a", scene.patient)
    @test !any(t -> t in ["\u0639\u0644\u064a", "\u0644\u0627", "\u062c\u0648\u0627\u0628"], scene.effect_candidates)
    @test any(t -> t in scene.effect_candidates, ["\u062a\u062d\u0631\u0643", "\u0645\u0643\u0627\u0646\u0647", "\u062d\u0631\u0643\u0629"])

    long_scene = Physics.extract_semantic_scene(
        calculus,
        "\u0636\u0631\u0628 \u062e\u0627\u0644\u062f \u0627\u0644\u0643\u0631\u0629 \u0628\u0627\u0644\u0645\u0636\u0631\u0628 \u0644\u0643\u064a \u062a\u062a\u062d\u0631\u0643 \u0627\u0644\u0643\u0631\u0629 \u0628\u0639\u064a\u062f\u0627\u064b \u0648\u064a\u062a\u063a\u064a\u0631 \u0645\u0648\u0636\u0639\u0647\u0627",
    )
    @test long_scene.patient == "\u0627\u0644\u0643\u0631\u0629"
    @test long_scene.instrument == "\u0627\u0644\u0645\u0636\u0631\u0628"
    @test !occursin("\u0644\u0643\u064a", long_scene.patient)
    @test !occursin("\u062a\u062a\u062d\u0631\u0643", long_scene.patient)
end

@testset "semantic imagination effect noise cleanup" begin
    scene = Physics.SemanticScene(
        "\u062f\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062d\u062c\u0631",
        "\u0627\u0644\u0644\u0627\u0639\u0628",
        "\u062f\u0641\u0639",
        "\u0627\u0644\u062d\u062c\u0631",
        "",
        "",
        "",
        "\u0633\u0643\u0648\u0646",
        "\u062d\u0631\u0643\u0629 \u0648\u062a\u063a\u064a\u0631 \u0645\u0648\u0636\u0639",
        "",
        ["\u0643\u0644", "\u0648\u0644\u0627", "\u062d\u0631\u0643\u0629", "\u0644\u0627"],
        1.0,
        "semantic_continuation",
        "fixture",
    )
    effects = Physics.scene_effect_terms(scene)
    @test "\u062d\u0631\u0643\u0629" in effects
    @test !("\u0643\u0644" in effects)
    @test !("\u0648\u0644\u0627" in effects)
    @test !("\u0644\u0627" in effects)
end

@testset "semantic imagination probe matrix fixture" begin
    _decode_unicode_escapes(s) = replace(String(s), r"\\u[0-9A-Fa-f]{4}" => m -> string(Char(parse(Int, String(m)[3:end]; base=16))))
    matrix_path = joinpath(@__DIR__, "fixtures", "semantic_scene_probe_matrix.tsv")
    @test isfile(matrix_path)
    rows = Tuple{String,String,String,String}[]
    for raw in eachline(matrix_path)
        line = strip(raw)
        isempty(line) && continue
        startswith(line, "#") && continue
        cols = split(line, '\t')
        @test length(cols) == 4
        length(cols) == 4 || continue
        push!(rows, (_decode_unicode_escapes(strip(cols[1])),
                     _decode_unicode_escapes(strip(cols[2])),
                     _decode_unicode_escapes(strip(cols[3])),
                     strip(cols[4])))
    end
    @test length(rows) >= 10

    calculus = Physics.SemanticCalculusMemory()
    for (source, target, _, _) in rows
        Physics.learn_semantic_calculus_from_pair!(calculus, source, target)
    end

    checked = 0
    for (source, target, prompt, expected) in rows
        expected in ("aligned", "partial") || continue
        pair_calc = Physics.SemanticCalculusMemory()
        scene_mem = Physics.SemanticSceneMemory()
        Physics.learn_semantic_calculus_from_pair!(pair_calc, source, target)
        Physics.learn_semantic_scene_from_text!(scene_mem, pair_calc, source)
        cmp = Physics.compare_semantic_scene_with_calculus(scene_mem, calculus, prompt)
        @test cmp.agreement in ["aligned", "partial"]
        @test cmp.overlap_score > 0.0
        checked += 1
    end
    @test checked >= 6
end

@testset "semantic imagination calculus comparison" begin
    calculus = Physics.SemanticCalculusMemory()
    @test Physics.learn_semantic_calculus_from_pair!(
        calculus,
        "Khalid hit the ball",
        "The ball moved away and changed position.",
    )

    mem = Physics.SemanticSceneMemory()
    @test Physics.learn_semantic_scene_from_text!(mem, calculus, "Khalid hit the ball.") == 1

    cmp = Physics.compare_semantic_scene_with_calculus(mem, calculus, "Khalid hit the ball")
    @test cmp isa Physics.SemanticSceneComparison
    @test cmp.scene !== nothing
    @test cmp.agreement in ["aligned", "partial"]
    @test cmp.overlap_score > 0.0
    @test cmp.scene_confidence > 0.0
    @test cmp.guidance_confidence > 0.0
    @test !isempty(cmp.scene_effect_terms)
    @test !isempty(cmp.guidance_terms)
    @test !isempty(cmp.raw_guidance_terms)

    diag = Physics.semantic_scene_comparison_diagnostic(mem, calculus, "Khalid hit the ball")
    @test occursin("agreement", diag)
    @test occursin("overlap", diag)

    scene_only_calculus = Physics.SemanticCalculusMemory()
    scene_only = Physics.compare_semantic_scene_with_calculus(mem, scene_only_calculus, "Khalid hit the ball")
    @test scene_only.agreement == "scene_only"
    @test scene_only.scene !== nothing
    @test isempty(scene_only.guidance_terms)

    empty_mem = Physics.SemanticSceneMemory()
    calculus_only = Physics.compare_semantic_scene_with_calculus(empty_mem, calculus, "Khalid hit the ball")
    @test calculus_only.agreement == "calculus_only"
    @test calculus_only.scene === nothing
    @test !isempty(calculus_only.guidance_terms)

    non_scene = Physics.compare_semantic_scene_with_calculus(empty_mem, calculus, "What is peace?")
    @test non_scene.agreement == "none"
    @test isempty(non_scene.guidance_terms)
    @test non_scene.guidance_confidence == 0.0

    noisy_calculus = Physics.SemanticCalculusMemory()
    @test Physics.learn_semantic_calculus_from_pair!(
        noisy_calculus,
        "Khalid hit the ball",
        "The ball moved away.",
    )
    @test Physics.learn_semantic_calculus_from_pair!(
        noisy_calculus,
        "The lamp illuminated the room",
        "The room became clear.",
    )
    filtered = Physics.compare_semantic_scene_with_calculus(mem, noisy_calculus, "Khalid hit the ball")
    @test length(filtered.raw_guidance_terms) >= length(filtered.guidance_terms)
    @test !("room" in filtered.guidance_terms)
    @test !("clear" in filtered.guidance_terms)

    arabic_scene_mem = Physics.SemanticSceneMemory()
    arabic_pair_calc = Physics.SemanticCalculusMemory()
    @test Physics.learn_semantic_calculus_from_pair!(
        arabic_pair_calc,
        "\u0636\u0631\u0628 \u062e\u0627\u0644\u062f \u0627\u0644\u0643\u0631\u0629",
        "\u0627\u0628\u062a\u0639\u062f\u062a \u0627\u0644\u0643\u0631\u0629 \u0648\u062a\u063a\u064a\u0631 \u0645\u0648\u0636\u0639\u0647\u0627.",
    )
    @test Physics.learn_semantic_scene_from_text!(
        arabic_scene_mem,
        arabic_pair_calc,
        "\u0636\u0631\u0628 \u062e\u0627\u0644\u062f \u0627\u0644\u0643\u0631\u0629",
    ) == 1
    english_calc = Physics.SemanticCalculusMemory()
    @test Physics.learn_semantic_calculus_from_pair!(
        english_calc,
        "Khalid hit the ball",
        "The ball moved away and changed position.",
    )
    bridged = Physics.compare_semantic_scene_with_calculus(
        arabic_scene_mem,
        english_calc,
        "\u0645\u0627\u0630\u0627 \u064a\u062d\u062f\u062b \u0639\u0646\u062f\u0645\u0627 \u064a\u0636\u0631\u0628 \u062e\u0627\u0644\u062f \u0627\u0644\u0643\u0631\u0629\u061f",
    )
    @test bridged.scene !== nothing
    @test bridged.agreement in ["aligned", "partial"]
    @test bridged.overlap_score > 0.0
    @test any(t -> t in bridged.guidance_terms, ["moved", "away", "changed", "position"])
end

@testset "semantic imagination scene memory" begin
    calculus = Physics.SemanticCalculusMemory()
    mem = Physics.SemanticSceneMemory(max_scenes=3)

    learned = Physics.learn_semantic_scene_from_text!(
        mem,
        calculus,
        "Khalid hit the ball. Peace is safety. The player pushed the stone.",
    )
    @test learned == 2
    @test Physics.has_semantic_scenes(mem)
    @test length(mem.scenes) == 2

    selected = Physics.select_semantic_scene(mem, "what happened when Khalid hit the ball?")
    @test selected !== nothing
    @test selected.actor == "khalid"
    @test selected.action == "hit"
    @test selected.patient == "ball"
    @test !isempty(selected.effect_candidates)

    pushed = Physics.select_semantic_scene(mem, "stone moved after the push")
    @test pushed !== nothing
    @test pushed.patient == "stone"

    diag = Physics.semantic_scene_diagnostic(mem, "Khalid hit the ball")
    @test occursin("khalid", lowercase(diag))
    @test occursin("hit", lowercase(diag))

    empty = Physics.SemanticSceneMemory()
    @test !Physics.has_semantic_scenes(empty)
    @test Physics.select_semantic_scene(empty, "hit ball") === nothing
    @test !isempty(Physics.semantic_scene_diagnostic(empty, "hit ball"))

    capped = Physics.SemanticSceneMemory(max_scenes=1)
    @test Physics.learn_semantic_scene_from_text!(capped, calculus, "Khalid hit the ball. The player pushed the stone.") == 2
    @test length(capped.scenes) == 1
    @test capped.scenes[1].patient == "stone"

    path = joinpath(mktempdir(), "semantic_scenes.json")
    @test Physics.save_semantic_scenes(mem, path) == path
    loaded = Physics.load_semantic_scenes(path)
    @test Physics.has_semantic_scenes(loaded)
    @test length(loaded.scenes) == length(mem.scenes)
    loaded_hit = Physics.select_semantic_scene(loaded, "Khalid hit the ball")
    @test loaded_hit !== nothing
    @test loaded_hit.action == "hit"
    @test loaded_hit.patient == "ball"
    @test !isempty(Physics.semantic_scenes_to_dict(loaded)["scenes"])

    event_mem = Physics.SemanticSceneMemory()
    @test Physics.learn_semantic_scene_from_text!(
        event_mem,
        calculus,
        "The child broke the cup. The lamp illuminated the room.",
    ) == 2
    broke = Physics.select_semantic_scene(event_mem, "The child broke the cup")
    @test broke !== nothing
    @test broke.action == "broke"
    lit = Physics.select_semantic_scene(event_mem, "The lamp illuminated the room")
    @test lit !== nothing
    @test lit.action == "illuminated"

    hit_only = Physics.SemanticSceneMemory()
    @test Physics.learn_semantic_scene_from_text!(hit_only, calculus, "Khalid hit the ball.") == 1
    @test Physics.select_semantic_scene(hit_only, "The lamp illuminated the room") === nothing

    broke_only = Physics.SemanticSceneMemory()
    @test Physics.learn_semantic_scene_from_text!(broke_only, calculus, "The child broke the cup.") == 1
    @test isempty(Physics.semantic_scene_answer(broke_only, calculus, "What happens when Khalid hit the ball?"))
end

@testset "semantic imagination independent answer" begin
    calculus = Physics.SemanticCalculusMemory()
    @test Physics.learn_semantic_calculus_from_pair!(
        calculus,
        "Khalid hit the ball",
        "The ball moved away and changed position.",
    )
    mem = Physics.SemanticSceneMemory()
    @test Physics.learn_semantic_scene_from_text!(mem, calculus, "Khalid hit the ball.") == 1

    answer = Physics.semantic_scene_answer(mem, calculus, "What happens when Khalid hit the ball?")
    @test !isempty(answer)
    @test occursin("hit", answer)
    @test occursin("ball", answer)
    @test any(t -> occursin(t, answer), ["moved", "away", "changed", "position"])

    rich_mem = Physics.SemanticSceneMemory()
    @test Physics.learn_semantic_scene_from_text!(
        rich_mem,
        calculus,
        "Khalid hit the ball with bat in yard before sunset.",
    ) == 1
    rich_answer = Physics.semantic_scene_answer(
        rich_mem,
        calculus,
        "What happens when Khalid hit the ball with bat in yard before sunset?",
    )
    @test occursin("using bat", rich_answer)
    @test occursin("in yard", rich_answer)
    @test occursin("sunset", rich_answer)
    @test occursin("stable", rich_answer)
    @test occursin("moved and changed position", rich_answer)

    @test isempty(Physics.semantic_scene_answer(mem, calculus, "What is peace?"))
    @test isempty(Physics.semantic_scene_answer(mem, calculus, "Why does knowledge increase understanding?"))
    @test isempty(Physics.semantic_scene_answer(mem, calculus, "Does Khalid hit the ball?"))

    arabic_pair_calc = Physics.SemanticCalculusMemory()
    arabic_mem = Physics.SemanticSceneMemory()
    @test Physics.learn_semantic_calculus_from_pair!(
        arabic_pair_calc,
        "\u0636\u0631\u0628 \u062e\u0627\u0644\u062f \u0627\u0644\u0643\u0631\u0629",
        "\u0627\u0628\u062a\u0639\u062f\u062a \u0627\u0644\u0643\u0631\u0629 \u0648\u062a\u063a\u064a\u0631 \u0645\u0648\u0636\u0639\u0647\u0627.",
    )
    @test Physics.learn_semantic_scene_from_text!(
        arabic_mem,
        arabic_pair_calc,
        "\u0636\u0631\u0628 \u062e\u0627\u0644\u062f \u0627\u0644\u0643\u0631\u0629",
    ) == 1
    english_calc = Physics.SemanticCalculusMemory()
    @test Physics.learn_semantic_calculus_from_pair!(
        english_calc,
        "Khalid hit the ball",
        "The ball moved away and changed position.",
    )
    ar_answer = Physics.semantic_scene_answer(
        arabic_mem,
        english_calc,
        "\u0645\u0627\u0630\u0627 \u064a\u062d\u062f\u062b \u0639\u0646\u062f\u0645\u0627 \u064a\u0636\u0631\u0628 \u062e\u0627\u0644\u062f \u0627\u0644\u0643\u0631\u0629\u061f",
    )
    @test !isempty(ar_answer)
    @test occursin("\u0636\u0631\u0628", ar_answer)
    @test occursin("\u0627\u0644\u0643\u0631\u0629", ar_answer)
    @test any(t -> occursin(t, ar_answer), ["\u0627\u0628\u062a\u0639\u062f", "\u0627\u0628\u062a\u0639\u0627\u062f", "\u062d\u0631\u0643\u0629", "\u062a\u063a\u064a\u0631"])
    @test !occursin("\u0623\u062b\u0631 \u062f\u0644\u0627\u0644\u064a", ar_answer)
    @test occursin("\u0641\u062a\u063a\u064a\u0631 \u062d\u0627\u0644", ar_answer)
end

@testset "semantic imagination answer comparison" begin
    calculus = Physics.SemanticCalculusMemory()
    @test Physics.learn_semantic_calculus_from_pair!(
        calculus,
        "Khalid hit the ball",
        "The ball moved away and changed position.",
    )
    mem = Physics.SemanticSceneMemory()
    @test Physics.learn_semantic_scene_from_text!(mem, calculus, "Khalid hit the ball.") == 1

    cmp = Physics.compare_semantic_scene_strategies(
        mem,
        calculus,
        p -> "general answer: " * p,
        "What happens when Khalid hit the ball?",
    )
    @test cmp isa Physics.SemanticSceneAnswerComparison
    @test !isempty(cmp.scene_answer)
    @test occursin("general answer", cmp.generate_answer)
    @test cmp.memory_has_scene
    @test cmp.question_allowed
    @test cmp.agreement in ["aligned", "partial"]
    @test cmp.overlap_score > 0.0
    @test cmp.scene_confidence > 0.0
    @test cmp.guidance_confidence > 0.0

    non_scene = Physics.compare_semantic_scene_strategies(
        mem,
        calculus,
        p -> "general answer: " * p,
        "What is peace?",
    )
    @test isempty(non_scene.scene_answer)
    @test !non_scene.question_allowed
    @test !isempty(non_scene.generate_answer)

    failed_generate = Physics.compare_semantic_scene_strategies(
        mem,
        calculus,
        p -> error("synthetic generation failure"),
        "What happens when Khalid hit the ball?",
    )
    @test !isempty(failed_generate.scene_answer)
    @test isempty(failed_generate.generate_answer)
end

@testset "semantic imagination strategy gate" begin
    calculus = Physics.SemanticCalculusMemory()
    @test Physics.learn_semantic_calculus_from_pair!(
        calculus,
        "Khalid hit the ball",
        "The ball moved away and changed position.",
    )
    mem = Physics.SemanticSceneMemory()
    @test Physics.learn_semantic_scene_from_text!(mem, calculus, "Khalid hit the ball.") == 1

    saved_mem = Gen._LEARNED_SEMANTIC_SCENE_MEMORY[]
    model_dir = mktempdir()
    Physics.save_semantic_scenes(mem, joinpath(model_dir, "semantic_scenes.json"))
    gen = Physics.MirnanGenerator(Dict{String,Int}(
        "Khalid" => 1, "hit" => 2, "the" => 3, "ball" => 4,
        "moved" => 5, "away" => 6, "changed" => 7, "position" => 8,
    ); model_dir=model_dir)
    gen.hisban = calculus
    @test Physics.has_semantic_scenes(gen.semantic_scenes)
    @test Physics.select_semantic_scene(gen.semantic_scenes, "Khalid hit the ball") !== nothing

    try
        Gen._LEARNED_SEMANTIC_SCENE_MEMORY[] = mem
        prompt = "What happens when Khalid hit the ball?"
        pt = String.(split(prompt))
        co = Gen.observe_prompt(gen.cerebellum, pt; prompt=prompt, vocab_size=length(gen.vocab))
        cp = Gen.choose_policy!(gen.cerebellum, co; requested_mode="auto")
        rp = Gen.detect_response_intent(prompt)
        ap = Gen._get_active_paragraphs(gen, pt)

        direct = Gen.try_generate(Gen.SemanticSceneStrategy(), gen, prompt,
                                  pt, "auto", co, cp, rp, ap)
        @test direct !== nothing
        @test occursin("semantic effect", direct)
        @test occursin("ball", direct)

        blocked_yesno = Gen.try_generate(Gen.SemanticSceneStrategy(), gen,
                                         "Does Khalid hit the ball?",
                                         String.(split("Does Khalid hit the ball?")),
                                         "auto", co, cp, rp, ap)
        @test blocked_yesno === nothing

        saved_scene_field = gen.semantic_scenes
        Gen._LEARNED_SEMANTIC_SCENE_MEMORY[] = Physics.SemanticSceneMemory()
        gen.semantic_scenes = Physics.SemanticSceneMemory()
        empty_mem = Gen.try_generate(Gen.SemanticSceneStrategy(), gen, prompt,
                                     pt, "auto", co, cp, rp, ap)
        @test empty_mem === nothing

        Gen._LEARNED_SEMANTIC_SCENE_MEMORY[] = nothing
        gen.semantic_scenes = saved_scene_field
        from_generator_field = Gen.try_generate(Gen.SemanticSceneStrategy(), gen, prompt,
                                                pt, "auto", co, cp, rp, ap)
        @test from_generator_field !== nothing
        @test occursin("semantic effect", from_generator_field)

        old_gate = get(ENV, "MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY", nothing)
        old_strict = get(ENV, "MIRNAN_STRICT_NO_TEMPLATES", nothing)
        try
            ENV["MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY"] = "1"
            ENV["MIRNAN_STRICT_NO_TEMPLATES"] = "0"
            generated = Physics.generate!(gen, prompt; mode="auto", max_words=20)
            @test occursin("semantic effect", generated)
            @test occursin("ball", generated)
        finally
            if old_gate === nothing
                delete!(ENV, "MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY")
            else
                ENV["MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY"] = old_gate
            end
            if old_strict === nothing
                delete!(ENV, "MIRNAN_STRICT_NO_TEMPLATES")
            else
                ENV["MIRNAN_STRICT_NO_TEMPLATES"] = old_strict
            end
        end
    finally
        Gen._LEARNED_SEMANTIC_SCENE_MEMORY[] = saved_mem
    end
end
