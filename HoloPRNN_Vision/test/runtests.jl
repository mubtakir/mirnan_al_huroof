using Test
using Images
using Serialization

include(joinpath(@__DIR__, "..", "src", "HoloPRNN_Vision.jl"))
using .HoloPRNN_Vision

@testset "HoloPRNN Vision core safety" begin
    img = [RGB(1.0, 0.0, 0.0) RGB(0.0, 1.0, 0.0);
           RGB(0.0, 0.0, 1.0) RGB(1.0, 1.0, 1.0)]

    z = image_to_wavefield(img)
    @test size(z) == size(img)
    @test eltype(z) == ComplexF64

    img2 = wavefield_to_image(z)
    @test size(img2) == size(img)

    params = OscillatorParams()
    profile = extract_style_profile(img, params)
    @test profile.H == 2
    @test profile.W == 2
    @test size(profile.K_h) == (2, 2)
    @test size(profile.K_v) == (2, 2)

    z_copy = copy(z)
    styled = apply_style(z_copy, profile; steps=1, inplace=false)
    @test styled isa Matrix{ComplexF64}
    @test z_copy == z
    @test apply_style(z_copy, profile; steps=0, inplace=true) === z_copy
    @test_throws DimensionMismatch apply_style(zeros(ComplexF64, 1, 1), profile; steps=1)

    profile_path = joinpath(mktempdir(), "style_profile.bin")
    @test save_style_profile(profile, profile_path) == profile_path
    loaded_profile = load_style_profile(profile_path)
    @test loaded_profile.H == profile.H
    @test loaded_profile.W == profile.W
    @test loaded_profile.K_h == profile.K_h
    @test loaded_profile.K_v == profile.K_v

    bad_profile_path = joinpath(mktempdir(), "bad_style_profile.bin")
    open(bad_profile_path, "w") do io
        serialize(io, (2, 2, zeros(ComplexF64, 1, 1), zeros(ComplexF64, 2, 2), params))
    end
    @test_throws DimensionMismatch load_style_profile(bad_profile_path)

    semantic_motion = (
        actor = "Khalid",
        action = "hit",
        patient = "ball",
        effect_candidates = ["moved", "away", "changed"],
        confidence = 0.9,
    )
    visual_motion = semantic_scene_to_visual_scene(semantic_motion; width=24, height=16)
    @test visual_motion.width == 24
    @test visual_motion.height == 16
    @test visual_motion.action == "hit"
    @test visual_motion.motion[1] > 0
    @test any(o -> o.kind == "actor", visual_motion.objects)
    @test any(o -> o.kind == "patient", visual_motion.objects)
    @test any(o -> o.shape == "person", visual_motion.objects)
    @test any(o -> o.shape == "ball", visual_motion.objects)

    motion_wave = visual_scene_to_wavefield(visual_motion)
    @test size(motion_wave) == (16, 24)
    @test maximum(abs.(motion_wave)) <= 1.0
    motion_image = render_visual_scene(visual_motion)
    @test size(motion_image) == (16, 24)
    @test eltype(motion_image) <: Colorant

    semantic_break = (
        actor = "child",
        action = "broke",
        patient = "cup",
        effect_candidates = ["pieces", "separated"],
        confidence = 1.0,
    )
    visual_break = semantic_scene_to_visual_scene(semantic_break)
    @test any(o -> o.kind == "effect", visual_break.objects)
    @test any(o -> o.shape == "cup", visual_break.objects)
    @test any(o -> o.shape == "fragment", visual_break.objects)

    semantic_light = (
        actor = "lamp",
        action = "illuminated",
        patient = "room",
        effect_candidates = ["clear", "visible"],
        confidence = 1.0,
    )
    visual_light = semantic_scene_to_visual_scene(semantic_light)
    @test visual_light.motion[2] < 0
    @test any(o -> o.kind == "effect", visual_light.objects)
    @test any(o -> o.shape == "room", visual_light.objects)
    @test any(o -> o.shape == "glow", visual_light.objects)

    arabic_motion = (
        actor = "\u062e\u0627\u0644\u062f",
        action = "\u0636\u0631\u0628",
        patient = "\u0627\u0644\u0643\u0631\u0629",
        effect_candidates = ["\u062d\u0631\u0643\u0629", "\u0627\u0628\u062a\u0639\u0627\u062f", "\u062a\u063a\u064a\u0631 \u0645\u0648\u0636\u0639"],
        confidence = 1.0,
    )
    visual_arabic_motion = semantic_scene_to_visual_scene(arabic_motion)
    @test visual_arabic_motion.motion[1] > 0
    @test any(o -> o.shape == "person", visual_arabic_motion.objects)
    @test any(o -> o.shape == "ball", visual_arabic_motion.objects)

    arabic_push_stone = (
        actor = "\u0627\u0644\u0644\u0627\u0639\u0628",
        action = "\u062f\u0641\u0639",
        patient = "\u0627\u0644\u062d\u062c\u0631",
        effect_candidates = ["\u062a\u062d\u0631\u0643", "\u062a\u063a\u064a\u0631 \u0645\u0648\u0636\u0639"],
        confidence = 1.0,
    )
    visual_arabic_stone = semantic_scene_to_visual_scene(arabic_push_stone)
    @test visual_arabic_stone.motion[1] > 0
    @test any(o -> o.shape == "stone", visual_arabic_stone.objects)

    arabic_break = (
        actor = "\u0627\u0644\u0637\u0641\u0644",
        action = "\u0643\u0633\u0631",
        patient = "\u0627\u0644\u0643\u0623\u0633",
        effect_candidates = ["\u0627\u0646\u0641\u0635\u0627\u0644", "\u062a\u0644\u0641", "\u062a\u063a\u064a\u0631 \u0647\u064a\u0626\u0629"],
        confidence = 1.0,
    )
    visual_arabic_break = semantic_scene_to_visual_scene(arabic_break)
    @test any(o -> o.shape == "cup", visual_arabic_break.objects)
    @test any(o -> o.shape == "fragment", visual_arabic_break.objects)

    arabic_light = (
        actor = "\u0627\u0644\u0645\u0635\u0628\u0627\u062d",
        action = "\u0623\u0636\u0627\u0621",
        patient = "\u0627\u0644\u063a\u0631\u0641\u0629",
        effect_candidates = ["\u0638\u0647\u0648\u0631", "\u0627\u0646\u0643\u0634\u0627\u0641", "\u0648\u0636\u0648\u062d"],
        confidence = 1.0,
    )
    visual_arabic_light = semantic_scene_to_visual_scene(arabic_light)
    @test visual_arabic_light.motion[2] < 0
    @test any(o -> o.kind == "actor" && o.shape == "lamp", visual_arabic_light.objects)
    @test any(o -> o.shape == "room", visual_arabic_light.objects)
    @test any(o -> o.shape == "glow", visual_arabic_light.objects)

    arabic_rich_motion = (
        actor = "\u062e\u0627\u0644\u062f",
        action = "\u0636\u0631\u0628",
        patient = "\u0627\u0644\u0643\u0631\u0629",
        instrument = "\u0627\u0644\u0645\u0636\u0631\u0628",
        place = "\u0627\u0644\u062d\u062f\u064a\u0642\u0629",
        time_marker = "\u0627\u0644\u0641\u062c\u0631",
        effect_candidates = ["\u062d\u0631\u0643\u0629", "\u0627\u0628\u062a\u0639\u0627\u062f", "\u062a\u063a\u064a\u0631 \u0645\u0648\u0636\u0639"],
        confidence = 1.0,
    )
    visual_arabic_rich = semantic_scene_to_visual_scene(arabic_rich_motion)
    @test any(o -> o.kind == "patient" && o.shape == "ball", visual_arabic_rich.objects)
    @test any(o -> o.kind == "instrument" && o.shape == "bat", visual_arabic_rich.objects)
    @test any(o -> o.kind == "place" && o.shape == "garden", visual_arabic_rich.objects)
    @test any(o -> o.kind == "time" && o.shape == "dawn", visual_arabic_rich.objects)
    @test render_visual_scene(visual_arabic_rich) isa Matrix

    text_visual = semantic_text_to_visual_scene(
        "\u0636\u0631\u0628 \u062e\u0627\u0644\u062f \u0627\u0644\u0643\u0631\u0629 \u0628\u0627\u0644\u0645\u0636\u0631\u0628 \u0641\u064a \u0627\u0644\u062d\u062f\u064a\u0642\u0629 \u0642\u0628\u0644 \u0627\u0644\u0641\u062c\u0631";
        width=24, height=16)
    @test any(o -> o.kind == "patient" && o.shape == "ball", text_visual.objects)
    @test any(o -> o.kind == "instrument" && o.shape == "bat", text_visual.objects)
    @test any(o -> o.kind == "place" && o.shape == "garden", text_visual.objects)
    @test any(o -> o.kind == "time" && o.shape == "dawn", text_visual.objects)

    extracted_scene = (
        actor = "\u062e\u0627\u0644\u062f",
        action = "\u0636\u0631\u0628",
        patient = "\u0627\u0644\u0643\u0631\u0629",
        instrument = "",
        place = "",
        time_marker = "",
        effect_candidates = ["\u062d\u0631\u0643\u0629"],
        confidence = 1.0,
    )
    bridged_visual = semantic_text_to_visual_scene(
        "\u0636\u0631\u0628 \u062e\u0627\u0644\u062f \u0627\u0644\u0643\u0631\u0629 \u0628\u0627\u0644\u0645\u0636\u0631\u0628 \u0641\u064a \u0627\u0644\u062d\u062f\u064a\u0642\u0629 \u0642\u0628\u0644 \u0627\u0644\u0641\u062c\u0631";
        extractor = _ -> extracted_scene)
    @test any(o -> o.kind == "instrument" && o.shape == "bat", bridged_visual.objects)
    @test any(o -> o.kind == "place" && o.shape == "garden", bridged_visual.objects)
    @test any(o -> o.kind == "time" && o.shape == "dawn", bridged_visual.objects)

    @test_throws ArgumentError semantic_scene_to_visual_scene(semantic_motion; width=0)

    a = zeros(Float64, size(z))
    omega = zeros(Float64, size(z))
    @test simulate_wave_field!(copy(z), copy(a), copy(omega), params; steps=1) isa Matrix{ComplexF64}
    @test maximum(abs.(HoloPRNN_Vision.laplacian2d(fill(1.0 + 0.0im, 3, 3)))) == 0.0
    @test_throws DimensionMismatch simulate_wave_field!(copy(z), zeros(Float64, 1, 1), copy(omega), params; steps=1)
    @test_throws DimensionMismatch run_inpainting(img, zeros(Bool, 1, 1), params; steps=1)
    @test_throws ArgumentError run_pattern_crystallization(zeros(Bool, 0, 2), params; steps=1)
    @test_throws DomainError simulate_wave_field!(copy(z), copy(a), copy(omega), OscillatorParams(dt=Inf); steps=1)

    target = [true false; false true]
    crystallized_a = run_pattern_crystallization(target, params; steps=1, noise_level=0.1, seed=7)
    crystallized_b = run_pattern_crystallization(target, params; steps=1, noise_level=0.1, seed=7)
    @test crystallized_a == crystallized_b

    clf = PhasNetClassifier(4, 3, 2)
    @test isempty(clf.loss_history)
    loss = compute_classification_loss(clf, [1.0, 0.0, 0.0, 1.0], [1.0, 0.0])
    @test isfinite(loss)
    @test loss >= 0.0
    @test length(predict_batch(clf, [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])) == 2
    history = train_classifier!(clf, [[1.0, 0.0, 0.0, 1.0]], [[1.0, 0.0]]; epochs=2, lr=0.0, reset_loss_history=true)
    @test history === clf.loss_history
    @test length(clf.loss_history) == 2
    @test all(isfinite, clf.loss_history)
    early_clf = PhasNetClassifier(4, 3, 2)
    early_history = train_classifier!(early_clf, [[1.0, 0.0, 0.0, 1.0]], [[1.0, 0.0]];
                                      epochs=10, lr=0.0, early_stopping=true, patience=2,
                                      min_delta=1e9, reset_loss_history=true)
    @test 1 <= length(early_history) <= 10
    @test length(early_history) < 10
    @test_throws DimensionMismatch train_classifier!(clf, [[1.0, 0.0]], [[1.0, 0.0]]; epochs=1)
    @test_throws DimensionMismatch train_classifier!(clf, [[1.0, 0.0, 0.0, 1.0]], [[1.0]]; epochs=1)
    @test_throws ArgumentError train_classifier!(clf, [[1.0, 0.0, 0.0, 1.0]], [[1.0, 0.0]]; epochs=1, weight_decay=1.5)
    @test_throws ArgumentError train_classifier!(clf, [[1.0, 0.0, 0.0, 1.0]], [[1.0, 0.0]]; epochs=1, patience=0)
    @test_throws ArgumentError train_classifier!(clf, [[1.0, 0.0, 0.0, 1.0]], [[1.0, 0.0]]; epochs=1, min_delta=-1.0)
    @test_throws DimensionMismatch predict_classifier!(clf, [1.0, 0.0])
end
