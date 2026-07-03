# مرنان — اختبارات الوحدة والتكامل

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "MirnanNew.jl")); using .MirnanNew
import .MirnanNew: SimulationSpace, CausalFrame, PropertyEffect, DynamicVerb, CausalRule,
    CausalTemplate, ProcessStep, ProcessConcept, Circumstance, TemporalRelation,
    ChainStep, EventChain, RelationFact, SemanticRelation, QuantifiedFact, ComparisonFact,
    MetaphorFact, IntentFact, SpeechActFact, ExceptionRule,
    AqlGovernanceRule, RejectionLesson, CorpusAnnotation,
    register_entity!, register_verb!, add_rule!, apply_verb!, evaluate_idea!,
    interact!, record_frame!, step_simulation!, register_template!, matching_templates,
    infer_event!, register_process!, get_process, instantiate_process!,
    register_opposite!, opposite_of, apply_signed_action!,
    set_circumstance!, get_circumstance, set_location!, set_time!,
    relate_events!, register_event_chain!, run_event_chain!,
    assert_relation!, relations_from, relations_to, relations_between,
    assert_quantified_fact!, quantified_facts_for,
    assert_comparison!, comparisons_for, compare_entities!,
    assert_metaphor!, metaphors_for,
    assert_intent!, intents_for,
    assert_speech_act!, speech_acts_for, speech_responses_for,
    add_exception_rule!, exceptions_for, active_exception,
    curate_rule!, propose_rule!, approve_rule!, reject_rule!,
    learn_rejection_lesson!, score_rule_proposal,
    annotate_corpus_sentence!, critical_corpus_pass!,
    aql_memory_audit, annotation_weight,
    aql_inhibition_score, aql_inhibition_reason,
    property_key, compile_adl!, train_from_text!,
    MorphAnalysis, TemporalInfo, BinaryRelation, InferredEvent, EntityKnowledge,
    AdvancedKnowledgeBase, strip_diacritics, strip_al, strip_prep_prefix,
    has_tanwin_nasb, tokenize_arabic, analyze_arabic_word, analyze_arabic_sentence,
    segment_sentences, deep_understand, get_or_create_entity!,
    add_relation!, add_role!, add_temporal_info!, get_entity_relations,
    get_entities_with_attribute, find_relations_between,
    Idea, compute_idea_vector, match_sentence_to_idea, score_candidate_for_idea,
    AqlEntity, Thing, Lion, Gazelle, Earth, Furniture,
    get_property, set_property!, get_attribute, set_attribute!,
    register_action!, get_action, has_action, print_entity, PROPERTY_MAP,
    ConceptClass, ConceptTaxonomy, default_aql_taxonomy,
    register_class!, add_subclass!, assign_class!, assigned_classes,
    class_lineage, inherited_attributes, apply_taxonomy!, is_a,
    explain_classification, known_classes, load_taxonomy, save_taxonomy,
    DEFAULT_TAXONOMY_FILE,
    roar!, shake!, run!, fall!, become!, emit_scent!, attract_beneficial_insects!,
    WordBalanceConfig, build_word_balance_weights, balance_summary, pair_balance_weight,
    init_rapg_db!, RAPGKnowledgeBase, store_passage!, load_rapg_kb, retrieve,
    update_marker_confidence!, IstinbatAttentionMemory,
    compute_word_phase_vector, compute_word_mass, compute_word_frequency,
    compute_word_frequency_with_irab, phase_similarity, compute_extended_phase_vector,
    MirnanGenerator, generate!,
    word_physics_profile,
    get_physics_report,
    phonosemantic_quality, nearest_phase_words, analyze_unknown_word, coin_word_for_concept,
    save_runtime_learning!
using Test
using LinearAlgebra
using SparseArrays

@testset "Mirnan V8 — Physics Engine" begin

    @testset "Constants" begin
        @test MirnanNew.Physics.Constants.PLANCK_H == 1.0
        @test MirnanNew.Physics.Constants.GRAVITY_G == 1.0
        @test MirnanNew.Physics.Constants.TOTAL_DIM == 10000
        @test MirnanNew.Physics.Constants.PHASE_DIM == 9958
    end

    @testset "Letter Database" begin
        db = MirnanNew.Physics.LetterDB.LetterDatabase()
        v = MirnanNew.Physics.LetterDB.get_vector(db, "ا")
        @test length(v) == MirnanNew.Physics.Constants.PHASE_DIM
        @test all(x -> x in (-1, 1), v)

        omega = MirnanNew.Physics.LetterDB.get_omega_0(db, "ا")
        @test omega > 0.0
    end

    @testset "Word Physics" begin
        pv = compute_word_phase_vector("سلام")
        @test length(pv) == MirnanNew.Physics.Constants.PHASE_DIM

        mass = compute_word_mass("سلام")
        @test mass > 0.0

        freq = compute_word_frequency("سلام")
        @test freq > 0.0

        epv = compute_extended_phase_vector("سلام")
        @test length(epv) == MirnanNew.Physics.Constants.TOTAL_DIM
    end

    @testset "Phase Similarity" begin
        v1 = compute_word_phase_vector("علم")
        v2 = compute_word_phase_vector("نور")
        sim = phase_similarity(v1, v2)
        @test -1.0 <= sim <= 1.0

        # الكلمة مع نفسها = 1.0
        sim_self = phase_similarity(v1, v1)
        @test abs(sim_self - 1.0) < 1e-6
    end

    @testset "Gravity" begin
        m_i, m_j = 1.0, 2.0
        v_i = ones(9958) * 0.1
        v_j = ones(9958) * 0.2
        f = MirnanNew.Physics.GravityEngine.gravitational_force(m_i, v_i, m_j, v_j, 1.0)
        @test length(f) == 9958
        @test norm(f) > 0.0
    end

    @testset "Clifford Math" begin
        mv1 = MirnanNew.Physics.CliffordMath.from_vector(ones(22))
        mv2 = MirnanNew.Physics.CliffordMath.from_vector(ones(22) * 0.5)
        prod = mv1 * mv2
        @test MirnanNew.Physics.CliffordMath.norm(prod) > 0.0
        @test MirnanNew.Physics.CliffordMath.get_scalar_essence(prod) > 0.0
    end

    @testset "Resonant Chain" begin
        chain = MirnanNew.Physics.ResonantChain.ResonantChainRLC()
        m_i, m_j = 2.0, 3.0
        pv_i = compute_word_phase_vector("علم")
        pv_j = compute_word_phase_vector("نور")
        f = MirnanNew.Physics.ResonantChain.pair_freq(chain, m_i, m_j, pv_i, pv_j)
        @test f > 0.0
    end

    @testset "Entropy Gate" begin
        gate = MirnanNew.Physics.EntropyGateModule.EntropyGate()
        pvs = [compute_extended_phase_vector(w) for w in ["علم", "نور", "حياة"]]
        S = MirnanNew.Physics.EntropyGateModule.compute_S(gate, pvs, pvs[1])
        @test S >= 0.0
    end

    @testset "RAM Core" begin
        ram = MirnanNew.Physics.RAMCore.AttractorMemory()
        idx = MirnanNew.Physics.RAMCore.observe!(ram, ["علم", "نور", "حياة"])
        @test idx >= 0
        @test length(ram.centers) >= 1

        hits = MirnanNew.Physics.RAMCore.resonate(ram, compute_extended_phase_vector("علم"))
        @test !isempty(hits)
    end

    @testset "Phase Reinforcement" begin
        pr = MirnanNew.Physics.PhaseReinforcement.PhaseReinforcer()
        pv = compute_extended_phase_vector("علم")
        MirnanNew.Physics.PhaseReinforcement.reinforce!(pr, "علم", pv; reward=1.0)
        @test MirnanNew.Physics.PhaseReinforcement.get_strength(pr, "علم") > 0.0
    end

    @testset "DCCF" begin
        dccf = MirnanNew.Physics.DCCF.DynamicCCF()
        words = ["علم", "نور", "حياة"]
        boost = MirnanNew.Physics.DCCF.get_context_boost(dccf, "علم", words)
        @test boost >= 0.0
    end

    @testset "PPM" begin
        pf = MirnanNew.Physics.PPM.PromptField()
        MirnanNew.Physics.PPM.absorb!(pf, "العلم نور")
        @test pf.active
        pv = compute_extended_phase_vector("علم")
        s = MirnanNew.Physics.PPM.score(pf, pv)
        @test s >= 0.0
    end

    @testset "AMFS" begin
        result = MirnanNew.Physics.AMFS.adapt_word("علم"; context_words=["نور", "حياة"])
        @test result["mass"] > 0.0
        @test result["freq"] > 0.0
        @test haskey(result, "centrality")
    end

    @testset "Density Matrix" begin
        dm = MirnanNew.Physics.DensityMatrix.PhaseDensityMatrix()
        pvs = [Float64.(compute_extended_phase_vector(w)) for w in ["علم", "نور", "حياة"]]
        MirnanNew.Physics.DensityMatrix.build!(dm, pvs; dims=MirnanNew.Physics.Constants.TOTAL_DIM)
        @test dm.rho !== nothing
        trace = MirnanNew.Physics.DensityMatrix.get_trace(dm)
        @test abs(trace - 1.0) < 0.1

        res = MirnanNew.Physics.DensityMatrix.resonance(dm, pvs[1])
        @test res >= 0.0 && res <= 1.0
    end

    @testset "Causal Flow" begin
        cf = MirnanNew.Physics.CausalFlow.CausalFlowField()
        K_dummy = spzeros(10, 10)
        K_dummy[1, 2] = 0.8
        K_dummy[2, 3] = 0.6

        flow = MirnanNew.Physics.CausalFlow.compute_flow(
            cf, zeros(64), [zeros(64) for _ in 1:3], [1, 2, 3];
            causal_matrix=K_dummy,
        )
        @test haskey(flow, "flow_magnitude")
        @test haskey(flow, "logical_score")
    end

    @testset "Generator" begin
        vocab = Dict(
            "ال" => 1, "في" => 2, "من" => 3, "علم" => 4, "نور" => 5,
            "حياة" => 6, "كبير" => 7, "سماء" => 8, "أرض" => 9, "ماء" => 10,
            "الله" => 11, "سلام" => 12, "عالم" => 13, "كتاب" => 14, "قلب" => 15,
            "يد" => 16, "بيت" => 17, "حق" => 18, "جديد" => 19, "طريق" => 20,
        )
        K_sem = spzeros(20, 20)
        for (i, j) in [(4, 5), (4, 6), (4, 13), (5, 13), (6, 10)]
            K_sem[i, j] = 0.5
            K_sem[j, i] = 0.5
        end

        gen = MirnanGenerator(vocab, K_sem)

        # وضع قياسي
        result = generate!(gen, "العلم نور"; mode="standard", max_words=4)
        @test result isa String

        # وضع إبداعي
        result_creative = generate!(gen, "العلم نور"; mode="creative", max_words=4)
        @test result_creative isa String

        # تقرير فيزيائي
        report = get_physics_report(gen, ["علم", "نور"])
        @test haskey(report, "entropy")
        @test haskey(report, "mass_mean")
        @test haskey(report, "k_causal_density")
        @test report["word_count"] == 2

        K_causal = spzeros(20, 20)
        K_causal[4, 5] = 0.9
        causal_gen = MirnanGenerator(vocab, K_sem; K_causal=K_causal)
        causal_report = get_physics_report(causal_gen, ["علم", "نور"])
        @test causal_report["k_causal_density"] > 0.0
    end

    @testset "Generator Edge Cases" begin
        vocab = Dict("علم" => 1, "نور" => 2)
        K_sem = spzeros(2, 2)
        K_sem[1, 2] = 0.5
        K_sem[2, 1] = 0.5
        gen = MirnanGenerator(vocab, K_sem)

        # أمر فارغ
        result = generate!(gen, ""; max_words=3)
        @test result == ""

        # كلمة ناقصة
        result = generate!(gen, "xyz"; max_words=2)
        @test result isa String
    end

    @testset "Contextual Learning" begin
        vocab = Dict(
            "جيش" => 1, "الليل" => 2, "زحف" => 3, "غطى" => 4,
            "الظلام" => 5, "السكون" => 6, "قبيلة" => 7,
        )
        gen = MirnanGenerator(vocab)

        answer = generate!(gen, "ماذا فعل جيش الليل"; max_words=6)
        @test occursin("جيش الليل", answer)
        @test occursin("زحف", answer)
        @test occursin("غطى", answer)

        MirnanNew.Physics.learn_from_feedback!(
            gen,
            "ماذا فعل جيش الليل",
            "جيش الليل قبيلة في الرواية";
            rating=1.0,
            note="جيش الليل اسم قبيلة في روايتي",
        )
        entity_answer = generate!(gen, "ماذا فعل جيش الليل"; max_words=6)
        @test occursin("كيان", entity_answer)
        @test occursin("سياق", entity_answer)
    end

    @testset "Word Fusion Engine" begin
        db = MirnanNew.Physics.LetterDB.LetterDatabase()
        lvs = MirnanNew.Physics.WordFusion._get_letter_vectors("علم", db)
        @test !isempty(lvs)

        lin = MirnanNew.Physics.WordFusion.linear_fusion(lvs)
        @test norm(lin) > 0.0

        geo = MirnanNew.Physics.WordFusion.geometric_word_fusion(lvs)
        @test MirnanNew.Physics.CliffordMath.norm(geo) > 0.0

        dec = MirnanNew.Physics.WordFusion.decompose_fusion(geo)
        @test haskey(dec, "shared_essence")
    end

    @testset "Semantic Arithmetic" begin
        @test haskey(MirnanNew.Physics.SemanticArithmetic.FAST_ANTONYMS, "نور")
        @test MirnanNew.Physics.SemanticArithmetic.FAST_ANTONYMS["نور"] == "ظلام"

        field = MirnanNew.Physics.SemanticArithmetic.classify_word_field("نور")
        @test field in ("epistemology", "sensory", "state")

        operators = MirnanNew.Physics.SemanticArithmetic.build_general_antinomy_operator([("نور","ظلام"), ("علم","جهل")])
        @test !isempty(operators)

        # Build field-specific operators
        fl_ops = MirnanNew.Physics.SemanticArithmetic.build_field_specific_antinomy_operators(
            [("نور","ظلام"), ("علم","جهل"), ("قوة","ضعف")])

        # analyze semantic affinity
        aff = MirnanNew.Physics.SemanticArithmetic.analyze_semantic_affinity("علم", "جهل")
        @test haskey(aff, "affinity_class")
    end

    @testset "MathBridge Safe Evaluation" begin
        mb = MirnanNew.Physics.MathBridgeModule.MathBridge()
        @test MirnanNew.Physics.MathBridgeModule.evaluate_math(mb, "2 + 3 * 4") == 14.0
        @test MirnanNew.Physics.MathBridgeModule.evaluate_math(mb, "(2 + 3) * 4") == 20.0
        @test MirnanNew.Physics.MathBridgeModule.evaluate_math(mb, "2 ^ 3") == 8.0
        @test MirnanNew.Physics.MathBridgeModule.evaluate_math(mb, "2 / 0") === nothing
        @test MirnanNew.Physics.MathBridgeModule.evaluate_math(mb, "2 ^ 100") === nothing
    end

    @testset "API Generation Uses MirnanGenerator" begin
        api_gen = MirnanGenerator(Dict("alpha" => 1, "beta" => 2, "gamma" => 3, "delta" => 4))
        MirnanNew.APIServer.set_generator!(api_gen)
        generated = MirnanNew.APIServer._generate_text("alpha", 3)
        @test generated isa String
        @test !isempty(strip(generated))
        @test !startswith(generated, "Generated text based on:")
    end

    @testset "CodeEngine K_code Roundtrip" begin
        code_samples = ["def foo ( ) :\n    return 1"]
        code_vocab, K_code, cpv = MirnanNew.Physics.CodeEngineModule.build_K_code(code_samples)
        ce = MirnanNew.Physics.CodeEngineModule.CodeEngine(code_vocab=code_vocab, K_code=K_code, cpv=cpv)
        dir = mktempdir()
        @test MirnanNew.Physics.CodeEngineModule.save_code_model(ce, dir)
        loaded = MirnanNew.Physics.CodeEngineModule.load_code_model(dir)
        @test loaded.code_vocab !== nothing
        @test loaded.code_vocab.next > 1
        @test nnz(loaded.K_code) > 0
    end

    @testset "Generator Al-Aql Message Bridge" begin
        vocab = Dict("القاهرة" => 1, "في" => 2, "مصر" => 3, "العلاقة" => 4)
        gen = MirnanGenerator(vocab)
        learned_response = generate!(gen, "القاهرة في مصر"; max_words=2)
        @test learned_response isa String

        answer = generate!(gen, "ما علاقة القاهرة بمصر؟"; max_words=2)
        @test occursin("فضاء العقل", answer)
        @test occursin("القاهرة", answer)
        @test occursin("مصر", answer)
    end

    @testset "Generator Al-Aql Leads Meaning" begin
        vocab = Dict(
            "ملح" => 1, "ماء" => 2, "إضافة" => 3, "الغليان" => 4,
            "زاد" => 5, "العدل" => 6, "سماع" => 7, "حكم" => 8,
            "نتيجة" => 9, "درجة_الغليان_زادت" => 10,
        )
        gen = MirnanGenerator(vocab)
        compile_adl!(gen.aql_space, """
        شيء ملح { kind: مادة }
        شيء ماء { kind: مادة }
        نمط رفع_الغليان { فعل: إضافة, فاعل_النتيجة: الهدف, نتيجة: درجة_الغليان_زادت, ثقة: 0.9 }
        عملية العدل { مجال: أخلاق, خطوة: source يسمع target -> سماع_الأطراف, خطوة: source يحكم target -> إعطاء_الحق }
        """)

        event_answer = generate!(gen, "ماذا يحدث إذا إضافة ملح إلى ماء؟"; max_words=4)
        @test occursin("فضاء العقل", event_answer)
        @test occursin("درجة الغليان زادت", event_answer)

        process_answer = generate!(gen, "ما معنى العدل؟"; max_words=4)
        @test occursin("عملية", process_answer)
        @test occursin("يسمع", process_answer)

        gen.aql_bias = MirnanNew.Physics.Generator._aql_bias_from_prompt(gen, "إضافة ملح إلى ماء")
        prompted = [MirnanNew.Physics.Generator._pv(gen, "إضافة"),
                    MirnanNew.Physics.Generator._pv(gen, "ملح"),
                    MirnanNew.Physics.Generator._pv(gen, "ماء")]
        guided_score, _ = MirnanNew.Physics.Generator._score(
            gen, "درجة_الغليان_زادت", Set(["إضافة", "ملح", "ماء"]),
            prompted, prompted; prev_word="ماء", context_words=["إضافة", "ملح", "ماء"])
        empty!(gen.aql_bias)
        unguided_score, _ = MirnanNew.Physics.Generator._score(
            gen, "درجة_الغليان_زادت", Set(["إضافة", "ملح", "ماء"]),
            prompted, prompted; prev_word="ماء", context_words=["إضافة", "ملح", "ماء"])
        @test guided_score > unguided_score
    end

    @testset "Letter Physics Lexical Oracle" begin
        vocab = Dict(
            "نور" => 1, "نار" => 2, "ماء" => 3, "علم" => 4,
            "ظلام" => 5, "سلام" => 6, "منير" => 7, "نوار" => 8,
        )

        profile = word_physics_profile("نوار")
        @test profile.word == "نوار"
        @test profile.phase_norm > 0.0
        @test profile.frequency > 0.0

        quality = phonosemantic_quality("نوار")
        @test haskey(quality, "aesthetic_score")
        @test 0.0 <= quality["aesthetic_score"] <= 1.0

        neighbors = nearest_phase_words("نوار", vocab; top_k=3)
        @test !isempty(neighbors)
        @test neighbors[1].hybrid_score >= 0.0

        analysis = analyze_unknown_word("نوارية", vocab; top_k=3)
        @test haskey(analysis, "nearest_words")
        @test haskey(analysis, "profile")

        coined = coin_word_for_concept(["نور", "علم"]; vocab=vocab, max_candidates=5)
        @test !isempty(coined)
        @test coined[1].score >= 0.0

        gen = MirnanGenerator(vocab)
        answer = generate!(gen, "حلل كلمة نوارية"; max_words=2)
        @test occursin("فيزياء الحرف", answer)
        @test occursin("أقرب كلمات", answer)
    end

    @testset "Runtime Learning Persistence" begin
        dir = mktempdir()
        vocab = Dict("القاهرة" => 1, "في" => 2, "مصر" => 3, "العلاقة" => 4)
        gen = MirnanGenerator(vocab; model_dir=dir)
        generate!(gen, "القاهرة في مصر"; max_words=2)
        compile_adl!(gen.aql_space, """
        شيء ملح { kind: مادة }
        شيء ماء { kind: مادة }
        نمط رفع_الغليان { فعل: إضافة, فاعل_النتيجة: الهدف, نتيجة: درجة_الغليان_زادت, ثقة: 0.9 }
        عملية العدل { مجال: أخلاق, خطوة: source يسمع target -> سماع_الأطراف }
        """)
        saved_path = save_runtime_learning!(gen)
        @test isfile(saved_path)

        restored = MirnanGenerator(vocab; model_dir=dir)
        answer = generate!(restored, "ما علاقة القاهرة بمصر؟"; max_words=2)
        @test occursin("فضاء العقل", answer)
        @test occursin("القاهرة", answer)
        @test occursin("مصر", answer)
        @test occursin("درجة الغليان زادت", generate!(restored, "ماذا يحدث إذا إضافة ملح إلى ماء؟"; max_words=2))
        @test occursin("عملية", generate!(restored, "ما معنى العدل؟"; max_words=2))
    end

    @testset "Al-Aql Governance and Inhibition" begin
        dir = mktempdir()
        vocab = Dict("حديد" => 1, "حرارة" => 2, "تمدد" => 3,
                     "ضغط" => 4, "مقيد" => 5, "حر" => 6,
                     "يسبب" => 7, "النار" => 8)
        gen = MirnanGenerator(vocab; model_dir=dir)
        rule = curate_rule!(gen.aql_space, "حديد مقيد", "ينتج", "تمدد";
                            polarity=-1, confidence=1.0,
                            tags=["تمدد"], id="no_expand_when_bound")
        @test haskey(gen.aql_space.curated_rules, rule.id)

        gen.aql_inhibition = MirnanNew.Physics.Generator._aql_inhibition_from_prompt(
            gen, "حديد مقيد مع حرارة")
        @test MirnanNew.Physics.Generator._aql_candidate_inhibition(gen, "تمدد") > 0.95

        ctx = ["حديد", "مقيد", "حرارة"]
        ctx_pv = [MirnanNew.Physics.Generator._pv(gen, w) for w in ctx]
        inhibited_score, _ = MirnanNew.Physics.Generator._score(
            gen, "تمدد", Set(ctx), ctx_pv, ctx_pv;
            prev_word="حرارة", context_words=ctx)
        empty!(gen.aql_inhibition)
        free_score, _ = MirnanNew.Physics.Generator._score(
            gen, "تمدد", Set(ctx), ctx_pv, ctx_pv;
            prev_word="حرارة", context_words=ctx)
        @test inhibited_score < free_score

        lesson = learn_rejection_lesson!(gen.aql_space, "causal_direction_error";
                                         pattern="يسبب", penalty_tags=["causal"],
                                         confidence=1.0)
        proposed = propose_rule!(gen.aql_space, "الدخان", "يسبب", "النار";
                                 confidence=0.9, tags=["causal"],
                                 id="smoke_causes_fire")
        @test proposed.status == "rejected"
        @test haskey(gen.aql_space.rejected_rules, proposed.id)
        @test !isempty(lesson.id)

        n = critical_corpus_pass!(gen.aql_space,
            ["إذا زادت الحرارة فإن الحديد يتمدد إلا إذا كان مقيداً",
             "جيش الليل زحف"])
        @test n >= 3
        tags = Set(a.tag for a in gen.aql_space.corpus_annotations)
        @test "causal" in tags
        @test "requires_condition" in tags
        @test "metaphoric" in tags
        audit = aql_memory_audit(gen.aql_space)
        @test audit["annotations"] == length(gen.aql_space.corpus_annotations)
        @test annotation_weight(gen.aql_space, "runtime:1"; matrix=:causal) < 1.0

        saved_path = save_runtime_learning!(gen)
        @test isfile(saved_path)
        restored = MirnanGenerator(vocab; model_dir=dir)
        @test haskey(restored.aql_space.curated_rules, "no_expand_when_bound")
        @test haskey(restored.aql_space.rejected_rules, "smoke_causes_fire")
        @test any(a -> a.tag == "metaphoric", restored.aql_space.corpus_annotations)
    end

    @testset "Weight Resonance" begin
        eng = MirnanNew.Physics.WeightResonance.WeightResonanceEngine()
        pv = MirnanNew.Physics.WeightResonance.get_weight_pv(eng, "فَعَلَ")
        @test length(pv) == 22

        res = MirnanNew.Physics.WeightResonance.resonance(eng, "فَعَلَ", "مَفْعُول")
        @test res >= -1.0 && res <= 1.0

        ts = MirnanNew.Physics.WeightResonance.transition_score(eng, "فَعَلَ", "مَفْعُول")
        @test ts >= 0.0
    end

    @testset "Syntax Field" begin
        sv = MirnanNew.Physics.SyntaxField.compute_syntax_vector("كان")
        @test length(sv) == MirnanNew.Physics.Constants.SYNTAX_DIMS
    end

    @testset "MorphoPhasic" begin
        eng = MirnanNew.Physics.MorphoPhasic.MorphoPhasicEngine()
        analysis = MirnanNew.Physics.MorphoPhasic.analyze_morpho(eng, "عالم")
        @test haskey(analysis, "root")
        @test haskey(analysis, "weight")
        @test haskey(analysis, "has_morph")
    end

    @testset "Dialogue Engine" begin
        eng = MirnanNew.Physics.DialogueEngineFull.DialogueEngine()
        pv = compute_extended_phase_vector("سلام")
        score = MirnanNew.Physics.DialogueEngineFull.compute_dialogue_score(
            eng, pv, "وعليكم"; context_words=["سلام"])
        @test score > 0.0  # starter bonus
    end

    @testset "Poetic Gravity" begin
        syl = MirnanNew.Physics.PoeticGravity.syllabify_arabic("سلام")
        @test !isempty(syl)

        count = MirnanNew.Physics.PoeticGravity.count_syllables_english("hello")
        @test count >= 1

        meter = MirnanNew.Physics.PoeticGravity.detect_meter(["سلام", "عليكم"])
        @test meter isa String
    end

    @testset "Contextual Operators" begin
        eng = MirnanNew.Physics.ContextualOperators.ContextualOperatorEngine()
        MirnanNew.Physics.ContextualOperators.init_builtin!(eng)

        @test MirnanNew.Physics.ContextualOperators.has_operator(eng, "لا")
        chain = MirnanNew.Physics.ContextualOperators.detect_chain(eng, ["غياب", "الشمس"])
        @test !isempty(chain)

        pv = compute_word_phase_vector("شمس")
        adjusted, boost = MirnanNew.Physics.ContextualOperators.apply(eng, pv, ["غياب", "الشمس"])
        @test boost >= 0.0
    end

    @testset "Spectral Wave Engine" begin
        ww = MirnanNew.Physics.SpectralWave.compute_word_wave("سلام")
        @test haskey(ww, "omega_0")
        @test ww["amplitude"] > 0.0

        val = MirnanNew.Physics.SpectralWave.wave_at(ww, 0.5)
        @test val isa Float64

        freqs, mags, phases = MirnanNew.Physics.SpectralWave.wave_spectrum(ww)
        @test length(mags) > 0
    end

    @testset "Kuramoto Oscillator & Damping Physics" begin
        using MirnanNew.Physics.KuramotoOscillator: OscillatorEngine, simulate, _derivative
        
        # 1. تهيئة المحرك بالمعاملات الافتراضية والتحقق من التخميد
        engine = OscillatorEngine()
        @test engine.dim == 27
        @test engine.damping_coeff == 0.1
        
        # 2. إعداد قيم الاختبار
        n = 3
        omega = [0.1, 0.2, 0.3]
        phases = rand(n, engine.dim) .* 2*pi .- pi
        masses = [1.0, 1.5, 2.0]
        pvs = [rand(engine.dim) for _ in 1:n]
        coupling_sub = [1.0 0.5 0.2; 0.5 1.0 0.4; 0.2 0.4 1.0]
        
        # 3. تشغيل الاشتقاق وفحص وجود التخميد والجاذبية
        dphi = _derivative(engine, omega, phases, masses, pvs, coupling_sub; temperature=0.1)
        @test size(dphi) == (n, engine.dim)
        @test all(isfinite, dphi)
        
        # 4. التحقق من قص المطال (Amplitude Clipping)
        # نضع قيم ضخمة جداً لتفعيل الـ clipping
        phases_large = phases .* 1000.0
        dphi_large = _derivative(engine, omega, phases_large, masses, pvs, coupling_sub; temperature=0.0)
        for i in 1:n
            @test norm(dphi_large[i, :]) <= 50.0 + 1e-9
        end
        
        # 5. تشغيل محاكاة كاملة
        phases_final, history = simulate(engine, omega, phases, masses, pvs, coupling_sub; dt=0.01, steps=10, temperature=0.05)
        @test size(phases_final) == (n, engine.dim)
        @test size(history) == (11, n, engine.dim)
    end

    @testset "Cerebellum PID Feedback Control" begin
        using MirnanNew.Physics.MirnanCerebellumModule: PIDController, MirnanCerebellum,
              correct!, record_error!, update_integral!, reset_pid!
        
        # 1. اختبار تهيئة PID ومعامِلاته الافتراضية
        pid = PIDController()
        @test pid.Kp == 0.6
        @test pid.Ki == 0.0
        @test pid.Kd == 0.0
        @test pid.setpoint == 0.5
        @test pid.integral == 0.0
        
        # 2. اختبار تسجيل الخطأ والتكامل
        record_error!(pid, 0.2)
        @test length(pid.error_history) == 1
        @test pid.error_history[1] == 0.2
        
        update_integral!(pid)
        @test pid.integral == 0.2
        
        # 3. اختبار تكامل مانع التراكم (anti-windup clamping ±10)
        reset_pid!(pid)
        @test pid.integral == 0.0
        @test isempty(pid.error_history)
        
        for _ in 1:15
            record_error!(pid, 2.0)
            update_integral!(pid)
        end
        # يجب ألا يتجاوز 10.0 بفضل الـ anti-windup clamping
        @test pid.integral == 10.0
        
        # 4. اختبار تصحيح الأوزان (correct!)
        weights = Dict("align" => 5.0, "syntax" => 5.5, "diversity" => 3.0)
        signal = Dict("target" => 0.5, "current" => 0.3) # الخطأ = 0.2
        
        # تصفير المذبذب للاختبار النقي لـ Kp
        pid_p = PIDController(Kp=0.5, Ki=0.0, Kd=0.0, setpoint=0.5)
        out = correct!(pid_p, weights, signal)
        @test out == 0.5 * 0.2 # Kp * err
        # عامل التصحيح = 1.0 + 0.1 * out = 1.0 + 0.1 * 0.1 = 1.01
        @test weights["align"] ≈ 5.05 atol=1e-5
        @test weights["syntax"] ≈ 5.555 atol=1e-5
        
        # 5. اختبار الـ Toggle لتعطيل الـ PID في المخيخ
        cb = MirnanCerebellum(pid_enabled=false)
        @test cb.pid_enabled == false
    end

    @testset "Continuous Wave Oscillator & Phase Coupling Physics" begin
        # 1. اختبار انشطار الموجة (split_wave)
        v_test = [1.0 + 0.0im, 0.0 + 1.0im, -1.0 + 0.0im, 0.0 - 1.0im]
        theta = pi/4
        phi_shift = pi/2
        
        v1, v2, v_anti = MirnanNew.Physics.ChaosEntanglement.split_wave(v_test, theta, phi_shift)
        
        @test length(v1) == 4
        @test length(v2) == 4
        @test length(v_anti) == 4
        
        @test abs(norm(v1)^2 + norm(v2)^2 - norm(v_test)^2) < 1e-10
        @test norm(v1 + v_anti) < 1e-10
        
        # 2. اختبار دمج الموجات والتداخل (merge_waves)
        w_constructive = MirnanNew.Physics.ChaosEntanglement.merge_waves(v1, v2; phase_shift=0.0)
        w_destructive = MirnanNew.Physics.ChaosEntanglement.merge_waves(v1, v2; phase_shift=pi)
        
        @test abs(norm(w_constructive) - 1.0) < 1e-6
        @test abs(norm(w_destructive) - 1.0) < 1e-6
        
        # 3. اختبار التشابك والقياس (couple_waves & project_coupled_wave)
        v_a = [1.0 + 0.0im, 0.0 + 1.0im]
        v_b = [0.0 + 1.0im, 1.0 + 0.0im]
        
        E = MirnanNew.Physics.ChaosEntanglement.couple_waves(v_a, v_b)
        @test size(E) == (2, 2)
        
        v_measured = MirnanNew.Physics.ChaosEntanglement.project_coupled_wave(E, v_b)
        @test length(v_measured) == 2
        @test abs(abs(dot(v_measured, v_a) / (norm(v_measured) * norm(v_a))) - 1.0) < 1e-6
    end

    @testset "Al-Aql Cognitive Causal Engine" begin
        space = SimulationSpace()
        
        # Scenario 1: Lion and Gazelle
        أسد = Lion("الأسد", 2.0)
        set_property!(أسد, "energy", 0.8, 0.0)
        
        غزال = Gazelle("الغزال", 0.6)
        set_property!(غزال, "fear", 0.1, 0.0)
        set_property!(غزال, "awareness", 0.7, 0.0)
        set_property!(غزال, "motion", 0.1, 0.0)
        
        register_entity!(space, أسد)
        register_entity!(space, غزال)
        
        interact!(space, "الأسد", roar!, "الغزال")
        
        fear_amp, _ = get_property(غزال, "fear")
        motion_amp, _ = get_property(غزال, "motion")
        @test isapprox(fear_amp, 0.1)
        @test isapprox(motion_amp, 0.8)
        
        # Scenario 2: Earth and Furniture
        أرض = Earth("الأرض", 10.0)
        set_property!(أرض, "energy", 0.9, 0.0)
        
        أثاث = Furniture("الأثاث", 1.2)
        set_property!(أثاث, "stability", 0.95, 0.0)
        set_property!(أثاث, "integrity", 0.9, 0.0)
        set_property!(أثاث, "motion", 0.0, 0.0)
        
        register_entity!(space, أرض)
        register_entity!(space, أثاث)
        
        interact!(space, "الأرض", shake!, "الأثاث")
        
        stability_amp, _ = get_property(أثاث, "stability")
        motion_amp_f, _ = get_property(أثاث, "motion")
        integrity_amp, _ = get_property(أثاث, "integrity")
        @test isapprox(stability_amp, 0.0)
        @test isapprox(motion_amp_f, 0.7)
        @test isapprox(integrity_amp, 0.9 * 0.6)

        # Scenario 3: Generic thing/action meaning frame
        وردة = Thing("وردة"; kind="plant", attributes=Dict{String,Any}(
            "color" => "أحمر",
            "scent" => "فواحة",
            "beauty" => 0.8,
        ))
        حشرة = Thing("حشرة مفيدة"; kind="insect")
        register_action!(وردة, "إرسال رائحة", emit_scent!)
        register_action!(وردة, "جلب الحشرات المفيدة", attract_beneficial_insects!)

        register_entity!(space, وردة)
        register_entity!(space, حشرة)
        @test interact!(space, "وردة", "إرسال رائحة", "حشرة مفيدة"; strength=0.7)
        @test interact!(space, "وردة", "جلب الحشرات المفيدة", "حشرة مفيدة"; strength=0.5)

        motion_insect, _ = get_property(حشرة, "motion")
        awareness_insect, _ = get_property(حشرة, "awareness")
        @test get_attribute(حشرة, "attracted_to") == "وردة"
        @test isapprox(motion_insect, 0.5)
        @test isapprox(awareness_insect, 0.7)
        @test any(occursin("فكرة", line) for line in space.log)
    end

    @testset "Idea Engine" begin
        idea = Idea(["خالد", "أحمد"], "ضرب", "رد")
        idea_vector = compute_idea_vector(idea, compute_extended_phase_vector)

        @test length(idea_vector) == MirnanNew.Physics.Constants.TOTAL_DIM
        @test norm(idea_vector[1:MirnanNew.Physics.Constants.PHASE_DIM]) > 0.0

        matched = match_sentence_to_idea(
            ["خالد", "ضرب", "أحمد", "فرد", "أحمد"],
            idea,
            compute_extended_phase_vector,
        )
        candidate_score = score_candidate_for_idea(
            "رد",
            ["خالد", "ضرب", "أحمد"],
            idea,
            compute_extended_phase_vector,
        )

        @test 0.0 <= matched <= 1.0
        @test 0.0 <= candidate_score <= 1.0
    end

    @testset "Al-Aql ADL" begin
        space = SimulationSpace()
        compile_adl!(space, """
        أسد : كائن { كتلة: 2.0, طاقة: 0.9 }
        غزال : كائن { كتلة: 0.6, خوف: 0.1, حركة: 0.2 }
        زأر : فعل { الهدف: خوف, مقياس: 5.0 }
        هرب : فعل { الهدف: حركة, مقياس: 6.0; الهدف: خوف, مقياس: -2.0 }
        قاعدة : خوف > 0.5 -> هرب
        فكرة : أسد زأر غزال
        """)

        @test haskey(space.entities, "أسد")
        @test haskey(space.dynamic_verbs, "زأر")
        @test length(space.rules) == 1

        fear, _ = get_property(space.entities["غزال"], "fear")
        motion, _ = get_property(space.entities["غزال"], "motion")
        @test fear <= 1.0
        @test motion > 0.2
        @test any(occursin("فكرة", line) for line in space.log)
    end

    @testset "Al-Aql ADL Taxonomy Syntax" begin
        space = SimulationSpace()
        compile_adl!(space, """
        مفتاح صياد < إنسان { غذاء: سمك, أعضاء: الصياد }
        تصنيف خالد : صياد
        فارس : صياد { مهارة: صيد, طاقة: 0.6 }
        """)

        @test "صياد" in known_classes(space)
        @test is_a(space.taxonomy, "صياد", "إنسان")
        @test haskey(space.entities, "خالد")
        @test is_a(space, "خالد", "صياد")
        @test is_a(space, "فارس", "حيوان")
        @test get_attribute(space.entities["فارس"], "مهارة") == "صيد"

        energy, _ = get_property(space.entities["فارس"], "energy")
        @test energy >= 0.6
    end

    @testset "Al-Aql Cross-Domain Causal Templates" begin
        space = SimulationSpace()
        compile_adl!(space, """
        مفتاح كتلة < شيء { مجال: فيزياء }
        مفتاح شحنة < شيء { مجال: كهرباء }
        مفتاح شحنة_موجبة < شحنة { إشارة: موجبة }
        مفتاح متكلم < إنسان { قدرة: قول }
        مفتاح مخاطب < إنسان { قدرة: رد }

        كتلة1 : كتلة { مقدار: 1.0 }
        كتلة2 : كتلة { مقدار: 2.0 }
        شحنة1 : شحنة_موجبة { مقدار: 1.0 }
        شحنة2 : شحنة_موجبة { مقدار: 1.0 }
        سالم : متكلم { }
        محمود : مخاطب { }

        نمط تجاذب_الكتل { مجال: فيزياء, مصدر: كتلة, فعل: تجاذب, هدف: كتلة, فاعل_النتيجة: كلاهما, نتيجة: تقارب ثم التحام }
        نمط تنافر_الشحنات { مجال: كهرباء, مصدر: شحنة_موجبة, فعل: قرب, هدف: شحنة_موجبة, فاعل_النتيجة: كلاهما, نتيجة: تنافر }
        نمط رد_السلام { مجال: حوار, مصدر: متكلم, فعل: قال, نوع: السلام_عليكم, هدف: مخاطب, فاعل_النتيجة: الهدف, فعل_النتيجة: قال, هدف_النتيجة: المصدر, نوع_النتيجة: وعليكم_السلام }
        """)

        mass_frames = infer_event!(space, "كتلة1", "تجاذب", "كتلة2")
        charge_frames = infer_event!(space, "شحنة1", "قرب", "شحنة2")
        greeting_frames = infer_event!(space, "سالم", "قال", "محمود"; kind="السلام_عليكم")

        @test length(space.templates) == 3
        @test occursin("تقارب", mass_frames[1].result)
        @test occursin("تنافر", charge_frames[1].result)
        @test occursin("وعليكم_السلام", greeting_frames[1].result)
    end

    @testset "Al-Aql Conditional Probability Templates" begin
        space = SimulationSpace()
        compile_adl!(space, """
        مفتاح شحنة < شيء { مجال: كهرباء }
        مفتاح شحنة_موجبة < شحنة { إشارة: موجبة }
        شحنة1 : شحنة_موجبة { distance: 0.8 }
        شحنة2 : شحنة_موجبة { distance: 0.8 }
        نمط مشروط تنافر_قريب { مجال: كهرباء, مصدر: شحنة_موجبة, فعل: قرب, هدف: شحنة_موجبة, شرط: source.distance < 0.5, فاعل_النتيجة: كلاهما, نتيجة: تنافر, ثقة: 0.95 }
        """)

        @test isempty(infer_event!(space, "شحنة1", "قرب", "شحنة2"))
        set_attribute!(space.entities["شحنة1"], "distance", 0.3)
        frames = infer_event!(space, "شحنة1", "قرب", "شحنة2")

        @test length(frames) == 1
        @test occursin("تنافر", frames[1].result)
        @test occursin("confidence=0.95", frames[1].result)
        @test occursin("condition=source.distance < 0.5", frames[1].result)
    end

    @testset "Al-Aql Template Result Action Application" begin
        space = SimulationSpace()
        compile_adl!(space, """
        مفتاح مادة < شيء { }
        ماء : مادة { boiling_point: 100.0 }
        ملح : مادة { quantity: 1.0 }
        رفع_الغليان : فعل { الهدف: boiling_point, مقياس: 2.0 }
        علاقة boiling_point خاصية_لـ ماء
        نمط مشروط اضافة_ملح_ترفع_الغليان { مجال: كيمياء, مصدر: مادة, فعل: اضافة, هدف: مادة, شرط: source.quantity > 0, فاعل_النتيجة: الهدف, فعل_النتيجة: رفع_الغليان, نتيجة: زيادة_درجة_الغليان, ثقة: 0.9 }
        """)

        before = get_attribute(space.entities["ماء"], "boiling_point")
        frames = infer_event!(space, "ملح", "اضافة", "ماء")
        after = get_attribute(space.entities["ماء"], "boiling_point")

        @test length(frames) == 1
        @test after > before
        @test any(occursin("template result applied", line) for line in space.log)
        @test any(rel -> rel.source == "boiling_point" && rel.relation == "خاصية_لـ" && rel.target == "ماء",
                  relations_from(space, "boiling_point"))
    end

    @testset "Al-Aql Extracts Causal Increase From Text" begin
        space = SimulationSpace()
        train_from_text!(space, "إضافة كمية من ملح إلى الماء فإنها تزيد من درجة غليانه.")

        @test haskey(space.entities, "ملح")
        @test haskey(space.entities, "ماء")
        @test haskey(space.dynamic_verbs, "رفع_درجة_الغليان")
        frames = infer_event!(space, "ملح", "إضافة", "ماء")
        @test !isempty(frames)
        @test occursin("زيادة_درجة_الغليان", frames[1].result)
    end

    @testset "Al-Aql English ADL Syntax" begin
        space = SimulationSpace()
        compile_adl!(space, """
        class Material < thing { }
        water : Material { boiling_point: 100.0 }
        salt : Material { quantity: 1.0 }
        verb raise_boiling { target: boiling_point, scale: 2.0 }
        relation boiling_point property_of water
        template conditional salt_raises_boiling {
          domain: chemistry,
          source: Material,
          action: add,
          target: Material,
          condition: source.quantity > 0,
          result_actor: target,
          result_action: raise_boiling,
          result_state: boiling_point_increased,
          confidence: 0.9
        }
        class Service < thing { }
        class Database < thing { }
        api : Service { latency: 0.3 }
        db : Database { latency: 0.6 }
        relation api calls db
        quantifier all Service has endpoint
        comparison db greater than api in latency
        intent { actor: api, action: query, target: db, intent: fetch_user, goal: return_profile, actual_result: rows_loaded }
        exception { rule: cache_read, condition: cache.enabled == true, exception: cache_miss, priority: 5 }
        metaphor { expression: data flows through pipeline, source_domain: river, target_domain: software, literal_subject: data, borrowed_actor: river, action: flows, transferred_property: movement }
        """)

        before = get_attribute(space.entities["water"], "boiling_point")
        frames = infer_event!(space, "salt", "add", "water")
        after = get_attribute(space.entities["water"], "boiling_point")

        @test length(frames) == 1
        @test after > before
        @test any(rel -> rel.source == "api" && rel.relation == "calls" && rel.target == "db",
                  relations_from(space, "api"))
        @test quantified_facts_for(space, "Service")[1].quantifier == "all"
        @test comparisons_for(space, "db")[1].comparator == "greater"
        @test intents_for(space, "api")[1].goal == "return_profile"
        @test active_exception(space, "cache_read").exception == "cache_miss"
        @test metaphors_for(space, "data")[1].target_domain == "software"
    end

    @testset "Al-Aql Compound Sequential Events" begin
        space = SimulationSpace()
        compile_adl!(space, """
        مفتاح إنسان_اختبار < إنسان { }
        خالد : إنسان_اختبار { }
        أحمد : إنسان_اختبار { }
        سلسلة ضرب_ثم_نتائج { مصدر: إنسان_اختبار, فعل: ضرب, هدف: إنسان_اختبار, ثقة: 0.9; خطوة: target هرب source -> هروب_المضروب; خطوة: الكتاب سقط الأرض -> سقوط_تابع }
        """)

        frames = run_event_chain!(space, "ضرب_ثم_نتائج", "خالد", "أحمد")

        @test haskey(space.event_chains, "ضرب_ثم_نتائج")
        @test length(frames) == 2
        @test frames[1].things == ["أحمد", "خالد"]
        @test frames[1].event == "هرب"
        @test occursin("هروب_المضروب", frames[1].result)
        @test occursin("confidence=0.9", frames[1].result)
        @test frames[2].things == ["الكتاب", "الأرض"]
        @test frames[2].event == "سقط"
        @test haskey(space.entities, "الكتاب")
        @test haskey(space.entities, "الأرض")
        @test length(space.temporal_relations) == 2
        @test space.temporal_relations[1].relation == "بعد"
    end

    @testset "Al-Aql Implicit Be Inference" begin
        space = SimulationSpace()
        train_from_text!(space, """
        عصفور على الشجرة.
        باب البيت.
        بيت جميل.
        الماء ساخن.
        """)

        @test haskey(space.entities, "عصفور")
        @test haskey(space.entities, "الشجرة")
        @test get_attribute(space.entities["عصفور"], "implicit_relation") == "على"
        @test get_attribute(space.entities["عصفور"], "implicit_target") == "الشجرة"
        @test haskey(space.entities, "باب البيت")
        @test get_attribute(space.entities["البيت"], "has_part") == "باب"
        @test get_attribute(space.entities["باب"], "possessor") == "البيت"
        @test get_attribute(space.entities["باب البيت"], "head") == "باب"
        @test get_attribute(space.entities["باب البيت"], "implicit_relation") == "له"
        @test get_attribute(space.entities["بيت"], "implicit_attribute") == "جميل"
        @test get_attribute(space.entities["بيت"], "beauty") == 0.8
        @test get_attribute(space.entities["الماء"], "implicit_attribute") == "ساخن"
        @test get_attribute(space.entities["الماء"], "temperature") == 0.8

        adl_space = SimulationSpace()
        compile_adl!(adl_space, "فكرة : قلم في الحقيبة")
        @test haskey(adl_space.entities, "قلم")
        @test get_attribute(adl_space.entities["قلم"], "implicit_relation") == "في"

        compile_adl!(adl_space, "فكرة : بيت جميل")
        @test get_attribute(adl_space.entities["بيت"], "beauty") == 0.8
    end

    @testset "Al-Aql Process Concepts" begin
        space = SimulationSpace()
        compile_adl!(space, """
        عملية كرم { مجال: أخلاق, شدة: 1.0, خطوة: source أعطى target -> بلا_مقابل }
        عملية جود < كرم { شدة: 2.0 }
        """)

        @test haskey(space.processes, "كرم")
        @test haskey(space.processes, "جود")
        @test get_process(space, "جود").intensity == 2.0

        frames = instantiate_process!(space, "كرم", "سالم", "محمود")
        generous_frames = instantiate_process!(space, "جود", "سالم", "محمود")

        @test frames[1].event == "أعطى"
        @test occursin("بلا_مقابل", frames[1].result)
        @test occursin("intensity=2.0", generous_frames[1].result)
    end

    @testset "Al-Aql Negation And Opposites" begin
        space = SimulationSpace()
        compile_adl!(space, """
        ضد حسن : قبيح
        ضد خاف : ثبت_وهاجم
        غزال : كائن { خوف: 0.4, حركة: 0.2 }
        أسد : كائن { طاقة: 0.9 }
        ثبت_وهاجم : فعل { الهدف: خوف, مقياس: -3.0; الهدف: حركة, مقياس: 4.0 }
        فكرة : غزال -خاف أسد
        """)

        @test opposite_of(space, "حسن") == "قبيح"
        @test opposite_of(space, "قبيح") == "حسن"
        @test any(occursin("opposite=ثبت_وهاجم", line) for line in space.log)

        fear, _ = get_property(space.entities["غزال"], "fear")
        motion, _ = get_property(space.entities["غزال"], "motion")
        @test fear <= 0.1
        @test motion > 0.1
    end

    @testset "Al-Aql Circumstances And Time Order" begin
        space = SimulationSpace()
        register_entity!(space, Thing("سالم"))
        default_circ = get_circumstance(space, "سالم")
        @test default_circ.location == "غير_محدد"
        @test default_circ.time == "غير_محدد"

        compile_adl!(space, """
        ظرف سالم { مكان: بيته, زمان: يوم الجمعة }
        ترتيب سلام_سالم قبل رد_محمود
        """)

        circ = get_circumstance(space, "سالم")
        @test circ.location == "بيته"
        @test circ.time == "يوم الجمعة"
        @test length(space.temporal_relations) == 1
        @test space.temporal_relations[1].relation == "قبل"

        train_from_text!(space, "عصفور على الشجرة.")
        @test get_circumstance(space, "عصفور").location == "على الشجرة"
    end

    @testset "Al-Aql Static Semantic Relations" begin
        space = SimulationSpace()
        compile_adl!(space, """
        علاقة خالد أخو أحمد
        علاقة : القاهرة في مصر
        علاقة { مصدر: الباب, علاقة: جزء_من, هدف: البيت, ثقة: 0.95 }
        """)

        @test length(space.semantic_relations) == 3
        @test any(rel -> rel.source == "خالد" && rel.relation == "أخو" && rel.target == "أحمد",
                  relations_between(space, "خالد", "أحمد"))
        @test any(rel -> rel.source == "القاهرة" && rel.relation == "في" && rel.target == "مصر",
                  relations_from(space, "القاهرة"))
        @test any(rel -> rel.relation == "جزء_من" && rel.confidence ≈ 0.95,
                  relations_to(space, "البيت"))

        train_from_text!(space, "باب البيت. عصفور على الشجرة.")
        @test any(rel -> rel.source == "البيت" && rel.relation == "له" && rel.target == "باب",
                  relations_between(space, "البيت", "باب"))
        @test any(rel -> rel.source == "باب" && rel.relation == "جزء_من" && rel.target == "البيت",
                  relations_between(space, "باب", "البيت"))
        @test any(rel -> rel.source == "عصفور" && rel.relation == "على" && rel.target == "الشجرة",
                  relations_from(space, "عصفور"))
    end

    @testset "Al-Aql Quantifiers And Comparisons" begin
        space = SimulationSpace()
        compile_adl!(space, """
        كم كل حيوان يكون حي
        كم بعض الناس لا يردون السلام
        كتلة1 : كائن { mass: 5.0 }
        كتلة2 : كائن { mass: 2.0 }
        مقارنة كتلة1 أكبر من كتلة2 في mass
        مقارنة { طرف1: كتلة1, خاصية: mass, طرف2: كتلة2, احسب: true }
        """)

        @test length(space.quantified_facts) == 2
        @test any(f -> f.quantifier == "كل" && f.subject == "حيوان" &&
                       f.predicate == "يكون" && f.object == "حي" && f.polarity == 1,
                  quantified_facts_for(space, "حيوان"))
        @test any(f -> f.quantifier == "بعض" && f.subject == "الناس" &&
                       f.predicate == "يردون" && f.object == "السلام" && f.polarity == -1,
                  quantified_facts_for(space, "الناس"))
        @test length(space.comparisons) == 2
        @test any(c -> c.left == "كتلة1" && c.right == "كتلة2" &&
                       c.property == "mass" && c.comparator == "أكبر",
                  comparisons_for(space, "كتلة1"))
    end

    @testset "Al-Aql Metaphor Intent Exceptions" begin
        space = SimulationSpace()
        compile_adl!(space, """
        علاقة الحديد نوع_من معدن
        مجاز { تعبير: جيش الليل زحف, مجال_المصدر: حرب, مجال_الهدف: زمن, موضوع: الليل, مستعار: جيش, فعل: زحف, خاصية_منقولة: انتشار, ثقة: 0.8 }
        نية { فاعل: خالد, فعل: ذهب, هدف_الفعل: السوق, نية: شراء_خبز, غاية: إحضار_طعام, نتيجة_واقعة: وصل_السوق, ثقة: 0.9 }
        استثناء { قاعدة: تمدد_الحديد_بالحرارة, شرط: حرارة > 0.5, استثناء: الحديد_مقيد, أولوية: 10, ثقة: 0.95 }
        """)

        @test relations_from(space, "الحديد")[1] isa RelationFact
        @test length(space.metaphors) == 1
        @test metaphors_for(space, "الليل")[1].borrowed_actor == "جيش"
        @test metaphors_for(space, "الليل")[1].transferred_property == "انتشار"
        @test length(intents_for(space, "خالد")) == 1
        @test intents_for(space, "خالد")[1].goal == "إحضار_طعام"
        @test intents_for(space, "خالد")[1].actual_result == "وصل_السوق"
        @test length(exceptions_for(space, "تمدد_الحديد_بالحرارة")) == 1
        @test active_exception(space, "تمدد_الحديد_بالحرارة").priority == 10
    end

    @testset "Al-Aql Text Training" begin
        space = SimulationSpace()
        train_from_text!(space, """
        الأسد حيوان مفترس قوي وطاقته كبيرة.
        الغزال حيوان خائف وسريع.
        زأر الأسد على الغزال يرفع الخوف.
        إذا زاد الخوف عن 0.5 يهرب الغزال.
        الأسد زأر على الغزال.
        """)

        @test haskey(space.entities, "الأسد")
        @test haskey(space.entities, "الغزال")
        @test haskey(space.dynamic_verbs, "الأسد") == false
        @test !isempty(space.rules)
        @test is_a(space, "الأسد", "مفترس")
        @test is_a(space, "الأسد", "حيوان")
        @test is_a(space, "الغزال", "حيوان")
    end

    @testset "Al-Aql Abstract Taxonomy" begin
        space = SimulationSpace()
        @test isfile(DEFAULT_TAXONOMY_FILE)
        @test "حيوان" in known_classes(space)
        @test is_a(space.taxonomy, "مفترس", "حيوان")
        @test "آلة" in known_classes(load_taxonomy(DEFAULT_TAXONOMY_FILE))

        أسد = Thing("أسد"; kind="حيوان", attributes=Dict{String,Any}(
            "title" => "ملك الغابة",
        ))
        register_entity!(space, أسد; classes=["مفترس"])

        @test is_a(space, "أسد", "حيوان")
        @test is_a(space, "أسد", "حي")
        @test get_attribute(أسد, "diet") == "لحم"
        @test get_attribute(أسد, "has_body") == true
        @test get_attribute(أسد, "title") == "ملك الغابة"

        energy, _ = get_property(أسد, "energy")
        @test energy > 0.1

        register_class!(space, "آلة"; parent="جماد",
            attributes=Dict{String,Any}("programmable" => true))
        حاسوب = Thing("حاسوب")
        register_entity!(space, حاسوب)
        assign_class!(space, "حاسوب", "آلة")

        @test is_a(space, "حاسوب", "غير_ذي_روح")
        @test get_attribute(حاسوب, "programmable") == true
        @test get_attribute(حاسوب, "alive") == false
        @test any(occursin("شيء", line) for line in explain_classification(space.taxonomy, حاسوب))

        text_space = SimulationSpace()
        train_from_text!(text_space, """
        الحاسوب آلة مفيدة.
        الوردة نبات جميل.
        """)

        @test is_a(text_space, "الحاسوب", "آلة")
        @test is_a(text_space, "الحاسوب", "جماد")
        @test is_a(text_space, "الوردة", "نبات")

        saved_path = joinpath(mktempdir(), "taxonomy.json")
        save_taxonomy(text_space.taxonomy, saved_path)
        loaded = load_taxonomy(saved_path)
        @test is_a(loaded, "آلة", "جماد")
    end

    @testset "Arabic Analysis Tools" begin
        @test analyze_arabic_word("يكتبون").kind == "فعل"
        @test analyze_arabic_word("أحمر").kind == "صفة"
        @test analyze_arabic_word("راكضًا").kind == "حال"
        @test analyze_arabic_word("الكتاب").kind == "اسم"

        sentence_analysis = analyze_arabic_sentence("لم يذهب الطالب")
        @test sentence_analysis[2].kind == "فعل"

        kb = deep_understand("""
        تعلم الفتى على يد الأستاذ في العصر العباسي.
        الأمير هو ابن الملك.
        زيد صديق عمرو.
        الطبيب يملك عيادة.
        الطبيب هو طبيب.
        """)

        @test !isempty(find_relations_between(kb, "الفتى", "الأستاذ"))
        @test !isempty(find_relations_between(kb, "الأمير", "الملك"))
        @test !isempty(find_relations_between(kb, "زيد", "عمرو"))
        @test !isempty(find_relations_between(kb, "الطبيب", "عيادة"))
        @test "طبيب" in kb.entities["طبيب"].roles
        @test any(t -> t.kind == "historical_period", kb.temporal_expressions)
    end

    @testset "Training Frequency Balance" begin
        counts = [1000.0, 10.0, 6.0, 2.0, 0.0]
        cfg = WordBalanceConfig(; target_quantile=0.60, power=0.50,
                                  min_weight=0.25, max_weight=4.0,
                                  rare_count_floor=2.0, rare_max_weight=1.2)
        weights, meta = build_word_balance_weights(counts; config=cfg)

        @test length(weights) == length(counts)
        @test weights[1] < 1.0
        @test weights[3] > weights[1]
        @test weights[4] <= 1.2
        @test weights[5] == 1.0
        @test meta["boosted"] >= 1
        @test pair_balance_weight(weights, 2, 3) ≈ sqrt(weights[2] * weights[3])
        @test pair_balance_weight(nothing, 2, 3) == 1.0
    end

    @testset "Retrieval-Augmented Physical Generation (RAPG)" begin
        db_path = joinpath(mktempdir(), "rapg_test.db")
        init_rapg_db!(db_path)
        @test isfile(db_path)
        
        # 1. Store test passage
        vec_in = zeros(Float32, 10000)
        vec_in[1] = 1.0f0
        vec_in[10] = 2.0f0
        nrm = norm(vec_in)
        vec_in ./= nrm
        
        store_passage!(db_path, "العلم ضياء ونور يكشف الطريق", vec_in, "test_source")
        
        # 2. Load database to memory
        kb = load_rapg_kb(db_path)
        @test length(kb.passages) == 1
        @test kb.passages[1] == "العلم ضياء ونور يكشف الطريق"
        @test kb.embeddings[:, 1] ≈ vec_in
        @test kb.sources[1] == "test_source"
        
        # 3. Retrieve
        mock_pv = zeros(Float32, 10000)
        mock_pv[1] = 1.0f0
        mock_pv[10] = 2.0f0
        mock_pv ./= norm(mock_pv)
        
        ret_passages, ret_pvs, ret_sims = retrieve(kb, "العلم الطريق", 1, w -> mock_pv)
        @test length(ret_passages) == 1
        @test ret_passages[1] == "العلم ضياء ونور يكشف الطريق"
        @test ret_sims[1] ≈ 1.0 atol=1e-5
        
        # 4. Generator & scoring integration
        vocab = Dict("علم" => 1, "نور" => 2, "ضياء" => 3)
        K_sem = spzeros(3, 3)
        gen = MirnanGenerator(vocab, K_sem; model_dir=mktempdir())
        
        # Populate retrieved context manually
        gen.rapg_kb = kb
        push!(gen.retrieved_passages, "العلم ضياء ونور يكشف الطريق")
        push!(gen.retrieved_pvs, vec_in)
        push!(gen.retrieved_similarities, 0.95)
        
        w_pv = zeros(Float64, 10000)
        w_pv[1] = 1.0
        w_pv[10] = 2.0
        w_pv ./= norm(w_pv)
        
        gen.scoring_weights["kb_knowledge"] = 3.0
        
        prompted = [w_pv]
        prob, total_wave, _ = MirnanNew.Physics.Generator._score(
            gen, "نور", Set(["علم"]), prompted, prompted; prev_word="علم", context_words=["علم"])
            
        @test prob >= 0.0
    end

    @testset "Marker Discovery Engine" begin
        mock_texts = [
            "هل العلم نور ؟",
            "هل الجهل ظلام ؟",
            "هل الحقيقة واضحة ؟",
            "هل العمل مفيد ؟",
            "أنا لا أحب الكذب .",
            "هو لا يريد التراجع .",
            "نحن لا نعرف الفشل .",
            "هم لا يخشون الصعاب .",
            "سافرنا لأن الجو جميل .",
            "ضحكنا لأن الخبر مفرح .",
            "نجحنا لأن العمل مستمر .",
            "قرأنا لأن الكتاب مفيد .",
            "العلم و المعرفة و الحكمة و القوة و العمل ."
        ]
        
        vocab = Dict{String,Int}()
        next_id = 1
        for text in mock_texts
            # simple tokenization matching _tokenize_with_boundaries
            cleaned = replace(text, r"([؟\.\!\?،؛:])" => s" \1 ")
            for token in split(cleaned)
                token = strip(token)
                if !isempty(token) && !haskey(vocab, token)
                    vocab[token] = next_id
                    next_id += 1
                end
            end
        end
        
        id2word = Dict{Int,String}(v => k for (k, v) in vocab)
        
        discovered = MirnanNew.Physics.discover_markers(mock_texts, vocab, id2word)
        
        @test haskey(discovered, "هل")
        @test haskey(discovered, "لا")
        @test haskey(discovered, "لأن")
        @test haskey(discovered, "و")
        
        @test discovered["هل"].category == "question"
        @test discovered["لا"].category == "negation"
        @test discovered["لأن"].category == "causal"
        @test discovered["و"].category == "connector"
        
        # Test serialization
        temp_dir = mktempdir()
        file_path = joinpath(temp_dir, "discovered_markers.json")
        MirnanNew.Physics.save_discovered_markers(file_path, discovered)
        @test isfile(file_path)
        
        loaded = MirnanNew.Physics.load_discovered_markers(file_path)
        @test loaded["هل"] == "question"
        @test loaded["لا"] == "negation"
        @test loaded["لأن"] == "causal"
        @test loaded["و"] == "connector"
    end

    @testset "Phase 7 — Multi-Engine Integration Loop" begin
        PIDController = MirnanNew.Physics.MirnanCerebellumModule.PIDController
        RAPGKnowledgeBase = MirnanNew.Physics.RAPGModule.RAPGKnowledgeBase
        retrieve_by_category = MirnanNew.Physics.RAPGModule.retrieve_by_category
        IstinbatAttentionMemory = MirnanNew.Physics.AlIstinbat.IstinbatAttentionMemory
        _LEARNED_ISTINBAT_MEMORY = MirnanNew.Physics.Generator._LEARNED_ISTINBAT_MEMORY
        generate! = MirnanNew.generate!

        # 1. Test correct_weight! and target correction
        pid = MirnanNew.Physics.MirnanCerebellumModule.PIDController(Kp=0.5, Ki=0.0, Kd=0.0, setpoint=0.5)
        weights = Dict("kb_knowledge" => 3.0, "align" => 5.0)
        
        p_out = MirnanNew.Physics.MirnanCerebellumModule.correct_weight!(pid, weights, "kb_knowledge", 0.4)
        @test p_out == 0.2
        @test weights["kb_knowledge"] ≈ 3.06 atol=1e-5
        @test weights["align"] == 5.0

        # 2. Test retrieve_by_category with mock RAPG DB
        db_path = joinpath(mktempdir(), "rapg_test_p7.db")
        init_rapg_db!(db_path)
        
        vec1 = compute_extended_phase_vector("العلم")
        vec2 = compute_extended_phase_vector("المطر")
        store_passage!(db_path, "العلم هو النور الكاشف للعقل", vec1, "definitions")
        store_passage!(db_path, "يسقط المطر لأن الغيوم مكثفة بالماء", vec2, "istinbat_attention")
        
        kb = load_rapg_kb(db_path)
        @test length(kb.passages) == 2
        
        pv_fn = w -> compute_extended_phase_vector(String(w))
        
        p1, _, s1 = retrieve_by_category(kb, "العلم", "question", 2, pv_fn)
        @test length(p1) == 1
        @test occursin("العلم هو النور", p1[1])
        
        p2, _, s2 = retrieve_by_category(kb, "المطر", "causal", 2, pv_fn)
        @test length(p2) == 1
        @test occursin("يسقط المطر", p2[1])

        p3, _, s3 = retrieve_by_category(kb, "العلم", "other_category", 2, pv_fn)
        @test length(p3) >= 1

        # 3. Test update_marker_confidence!
        istinbat = IstinbatAttentionMemory()
        istinbat.discovered_markers["هل"] = "question"
        istinbat.discovered_confidences["هل"] = 0.6
        istinbat.discovered_markers["ما"] = "question"
        istinbat.discovered_confidences["ما"] = 0.6
        
        update_marker_confidence!(istinbat, ["هل"], 0.9)
        @test istinbat.discovered_confidences["هل"] ≈ 0.63 atol=1e-5
        
        update_marker_confidence!(istinbat, ["هل"], 0.1)
        @test istinbat.discovered_confidences["هل"] ≈ 0.577 atol=1e-5

        # 4. Integration & Diagnostics test with MirnanGenerator
        vocab = Dict("ما" => 1, "هو" => 2, "العلم" => 3, "؟" => 4, "النور" => 5, "الكاشف" => 6, "العقل" => 7)
        K_sem = spzeros(7, 7)
        K_sem[3, 5] = 0.8  # العلم -> النور
        K_sem[5, 3] = 0.8  # النور -> العلم
        K_sem[5, 6] = 0.8  # النور -> الكاشف
        K_sem[6, 7] = 0.8  # الكاشف -> العقل
        gen = MirnanGenerator(vocab, K_sem; model_dir=mktempdir())
        gen.rapg_kb = kb
        
        push!(gen.retrieved_passages, "العلم هو النور الكاشف للعقل")
        push!(gen.retrieved_pvs, vec1)
        push!(gen.retrieved_similarities, 0.8)
        
        _LEARNED_ISTINBAT_MEMORY[] = istinbat
        
        empty!(gen.cerebellum.integration_log)
        ans = generate!(gen, "ما هو العلم ؟"; mode="resonant", max_words=3)
        @show ans
        @show gen.cerebellum.pid_enabled
        @show gen.cerebellum.integration_log
        @test !isempty(gen.cerebellum.integration_log)
        
        last_log = gen.cerebellum.integration_log[1]
        @test haskey(last_log, "step")
        @test haskey(last_log, "word")
        @test haskey(last_log, "markers_active")
        @test haskey(last_log, "retrieval_similarity")
        @test haskey(last_log, "weights")
        @test last_log["retrieval_similarity"] > 0.4
        @test "question" in last_log["markers_active"]
    end

    @testset "Phase 8 — Pure Self-Discovery & Evolutionary Weights" begin
        Chromosome = MirnanNew.Physics.AlTawweerModule.Chromosome
        evolve_weights! = MirnanNew.Physics.AlTawweerModule.evolve_weights!
        evaluate_fitness! = MirnanNew.Physics.AlTawweerModule.evaluate_fitness!
        IstinbatAttentionMemory = MirnanNew.Physics.AlIstinbat.IstinbatAttentionMemory
        _marker_hit = MirnanNew.Physics.AlIstinbat._marker_hit
        crossover = MirnanNew.Physics.AlTawweerModule.crossover
        mutate! = MirnanNew.Physics.AlTawweerModule.mutate!
        selection = MirnanNew.Physics.AlTawweerModule.selection
        using SparseArrays
        
        # 1. Test Pure Self-Discovery
        istinbat = IstinbatAttentionMemory()
        istinbat.discovered_markers["لأن"] = "causal"
        
        rtype, word = _marker_hit("الغيوم لأن المطر يسقط", istinbat.discovered_markers)
        @test rtype == "causal"
        @test word == "لأن"
        
        rtype2, word2 = _marker_hit("يسقط المطر بسبب الغيوم", istinbat.discovered_markers; include_defaults=false)
        @test rtype2 == ""
        @test word2 == ""
        
        # 2. Test Evolutionary weight structures
        w1 = Dict(k => 5.0 for k in MirnanNew.Physics.AlTawweerModule.WEIGHT_KEYS)
        w2 = Dict(k => 3.0 for k in MirnanNew.Physics.AlTawweerModule.WEIGHT_KEYS)
        
        c1 = Chromosome(w1)
        c2 = Chromosome(w2)
        
        child1, child2 = crossover(c1, c2)
        @test haskey(child1.weights, "align")
        @test child1.weights["align"] >= 3.0 && child1.weights["align"] <= 5.0
        
        mutate!(child1, 1.0, 0.5)
        @test child1.weights["align"] >= 0.0 && child1.weights["align"] <= 15.0
        
        # 3. Running evolve_weights! on mock generator
        vocab = Dict("ما" => 1, "هو" => 2, "العلم" => 3, "؟" => 4, "النور" => 5)
        K_sem = spzeros(5, 5)
        K_sem[3, 5] = 0.8
        gen = MirnanGenerator(vocab, K_sem; model_dir=mktempdir())
        
        validation_prompts = ["ما هو العلم ؟", "العلم هو النور"]
        evolved_w = evolve_weights!(gen, validation_prompts; generations=2, pop_size=4)
        
        @test length(evolved_w) >= length(MirnanNew.Physics.AlTawweerModule.WEIGHT_KEYS)
        for key in MirnanNew.Physics.AlTawweerModule.WEIGHT_KEYS
            @test haskey(evolved_w, key)
            @test evolved_w[key] >= 0.0 && evolved_w[key] <= 15.0
        end
    end

    @testset "Phase 9 — PRNN 2.0 Stuart-Landau Generation Strategy" begin
        # 1. Verification of the PRNNStrategy struct and its registration
        @test MirnanNew.PRNNStrategy <: MirnanNew.Physics.Generator.GenerationStrategy
        
        # 2. Testing try_generate under PRNNStrategy
        vocab = Dict("العلم" => 1, "نور" => 2, "و" => 3, "الجهل" => 4, "ظلام" => 5)
        # Create a simple transition matrix K_sem
        K_sem = spzeros(5, 5)
        # transitions: العلم -> نور (0.9), الجهل -> ظلام (0.85)
        K_sem[1, 2] = 0.9
        K_sem[4, 5] = 0.85
        
        gen = MirnanGenerator(vocab, K_sem; model_dir=mktempdir())
        
        # Test PRNN Generation directly via generate!
        # Set prnn parameters in scoring_weights
        gen.scoring_weights["prnn_coupling_beta"] = 3.0
        gen.scoring_weights["prnn_mu"] = 1.0
        gen.scoring_weights["prnn_g_inh"] = 0.5
        gen.scoring_weights["prnn_gamma"] = 2.0
        gen.scoring_weights["prnn_tau_a"] = 1.5
        gen.scoring_weights["prnn_steps"] = 40.0
        gen.scoring_weights["prnn_dt"] = 0.02
        gen.scoring_weights["prnn_overlap_threshold"] = 0.05
        gen.scoring_weights["prnn_noise_amp"] = 0.1
        gen.scoring_weights["prnn_anneal_rate"] = 0.01
        
        # Test direct call to generate! with mode="prnn"
        res = generate!(gen, "العلم"; mode="prnn", max_words=5)
        @test occursin("نور", res)
        
        # Test automatic creative routing in generate! (mode="auto" but creative prompts with K_sem should trigger prnn)
        res_auto = generate!(gen, "قصيدة العلم"; mode="auto", max_words=5)
        @test occursin("نور", res_auto)
    end

end # main testset

