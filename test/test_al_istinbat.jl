include("../src/MirnanNew.jl")

using .MirnanNew
using Test

const Physics = MirnanNew.Physics

@testset "al_istinbat inference attention memory" begin
    mem = Physics.IstinbatAttentionMemory()
    texts = [
        "العلم نور لأنه يكشف الطريق.",
        "الجهل ليس نورا؛ بل يحجب الرؤية ويوقع الإنسان في المتاهات.",
        "النظريات العلمية تتحول إلى قناعات راسخة حين تتصل المعرفة بالتجربة والمراجعة المستمرة.",
        "تتحول القوة إلى ظلم حين تفارق الرحمة والعدل.",
        "الفرق بين العدل والرحمة أن العدل يعطي الحق، أما الرحمة فتخفف الأذى.",
        "لا يكفي العلم وحده لأنه يحتاج إلى فهم وعدل ورحمة.",
    ]

    learned = Physics.train_istinbat_from_texts!(mem, texts)
    @test learned >= 3
    @test Physics.has_istinbat_records(mem)

    rec = Physics.select_istinbat_attention(mem, "لماذا يكشف العلم الطريق؟")
    @test rec !== nothing
    @test rec.relation_type in ("causal", "analogy", "need", "transform", "difference", "prevention", "contradiction", "negation")
    @test !isempty(Physics.istinbat_focus_terms(rec))

    contradiction = Physics.select_contradiction_attention(mem, "هل الجهل نور؟")
    @test contradiction !== nothing
    @test contradiction.polarity < 0
    answer = Physics.contradiction_answer_from_attention(contradiction)
    @test startswith(answer, "لا")
    @test occursin("الجهل", answer)

    anchor = Physics.select_causal_anchor_attention(mem, "كيف تتحول النظريات العلمية إلى قناعات؟")
    @test anchor !== nothing
    @test anchor.relation_type == "causal_anchor"
    anchor_answer = Physics.causal_anchor_answer_from_attention(anchor)
    @test occursin("النظريات العلمية", anchor_answer)
    @test occursin("التجربة", anchor_answer)

    tmp = tempname() * ".json"
    Physics.save_istinbat(mem, tmp)
    loaded = Physics.load_istinbat(tmp)
    @test Physics.has_istinbat_records(loaded)
    @test length(loaded.records) == length(mem.records)

    opposition_mem = Physics.IstinbatAttentionMemory()
    opposition_learned = Physics.learn_opposition_from_text!(
        opposition_mem,
        "العقل يميز بين الخير والشر. العدل والظلم طريقان لا يلتقيان."
    )
    @test opposition_learned >= 2
    @test Physics.terms_are_opposed(opposition_mem, "الخير", "الشر")
    @test Physics.terms_are_opposed(opposition_mem, "العدل", "الظلم")
    @test !Physics.terms_are_opposed(opposition_mem, "العدل", "الخير")

    negation_mem = Physics.IstinbatAttentionMemory()
    negation_learned = Physics.learn_direct_negation_from_text!(
        negation_mem,
        "الجهل ليس نوراً؛ بل يحجب الرؤية."
    )
    @test negation_learned >= 1
    @test Physics.terms_are_negated(negation_mem, "الجهل", "نور")
    @test Physics.terms_are_negated(negation_mem, "الجهل", "نوراً")
    @test !Physics.terms_are_negated(negation_mem, "العلم", "نور")

    negative_operator_mem = Physics.IstinbatAttentionMemory()
    @test Physics.learn_istinbat_from_text!(negative_operator_mem, "الجهل يحجب الفهم.") >= 1
    @test Physics.learn_istinbat_from_text!(negative_operator_mem, "الظلم يفسد الثقة.") >= 1
    negative_operator_records = collect(values(negative_operator_mem.records))
    @test any(r -> r.relation_type == "contradiction" && r.polarity < 0 &&
                   "جهل" in r.before_terms && "فهم" in r.after_terms,
              negative_operator_records)
    @test any(r -> r.relation_type == "contradiction" && r.polarity < 0 &&
                   "ظلم" in r.before_terms && "ثقه" in r.after_terms,
              negative_operator_records)

    persistent = Physics.IstinbatAttentionMemory()
    manual = Physics.IstinbatAttentionRecord(
        "manual|1|علم=>فهم",
        "manual",
        "يدوي",
        ["علم"],
        ["فهم"],
        ["فهم", "تعلم"],
        1,
        0.95,
        ["العلم يفتح الفهم."],
        1,
    )
    persistent.records[manual.record_id] = manual
    Physics.merge_istinbat!(loaded, persistent)
    @test haskey(loaded.records, "manual|1|علم=>فهم")
    @test "تعلم" in loaded.records["manual|1|علم=>فهم"].focus_terms
end
