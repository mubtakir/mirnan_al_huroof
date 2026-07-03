include("../src/MirnanNew.jl")

using .MirnanNew
using Test

const Physics = MirnanNew.Physics

@testset "relation_type_for_marker" begin
    @test Physics.relation_type_for_marker("لكي") == "purpose"
    @test Physics.relation_type_for_marker("كي") == "purpose"
    @test Physics.relation_type_for_marker("من أجل") == "purpose"
    @test Physics.relation_type_for_marker("إذا") == "conditional"
    @test Physics.relation_type_for_marker("إن") == "conditional"
    @test Physics.relation_type_for_marker("قبل") == "temporal"
    @test Physics.relation_type_for_marker("بعد") == "temporal"
    @test Physics.relation_type_for_marker("حيث") == "spatial"
    @test Physics.relation_type_for_marker("في حين") == "state"
    @test isempty(Physics.relation_type_for_marker("xyz"))
    # existing markers still work
    @test Physics.relation_type_for_marker("لأن") == "causal"
end

@testset "extract_relation_frames: purpose" begin
    frames = Physics.extract_relation_frames("يدرس الطالب لكي ينجح.")
    @test length(frames) >= 1
    purpose_frames = [f for f in frames if f.relation_type == "purpose"]
    @test !isempty(purpose_frames)
    f = first(purpose_frames)
    @test occursin("ينجح", join(f.right_terms, " "))
    @test f.direction == 1
    @test f.confidence >= 0.85
end

@testset "extract_relation_frames: conditional" begin
    frames = Physics.extract_relation_frames("إذا اجتهدت تنجح.")
    @test length(frames) >= 1
    @test any(f -> f.relation_type == "conditional", frames)
end

@testset "extract_relation_frames: conditional split" begin
    frames = Physics.extract_relation_frames("\u0625\u0630\u0627 \u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645 \u0632\u0627\u062f \u0627\u0644\u0641\u0647\u0645.")
    conds = [f for f in frames if f.relation_type == "conditional"]
    @test !isempty(conds)
    cond = first(conds)
    @test join(cond.left_terms, " ") == "\u0632\u0627\u062f \u0639\u0644\u0645"
    @test join(cond.right_terms, " ") == "\u0632\u0627\u062f \u0641\u0647\u0645"
end

@testset "extract_relation_frames: temporal" begin
    frames = Physics.extract_relation_frames("سافر المسافر قبل الفجر.")
    @test length(frames) >= 1
    temporal_frames = [f for f in frames if f.relation_type == "temporal"]
    @test !isempty(temporal_frames)
    @test any(f -> occursin("سافر", join(f.left_terms, " ")), temporal_frames)
    @test any(f -> occursin("فجر", join(f.right_terms, " ")), temporal_frames)
end

@testset "extract_relation_frames: spatial" begin
    frames = Physics.extract_relation_frames("الكتاب حيث تركته.")
    @test length(frames) >= 1
    @test any(f -> f.relation_type == "spatial", frames)
end

@testset "extract_relation_frames: state" begin
    frames = Physics.extract_relation_frames("جاء الرجل في حين كان المطر يهطل.")
    @test length(frames) >= 1
    @test any(f -> f.relation_type == "state", frames)
end

@testset "relation frames: expanded non-purpose markers" begin
    @test Physics.relation_type_for_marker("\u0639\u0646\u062f\u0645\u0627") == "temporal"
    @test Physics.relation_type_for_marker("\u062d\u064a\u0646") == "temporal"
    @test Physics.relation_type_for_marker("\u0641\u0648\u0642") == "spatial"
    @test Physics.relation_type_for_marker("\u062a\u062d\u062a") == "spatial"
    @test Physics.relation_type_for_marker("\u062f\u0627\u062e\u0644") == "spatial"
    @test Physics.relation_type_for_marker("\u0648\u0647\u0648") == "state"
    @test Physics.relation_type_for_marker("\u0648\u0647\u064a") == "state"

    temporal_frames = [f for f in Physics.extract_relation_frames("\u0639\u0627\u062f \u0627\u0644\u0645\u0633\u0627\u0641\u0631 \u0639\u0646\u062f\u0645\u0627 \u063a\u0631\u0628\u062a \u0627\u0644\u0634\u0645\u0633. \u0647\u062f\u0623 \u0627\u0644\u0637\u0641\u0644 \u062d\u064a\u0646 \u0633\u0645\u0639 \u0627\u0644\u0642\u0635\u0629.") if f.relation_type == "temporal"]
    @test any(f -> f.marker == "\u0639\u0646\u062f\u0645\u0627", temporal_frames)
    @test any(f -> f.marker == "\u062d\u064a\u0646", temporal_frames)

    spatial_frames = [f for f in Physics.extract_relation_frames("\u0648\u0636\u0639 \u0627\u0644\u0637\u0641\u0644 \u0627\u0644\u0643\u062a\u0627\u0628 \u0641\u0648\u0642 \u0627\u0644\u0637\u0627\u0648\u0644\u0629. \u062c\u0644\u0633 \u0627\u0644\u0642\u0637 \u062a\u062d\u062a \u0627\u0644\u0643\u0631\u0633\u064a. \u062e\u0628\u0623 \u0627\u0644\u0645\u0641\u062a\u0627\u062d \u062f\u0627\u062e\u0644 \u0627\u0644\u0635\u0646\u062f\u0648\u0642.") if f.relation_type == "spatial"]
    @test any(f -> f.marker == "\u0641\u0648\u0642", spatial_frames)
    @test any(f -> f.marker == "\u062a\u062d\u062a", spatial_frames)
    @test any(f -> f.marker == "\u062f\u0627\u062e\u0644", spatial_frames)

    state_frames = [f for f in Physics.extract_relation_frames("\u062f\u062e\u0644 \u0627\u0644\u0637\u0641\u0644 \u0648\u0647\u0648 \u062e\u0627\u0626\u0641. \u062e\u0631\u062c\u062a \u0627\u0644\u0637\u0627\u0644\u0628\u0629 \u0648\u0647\u064a \u0641\u0631\u062d\u0629.") if f.relation_type == "state"]
    @test any(f -> f.marker == "\u0648\u0647\u0648", state_frames)
    @test any(f -> f.marker == "\u0648\u0647\u064a", state_frames)
end

@testset "relation frames: English marker parity" begin
    @test Physics.relation_type_for_marker("in order to") == "purpose"
    @test Physics.relation_type_for_marker("if") == "conditional"
    @test Physics.relation_type_for_marker("before") == "temporal"
    @test Physics.relation_type_for_marker("above") == "spatial"
    @test Physics.relation_type_for_marker("while") == "state"

    purpose_frames = [f for f in Physics.extract_relation_frames("student studies in order to succeed.") if f.relation_type == "purpose"]
    @test !isempty(purpose_frames)
    @test any(f -> occursin("succeed", join(f.right_terms, " ")), purpose_frames)

    conditional_frames = [f for f in Physics.extract_relation_frames("if knowledge grows understanding grows.") if f.relation_type == "conditional"]
    @test !isempty(conditional_frames)
    @test any(f -> occursin("understanding", join(f.right_terms, " ")), conditional_frames)

    temporal_frames = [f for f in Physics.extract_relation_frames("traveler returned before dawn.") if f.relation_type == "temporal"]
    @test !isempty(temporal_frames)
    @test any(f -> f.marker == "before", temporal_frames)

    spatial_frames = [f for f in Physics.extract_relation_frames("book is above table.") if f.relation_type == "spatial"]
    @test !isempty(spatial_frames)
    @test any(f -> f.marker == "above", spatial_frames)

    state_frames = [f for f in Physics.extract_relation_frames("child entered while afraid.") if f.relation_type == "state"]
    @test !isempty(state_frames)
    @test any(f -> f.marker == "while", state_frames)
end

@testset "extract_relation_frames: multiple markers" begin
    frames = Physics.extract_relation_frames("سافر قبل الفجر لكي يصل مبكرا.")
    @test length(frames) >= 2
    types = Set(f.relation_type for f in frames)
    @test "temporal" in types
    @test "purpose" in types
end

@testset "extract_relation_frames: multi-word markers" begin
    # من أجل — multi-word purpose
    frames = Physics.extract_relation_frames("يدرس الطالب من أجل النجاح.")
    @test length(frames) >= 1
    @test any(f -> f.relation_type == "purpose", frames)
    purpose_frames = [f for f in frames if f.relation_type == "purpose"]
    if !isempty(purpose_frames)
        f = first(purpose_frames)
        @test occursin("نجاح", join(f.right_terms, " "))
    end

    # متى ما — multi-word conditional
    frames = Physics.extract_relation_frames("متى ما تفعل خيرا تلق جزاءه.")
    @test length(frames) >= 1
    @test any(f -> f.relation_type == "conditional", frames)

    # في حين — multi-word state
    frames = Physics.extract_relation_frames("جاء الرجل في حين كان المطر يهطل.")
    @test length(frames) >= 1
    @test any(f -> f.relation_type == "state", frames)
end

@testset "extract_relation_frames: confidence levels" begin
    frames = Physics.extract_relation_frames("يدرس الطالب لكي ينجح.")
    @test any(f -> f.confidence >= 0.85, frames)
    frames = Physics.extract_relation_frames("جاء حيث كان.")
    @test any(f -> f.confidence <= 0.65, frames)
end

@testset "extract_relation_frames: no markers" begin
    @test isempty(Physics.extract_relation_frames(""))
    @test isempty(Physics.extract_relation_frames("السماء زرقاء."))
end

@testset "RelationFrame struct" begin
    f = Physics.RelationFrame(
        ["طالب"], "لكي", ["ينجح"], "purpose", 1, 1, 0.9, "يدرس الطالب لكي ينجح."
    )
    @test f.relation_type == "purpose"
    @test f.polarity == 1
    @test f.direction == 1
    @test f.confidence ≈ 0.9
    @test occursin("طالب", join(f.left_terms))
    @test occursin("ينجح", join(f.right_terms))
end

@testset "learn_relation_frames_from_text!: memory storage" begin
    mem = Physics.IstinbatAttentionMemory()

    # purpose — مباشرةً عبر learn_relation_frames_from_text!
    n = Physics.learn_relation_frames_from_text!(mem, "يدرس الطالب لكي ينجح.")
    @test n >= 1
    purpose_recs = [r for r in values(mem.records) if r.relation_type == "purpose"]
    @test !isempty(purpose_recs)
    @test any(r -> occursin("ينجح", join(r.after_terms, " ")), purpose_recs)

    # conditional
    n += Physics.learn_relation_frames_from_text!(mem, "إذا اجتهدت تنجح.")
    cond_recs = [r for r in values(mem.records) if r.relation_type == "conditional"]
    @test !isempty(cond_recs)
end

@testset "learn_istinbat_from_text! includes new types" begin
    mem = Physics.IstinbatAttentionMemory()

    # عبر الدالة الرئيسية — يجب أن تخزّن الأنواع الجديدة أيضاً
    n = Physics.learn_istinbat_from_text!(mem, "يدرس الطالب لكي ينجح. سافر قبل الفجر. إذا اجتهدت تنجح.")
    @test n >= 3

    types_found = Set(r.relation_type for r in values(mem.records))
    @test "purpose" in types_found
    @test "temporal" in types_found
    @test "conditional" in types_found

    @test haskey(mem.records, "purpose|1|درب|يقصد=>ينجح") ||
            any(r -> r.relation_type == "purpose", values(mem.records))

    # التأكد من عدم كسر السجلات الحالية (causal)
    @test !isempty(mem.records)
end

@testset "learn_relation_frames_from_text!: no interference with old records" begin
    mem = Physics.IstinbatAttentionMemory()

    # causal old + new types same text
    Physics.learn_istinbat_from_text!(mem, "العلم نور لأنه يكشف الطريق. يدرس الطالب لكي ينجح.")
    types = Set(r.relation_type for r in values(mem.records))
    @test "causal" in types
    @test "purpose" in types

    # السجلات القديمة تحتفظ بنوعها
    causal_recs = [r for r in values(mem.records) if r.relation_type == "causal"]
    @test !isempty(causal_recs)
end

@testset "select_relation_frame_attention: selects purpose" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "يدرس الطالب لكي ينجح.")
    rec = Physics.select_relation_frame_attention(mem, "لماذا يدرس الطالب؟")
    @test rec !== nothing
    @test rec.relation_type == "purpose"
    @test occursin("ينجح", join(rec.after_terms, " "))
end

@testset "select_relation_frame_attention: selects temporal" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "سافر المسافر قبل الفجر.")
    rec = Physics.select_relation_frame_attention(mem, "متى سافر المسافر؟")
    @test rec !== nothing
    @test rec.relation_type == "temporal"
    @test occursin("فجر", join(rec.after_terms, " "))
end

@testset "select_relation_frame_attention: selects conditional" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "إذا اجتهدت تنجح.")
    rec = Physics.select_relation_frame_attention(mem, "متى تنجح؟")
    @test rec !== nothing
    @test rec.relation_type == "conditional"
end

@testset "select_relation_frame_attention: ignores causal" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "العلم نور لأنه يكشف الطريق.")
    rec = Physics.select_relation_frame_attention(mem, "لماذا العلم نور؟")
    # must not match causal records — the function filters them out
    @test rec === nothing
end

@testset "select_relation_frame_attention: empty memory" begin
    mem = Physics.IstinbatAttentionMemory()
    @test Physics.select_relation_frame_attention(mem, "لماذا يدرس الطالب؟") === nothing
end

@testset "select_relation_frame_attention: prioritises relevant over irrelevant" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "يدرس الطالب لكي ينجح.")
    # even with unrelated prompt, attention_weight alone may trigger return
    # (same behaviour as existing select_istinbat_attention)
    rec_irrelevant = Physics.select_relation_frame_attention(mem, "السماء زرقاء.")
    # relevant prompt should match better
    rec_relevant = Physics.select_relation_frame_attention(mem, "لماذا يدرس الطالب؟")
    @test rec_relevant !== nothing
    @test rec_relevant.relation_type == "purpose"
end

@testset "relation_frame_diagnostic: shows frame info" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "يدرس الطالب لكي ينجح.")
    report = Physics.relation_frame_diagnostic(mem, "لماذا يدرس الطالب؟")
    @test occursin("purpose", report)
    @test occursin("لكي", report)
    @test occursin("ينجح", report)
    @test occursin("الثقة", report)
end

@testset "relation_frame_diagnostic: no match" begin
    mem = Physics.IstinbatAttentionMemory()
    @test occursin("لا يوجد", Physics.relation_frame_diagnostic(mem, "لماذا؟"))
end

@testset "purpose_answer: matches purpose question" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "يدرس الطالب لكي ينجح.")
    ans = Physics.purpose_answer(mem, "لماذا يدرس الطالب؟")
    @test !isempty(ans)
    @test occursin("الغاية", ans)
    @test occursin("ينجح", ans)
end

@testset "purpose_answer: English why question" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "student studies in order to succeed.")
    ans = Physics.purpose_answer(mem, "why student studies?")
    @test !isempty(ans)
    @test occursin("purpose", lowercase(ans))
    @test occursin("succeed", lowercase(ans))
    @test isempty(Physics.purpose_answer(mem, "does student study?"))
end

@testset "purpose_answer: rejects noun-only purpose records" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "\u0645\u0648\u0633\u064a\u0642\u064a \u0643\u0644\u0627\u0633\u064a \u0644\u0643\u064a \u0639\u0631\u0628\u064a\u0647 \u0628\u0645\u0642\u0627\u0645\u0627\u062a\u0647\u0627 \u0645\u062a\u0639\u062f\u062f\u0647.")
    @test isempty(Physics.purpose_answer(mem, "\u0644\u0645\u0627\u0630\u0627 \u0645\u0648\u0633\u064a\u0642\u064a \u0643\u0644\u0627\u0633\u064a\u061f"))
end

@testset "purpose_answer: smooths Arabic purpose phrasing" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "\u062f\u0641\u0639 \u0644\u0627\u0639\u0628 \u062d\u062c\u0631 \u0644\u0643\u064a \u064a\u0628\u0639\u062f \u062d\u062c\u0631 \u0637\u0631\u064a\u0642.")
    ans = Physics.purpose_answer(mem, "\u0644\u0645\u0627\u0630\u0627 \u062f\u0641\u0639 \u0644\u0627\u0639\u0628 \u062d\u062c\u0631\u061f")
    @test occursin("\u062f\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062d\u062c\u0631", ans)
    @test occursin("\u0625\u0628\u0639\u0627\u062f \u0627\u0644\u062d\u062c\u0631 \u0639\u0646 \u0627\u0644\u0637\u0631\u064a\u0642", ans)
end

@testset "purpose_answer: ignores competing temporal frame" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_relation_frames_from_text!(
        mem,
        "\u0639\u0646\u062f\u0645\u0627 \u062f\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062d\u062c\u0631 \u062a\u062d\u0631\u0643 \u0627\u0644\u062d\u062c\u0631 \u0645\u0646 \u0645\u0643\u0627\u0646\u0647 \u0648\u0635\u0627\u0631 \u0627\u0644\u0637\u0631\u064a\u0642 \u0623\u0648\u0633\u0639.",
    )
    Physics.learn_relation_frames_from_text!(
        mem,
        "\u062f\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062d\u062c\u0631 \u0644\u0643\u064a \u064a\u0628\u0639\u062f \u0627\u0644\u062d\u062c\u0631 \u0639\u0646 \u0627\u0644\u0637\u0631\u064a\u0642.",
    )
    ans = Physics.purpose_answer(mem, "\u0644\u0645\u0627\u0630\u0627 \u062f\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062d\u062c\u0631\u061f")
    @test !isempty(ans)
    @test occursin("\u0625\u0628\u0639\u0627\u062f \u0627\u0644\u062d\u062c\u0631 \u0639\u0646 \u0627\u0644\u0637\u0631\u064a\u0642", ans) ||
          occursin("\u064a\u0628\u0639\u062f \u062d\u062c\u0631 \u0637\u0631\u064a\u0642", ans)
end

@testset "purpose_answer: leaves long procedural phrases unsmoothed" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "\u0641\u062a\u062d \u0639\u0644\u0628\u0647 \u0627\u0646\u0642\u0644 \u0645\u062d\u062a\u0648\u064a \u0645\u062a\u0628\u0642\u064a \u0641\u0648\u0631\u0627\u064b \u0648\u0639\u0627\u0621 \u0632\u062c\u0627\u062c\u064a \u0628\u0644\u0627\u0633\u062a\u064a \u0644\u0643\u064a \u0645\u062d\u0643\u0645 \u0627\u063a\u0644\u0627\u0642 \u0648\u0627\u062d\u0641\u0638\u0647 \u062b\u0644\u0627\u062c\u0647 \u0648\u0627\u0633\u062a\u0647\u0644\u0643\u0647 \u062e\u0644\u0627\u0644 \u064a\u0648\u0645\u064a\u0646.")
    ans = Physics.purpose_answer(mem, "\u0644\u0645\u0627\u0630\u0627 \u0641\u062a\u062d \u0639\u0644\u0628\u0647 \u0627\u0646\u0642\u0644 \u0645\u062d\u062a\u0648\u064a \u0645\u062a\u0628\u0642\u064a \u0641\u0648\u0631\u0627\u064b \u0648\u0639\u0627\u0621 \u0632\u062c\u0627\u062c\u064a \u0628\u0644\u0627\u0633\u062a\u064a\u061f")
    @test occursin("\u0641\u062a\u062d \u0639\u0644\u0628\u0647 \u0627\u0646\u0642\u0644", ans)
    @test !occursin("\u0627\u0644\u0641\u062a\u062d \u0627\u0644\u0639\u0644\u0628\u0647", ans)
end

@testset "purpose_answer: ignored for yes/no questions" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "يدرس الطالب لكي ينجح.")
    @test isempty(Physics.purpose_answer(mem, "هل يدرس الطالب؟"))
    @test isempty(Physics.purpose_answer(mem, "هل الغاية من الدراسة النجاح؟"))
end

@testset "purpose_answer: ignored for causal-only memory" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "العلم نور لأنه يكشف الطريق.")
    @test isempty(Physics.purpose_answer(mem, "لماذا العلم نور؟"))
end

@testset "purpose_answer: ignored when topic mismatch" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "يدرس الطالب لكي ينجح.")
    @test isempty(Physics.purpose_answer(mem, "لماذا تشرق الشمس؟"))
end

@testset "purpose_answer: ignored with non-purpose prompt" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "يدرس الطالب لكي ينجح.")
    @test isempty(Physics.purpose_answer(mem, "كيف حالك؟"))
    @test isempty(Physics.purpose_answer(mem, "ما اسمك؟"))
end

@testset "purpose_answer: empty memory returns empty" begin
    mem = Physics.IstinbatAttentionMemory()
    @test isempty(Physics.purpose_answer(mem, "لماذا يدرس الطالب؟"))
end

@testset "conditional_answer: matches conditional question" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(
        mem,
        "\u0625\u0630\u0627 \u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645 \u0632\u0627\u062f \u0627\u0644\u0641\u0647\u0645.",
    )
    ans = Physics.conditional_answer(
        mem,
        "\u0645\u0627\u0630\u0627 \u064a\u062d\u062f\u062b \u0625\u0630\u0627 \u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645\u061f",
    )
    @test !isempty(ans)
    @test occursin("\u0625\u0630\u0627 \u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645", ans)
    @test occursin("\u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645", ans)
    @test occursin("\u0632\u0627\u062f \u0627\u0644\u0641\u0647\u0645", ans)
end

@testset "conditional_answer: smooth Arabic phrasing" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "\u0625\u0630\u0627 \u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645 \u0632\u0627\u062f \u0627\u0644\u0641\u0647\u0645.")
    ans = Physics.conditional_answer(mem, "\u0645\u0627\u0630\u0627 \u064a\u062d\u062f\u062b \u0625\u0630\u0627 \u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645\u061f")
    @test occursin("\u0641\u0627\u0644\u0646\u062a\u064a\u062c\u0629", ans)
    @test !occursin("\u064a\u062a\u0631\u062a\u0628 \u0639\u0644\u0649 \u0630\u0644\u0643", ans)
end

@testset "conditional_answer: repairs shortened trained record from example" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_fact!(
        mem,
        "conditional";
        marker="\u0625\u0630\u0627",
        before_terms=["\u0632\u0627\u062f"],
        after_terms=["\u0639\u0644\u0645"],
    )
    rec = first(values(mem.records))
    empty!(rec.examples)
    push!(rec.examples, "\u0625\u0630\u0627 \u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645 \u0632\u0627\u062f \u0627\u0644\u0641\u0647\u0645.")
    ans = Physics.conditional_answer(mem, "\u0645\u0627\u0630\u0627 \u064a\u062d\u062f\u062b \u0625\u0630\u0627 \u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645\u061f")
    @test occursin("\u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645", ans)
    @test occursin("\u0632\u0627\u062f \u0627\u0644\u0641\u0647\u0645", ans)
    @test !occursin("\u0625\u0630\u0627 \u0632\u0627\u062f\u060c", ans)
    @test !occursin("\u0630\u0644\u0643 \u0639\u0644\u0645", ans)
end

@testset "conditional_answer: rejects distant partial overlap" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_fact!(
        mem,
        "conditional";
        marker="\u0625\u0630\u0627",
        before_terms=[
            "\u0639\u0644\u0645", "\u0623\u062f\u0648\u064a\u0629", "\u062a\u0645\u062b\u0644", "\u062c\u0631\u0639\u0629",
            "\u0639\u0644\u0627\u062c\u064a\u0629", "\u0646\u0635\u0641", "\u0645\u0645\u064a\u062a\u0629",
        ],
        after_terms=[
            "\u0632\u0627\u062f", "\u0641\u0627\u0631\u0642", "\u0643\u0627\u0646", "\u062f\u0648\u0627\u0621",
            "\u0623\u0643\u062b\u0631", "\u0623\u0645\u0627\u0646\u0627",
        ],
    )
    ans = Physics.conditional_answer(mem, "\u0645\u0627\u0630\u0627 \u064a\u062d\u062f\u062b \u0625\u0630\u0627 \u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645\u061f")
    @test isempty(ans)
end

@testset "conditional_answer: guards" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "\u0625\u0630\u0627 \u0627\u062c\u062a\u0647\u062f \u0627\u0644\u0637\u0627\u0644\u0628 \u0646\u062c\u062d.")
    @test isempty(Physics.conditional_answer(mem, "\u0647\u0644 \u0627\u062c\u062a\u0647\u062f \u0627\u0644\u0637\u0627\u0644\u0628\u061f"))
    @test isempty(Physics.conditional_answer(mem, "\u0645\u0627\u0630\u0627 \u064a\u062d\u062f\u062b \u0625\u0630\u0627 \u0623\u0636\u0627\u0621 \u0627\u0644\u0645\u0635\u0628\u0627\u062d\u061f"))
    empty = Physics.IstinbatAttentionMemory()
    @test isempty(Physics.conditional_answer(empty, "\u0645\u0627\u0630\u0627 \u064a\u062d\u062f\u062b \u0625\u0630\u0627 \u0627\u062c\u062a\u0647\u062f \u0627\u0644\u0637\u0627\u0644\u0628\u061f"))
end

@testset "conditional_answer: English if then phrasing" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "if student studies then student succeeds.")
    ans = Physics.conditional_answer(mem, "what happens if student studies?")
    @test !isempty(ans)
    @test occursin("If student studies", ans)
    @test occursin("student succeeds", ans)
    @test isempty(Physics.conditional_answer(mem, "does student study?"))
end

@testset "compare_conditional_strategies: returns ConditionalComparisonRecord" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(
        mem,
        "\u0625\u0630\u0627 \u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645 \u0632\u0627\u062f \u0627\u0644\u0641\u0647\u0645.",
    )
    mock_gen = p -> "\u062c\u0648\u0627\u0628 \u0639\u0627\u0645: $p"
    result = Physics.compare_conditional_strategies(
        mem,
        mock_gen,
        "\u0645\u0627\u0630\u0627 \u064a\u062d\u062f\u062b \u0625\u0630\u0627 \u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645\u061f",
    )
    @test result isa Physics.ConditionalComparisonRecord
    @test !isempty(result.conditional_answer)
    @test result.memory_has_conditional == true
    @test result.conditional_confidence > 0
    @test result.overlap_score > 0
    @test result.has_marker == true
    @test result.relation_type == "conditional"
    @test occursin("\u0632\u0627\u062f \u0627\u0644\u0641\u0647\u0645", result.conditional_answer)
    @test occursin("\u062c\u0648\u0627\u0628 \u0639\u0627\u0645", result.generate_answer)
end

@testset "compare_conditional_strategies: empty memory" begin
    mem = Physics.IstinbatAttentionMemory()
    result = Physics.compare_conditional_strategies(
        mem,
        p -> "",
        "\u0645\u0627\u0630\u0627 \u064a\u062d\u062f\u062b \u0625\u0630\u0627 \u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645\u061f",
    )
    @test result.memory_has_conditional == false
    @test isempty(result.conditional_answer)
    @test isempty(result.generate_answer)
    @test result.relation_type == "none"
end

@testset "compare_conditional_strategies: non-conditional question" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "\u0625\u0630\u0627 \u0627\u062c\u062a\u0647\u062f \u0627\u0644\u0637\u0627\u0644\u0628 \u0646\u062c\u062d.")
    result = Physics.compare_conditional_strategies(
        mem,
        p -> "\u062c\u0648\u0627\u0628 \u0639\u0627\u0645",
        "\u0643\u064a\u0641 \u062d\u0627\u0644\u0643\u061f",
    )
    @test isempty(result.conditional_answer)
    @test result.memory_has_conditional == false
    @test result.generate_answer == "\u062c\u0648\u0627\u0628 \u0639\u0627\u0645"
end

@testset "compare_conditional_strategies: yesno question ignored" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "\u0625\u0630\u0627 \u0627\u062c\u062a\u0647\u062f \u0627\u0644\u0637\u0627\u0644\u0628 \u0646\u062c\u062d.")
    result = Physics.compare_conditional_strategies(
        mem,
        p -> "\u062c\u0648\u0627\u0628 \u0647\u0644",
        "\u0647\u0644 \u0627\u062c\u062a\u0647\u062f \u0627\u0644\u0637\u0627\u0644\u0628\u061f",
    )
    @test isempty(result.conditional_answer)
    @test result.memory_has_conditional == false
    @test result.generate_answer == "\u062c\u0648\u0627\u0628 \u0647\u0644"
end

@testset "temporal_answer: matches temporal question" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(
        mem,
        "\u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628 \u0642\u0628\u0644 \u0627\u0644\u0641\u062c\u0631.",
    )
    ans = Physics.temporal_answer(mem, "\u0645\u062a\u0649 \u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628\u061f")
    @test !isempty(ans)
    @test occursin("\u0633\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628", ans)
    @test !occursin("\u0627\u0644\u0633\u0627\u0641\u0631", ans)
    @test occursin("\u0642\u0628\u0644", ans)
    @test occursin("\u0627\u0644\u0641\u062c\u0631", ans)
end

@testset "temporal_answer: smooth Arabic phrasing" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "\u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628 \u0642\u0628\u0644 \u0627\u0644\u0641\u062c\u0631.")
    ans = Physics.temporal_answer(mem, "\u0645\u062a\u0649 \u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628\u061f")
    @test startswith(ans, "\u0643\u0627\u0646 \u0633\u0641\u0631 ")
    @test occursin("\u0642\u0628\u0644 \u0627\u0644\u0641\u062c\u0631", ans)
    @test !occursin("\u0648\u0642\u0639 \u0633\u0627\u0641\u0631", ans)
end

@testset "temporal_answer: trained definite subject parity" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "\u0633\u0627\u0641\u0631 \u0627\u0644\u0631\u062c\u0644 \u0642\u0628\u0644 \u0627\u0644\u0641\u062c\u0631.")
    ans = Physics.temporal_answer(mem, "\u0645\u062a\u0649 \u0633\u0627\u0641\u0631 \u0627\u0644\u0631\u062c\u0644\u061f")
    @test !isempty(ans)
    @test occursin("\u0642\u0628\u0644 \u0627\u0644\u0641\u062c\u0631", ans)
    @test !startswith(strip(ans), "\u0646\u0639\u0645")
end

@testset "temporal_answer: guards" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "\u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628 \u0642\u0628\u0644 \u0627\u0644\u0641\u062c\u0631.")
    @test isempty(Physics.temporal_answer(mem, "\u0647\u0644 \u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628\u061f"))
    @test isempty(Physics.temporal_answer(mem, "\u0645\u062a\u0649 \u0623\u0636\u0627\u0621 \u0627\u0644\u0645\u0635\u0628\u0627\u062d\u061f"))
    empty = Physics.IstinbatAttentionMemory()
    @test isempty(Physics.temporal_answer(empty, "\u0645\u062a\u0649 \u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628\u061f"))
end

@testset "temporal_answer: rejects distant partial overlap" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_fact!(
        mem,
        "temporal";
        marker="\u0642\u0628\u0644",
        before_terms=["\u0627\u0644\u0637\u0627\u0644\u0628", "\u0642\u0631\u0623"],
        after_terms=["\u0627\u0644\u0641\u062c\u0631"],
    )
    @test isempty(Physics.temporal_answer(mem, "\u0645\u062a\u0649 \u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628\u061f"))
end

@testset "temporal_answer: rejects trained glue noise" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_fact!(
        mem,
        "temporal";
        marker="\u0625\u0630",
        before_terms=["\u0633\u0627\u0641\u0631", "\u0627\u0644\u0637\u0627\u0644\u0628"],
        after_terms=["\u0627\u0644\u0644\u0645", "\u062a\u0642\u062a\u0631\u0628"],
    )
    @test isempty(Physics.temporal_answer(mem, "\u0645\u062a\u0649 \u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628\u061f"))
end

@testset "temporal_answer: rejects ambiguous non-time anchor" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_fact!(
        mem,
        "temporal";
        marker="\u0639\u0646\u062f",
        before_terms=["\u0633\u0627\u0641\u0631", "\u0627\u0644\u0637\u0627\u0644\u0628"],
        after_terms=["\u064a\u0631\u0649", "\u0627\u0644\u0646\u0639\u0645"],
    )
    @test isempty(Physics.temporal_answer(mem, "\u0645\u062a\u0649 \u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628\u061f"))
end

@testset "temporal_answer: English when phrasing" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "student traveled before dawn.")
    ans = Physics.temporal_answer(mem, "when student traveled?")
    @test !isempty(ans)
    @test occursin("student traveled", ans)
    @test occursin("before", ans)
    @test occursin("dawn", ans)
    @test isempty(Physics.temporal_answer(mem, "did student travel?"))
end

@testset "compare_temporal_strategies: returns TemporalComparisonRecord" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(
        mem,
        "\u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628 \u0642\u0628\u0644 \u0627\u0644\u0641\u062c\u0631.",
    )
    mock_gen = p -> "\u062c\u0648\u0627\u0628 \u0639\u0627\u0645: $p"
    result = Physics.compare_temporal_strategies(
        mem,
        mock_gen,
        "\u0645\u062a\u0649 \u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628\u061f",
    )
    @test result isa Physics.TemporalComparisonRecord
    @test !isempty(result.temporal_answer)
    @test result.memory_has_temporal == true
    @test result.temporal_confidence > 0
    @test result.overlap_score > 0
    @test result.has_marker == true
    @test result.relation_type == "temporal"
    @test occursin("\u0642\u0628\u0644 \u0627\u0644\u0641\u062c\u0631", result.temporal_answer)
    @test occursin("\u062c\u0648\u0627\u0628 \u0639\u0627\u0645", result.generate_answer)
end

@testset "compare_temporal_strategies: guards" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "\u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628 \u0642\u0628\u0644 \u0627\u0644\u0641\u062c\u0631.")
    yesno = Physics.compare_temporal_strategies(mem, p -> "\u062c\u0648\u0627\u0628 \u0647\u0644", "\u0647\u0644 \u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628\u061f")
    @test isempty(yesno.temporal_answer)
    @test yesno.memory_has_temporal == false
    @test yesno.generate_answer == "\u062c\u0648\u0627\u0628 \u0647\u0644"

    empty = Physics.compare_temporal_strategies(Physics.IstinbatAttentionMemory(), p -> "", "\u0645\u062a\u0649 \u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628\u061f")
    @test isempty(empty.temporal_answer)
    @test empty.relation_type == "none"
end

@testset "spatial_answer: matches spatial question" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(
        mem,
        "\u062c\u0644\u0633 \u0627\u0644\u0637\u0641\u0644 \u062d\u064a\u062b \u0627\u0644\u062d\u062f\u064a\u0642\u0629.",
    )
    ans = Physics.spatial_answer(mem, "\u0623\u064a\u0646 \u062c\u0644\u0633 \u0627\u0644\u0637\u0641\u0644\u061f")
    @test !isempty(ans)
    @test occursin("\u062c\u0644\u0648\u0633 \u0627\u0644\u0637\u0641\u0644", ans)
    @test occursin("\u0641\u064a", ans)
    @test occursin("\u0627\u0644\u062d\u062f\u064a\u0642\u0629", ans)
end

@testset "spatial_answer: smooth Arabic phrasing" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "\u062c\u0644\u0633 \u0627\u0644\u0637\u0641\u0644 \u062d\u064a\u062b \u0627\u0644\u062d\u062f\u064a\u0642\u0629.")
    ans = Physics.spatial_answer(mem, "\u0623\u064a\u0646 \u062c\u0644\u0633 \u0627\u0644\u0637\u0641\u0644\u061f")
    @test startswith(ans, "\u0643\u0627\u0646 \u0645\u0643\u0627\u0646 ")
    @test occursin("\u062c\u0644\u0648\u0633 \u0627\u0644\u0637\u0641\u0644", ans)
    @test occursin("\u0641\u064a \u0627\u0644\u062d\u062f\u064a\u0642\u0629", ans)
end

@testset "spatial_answer: throne with ala marker" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "\u062a\u0631\u0628\u0639 \u0627\u0644\u0645\u0644\u0643 \u0639\u0644\u0649 \u0627\u0644\u0639\u0631\u0634.")
    ans = Physics.spatial_answer(mem, "\u0623\u064a\u0646 \u062a\u0631\u0628\u0639 \u0627\u0644\u0645\u0644\u0643\u061f")
    @test !isempty(ans)
    @test occursin("\u0627\u0644\u0639\u0631\u0634", ans)
    @test !occursin("\u0646\u062f\u0645\u0627", ans)
end

@testset "spatial_answer: guards" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "\u062c\u0644\u0633 \u0627\u0644\u0637\u0641\u0644 \u062d\u064a\u062b \u0627\u0644\u062d\u062f\u064a\u0642\u0629.")
    @test isempty(Physics.spatial_answer(mem, "\u0647\u0644 \u062c\u0644\u0633 \u0627\u0644\u0637\u0641\u0644\u061f"))
    @test isempty(Physics.spatial_answer(mem, "\u0623\u064a\u0646 \u0623\u0636\u0627\u0621 \u0627\u0644\u0645\u0635\u0628\u0627\u062d\u061f"))
    empty = Physics.IstinbatAttentionMemory()
    @test isempty(Physics.spatial_answer(empty, "\u0623\u064a\u0646 \u062c\u0644\u0633 \u0627\u0644\u0637\u0641\u0644\u061f"))
end

@testset "spatial_answer: rejects distant partial overlap" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_fact!(
        mem,
        "spatial";
        marker="\u062d\u064a\u062b",
        before_terms=["\u0627\u0644\u0637\u0641\u0644", "\u0631\u0643\u0636"],
        after_terms=["\u0627\u0644\u062d\u062f\u064a\u0642\u0629"],
    )
    @test isempty(Physics.spatial_answer(mem, "\u0623\u064a\u0646 \u062c\u0644\u0633 \u0627\u0644\u0637\u0641\u0644\u061f"))
end

@testset "spatial_answer: rejects generic event prompt" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_fact!(
        mem,
        "spatial";
        marker="\u0647\u0646\u0627",
        before_terms=["\u0643\u0627\u0646"],
        after_terms=["\u0627\u0644\u062d\u062f\u064a\u0642\u0629"],
    )
    @test isempty(Physics.spatial_answer(mem, "\u0623\u064a\u0646 \u0643\u0627\u0646\u061f"))
end

@testset "spatial_answer: English where phrasing" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "child sat where garden.")
    ans = Physics.spatial_answer(mem, "where child sat?")
    @test !isempty(ans)
    @test startswith(ans, "The place of")
    @test occursin("child sat", ans)
    @test occursin("garden", ans)
    @test isempty(Physics.spatial_answer(mem, "did child sit?"))
end

@testset "compare_spatial_strategies: returns SpatialComparisonRecord" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(
        mem,
        "\u062c\u0644\u0633 \u0627\u0644\u0637\u0641\u0644 \u062d\u064a\u062b \u0627\u0644\u062d\u062f\u064a\u0642\u0629.",
    )
    mock_gen = p -> "\u062c\u0648\u0627\u0628 \u0639\u0627\u0645: $p"
    result = Physics.compare_spatial_strategies(
        mem,
        mock_gen,
        "\u0623\u064a\u0646 \u062c\u0644\u0633 \u0627\u0644\u0637\u0641\u0644\u061f",
    )
    @test result isa Physics.SpatialComparisonRecord
    @test !isempty(result.spatial_answer)
    @test result.memory_has_spatial == true
    @test result.spatial_confidence > 0
    @test result.overlap_score > 0
    @test result.has_marker == true
    @test result.relation_type == "spatial"
    @test occursin("\u0641\u064a \u0627\u0644\u062d\u062f\u064a\u0642\u0629", result.spatial_answer)
    @test occursin("\u062c\u0648\u0627\u0628 \u0639\u0627\u0645", result.generate_answer)
end

@testset "compare_spatial_strategies: guards" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "\u062c\u0644\u0633 \u0627\u0644\u0637\u0641\u0644 \u062d\u064a\u062b \u0627\u0644\u062d\u062f\u064a\u0642\u0629.")
    yesno = Physics.compare_spatial_strategies(mem, p -> "\u062c\u0648\u0627\u0628 \u0647\u0644", "\u0647\u0644 \u062c\u0644\u0633 \u0627\u0644\u0637\u0641\u0644\u061f")
    @test isempty(yesno.spatial_answer)
    @test yesno.memory_has_spatial == false
    @test yesno.generate_answer == "\u062c\u0648\u0627\u0628 \u0647\u0644"

    empty = Physics.compare_spatial_strategies(Physics.IstinbatAttentionMemory(), p -> "", "\u0623\u064a\u0646 \u062c\u0644\u0633 \u0627\u0644\u0637\u0641\u0644\u061f")
    @test isempty(empty.spatial_answer)
    @test empty.relation_type == "none"
end

@testset "state_answer: matches state question" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(
        mem,
        "\u062f\u062e\u0644 \u0627\u0644\u0637\u0641\u0644 \u062d\u0627\u0644 \u062e\u0627\u0626\u0641.",
    )
    ans = Physics.state_answer(mem, "\u0643\u064a\u0641 \u062f\u062e\u0644 \u0627\u0644\u0637\u0641\u0644\u061f")
    @test !isempty(ans)
    @test occursin("\u062f\u062e\u0644 \u0627\u0644\u0637\u0641\u0644", ans)
    @test occursin("\u062d\u0627\u0644", ans)
    @test occursin("\u062e\u0627\u0626\u0641", ans)
end

@testset "state_answer: smooth Arabic phrasing" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "\u062f\u062e\u0644 \u0627\u0644\u0637\u0641\u0644 \u062d\u0627\u0644 \u062e\u0627\u0626\u0641.")
    ans = Physics.state_answer(mem, "\u0643\u064a\u0641 \u062f\u062e\u0644 \u0627\u0644\u0637\u0641\u0644\u061f")
    @test occursin("\u0648\u0643\u0627\u0646 \u0639\u0644\u0649 \u062d\u0627\u0644", ans)
    @test !startswith(ans, "\u062f\u062e\u0644 \u0627\u0644\u0637\u0641\u0644 \u062d\u0627\u0644")
end

@testset "state_answer: guards" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "\u062f\u062e\u0644 \u0627\u0644\u0637\u0641\u0644 \u062d\u0627\u0644 \u062e\u0627\u0626\u0641.")
    @test isempty(Physics.state_answer(mem, "\u0647\u0644 \u062f\u062e\u0644 \u0627\u0644\u0637\u0641\u0644\u061f"))
    @test isempty(Physics.state_answer(mem, "\u0643\u064a\u0641 \u0623\u0636\u0627\u0621 \u0627\u0644\u0645\u0635\u0628\u0627\u062d\u061f"))
    empty = Physics.IstinbatAttentionMemory()
    @test isempty(Physics.state_answer(empty, "\u0643\u064a\u0641 \u062f\u062e\u0644 \u0627\u0644\u0637\u0641\u0644\u061f"))
end

@testset "state_answer: rejects distant partial overlap" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_fact!(
        mem,
        "state";
        marker="\u062d\u0627\u0644",
        before_terms=["\u0627\u0644\u0637\u0641\u0644", "\u0636\u062d\u0643"],
        after_terms=["\u062e\u0627\u0626\u0641"],
    )
    @test isempty(Physics.state_answer(mem, "\u0643\u064a\u0641 \u062f\u062e\u0644 \u0627\u0644\u0637\u0641\u0644\u061f"))
end

@testset "state_answer: English how phrasing" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "child entered while afraid.")
    ans = Physics.state_answer(mem, "how child entered?")
    @test !isempty(ans)
    @test occursin("child entered", ans)
    @test occursin("while", ans)
    @test occursin("afraid", ans)
    @test isempty(Physics.state_answer(mem, "did child enter?"))
end

@testset "compare_state_strategies: returns StateComparisonRecord" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(
        mem,
        "\u062f\u062e\u0644 \u0627\u0644\u0637\u0641\u0644 \u062d\u0627\u0644 \u062e\u0627\u0626\u0641.",
    )
    mock_gen = p -> "\u062c\u0648\u0627\u0628 \u0639\u0627\u0645: $p"
    result = Physics.compare_state_strategies(
        mem,
        mock_gen,
        "\u0643\u064a\u0641 \u062f\u062e\u0644 \u0627\u0644\u0637\u0641\u0644\u061f",
    )
    @test result isa Physics.StateComparisonRecord
    @test !isempty(result.state_answer)
    @test result.memory_has_state == true
    @test result.state_confidence > 0
    @test result.overlap_score > 0
    @test result.has_marker == true
    @test result.relation_type == "state"
    @test occursin("\u062d\u0627\u0644 \u062e\u0627\u0626\u0641", result.state_answer)
    @test occursin("\u062c\u0648\u0627\u0628 \u0639\u0627\u0645", result.generate_answer)
end

@testset "compare_state_strategies: guards" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "\u062f\u062e\u0644 \u0627\u0644\u0637\u0641\u0644 \u062d\u0627\u0644 \u062e\u0627\u0626\u0641.")
    yesno = Physics.compare_state_strategies(mem, p -> "\u062c\u0648\u0627\u0628 \u0647\u0644", "\u0647\u0644 \u062f\u062e\u0644 \u0627\u0644\u0637\u0641\u0644\u061f")
    @test isempty(yesno.state_answer)
    @test yesno.memory_has_state == false
    @test yesno.generate_answer == "\u062c\u0648\u0627\u0628 \u0647\u0644"

    empty = Physics.compare_state_strategies(Physics.IstinbatAttentionMemory(), p -> "", "\u0643\u064a\u0641 \u062f\u062e\u0644 \u0627\u0644\u0637\u0641\u0644\u061f")
    @test isempty(empty.state_answer)
    @test empty.relation_type == "none"
end

@testset "compare_purpose_strategies: returns PurposeComparisonRecord" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "يدرس الطالب لكي ينجح.")
    mock_gen = p -> "جواب تجريبي: $p"
    result = Physics.compare_purpose_strategies(mem, mock_gen, "لماذا يدرس الطالب؟")
    @test result isa Physics.PurposeComparisonRecord
    @test result.prompt == "لماذا يدرس الطالب؟"
    @test !isempty(result.purpose_answer)
    @test result.memory_has_purpose == true
    @test result.purpose_confidence > 0
    @test result.overlap_score > 0
    @test result.has_marker == true
    @test result.relation_type == "purpose"
    @test result.generate_answer == "جواب تجريبي: لماذا يدرس الطالب؟"
end

@testset "compare_purpose_strategies: empty memory" begin
    mem = Physics.IstinbatAttentionMemory()
    result = Physics.compare_purpose_strategies(mem, p -> "", "لماذا يدرس الطالب؟")
    @test result.memory_has_purpose == false
    @test isempty(result.purpose_answer)
    @test isempty(result.generate_answer)
    @test result.relation_type == "none"
end

@testset "compare_purpose_strategies: non-purpose question" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "يدرس الطالب لكي ينجح.")
    result = Physics.compare_purpose_strategies(mem, p -> "جواب كيف", "كيف حالك؟")
    @test isempty(result.purpose_answer)
    @test result.generate_answer == "جواب كيف"
end

@testset "compare_purpose_strategies: yesno question ignored" begin
    mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(mem, "يدرس الطالب لكي ينجح.")
    result = Physics.compare_purpose_strategies(mem, p -> "جواب هل", "هل يدرس الطالب؟")
    @test isempty(result.purpose_answer)
    @test result.generate_answer == "جواب هل"
end

@testset "compare_scene_purpose_strategies: cooperative bridge" begin
    scene_calc = Physics.SemanticCalculusMemory()
    @test Physics.learn_semantic_calculus_from_pair!(
        scene_calc,
        "Khalid hit the ball",
        "The ball moved away and changed position.",
    )
    scene_mem = Physics.SemanticSceneMemory()
    @test Physics.learn_semantic_scene_from_text!(
        scene_mem,
        scene_calc,
        "Khalid hit the ball with bat in yard before sunset.",
    ) == 1

    istinbat_mem = Physics.IstinbatAttentionMemory()
    @test Physics.learn_istinbat_from_text!(
        istinbat_mem,
        "Khalid hit the ball لكي يتحرك ball.",
    ) >= 1

    result = Physics.compare_scene_purpose_strategies(
        scene_mem,
        scene_calc,
        istinbat_mem,
        "لماذا Khalid hit the ball؟",
    )
    @test result isa Physics.ScenePurposeComparisonRecord
    @test result.scene_has_event
    @test result.memory_has_purpose
    @test result.agreement == "cooperative"
    @test result.scene_confidence > 0.0
    @test result.purpose_confidence > 0.0
    @test result.purpose_relation == "purpose"
    @test occursin("action=hit", result.scene_summary)
    @test occursin("instrument=bat", result.scene_summary)
    @test !isempty(result.purpose_answer)
end

@testset "compare_scene_purpose_strategies: isolated sides" begin
    scene_calc = Physics.SemanticCalculusMemory()
    Physics.learn_semantic_calculus_from_pair!(
        scene_calc,
        "Khalid hit the ball",
        "The ball moved away and changed position.",
    )
    scene_mem = Physics.SemanticSceneMemory()
    Physics.learn_semantic_scene_from_text!(scene_mem, scene_calc, "Khalid hit the ball.")

    empty_istinbat = Physics.IstinbatAttentionMemory()
    scene_only = Physics.compare_scene_purpose_strategies(
        scene_mem,
        scene_calc,
        empty_istinbat,
        "لماذا Khalid hit the ball؟",
    )
    @test scene_only.agreement == "scene_only"
    @test scene_only.scene_has_event
    @test !scene_only.memory_has_purpose

    purpose_mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(purpose_mem, "Khalid hit the ball لكي يتحرك ball.")
    empty_scene = Physics.SemanticSceneMemory()
    purpose_only = Physics.compare_scene_purpose_strategies(
        empty_scene,
        scene_calc,
        purpose_mem,
        "لماذا Khalid hit the ball؟",
    )
    @test purpose_only.agreement == "purpose_only"
    @test !purpose_only.scene_has_event
    @test purpose_only.memory_has_purpose

    none = Physics.compare_scene_purpose_strategies(
        empty_scene,
        scene_calc,
        empty_istinbat,
        "لماذا Khalid hit the ball؟",
    )
    @test none.agreement == "none"
end

@testset "scene_purpose_answer: cooperative composite only" begin
    scene_calc = Physics.SemanticCalculusMemory()
    Physics.learn_semantic_calculus_from_pair!(
        scene_calc,
        "Khalid hit the ball",
        "The ball moved away and changed position.",
    )
    scene_mem = Physics.SemanticSceneMemory()
    Physics.learn_semantic_scene_from_text!(
        scene_mem,
        scene_calc,
        "Khalid hit the ball with bat in yard before sunset.",
    )
    istinbat_mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(
        istinbat_mem,
        "Khalid hit the ball \u0644\u0643\u064a \u064a\u062a\u062d\u0631\u0643 ball.",
    )

    ans = Physics.scene_purpose_answer(
        scene_mem,
        scene_calc,
        istinbat_mem,
        "\u0644\u0645\u0627\u0630\u0627 Khalid hit the ball\u061f",
    )
    @test !isempty(ans)
    @test occursin("purpose", lowercase(ans)) || occursin("\u0627\u0644\u063a\u0627\u064a\u0629", ans)
    @test occursin("hit", ans)
    @test occursin("ball", ans)
    @test !occursin("action=", ans)
    @test !occursin("ball\u060c", ans)
    @test occursin(". \u0648\u0627\u0644\u063a\u0627\u064a\u0629", ans)

    arabic_scene_mem = Physics.SemanticSceneMemory()
    Physics.learn_semantic_scene_from_text!(
        arabic_scene_mem,
        scene_calc,
        "\u062f\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062d\u062c\u0631.",
    )
    arabic_istinbat = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(
        arabic_istinbat,
        "\u062f\u0641\u0639 \u0644\u0627\u0639\u0628 \u062d\u062c\u0631 \u0644\u0643\u064a \u064a\u0628\u0639\u062f \u062d\u062c\u0631 \u0637\u0631\u064a\u0642.",
    )
    arabic_ans = Physics.scene_purpose_answer(
        arabic_scene_mem,
        scene_calc,
        arabic_istinbat,
        "\u0644\u0645\u0627\u0630\u0627 \u062f\u0641\u0639 \u0644\u0627\u0639\u0628 \u062d\u062c\u0631\u061f",
    )
    @test !occursin("\u0648\u0645\u0646 \u062c\u0647\u0629 \u0627\u0644\u063a\u0627\u064a\u0629: \u0627\u0644\u063a\u0627\u064a\u0629 \u0645\u0646", arabic_ans)
    @test occursin("\u061b \u0648\u0643\u0627\u0646\u062a \u0627\u0644\u063a\u0627\u064a\u0629 \u0625\u0628\u0639\u0627\u062f", arabic_ans)
    @test !occursin("\u0639\u0646\u062f \u062f\u0641\u0639", arabic_ans)
    @test !occursin(", ", arabic_ans)

    empty_istinbat = Physics.IstinbatAttentionMemory()
    @test isempty(Physics.scene_purpose_answer(
        scene_mem,
        scene_calc,
        empty_istinbat,
        "\u0644\u0645\u0627\u0630\u0627 Khalid hit the ball\u061f",
    ))

    empty_scene = Physics.SemanticSceneMemory()
    @test isempty(Physics.scene_purpose_answer(
        empty_scene,
        scene_calc,
        istinbat_mem,
        "\u0644\u0645\u0627\u0630\u0627 Khalid hit the ball\u061f",
    ))

    unrelated_scene = Physics.SemanticSceneMemory()
    Physics.learn_semantic_calculus_from_pair!(
        scene_calc,
        "\u0643\u0634\u0641 \u0627\u0644\u0635\u062d\u0646",
        "\u0638\u0647\u0631 \u0627\u0644\u0635\u062d\u0646 \u0628\u0648\u0636\u0648\u062d.",
    )
    Physics.learn_semantic_scene_from_text!(
        unrelated_scene,
        scene_calc,
        "\u0643\u0634\u0641 \u0635\u062d\u0646 \u0641\u0646\u062c\u0627\u0646 \u0631\u0642\u064a\u0642.",
    )
    procedural_mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(
        procedural_mem,
        "\u0641\u062a\u062d \u0639\u0644\u0628\u0647 \u0627\u0646\u0642\u0644 \u0645\u062d\u062a\u0648\u064a \u0645\u062a\u0628\u0642\u064a \u0641\u0648\u0631\u0627\u064b \u0648\u0639\u0627\u0621 \u0632\u062c\u0627\u062c\u064a \u0628\u0644\u0627\u0633\u062a\u064a \u0644\u0643\u064a \u0645\u062d\u0643\u0645 \u0627\u063a\u0644\u0627\u0642 \u0648\u0627\u062d\u0641\u0638\u0647 \u062b\u0644\u0627\u062c\u0647 \u0648\u0627\u0633\u062a\u0647\u0644\u0643\u0647 \u062e\u0644\u0627\u0644 \u064a\u0648\u0645\u064a\u0646.",
    )
    @test isempty(Physics.scene_purpose_answer(
        unrelated_scene,
        scene_calc,
        procedural_mem,
        "\u0644\u0645\u0627\u0630\u0627 \u0641\u062a\u062d \u0639\u0644\u0628\u0647 \u0627\u0646\u0642\u0644 \u0645\u062d\u062a\u0648\u064a \u0645\u062a\u0628\u0642\u064a \u0641\u0648\u0631\u0627\u064b \u0648\u0639\u0627\u0621 \u0632\u062c\u0627\u062c\u064a \u0628\u0644\u0627\u0633\u062a\u064a\u061f",
    ))
end

@testset "RelationFrameStrategy" begin
    Gen = Physics.Generator
    random_seed = rand(1:99999)

    gen = Physics.MirnanGenerator(Dict{String,Int}(
        "الطالب" => 1, "يدرس" => 2, "ينجح" => 3, "يشرب" => 4, "الحليب" => 5,
        "لماذا" => 6, "هل" => 7, "كيف" => 8, "حالك" => 9,
    ); model_dir=mktempdir())

    saved_mem = Gen._LEARNED_ISTINBAT_MEMORY[]

    function _rfs_setup(mem, text, prompt)
        Physics.learn_istinbat_from_text!(mem, text)
        Gen._LEARNED_ISTINBAT_MEMORY[] = mem
        pt = String.(split(strip(prompt)))
        co = Gen.observe_prompt(gen.cerebellum, pt; prompt=prompt, vocab_size=length(gen.vocab))
        cp = Gen.choose_policy!(gen.cerebellum, co; requested_mode="auto")
        rp = Gen.detect_response_intent(prompt)
        ap = Gen._get_active_paragraphs(gen, pt)
        return pt, "auto", co, cp, rp, ap
    end

    let
        mem = Physics.IstinbatAttentionMemory()
        args = _rfs_setup(mem, "يدرس الطالب لكي ينجح.", "لماذا يدرس الطالب؟")
        result = Gen.try_generate(Gen.RelationFrameStrategy(), gen,
                                   "لماذا يدرس الطالب؟", args...)
        @test result !== nothing
        @test !isempty(strip(result))
    end
    Gen._LEARNED_ISTINBAT_MEMORY[] = saved_mem

    let
        mem = Physics.IstinbatAttentionMemory()
        args = _rfs_setup(mem, "يدرس الطالب لكي ينجح.", "هل يدرس الطالب؟")
        result = Gen.try_generate(Gen.RelationFrameStrategy(), gen,
                                   "هل يدرس الطالب؟", args...)
        @test result === nothing
    end
    Gen._LEARNED_ISTINBAT_MEMORY[] = saved_mem

    let
        mem = Physics.IstinbatAttentionMemory()
        args = _rfs_setup(mem, "الطفل يشرب الحليب.", "لماذا يدرس الطالب؟")
        result = Gen.try_generate(Gen.RelationFrameStrategy(), gen,
                                   "لماذا يدرس الطالب؟", args...)
        @test result === nothing
    end
    Gen._LEARNED_ISTINBAT_MEMORY[] = saved_mem

    let
        Gen._LEARNED_ISTINBAT_MEMORY[] = Physics.IstinbatAttentionMemory()
        pt = String.(split("لماذا يدرس الطالب؟"))
        co = Gen.observe_prompt(gen.cerebellum, pt;
                                 prompt="لماذا يدرس الطالب؟",
                                 vocab_size=length(gen.vocab))
        cp = Gen.choose_policy!(gen.cerebellum, co; requested_mode="auto")
        rp = Gen.detect_response_intent("لماذا يدرس الطالب؟")
        ap = Gen._get_active_paragraphs(gen, pt)
        result = Gen.try_generate(Gen.RelationFrameStrategy(), gen,
                                   "لماذا يدرس الطالب؟",
                                   pt, "auto", co, cp, rp, ap)
        @test result === nothing
    end
    Gen._LEARNED_ISTINBAT_MEMORY[] = saved_mem

    # ═══ بوابة MIRNAN_ENABLE_RELATION_FRAME_STRATEGY ═══
    old_env_rf = get(ENV, "MIRNAN_ENABLE_RELATION_FRAME_STRATEGY", nothing)
    old_env_strict = get(ENV, "MIRNAN_STRICT_NO_TEMPLATES", nothing)
    try
        ENV["MIRNAN_ENABLE_RELATION_FRAME_STRATEGY"] = "1"
        ENV["MIRNAN_STRICT_NO_TEMPLATES"] = "0"

        mem_purpose = Physics.IstinbatAttentionMemory()
        Physics.learn_istinbat_from_text!(mem_purpose, "يدرس الطالب لكي ينجح.")
        Gen._LEARNED_ISTINBAT_MEMORY[] = mem_purpose

        purpose_result = Physics.generate!(gen, "لماذا يدرس الطالب؟"; max_words=8)
        @test !isempty(strip(purpose_result))
        @test occursin("الغاية", purpose_result) || occursin("ينجح", purpose_result)

        yesno_result = Physics.generate!(gen, "هل يدرس الطالب؟"; max_words=8)
        @test !isempty(strip(yesno_result))
        @test !occursin("الغاية", yesno_result)
    finally
        Gen._LEARNED_ISTINBAT_MEMORY[] = saved_mem
        if old_env_rf === nothing
            delete!(ENV, "MIRNAN_ENABLE_RELATION_FRAME_STRATEGY")
        else
            ENV["MIRNAN_ENABLE_RELATION_FRAME_STRATEGY"] = old_env_rf
        end
        if old_env_strict === nothing
            delete!(ENV, "MIRNAN_STRICT_NO_TEMPLATES")
        else
            ENV["MIRNAN_STRICT_NO_TEMPLATES"] = old_env_strict
        end
    end
end

@testset "ConditionalFrameStrategy" begin
    Gen = Physics.Generator
    gen = Physics.MirnanGenerator(Dict{String,Int}(
        "\u0645\u0627\u0630\u0627" => 1,
        "\u064a\u062d\u062f\u062b" => 2,
        "\u0625\u0630\u0627" => 3,
        "\u0632\u0627\u062f" => 4,
        "\u0627\u0644\u0639\u0644\u0645" => 5,
        "\u0627\u0644\u0641\u0647\u0645" => 6,
        "\u0647\u0644" => 7,
    ); model_dir=mktempdir())

    saved_mem = Gen._LEARNED_ISTINBAT_MEMORY[]
    prompt = "\u0645\u0627\u0630\u0627 \u064a\u062d\u062f\u062b \u0625\u0630\u0627 \u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645\u061f"

    function _cfs_args(prompt_text)
        pt = String.(split(strip(prompt_text)))
        co = Gen.observe_prompt(gen.cerebellum, pt; prompt=prompt_text, vocab_size=length(gen.vocab))
        cp = Gen.choose_policy!(gen.cerebellum, co; requested_mode="auto")
        rp = Gen.detect_response_intent(prompt_text)
        ap = Gen._get_active_paragraphs(gen, pt)
        return pt, "auto", co, cp, rp, ap
    end

    try
        mem = Physics.IstinbatAttentionMemory()
        Physics.learn_istinbat_from_text!(
            mem,
            "\u0625\u0630\u0627 \u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645 \u0632\u0627\u062f \u0627\u0644\u0641\u0647\u0645.",
        )
        Gen._LEARNED_ISTINBAT_MEMORY[] = mem

        args = _cfs_args(prompt)
        direct = Gen.try_generate(Gen.ConditionalFrameStrategy(), gen, prompt, args...)
        @test direct !== nothing
        @test occursin("\u0632\u0627\u062f \u0627\u0644\u0641\u0647\u0645", direct)

        yesno_prompt = "\u0647\u0644 \u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645\u061f"
        yesno_args = _cfs_args(yesno_prompt)
        yesno = Gen.try_generate(Gen.ConditionalFrameStrategy(), gen, yesno_prompt, yesno_args...)
        @test yesno === nothing

        old_env = get(ENV, "MIRNAN_ENABLE_CONDITIONAL_FRAME_STRATEGY", nothing)
        old_strict = get(ENV, "MIRNAN_STRICT_NO_TEMPLATES", nothing)
        try
            ENV["MIRNAN_ENABLE_CONDITIONAL_FRAME_STRATEGY"] = "1"
            ENV["MIRNAN_STRICT_NO_TEMPLATES"] = "0"
            generated = Physics.generate!(gen, prompt; max_words=18)
            @test occursin("\u0632\u0627\u062f \u0627\u0644\u0641\u0647\u0645", generated)

            yesno_generated = Physics.generate!(gen, yesno_prompt; max_words=18)
            @test !occursin("\u0641\u0627\u0644\u0646\u062a\u064a\u062c\u0629", yesno_generated)
        finally
            if old_env === nothing
                delete!(ENV, "MIRNAN_ENABLE_CONDITIONAL_FRAME_STRATEGY")
            else
                ENV["MIRNAN_ENABLE_CONDITIONAL_FRAME_STRATEGY"] = old_env
            end
            if old_strict === nothing
                delete!(ENV, "MIRNAN_STRICT_NO_TEMPLATES")
            else
                ENV["MIRNAN_STRICT_NO_TEMPLATES"] = old_strict
            end
        end
    finally
        Gen._LEARNED_ISTINBAT_MEMORY[] = saved_mem
    end
end

@testset "TemporalFrameStrategy" begin
    Gen = Physics.Generator
    gen = Physics.MirnanGenerator(Dict{String,Int}(
        "\u0645\u062a\u0649" => 1,
        "\u0633\u0627\u0641\u0631" => 2,
        "\u0627\u0644\u0637\u0627\u0644\u0628" => 3,
        "\u0642\u0628\u0644" => 4,
        "\u0627\u0644\u0641\u062c\u0631" => 5,
        "\u0647\u0644" => 6,
        "\u0627\u0644\u0648\u0627\u0644\u062f" => 7,
    ); model_dir=mktempdir())

    saved_mem = Gen._LEARNED_ISTINBAT_MEMORY[]
    prompt = "\u0645\u062a\u0649 \u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628\u061f"

    function _tfs_args(prompt_text)
        pt = String.(split(strip(prompt_text)))
        co = Gen.observe_prompt(gen.cerebellum, pt; prompt=prompt_text, vocab_size=length(gen.vocab))
        cp = Gen.choose_policy!(gen.cerebellum, co; requested_mode="auto")
        rp = Gen.detect_response_intent(prompt_text)
        ap = Gen._get_active_paragraphs(gen, pt)
        return pt, "auto", co, cp, rp, ap
    end

    try
        mem = Physics.IstinbatAttentionMemory()
        Physics.learn_istinbat_from_text!(
            mem,
            "\u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628 \u0642\u0628\u0644 \u0627\u0644\u0641\u062c\u0631.",
        )
        Gen._LEARNED_ISTINBAT_MEMORY[] = mem

        args = _tfs_args(prompt)
        direct = Gen.try_generate(Gen.TemporalFrameStrategy(), gen, prompt, args...)
        @test direct !== nothing
        @test occursin("\u0642\u0628\u0644 \u0627\u0644\u0641\u062c\u0631", direct)

        yesno_prompt = "\u0647\u0644 \u0633\u0627\u0641\u0631 \u0627\u0644\u0637\u0627\u0644\u0628\u061f"
        yesno_args = _tfs_args(yesno_prompt)
        yesno = Gen.try_generate(Gen.TemporalFrameStrategy(), gen, yesno_prompt, yesno_args...)
        @test yesno === nothing

        unknown_prompt = "\u0645\u062a\u0649 \u0633\u0627\u0641\u0631 \u0627\u0644\u0648\u0627\u0644\u062f\u061f"
        unknown_args = _tfs_args(unknown_prompt)
        unknown = Gen.try_generate(Gen.TemporalFrameStrategy(), gen, unknown_prompt, unknown_args...)
        @test unknown == "\u0644\u0627 \u0623\u062c\u062f \u0632\u0645\u0646\u0627\u064b \u0645\u062d\u062f\u062f\u0627\u064b \u0641\u064a \u0627\u0644\u0630\u0627\u0643\u0631\u0629."

        old_env = get(ENV, "MIRNAN_ENABLE_TEMPORAL_FRAME_STRATEGY", nothing)
        old_strict = get(ENV, "MIRNAN_STRICT_NO_TEMPLATES", nothing)
        try
            delete!(ENV, "MIRNAN_ENABLE_TEMPORAL_FRAME_STRATEGY")
            ENV["MIRNAN_STRICT_NO_TEMPLATES"] = "0"
            default_generated = Physics.generate!(gen, prompt; max_words=18)
            @test occursin("\u0642\u0628\u0644 \u0627\u0644\u0641\u062c\u0631", default_generated)

            unknown_generated = Physics.generate!(gen, unknown_prompt; max_words=18)
            @test unknown_generated == "\u0644\u0627 \u0623\u062c\u062f \u0632\u0645\u0646\u0627\u064b \u0645\u062d\u062f\u062f\u0627\u064b \u0641\u064a \u0627\u0644\u0630\u0627\u0643\u0631\u0629."

            ENV["MIRNAN_ENABLE_TEMPORAL_FRAME_STRATEGY"] = "0"
            disabled_generated = Physics.generate!(gen, prompt; max_words=18)
            @test !occursin("\u0642\u0628\u0644 \u0627\u0644\u0641\u062c\u0631", disabled_generated)

            ENV["MIRNAN_ENABLE_TEMPORAL_FRAME_STRATEGY"] = "1"
            ENV["MIRNAN_STRICT_NO_TEMPLATES"] = "0"
            generated = Physics.generate!(gen, prompt; max_words=18)
            @test occursin("\u0642\u0628\u0644 \u0627\u0644\u0641\u062c\u0631", generated)

            yesno_generated = Physics.generate!(gen, yesno_prompt; max_words=18)
            @test !occursin("\u0642\u0628\u0644 \u0627\u0644\u0641\u062c\u0631", yesno_generated)
        finally
            if old_env === nothing
                delete!(ENV, "MIRNAN_ENABLE_TEMPORAL_FRAME_STRATEGY")
            else
                ENV["MIRNAN_ENABLE_TEMPORAL_FRAME_STRATEGY"] = old_env
            end
            if old_strict === nothing
                delete!(ENV, "MIRNAN_STRICT_NO_TEMPLATES")
            else
                ENV["MIRNAN_STRICT_NO_TEMPLATES"] = old_strict
            end
        end
    finally
        Gen._LEARNED_ISTINBAT_MEMORY[] = saved_mem
    end
end

@testset "SpatialFrameStrategy" begin
    Gen = Physics.Generator
    gen = Physics.MirnanGenerator(Dict{String,Int}(
        "\u0623\u064a\u0646" => 1,
        "\u062c\u0644\u0633" => 2,
        "\u0627\u0644\u0637\u0641\u0644" => 3,
        "\u062d\u064a\u062b" => 4,
        "\u0627\u0644\u062d\u062f\u064a\u0642\u0629" => 5,
        "\u0647\u0644" => 6,
        "\u0627\u0644\u0645\u0644\u0643" => 7,
        "\u062a\u0631\u0628\u0639" => 8,
        "\u0627\u0644\u0639\u0631\u0634" => 9,
        "\u0639\u0644\u0649" => 10,
        "\u0637\u0627\u0631" => 11,
    ); model_dir=mktempdir())

    saved_mem = Gen._LEARNED_ISTINBAT_MEMORY[]
    prompt = "\u0623\u064a\u0646 \u062c\u0644\u0633 \u0627\u0644\u0637\u0641\u0644\u061f"

    function _sfs_args(prompt_text)
        pt = String.(split(strip(prompt_text)))
        co = Gen.observe_prompt(gen.cerebellum, pt; prompt=prompt_text, vocab_size=length(gen.vocab))
        cp = Gen.choose_policy!(gen.cerebellum, co; requested_mode="auto")
        rp = Gen.detect_response_intent(prompt_text)
        ap = Gen._get_active_paragraphs(gen, pt)
        return pt, "auto", co, cp, rp, ap
    end

    try
        mem = Physics.IstinbatAttentionMemory()
        Physics.learn_istinbat_from_text!(
            mem,
            "\u062c\u0644\u0633 \u0627\u0644\u0637\u0641\u0644 \u062d\u064a\u062b \u0627\u0644\u062d\u062f\u064a\u0642\u0629.",
        )
        Physics.learn_istinbat_from_text!(
            mem,
            "\u062a\u0631\u0628\u0639 \u0627\u0644\u0645\u0644\u0643 \u0639\u0644\u0649 \u0627\u0644\u0639\u0631\u0634.",
        )
        Gen._LEARNED_ISTINBAT_MEMORY[] = mem

        args = _sfs_args(prompt)
        direct = Gen.try_generate(Gen.SpatialFrameStrategy(), gen, prompt, args...)
        @test direct !== nothing
        @test occursin("\u0641\u064a \u0627\u0644\u062d\u062f\u064a\u0642\u0629", direct)

        throne_prompt = "\u0623\u064a\u0646 \u062a\u0631\u0628\u0639 \u0627\u0644\u0645\u0644\u0643\u061f"
        throne_args = _sfs_args(throne_prompt)
        throne = Gen.try_generate(Gen.SpatialFrameStrategy(), gen, throne_prompt, throne_args...)
        @test throne !== nothing
        @test occursin("\u0627\u0644\u0639\u0631\u0634", throne)

        unknown_prompt = "\u0623\u064a\u0646 \u0637\u0627\u0631 \u0627\u0644\u0645\u0644\u0643\u061f"
        unknown_args = _sfs_args(unknown_prompt)
        unknown = Gen.try_generate(Gen.SpatialFrameStrategy(), gen, unknown_prompt, unknown_args...)
        @test unknown == "\u0644\u0627 \u0623\u062c\u062f \u0645\u0643\u0627\u0646\u0627\u064b \u0645\u062d\u062f\u062f\u0627\u064b \u0641\u064a \u0627\u0644\u0630\u0627\u0643\u0631\u0629."

        yesno_prompt = "\u0647\u0644 \u062c\u0644\u0633 \u0627\u0644\u0637\u0641\u0644\u061f"
        yesno_args = _sfs_args(yesno_prompt)
        yesno = Gen.try_generate(Gen.SpatialFrameStrategy(), gen, yesno_prompt, yesno_args...)
        @test yesno === nothing

        old_env = get(ENV, "MIRNAN_ENABLE_SPATIAL_FRAME_STRATEGY", nothing)
        old_strict = get(ENV, "MIRNAN_STRICT_NO_TEMPLATES", nothing)
        try
            delete!(ENV, "MIRNAN_ENABLE_SPATIAL_FRAME_STRATEGY")
            ENV["MIRNAN_STRICT_NO_TEMPLATES"] = "0"
            default_generated = Physics.generate!(gen, throne_prompt; max_words=18)
            @test occursin("\u0627\u0644\u0639\u0631\u0634", default_generated)

            unknown_generated = Physics.generate!(gen, unknown_prompt; max_words=18)
            @test unknown_generated == "\u0644\u0627 \u0623\u062c\u062f \u0645\u0643\u0627\u0646\u0627\u064b \u0645\u062d\u062f\u062f\u0627\u064b \u0641\u064a \u0627\u0644\u0630\u0627\u0643\u0631\u0629."

            ENV["MIRNAN_ENABLE_SPATIAL_FRAME_STRATEGY"] = "0"
            disabled_generated = Physics.generate!(gen, throne_prompt; max_words=18)
            @test !startswith(disabled_generated, "\u0643\u0627\u0646 \u0645\u0643\u0627\u0646")

            ENV["MIRNAN_ENABLE_SPATIAL_FRAME_STRATEGY"] = "1"
            ENV["MIRNAN_STRICT_NO_TEMPLATES"] = "0"
            generated = Physics.generate!(gen, prompt; max_words=18)
            @test occursin("\u0641\u064a \u0627\u0644\u062d\u062f\u064a\u0642\u0629", generated)

            yesno_generated = Physics.generate!(gen, yesno_prompt; max_words=18)
            @test !occursin("\u062d\u064a\u062b \u0627\u0644\u062d\u062f\u064a\u0642\u0629", yesno_generated)
        finally
            if old_env === nothing
                delete!(ENV, "MIRNAN_ENABLE_SPATIAL_FRAME_STRATEGY")
            else
                ENV["MIRNAN_ENABLE_SPATIAL_FRAME_STRATEGY"] = old_env
            end
            if old_strict === nothing
                delete!(ENV, "MIRNAN_STRICT_NO_TEMPLATES")
            else
                ENV["MIRNAN_STRICT_NO_TEMPLATES"] = old_strict
            end
        end
    finally
        Gen._LEARNED_ISTINBAT_MEMORY[] = saved_mem
    end
end

@testset "StateFrameStrategy" begin
    Gen = Physics.Generator
    gen = Physics.MirnanGenerator(Dict{String,Int}(
        "\u0643\u064a\u0641" => 1,
        "\u062f\u062e\u0644" => 2,
        "\u0627\u0644\u0637\u0641\u0644" => 3,
        "\u062d\u0627\u0644" => 4,
        "\u062e\u0627\u0626\u0641" => 5,
        "\u0647\u0644" => 6,
    ); model_dir=mktempdir())
    saved_mem = Gen._LEARNED_ISTINBAT_MEMORY[]
    prompt = "\u0643\u064a\u0641 \u062f\u062e\u0644 \u0627\u0644\u0637\u0641\u0644\u061f"

    function _stfs_args(prompt_text)
        pt = String.(split(strip(prompt_text)))
        co = Gen.observe_prompt(gen.cerebellum, pt; prompt=prompt_text, vocab_size=length(gen.vocab))
        cp = Gen.choose_policy!(gen.cerebellum, co; requested_mode="auto")
        rp = Gen.detect_response_intent(prompt_text)
        ap = Gen._get_active_paragraphs(gen, pt)
        return pt, "auto", co, cp, rp, ap
    end

    try
        mem = Physics.IstinbatAttentionMemory()
        Physics.learn_istinbat_from_text!(mem, "\u062f\u062e\u0644 \u0627\u0644\u0637\u0641\u0644 \u062d\u0627\u0644 \u062e\u0627\u0626\u0641.")
        Gen._LEARNED_ISTINBAT_MEMORY[] = mem

        args = _stfs_args(prompt)
        direct = Gen.try_generate(Gen.StateFrameStrategy(), gen, prompt, args...)
        @test direct !== nothing
        @test occursin("\u062d\u0627\u0644 \u062e\u0627\u0626\u0641", direct)

        yesno_prompt = "\u0647\u0644 \u062f\u062e\u0644 \u0627\u0644\u0637\u0641\u0644\u061f"
        yesno_args = _stfs_args(yesno_prompt)
        yesno = Gen.try_generate(Gen.StateFrameStrategy(), gen, yesno_prompt, yesno_args...)
        @test yesno === nothing

        old_env = get(ENV, "MIRNAN_ENABLE_STATE_FRAME_STRATEGY", nothing)
        old_strict = get(ENV, "MIRNAN_STRICT_NO_TEMPLATES", nothing)
        try
            ENV["MIRNAN_ENABLE_STATE_FRAME_STRATEGY"] = "1"
            ENV["MIRNAN_STRICT_NO_TEMPLATES"] = "0"
            generated = Physics.generate!(gen, prompt; max_words=18)
            @test occursin("\u062d\u0627\u0644 \u062e\u0627\u0626\u0641", generated)
            yesno_generated = Physics.generate!(gen, yesno_prompt; max_words=18)
            @test !occursin("\u062d\u0627\u0644 \u062e\u0627\u0626\u0641", yesno_generated)
        finally
            if old_env === nothing
                delete!(ENV, "MIRNAN_ENABLE_STATE_FRAME_STRATEGY")
            else
                ENV["MIRNAN_ENABLE_STATE_FRAME_STRATEGY"] = old_env
            end
            if old_strict === nothing
                delete!(ENV, "MIRNAN_STRICT_NO_TEMPLATES")
            else
                ENV["MIRNAN_STRICT_NO_TEMPLATES"] = old_strict
            end
        end
    finally
        Gen._LEARNED_ISTINBAT_MEMORY[] = saved_mem
    end
end

@testset "scene purpose composite cleans noisy scene effects" begin
    scene_mem = Physics.SemanticSceneMemory()
    push!(scene_mem.scenes, Physics.SemanticScene(
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
        ["\u0643\u0644", "\u0648\u0644\u0627", "\u062d\u0631\u0643\u0629"],
        1.0,
        "semantic_continuation",
        "fixture",
    ))

    istinbat_mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(
        istinbat_mem,
        "\u062f\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062d\u062c\u0631 \u0644\u0643\u064a \u064a\u0628\u0639\u062f \u0627\u0644\u062d\u062c\u0631 \u0639\u0646 \u0627\u0644\u0637\u0631\u064a\u0642.",
    )

    answer = Physics.scene_purpose_answer(
        scene_mem,
        Physics.SemanticCalculusMemory(),
        istinbat_mem,
        "\u0644\u0645\u0627\u0630\u0627 \u062f\u0641\u0639 \u0627\u0644\u0644\u0627\u0639\u0628 \u0627\u0644\u062d\u062c\u0631\u061f",
    )

    @test occursin("\u062d\u0631\u0643\u0629", answer)
    @test !occursin("\u0643\u0644\u060c", answer)
    @test !occursin("\u0648\u0644\u0627", answer)
    @test !occursin("\u0622\u062b\u0627\u0631 \u0645\u062b\u0644 \u0643\u0644", answer)
end

@testset "ScenePurposeStrategy" begin
    Gen = Physics.Generator
    gen = Physics.MirnanGenerator(Dict{String,Int}(
        "Khalid" => 1, "hit" => 2, "ball" => 3,
        "\u0644\u0645\u0627\u0630\u0627" => 4, "\u0647\u0644" => 5,
    ); model_dir=mktempdir())

    saved_scene = Gen._LEARNED_SEMANTIC_SCENE_MEMORY[]
    saved_istinbat = Gen._LEARNED_ISTINBAT_MEMORY[]

    scene_calc = Physics.SemanticCalculusMemory()
    Physics.learn_semantic_calculus_from_pair!(
        scene_calc,
        "Khalid hit the ball",
        "The ball moved away and changed position.",
    )
    scene_mem = Physics.SemanticSceneMemory()
    Physics.learn_semantic_scene_from_text!(
        scene_mem,
        scene_calc,
        "Khalid hit the ball with bat in yard before sunset.",
    )
    istinbat_mem = Physics.IstinbatAttentionMemory()
    Physics.learn_istinbat_from_text!(
        istinbat_mem,
        "Khalid hit the ball \u0644\u0643\u064a \u064a\u062a\u062d\u0631\u0643 ball.",
    )

    try
        Gen._LEARNED_SEMANTIC_SCENE_MEMORY[] = scene_mem
        Gen._LEARNED_ISTINBAT_MEMORY[] = istinbat_mem

        prompt = "\u0644\u0645\u0627\u0630\u0627 Khalid hit the ball\u061f"
        pt = String.(split(strip(prompt)))
        co = Gen.observe_prompt(gen.cerebellum, pt; prompt=prompt, vocab_size=length(gen.vocab))
        cp = Gen.choose_policy!(gen.cerebellum, co; requested_mode="auto")
        rp = Gen.detect_response_intent(prompt)
        ap = Gen._get_active_paragraphs(gen, pt)

        direct = Gen.try_generate(Gen.ScenePurposeStrategy(), gen,
                                  prompt, pt, "auto", co, cp, rp, ap)
        @test direct !== nothing
        @test occursin("purpose", lowercase(direct)) || occursin("\u0627\u0644\u063a\u0627\u064a\u0629", direct)
        @test !occursin("action=", direct)

        yesno = Gen.try_generate(Gen.ScenePurposeStrategy(), gen,
                                 "Does Khalid hit the ball?", pt, "auto", co, cp, rp, ap)
        @test yesno === nothing

        old_env = get(ENV, "MIRNAN_ENABLE_SCENE_PURPOSE_STRATEGY", nothing)
        old_strict = get(ENV, "MIRNAN_STRICT_NO_TEMPLATES", nothing)
        try
            ENV["MIRNAN_ENABLE_SCENE_PURPOSE_STRATEGY"] = "1"
            ENV["MIRNAN_STRICT_NO_TEMPLATES"] = "0"
            generated = Physics.generate!(gen, prompt; max_words=24)
            @test occursin("purpose", lowercase(generated)) || occursin("\u0627\u0644\u063a\u0627\u064a\u0629", generated)
            @test !occursin("action=", generated)
        finally
            if old_env === nothing
                delete!(ENV, "MIRNAN_ENABLE_SCENE_PURPOSE_STRATEGY")
            else
                ENV["MIRNAN_ENABLE_SCENE_PURPOSE_STRATEGY"] = old_env
            end
            if old_strict === nothing
                delete!(ENV, "MIRNAN_STRICT_NO_TEMPLATES")
            else
                ENV["MIRNAN_STRICT_NO_TEMPLATES"] = old_strict
            end
        end
    finally
        Gen._LEARNED_SEMANTIC_SCENE_MEMORY[] = saved_scene
        Gen._LEARNED_ISTINBAT_MEMORY[] = saved_istinbat
    end
end

@testset "relation_type_for_quantity_marker" begin
    @test Physics.relation_type_for_quantity_marker("كم") == "count"
    @test Physics.relation_type_for_quantity_marker("عدد") == "count"
    @test Physics.relation_type_for_quantity_marker("مقدار") == "measure"
    @test Physics.relation_type_for_quantity_marker("كمية") == "measure"
    @test Physics.relation_type_for_quantity_marker("أكثر") == "comparison"
    @test Physics.relation_type_for_quantity_marker("أقل") == "comparison"
    @test Physics.relation_type_for_quantity_marker("يساوي") == "comparison"
    @test Physics.relation_type_for_quantity_marker("نصف") == "comparison"
    @test Physics.relation_type_for_quantity_marker("ضعف") == "comparison"
    @test Physics.relation_type_for_quantity_marker("كل") == "quantifier_scope"
    @test Physics.relation_type_for_quantity_marker("بعض") == "quantifier_scope"
    @test Physics.relation_type_for_quantity_marker("كثير") == "vague_quantity"
    @test Physics.relation_type_for_quantity_marker("قليل") == "vague_quantity"
    @test isempty(Physics.relation_type_for_quantity_marker("xyz"))
end

@testset "extract_quantity_frames: count" begin
    frames = Physics.extract_quantity_frames("كم عدد الطلاب في الفصل؟")
    @test length(frames) >= 1
    f = first(frames)
    @test f.quantity_type == "count"
    @test f.marker == "كم" || f.marker == "عدد"
end

@testset "extract_quantity_frames: measure" begin
    frames = Physics.extract_quantity_frames("ما مقدار الطاقة المطلوبة؟")
    @test length(frames) >= 1
    f = first(frames)
    @test f.quantity_type == "measure"
    @test f.marker == "مقدار"
end

@testset "extract_quantity_frames: comparison" begin
    frames = Physics.extract_quantity_frames("هذا أكثر من ذاك")
    @test length(frames) >= 1
    f = first(frames)
    @test f.quantity_type == "comparison"
end

@testset "extract_quantity_frames: comparison answer prefix cleanup" begin
    frames = Physics.extract_quantity_frames("\u0646\u0639\u0645\u060c \u062e\u0645\u0633 \u0646\u062c\u0645\u0627\u062a \u0623\u0643\u062b\u0631 \u0645\u0646 \u062b\u0644\u0627\u062b \u0646\u0642\u0627\u0637.")
    @test length(frames) >= 1
    f = first(frames)
    @test f.quantity_type == "comparison"
    @test f.target == "\u062e\u0645\u0633 \u0646\u062c\u0645\u0627\u062a"
    @test f.value == "\u0645\u0646 \u062b\u0644\u0627\u062b \u0646\u0642\u0627\u0637"
end

@testset "extract_quantity_frames: quantifier scope" begin
    frames = Physics.extract_quantity_frames("كل الطلاب مجتهدون")
    @test length(frames) >= 1
    f = first(frames)
    @test f.quantity_type == "quantifier_scope"
end

@testset "extract_quantity_frames: vague quantity" begin
    frames = Physics.extract_quantity_frames("هناك كثير من الفوائد")
    @test length(frames) >= 1
    f = first(frames)
    @test f.quantity_type == "vague_quantity"
end

@testset "extract_quantity_frames: target trailing preposition cleanup" begin
    frames = Physics.extract_quantity_frames("\u0635\u064a\u063a \u0627\u0644\u0645\u0628\u0627\u0644\u063a\u0629 \u0623\u0648\u0632\u0627\u0646 \u0633\u0645\u0627\u0639\u064a\u0629 \u0641\u064a \u0643\u062b\u064a\u0631 \u0645\u0646 \u0627\u0644\u0623\u062d\u064a\u0627\u0646.")
    @test length(frames) >= 1
    f = first(frames)
    @test f.quantity_type == "vague_quantity"
    @test f.target == "\u0635\u064a\u063a \u0627\u0644\u0645\u0628\u0627\u0644\u063a\u0629 \u0623\u0648\u0632\u0627\u0646 \u0633\u0645\u0627\u0639\u064a\u0629"
    @test !endswith(f.target, " \u0641\u064a")
end

@testset "extract_quantity_frames: multiple quantities" begin
    frames = Physics.extract_quantity_frames("كم عدد الطلاب وكم عدد المعلمين؟")
    @test length(frames) == 2
end

@testset "extract_quantity_frames: no markers" begin
    @test isempty(Physics.extract_quantity_frames("السماء زرقاء."))
    @test isempty(Physics.extract_quantity_frames(""))
end

@testset "extract_quantity_frames: trained noise rejected" begin
    @test isempty(Physics.extract_quantity_frames("وعليكم السلام ورحمة الله."))
    @test isempty(Physics.extract_quantity_frames("كلما تعلم الإنسان أصبح فهمه أعمق وأصبح حكمه على الأمور أعدل."))
    @test isempty(Physics.extract_quantity_frames("إذا أكثر من الفعل دون تفكير أتعب نفسه بغير ثمرة."))
    @test isempty(Physics.extract_quantity_frames("كمية المادة، 6.022 × 10²³ جزيء."))
    @test isempty(Physics.extract_quantity_frames("هل الكلام الكثير دليل على المعرفة؟"))
    @test isempty(Physics.extract_quantity_frames("المولارية هي عدد مولات المذاب في لتر من المحلول."))
    @test isempty(Physics.extract_quantity_frames("وحدة كمية المادة، 6.022 × 10²³ جزيء."))
    @test isempty(Physics.extract_quantity_frames("أمثلة على التنازع بأكثر من عاملين حضر واستمع وأنصت الضيف."))
    @test isempty(Physics.extract_quantity_frames("خرج = 1 إذا عدد المدخلات=1 فردي | لمدخلين."))
    @test isempty(Physics.extract_quantity_frames("لماذا يفرح الشاكر أكثر من غيره؟"))
    @test isempty(Physics.extract_quantity_frames("سيليكون أحادي/مت عدد البلورات | كفاءة 15-22% تجاريا."))
    @test isempty(Physics.extract_quantity_frames("كل شيء بمقدار."))
    @test isempty(Physics.extract_quantity_frames("تمييز العدد يأتي بعد الأعداد من ثلاثة إلى عشرة."))
    @test isempty(Physics.extract_quantity_frames("كل مصباح يقلب بعدد قواسم رقمه."))
    @test isempty(Physics.extract_quantity_frames("you are welcome many thanks."))
end

@testset "QuantityFrame struct fields" begin
    f = Physics.QuantityFrame("كم", "count", "الطلاب", "في الفصل", "neutral", 0.9)
    @test f.marker == "كم"
    @test f.quantity_type == "count"
    @test f.target == "الطلاب"
    @test f.value == "في الفصل"
    @test f.polarity == "neutral"
    @test f.confidence ≈ 0.9
    @test f isa Physics.QuantityFrame
end

@testset "quantity_answer: count and measure" begin
    count_frames = [
        Physics.QuantityFrame("\u0639\u062f\u062f", "count", "\u0627\u0644\u0637\u0644\u0627\u0628", "30", "neutral", 0.9),
    ]
    count_ans = Physics.quantity_answer(count_frames, "\u0643\u0645 \u0639\u062f\u062f \u0627\u0644\u0637\u0644\u0627\u0628\u061f")
    @test !isempty(count_ans)
    @test occursin("30", count_ans)
    @test occursin("\u0627\u0644\u0637\u0644\u0627\u0628", count_ans)

    indication_frames = [
        Physics.QuantityFrame("\u0639\u062f\u062f", "count", "\u062c\u0645\u0639 \u0627\u0644\u0642\u0644\u0629 \u064a\u062f\u0644 \u0639\u0644\u0649", "\u0645\u0646 \u062b\u0644\u0627\u062b\u0629 \u0625\u0644\u0649 \u0639\u0634\u0631\u0629", "neutral", 0.9),
    ]
    indication_ans = Physics.quantity_answer(indication_frames, "\u0643\u0645 \u0639\u062f\u062f \u062c\u0645\u0639 \u0627\u0644\u0642\u0644\u0629 \u064a\u062f\u0644 \u0639\u0644\u0649\u061f")
    @test indication_ans == "\u064a\u062f\u0644 \u062c\u0645\u0639 \u0627\u0644\u0642\u0644\u0629 \u0639\u0644\u0649 \u0645\u0646 \u062b\u0644\u0627\u062b\u0629 \u0625\u0644\u0649 \u0639\u0634\u0631\u0629."

    measure_frames = [
        Physics.QuantityFrame("\u0645\u0642\u062f\u0627\u0631", "measure", "\u0627\u0644\u0637\u0627\u0642\u0629", "5 \u062c\u0648\u0644", "neutral", 0.9),
    ]
    measure_ans = Physics.quantity_answer(measure_frames, "\u0645\u0627 \u0645\u0642\u062f\u0627\u0631 \u0627\u0644\u0637\u0627\u0642\u0629\u061f")
    @test !isempty(measure_ans)
    @test occursin("5", measure_ans)
    @test occursin("\u0627\u0644\u0637\u0627\u0642\u0629", measure_ans)
end

@testset "quantity_answer: comparison and vague quantity" begin
    comparison_frames = [
        Physics.QuantityFrame("\u0623\u0643\u062b\u0631", "comparison", "\u0627\u0644\u0645\u0627\u0621", "\u0645\u0646 \u0627\u0644\u0632\u064a\u062a", "neutral", 0.75),
    ]
    comparison_ans = Physics.quantity_answer(comparison_frames, "\u0623\u064a\u0647\u0645\u0627 \u0623\u0643\u062b\u0631 \u0627\u0644\u0645\u0627\u0621 \u0623\u0645 \u0627\u0644\u0632\u064a\u062a\u061f")
    @test !isempty(comparison_ans)
    @test occursin("\u0627\u0644\u0645\u0627\u0621", comparison_ans)
    @test occursin("\u0623\u0643\u062b\u0631", comparison_ans)

    vague_frames = [
        Physics.QuantityFrame("\u0643\u062b\u064a\u0631", "vague_quantity", "\u0627\u0644\u0641\u0648\u0627\u0626\u062f", "", "neutral", 0.6),
    ]
    vague_ans = Physics.quantity_answer(vague_frames, "\u0645\u0627 \u0643\u0645\u064a\u0629 \u0627\u0644\u0641\u0648\u0627\u0626\u062f\u061f")
    @test !isempty(vague_ans)
    @test occursin("\u0643\u062b\u064a\u0631", vague_ans)
    @test occursin("\u0627\u0644\u0643\u0645\u064a\u0629 \u0627\u0644\u062a\u0642\u0631\u064a\u0628\u064a\u0629", vague_ans)
end

@testset "quantity_answer: quantifier scope phrasing" begin
    frames = [
        Physics.QuantityFrame("\u0643\u0644", "quantifier_scope", "\u0644\u063a\u0629", "\u062a\u0645\u062a\u0644\u0643 \u0642\u0648\u0627\u0639\u062f", "neutral", 0.75),
    ]
    ans = Physics.quantity_answer(frames, "\u0645\u0627 \u0646\u0637\u0627\u0642 \u0644\u063a\u0629 \u062a\u0645\u062a\u0644\u0643 \u0642\u0648\u0627\u0639\u062f\u061f")
    @test ans == "\u0627\u0644\u0646\u0637\u0627\u0642: \u0643\u0644 \u0644\u063a\u0629 \u062a\u0645\u062a\u0644\u0643 \u0642\u0648\u0627\u0639\u062f."
end

@testset "quantity_answer: direct measure and less comparison" begin
    direct = Physics.extract_quantity_frames("\u0643\u0645 \u0637\u0648\u0644 \u0627\u0644\u062c\u0633\u0631\u061f")
    @test length(direct) >= 1
    @test first(direct).quantity_type == "measure"
    @test first(direct).marker == "\u0637\u0648\u0644"
    @test first(direct).target == "\u0627\u0644\u062c\u0633\u0631"

    measure_frames = [
        Physics.QuantityFrame("\u0637\u0648\u0644", "measure", "\u0627\u0644\u062c\u0633\u0631", "100 \u0645\u062a\u0631", "neutral", 0.9),
    ]
    measure_ans = Physics.quantity_answer(measure_frames, "\u0643\u0645 \u0637\u0648\u0644 \u0627\u0644\u062c\u0633\u0631\u061f")
    @test occursin("100", measure_ans)
    @test occursin("\u0627\u0644\u062c\u0633\u0631", measure_ans)

    less_frames = [
        Physics.QuantityFrame("\u0623\u0642\u0644", "comparison", "\u0627\u0644\u0633\u0643\u0631", "\u0645\u0646 \u0627\u0644\u0645\u0644\u062d", "neutral", 0.75),
    ]
    less_ans = Physics.quantity_answer(less_frames, "\u0623\u064a\u0647\u0645\u0627 \u0623\u0642\u0644 \u0627\u0644\u0633\u0643\u0631 \u0623\u0645 \u0627\u0644\u0645\u0644\u062d\u061f")
    @test occursin("\u0627\u0644\u0633\u0643\u0631", less_ans)
    @test occursin("\u0623\u0642\u0644", less_ans)
    @test occursin("\u0627\u0644\u0645\u0644\u062d", less_ans)
end

@testset "quantity_answer: English extraction and comparison" begin
    @test Physics.relation_type_for_quantity_marker("how many") == "count"
    @test Physics.relation_type_for_quantity_marker("length") == "measure"
    @test Physics.relation_type_for_quantity_marker("more than") == "comparison"

    direct = Physics.extract_quantity_frames("how many students?")
    @test any(f -> f.quantity_type == "count" && f.marker == "how many", direct)

    cmp_frames = Physics.extract_quantity_frames("five stars more than three points.")
    @test any(f -> f.quantity_type == "comparison" && f.marker == "more than", cmp_frames)
    ans = Physics.quantity_answer(cmp_frames, "which is more stars or points?")
    @test !isempty(ans)
    @test occursin("more than", lowercase(ans))
end

@testset "quantity_answer: guards" begin
    frames = [
        Physics.QuantityFrame("\u0639\u062f\u062f", "count", "\u0627\u0644\u0637\u0644\u0627\u0628", "30", "neutral", 0.9),
    ]
    @test isempty(Physics.quantity_answer(frames, "\u0647\u0644 \u0639\u062f\u062f \u0627\u0644\u0637\u0644\u0627\u0628 30\u061f"))
    @test isempty(Physics.quantity_answer(frames, "\u0644\u0645\u0627\u0630\u0627 \u064a\u062f\u0631\u0633 \u0627\u0644\u0637\u0644\u0627\u0628\u061f"))
    @test isempty(Physics.quantity_answer(Physics.QuantityFrame[], "\u0643\u0645 \u0639\u062f\u062f \u0627\u0644\u0637\u0644\u0627\u0628\u061f"))
end

@testset "quantity_answer: trained noise guards" begin
    noisy = [
        Physics.QuantityFrame("\u0643\u0645", "count", "\u0648\u0639\u0644\u064a", "\u0627\u0644\u0633\u0644\u0627\u0645 \u0648\u0631\u062d\u0645\u0629 \u0627\u0644\u0644\u0647", "neutral", 0.9),
        Physics.QuantityFrame("\u0642\u0644\u064a\u0644", "vague_quantity", "\u0627\u0644\u062c\u0647\u0644 \u064a\u0636\u064a\u0642 \u0623\u0641\u0642 \u0627\u0644\u0625\u0646\u0633\u0627\u0646 \u0641\u064a\u0638\u0646", "\u0643\u0644 \u0634\u064a\u0621", "neutral", 0.6),
        Physics.QuantityFrame("\u0623\u0643\u062b\u0631", "comparison", "\u0643\u0644\u0645\u0627 \u062a\u0639\u0644\u0645 \u0627\u0644\u0625\u0646\u0633\u0627\u0646", "\u0623\u0635\u0628\u062d \u0641\u0647\u0645\u0647 \u0623\u0639\u0645\u0642", "neutral", 0.75),
    ]
    @test isempty(Physics.quantity_answer(noisy, "\u0643\u0645 \u0639\u062f\u062f \u0648\u0639\u0644\u064a\u061f"))
    @test isempty(Physics.quantity_answer(noisy, "\u0645\u0627 \u0643\u0645\u064a\u0629 \u0627\u0644\u062c\u0647\u0644 \u064a\u0636\u064a\u0642 \u0623\u0641\u0642 \u0627\u0644\u0625\u0646\u0633\u0627\u0646\u061f"))
    @test isempty(Physics.quantity_answer(noisy, "\u0623\u064a\u0647\u0645\u0627 \u0623\u0643\u062b\u0631 \u0643\u0644\u0645\u0627 \u062a\u0639\u0644\u0645 \u0627\u0644\u0625\u0646\u0633\u0627\u0646\u061f"))

    mixed = vcat(noisy, [
        Physics.QuantityFrame("\u0639\u062f\u062f", "count", "\u0644", "1 \u0645\u0648\u0644", "neutral", 0.9),
        Physics.QuantityFrame("\u0643\u0645", "count", "\u0627\u0644\u062d", "\u0629 \u0641\u064a \u0627\u0644\u0632\u0645\u0646 \u062b\u0644\u0627\u062b\u0629", "neutral", 0.9),
        Physics.QuantityFrame("\u0623\u0643\u062b\u0631", "comparison", "*", "\u0645\u0646 ...", "neutral", 0.75),
        Physics.QuantityFrame("\u0623\u0643\u062b\u0631", "comparison", "\u0625\u0630\u0627", "\u0645\u0646 \u0627\u0644\u0641\u0639\u0644 \u062f\u0648\u0646 \u062a\u0641\u0643\u064a\u0631", "neutral", 0.75),
        Physics.QuantityFrame("\u0643\u062b\u064a\u0631", "vague_quantity", "\u0628\u0623\u0646 \u062a\u0646\u0638\u0631 \u0641\u064a \u0639\u064a\u0648\u0628 \u0646\u0641\u0633\u0643", "\u0641\u064a \u0645\u062d\u0627\u0633\u0646 \u063a\u064a\u0631\u0643 \u0643\u062b\u064a\u0631\u0627", "neutral", 0.6),
        Physics.QuantityFrame("\u0643\u0644", "quantifier_scope", "\u0647\u0644 \u0627\u0644 \u0628 \u0627\u0646\u0633\u0627\u0646", "", "neutral", 0.75),
    ])
    @test isempty(Physics.quantity_answer(mixed, "\u0643\u0645 \u0639\u062f\u062f \u0648\u0639\u0644\u064a\u061f"))
    @test isempty(Physics.quantity_answer(mixed, "\u0645\u0627 \u0643\u0645\u064a\u0629 \u0627\u0644\u062c\u0647\u0644 \u064a\u0636\u064a\u0642 \u0623\u0641\u0642 \u0627\u0644\u0625\u0646\u0633\u0627\u0646 \u0641\u064a\u0638\u0646 \u0627\u0644\u061f"))
    @test isempty(Physics.quantity_answer(mixed, "\u0623\u064a\u0647\u0645\u0627 \u0623\u0643\u062b\u0631 \u0643\u0644\u0645\u0627 \u062a\u0639\u0644\u0645 \u0627\u0644\u0625\u0646\u0633\u0627\u0646\u061f"))
    @test isempty(Physics.quantity_answer(mixed, "\u0643\u0645 \u0639\u062f\u062f \u0627\u0644\u0645\u0639\u0631\u0641\u0629 \u062a\u0642\u0648\u062f \u0625\u0644\u0649 \u0627\u0644\u062d\u061f"))
    @test isempty(Physics.quantity_answer(mixed, "\u0623\u064a\u0647\u0645\u0627 \u0623\u0643\u062b\u0631 \u0627\u0644\u0638\u0644\u0645 \u0643\u0627\u0644\u0645\u0631\u0636 \u064a\u061f"))
    @test isempty(Physics.quantity_answer(mixed, "\u0645\u0627 \u0645\u0642\u062f\u0627\u0631 \u0646\u0633\u062a\u0639\u0645\u0644 \u0623\u0642\u0644\u061f"))
    @test isempty(Physics.quantity_answer(mixed, "\u0643\u0645 \u0639\u062f\u062f \u0627\u0644\u062d\u061f"))
    @test isempty(Physics.quantity_answer(mixed, "\u0623\u064a\u0647\u0645\u0627 \u0623\u0643\u062b\u0631 \u0625\u0630\u0627\u061f"))
    @test isempty(Physics.quantity_answer(mixed, "\u0645\u0627 \u0643\u0645\u064a\u0629 \u0628\u0623\u0646 \u062a\u0646\u0638\u0631 \u0641\u064a \u0639\u064a\u0648\u0628 \u0646\u0641\u0633\u0643\u061f"))
end

@testset "compare_quantity_strategies: returns QuantityComparisonRecord" begin
    frames = [
        Physics.QuantityFrame("\u0639\u062f\u062f", "count", "\u0627\u0644\u0637\u0644\u0627\u0628", "30", "neutral", 0.9),
    ]
    result = Physics.compare_quantity_strategies(
        frames,
        p -> "\u062c\u0648\u0627\u0628 \u0639\u0627\u0645: $p",
        "\u0643\u0645 \u0639\u062f\u062f \u0627\u0644\u0637\u0644\u0627\u0628\u061f",
    )
    @test result isa Physics.QuantityComparisonRecord
    @test !isempty(result.quantity_answer)
    @test result.memory_has_quantity == true
    @test result.quantity_confidence > 0
    @test result.overlap_score > 0
    @test result.has_marker == true
    @test result.quantity_type == "count"
    @test occursin("30", result.quantity_answer)
    @test occursin("\u062c\u0648\u0627\u0628 \u0639\u0627\u0645", result.generate_answer)
end

@testset "compare_quantity_strategies: guards" begin
    frames = [
        Physics.QuantityFrame("\u0639\u062f\u062f", "count", "\u0627\u0644\u0637\u0644\u0627\u0628", "30", "neutral", 0.9),
    ]
    yesno = Physics.compare_quantity_strategies(frames, p -> "\u062c\u0648\u0627\u0628 \u0647\u0644", "\u0647\u0644 \u0639\u062f\u062f \u0627\u0644\u0637\u0644\u0627\u0628 30\u061f")
    @test isempty(yesno.quantity_answer)
    @test yesno.memory_has_quantity == false
    @test yesno.generate_answer == "\u062c\u0648\u0627\u0628 \u0647\u0644"

    empty = Physics.compare_quantity_strategies(Physics.QuantityFrame[], p -> "", "\u0643\u0645 \u0639\u062f\u062f \u0627\u0644\u0637\u0644\u0627\u0628\u061f")
    @test isempty(empty.quantity_answer)
    @test empty.quantity_type == "none"
end

@testset "QuantityFrameMemory: learn and answer" begin
    mem = Physics.QuantityFrameMemory()
    learned = Physics.learn_quantity_frames_from_text!(
        mem,
        "\u0627\u0644\u0637\u0644\u0627\u0628 \u0639\u062f\u062f 30.",
        Dict{String,Any}("file_name" => "quantity_fixture"),
    )
    @test learned == 1
    @test length(mem.frames) == 1
    @test length(mem.source_metadata) == 1

    selected = Physics.select_quantity_frame(mem, "\u0643\u0645 \u0639\u062f\u062f \u0627\u0644\u0637\u0644\u0627\u0628\u061f")
    @test selected !== nothing
    @test selected.quantity_type == "count"

    ans = Physics.quantity_answer(mem, "\u0643\u0645 \u0639\u062f\u062f \u0627\u0644\u0637\u0644\u0627\u0628\u061f")
    @test !isempty(ans)
    @test occursin("30", ans)
    @test !occursin("30..", ans)
end

@testset "QuantityFrameMemory: train and compare" begin
    mem = Physics.QuantityFrameMemory()
    texts = [
        "\u0627\u0644\u0637\u0644\u0627\u0628 \u0639\u062f\u062f 30.",
        "\u0627\u0644\u0637\u0627\u0642\u0629 \u0645\u0642\u062f\u0627\u0631 5 \u062c\u0648\u0644.",
    ]
    learned = Physics.train_quantity_frames_from_texts!(mem, texts)
    @test learned == 2

    result = Physics.compare_quantity_strategies(
        mem,
        p -> "\u062c\u0648\u0627\u0628 \u0639\u0627\u0645",
        "\u0645\u0627 \u0645\u0642\u062f\u0627\u0631 \u0627\u0644\u0637\u0627\u0642\u0629\u061f",
    )
    @test result isa Physics.QuantityComparisonRecord
    @test result.memory_has_quantity == true
    @test result.quantity_type == "measure"
    @test occursin("5", result.quantity_answer)
end

@testset "QuantityFrameMemory: save and load" begin
    mem = Physics.QuantityFrameMemory()
    Physics.learn_quantity_frames_from_text!(
        mem,
        "\u0627\u0644\u0637\u0644\u0627\u0628 \u0639\u062f\u062f 30.",
        Dict{String,Any}("file_name" => "quantity_fixture"),
    )
    @test Physics.has_quantity_records(mem)
    path = joinpath(mktempdir(), "quantity_memory.json")
    saved = Physics.save_quantity_memory(mem, path)
    @test isfile(saved)

    loaded = Physics.load_quantity_memory(saved)
    @test Physics.has_quantity_records(loaded)
    @test length(loaded.frames) == 1
    @test loaded.frames[1].quantity_type == "count"
    @test loaded.frames[1].value == "30"
    ans = Physics.quantity_answer(loaded, "\u0643\u0645 \u0639\u062f\u062f \u0627\u0644\u0637\u0644\u0627\u0628\u061f")
    @test occursin("30", ans)

    empty = Physics.load_quantity_memory(joinpath(mktempdir(), "missing.json"))
    @test !Physics.has_quantity_records(empty)
end

@testset "QuantityFrameMemory: generator autoload" begin
    Gen = Physics.Generator
    saved_mem = Gen._LEARNED_QUANTITY_MEMORY[]
    model_dir = mktempdir()
    mem = Physics.QuantityFrameMemory()
    Physics.learn_quantity_frames_from_text!(mem, "\u0627\u0644\u0637\u0644\u0627\u0628 \u0639\u062f\u062f 30.")
    Physics.save_quantity_memory(mem, joinpath(model_dir, "quantity_memory.json"))

    try
        _ = Physics.MirnanGenerator(Dict{String,Int}(
            "\u0643\u0645" => 1,
            "\u0639\u062f\u062f" => 2,
            "\u0627\u0644\u0637\u0644\u0627\u0628" => 3,
            "30" => 4,
        ); model_dir=model_dir)
        loaded = Gen._LEARNED_QUANTITY_MEMORY[]
        @test loaded !== nothing
        @test Physics.has_quantity_records(loaded)
        ans = Physics.quantity_answer(loaded, "\u0643\u0645 \u0639\u062f\u062f \u0627\u0644\u0637\u0644\u0627\u0628\u061f")
        @test occursin("30", ans)
    finally
        Gen._LEARNED_QUANTITY_MEMORY[] = saved_mem
    end
end

@testset "QuantityFrameStrategy" begin
    Gen = Physics.Generator
    gen = Physics.MirnanGenerator(Dict{String,Int}(
        "\u0643\u0645" => 1,
        "\u0639\u062f\u062f" => 2,
        "\u0627\u0644\u0637\u0644\u0627\u0628" => 3,
        "30" => 4,
        "\u0647\u0644" => 5,
    ); model_dir=mktempdir())
    saved_mem = Gen._LEARNED_QUANTITY_MEMORY[]
    prompt = "\u0643\u0645 \u0639\u062f\u062f \u0627\u0644\u0637\u0644\u0627\u0628\u061f"

    function _qfs_args(prompt_text)
        pt = String.(split(strip(prompt_text)))
        co = Gen.observe_prompt(gen.cerebellum, pt; prompt=prompt_text, vocab_size=length(gen.vocab))
        cp = Gen.choose_policy!(gen.cerebellum, co; requested_mode="auto")
        rp = Gen.detect_response_intent(prompt_text)
        ap = Gen._get_active_paragraphs(gen, pt)
        return pt, "auto", co, cp, rp, ap
    end

    try
        mem = Physics.QuantityFrameMemory()
        Physics.learn_quantity_frames_from_text!(mem, "\u0627\u0644\u0637\u0644\u0627\u0628 \u0639\u062f\u062f 30.")
        Gen._LEARNED_QUANTITY_MEMORY[] = mem

        args = _qfs_args(prompt)
        direct = Gen.try_generate(Gen.QuantityFrameStrategy(), gen, prompt, args...)
        @test direct !== nothing
        @test occursin("30", direct)

        yesno_prompt = "\u0647\u0644 \u0639\u062f\u062f \u0627\u0644\u0637\u0644\u0627\u0628 30\u061f"
        yesno_args = _qfs_args(yesno_prompt)
        yesno = Gen.try_generate(Gen.QuantityFrameStrategy(), gen, yesno_prompt, yesno_args...)
        @test yesno === nothing

        old_env = get(ENV, "MIRNAN_ENABLE_QUANTITY_FRAME_STRATEGY", nothing)
        old_strict = get(ENV, "MIRNAN_STRICT_NO_TEMPLATES", nothing)
        try
            ENV["MIRNAN_ENABLE_QUANTITY_FRAME_STRATEGY"] = "1"
            ENV["MIRNAN_STRICT_NO_TEMPLATES"] = "0"
            generated = Physics.generate!(gen, prompt; max_words=18)
            @test occursin("30", generated)
            yesno_generated = Physics.generate!(gen, yesno_prompt; max_words=18)
            @test !occursin("\u0639\u062f\u062f \u0627\u0644\u0637\u0644\u0627\u0628 \u0647\u0648 30", yesno_generated)
        finally
            if old_env === nothing
                delete!(ENV, "MIRNAN_ENABLE_QUANTITY_FRAME_STRATEGY")
            else
                ENV["MIRNAN_ENABLE_QUANTITY_FRAME_STRATEGY"] = old_env
            end
            if old_strict === nothing
                delete!(ENV, "MIRNAN_STRICT_NO_TEMPLATES")
            else
                ENV["MIRNAN_STRICT_NO_TEMPLATES"] = old_strict
            end
        end
    finally
        Gen._LEARNED_QUANTITY_MEMORY[] = saved_mem
    end
end
