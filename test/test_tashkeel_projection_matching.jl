include("../src/MirnanNew.jl")

using .MirnanNew
using Test

const Physics = MirnanNew.Physics

@testset "tashkeel projection matching" begin
    ta3rif = Physics.Ta3rifMemory()
    Physics.learn_ta3rif_from_text!(ta3rif, "العلم هو معرفة منظمة تكشف أسباب الأشياء.")
    plain_answer = Physics.answer_ta3rif(ta3rif, "ما معنى العلم؟")
    diac_answer = Physics.answer_ta3rif(ta3rif, "ما معنى العِلْم؟")
    @test occursin("معرفة منظمة", plain_answer)
    @test occursin("معرفة منظمة", diac_answer)

    nisba = Physics.NisbaMemory()
    Physics.train_nisba_from_texts!(nisba, [
        "الرحمة تهذب القوة؛ لأن القوة بلا رحمة تصير طغياناً.",
        "العلم كالنور يكشف الطريق للسائر في الظلام.",
    ])
    mercy = Physics.select_nisba_relation(nisba, "كيف تهذب الرحمةُ القوةَ؟")
    science = Physics.select_nisba_relation(nisba, "هل العلمُ نورٌ؟")
    @test mercy !== nothing
    @test science !== nothing

    istinbat = Physics.IstinbatAttentionMemory()
    Physics.train_istinbat_from_texts!(istinbat, [
        "العلم نور لأنه يكشف الطريق.",
        "الرحمة تهذب القوة لأنها تمنع الطغيان.",
    ])
    rec = Physics.select_istinbat_attention(istinbat, "لماذا العلمُ نورٌ؟")
    @test rec !== nothing
    @test !isempty(Physics.istinbat_focus_terms(rec))
end
