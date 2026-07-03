include("../src/MirnanNew.jl")

using .MirnanNew
using Test
using SparseArrays

const Physics = MirnanNew.Physics

@testset "al_ta3rif general definition memory" begin
    dir = mktempdir()
    mem = Physics.Ta3rifMemory()
    texts = [
        "العلم هو معرفة منظمة تكشف أسباب الأشياء.",
        "العدل هو إعطاء كل ذي حق حقه.",
        "الحكمة هي وضع الشيء في موضعه الصحيح.",
        "العرش يرتبط ب الملك.",
        "الشجاعة صفة الثبات عند الخوف.",
        "Knowledge is organized understanding.",
    ]

    learned = Physics.train_ta3rif_from_texts!(mem, texts)
    @test learned >= 4
    @test Physics.has_ta3rif_records(mem)

    science = Physics.answer_ta3rif(mem, "ما هو العلم")
    @test occursin("العلم", science)
    @test occursin("معرفة منظمة", science)
    @test !occursin("في فضاء العقل", science)
    @test !occursin("__classes", science)

    throne = Physics.answer_ta3rif(mem, "ما هو العرش")
    @test occursin("العرش", throne)
    @test occursin("الملك", throne)

    path = Physics.save_ta3rif(mem, joinpath(dir, "al_ta3rif.json"))
    @test isfile(path)
    loaded = Physics.load_ta3rif(path)
    @test occursin("إعطاء كل ذي حق", Physics.answer_ta3rif(loaded, "ما هو العدل"))
    @test occursin("الحكمة هي وضع الشيء", Physics.answer_ta3rif(loaded, "ما معنى الحكمة؟"))
    @test occursin("Knowledge is organized understanding", Physics.answer_ta3rif(loaded, "what is Knowledge?"))

    gen = Physics.MirnanGenerator(Dict("العلم" => 1, "هو" => 2), spzeros(2, 2); model_dir=dir)
    gen.ta3rif = loaded
    gen.self_review.enabled = false
    generated = Physics.generate!(gen, "ما هو العدل"; max_words=10)
    @test occursin("إعطاء كل ذي حق", generated)
    @test !occursin("كيان مسجل", generated)

    loose = Physics.Ta3rifMemory()
    Physics.learn_ta3rif_from_text!(loose, "العدل ليس مساواة عمياء، بل هو إنصاف يضع النظائر معا ويفرق بين المختلفات.")
    loose_answer = Physics.answer_ta3rif(loose, "ما هو العدل؟")
    @test isempty(loose_answer)

    persistent = Physics.Ta3rifMemory()
    Physics.learn_ta3rif_from_text!(persistent, "العدل هو إنصاف يضع النظائر معا ويفرق بين المختلفات.")
    Physics.merge_ta3rif!(loose, persistent)
    merged_answer = Physics.answer_ta3rif(loose, "ما هو العدل؟")
    @test occursin("إنصاف", merged_answer)
end
