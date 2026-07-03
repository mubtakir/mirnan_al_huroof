include("../src/MirnanNew.jl")
using .MirnanNew
using Test

const Physics = MirnanNew.Physics

@testset "al_nisba semantic relation memory" begin
    dir = mktempdir()
    mem = Physics.NisbaMemory()
    texts = [
        "العلم كالنور يكشف الطريق للسائر في الظلام.",
        "الجهل ظلمة؛ لأنه يحجب الرؤية ويوقع الإنسان في المتاهات.",
        "الرحمة تهذب القوة؛ لأن القوة بلا رحمة تصير طغيانا.",
        "يحفظ العدل السلام لأنه يمنع التعدي ويصون الحقوق.",
        "تتحول القوة إلى ظلم حين تفارق الرحمة والعدل.",
        "لا يكفي العلم وحده؛ لأنه يحتاج إلى فهم وعدل ورحمة.",
        "الفرق بين العدل والرحمة أن العدل يعطي الحق، أما الرحمة فتخفف الأذى.",
    ]

    learned = Physics.train_nisba_from_texts!(mem, texts)
    @test learned >= 6
    @test Physics.has_nisba_relations(mem)

    light = Physics.select_nisba_relation(mem, "ما الذي يجعل العلم شبيها بالنور؟")
    @test light !== nothing
    @test light.relation_type == "analogy"
    @test "علم" in light.concepts || "العلم" in light.concepts
    @test "نور" in light.concepts || any(occursin("نور", c) for c in light.concepts)
    @test any(t -> t in ("يشبه", "يكشف", "يهدي"), Physics.nisba_guidance_terms(light))

    tyranny = Physics.select_nisba_relation(mem, "متى تتحول القوة إلى ظلم؟")
    @test tyranny !== nothing
    @test tyranny.relation_type == "transform"
    @test any(c -> occursin("قوه", c) || occursin("قوة", c), tyranny.concepts)

    mercy = Physics.select_nisba_relation(mem, "كيف تحمي الرحمة القوة من الطغيان؟")
    @test mercy !== nothing
    @test mercy.relation_type in ("causal", "prevention", "transform")
    @test any(c -> occursin("رحمه", c) || occursin("رحمة", c), mercy.concepts)

    difference = Physics.select_nisba_relation(mem, "بيّن الفرق بين العدل والرحمة.")
    @test difference !== nothing
    @test difference.relation_type == "difference"

    path = Physics.save_nisba(mem, joinpath(dir, "al_nisba.json"))
    @test isfile(path)
    loaded = Physics.load_nisba(path)
    @test Physics.has_nisba_relations(loaded)
    loaded_light = Physics.select_nisba_relation(loaded, "هل يكشف العلم الطريق كما يكشف الضوء المكان؟")
    @test loaded_light !== nothing
    @test loaded_light.relation_type in ("analogy", "causal")
end
