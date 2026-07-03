using Test
using SparseArrays

include(joinpath(@__DIR__, "..", "src", "MirnanNew.jl"))
using .MirnanNew
const Physics = MirnanNew.Physics

@testset "al_intibah semantic attention field" begin
    words = ["العلم", "نور", "تعلم", "فهم", "كشف", "رؤية", "الماء", "النبات", "نمو", "حياة"]
    vocab = Dict(w => i for (i, w) in enumerate(words))
    id2word = Dict(i => w for (i, w) in enumerate(words))
    K = spzeros(Float64, length(words), length(words))

    K[vocab["العلم"], vocab["تعلم"]] = 4.0
    K[vocab["تعلم"], vocab["فهم"]] = 3.0
    K[vocab["فهم"], vocab["كشف"]] = 3.0
    K[vocab["كشف"], vocab["رؤية"]] = 2.5
    K[vocab["رؤية"], vocab["نور"]] = 2.5
    K[vocab["العلم"], vocab["فهم"]] = 2.2
    K[vocab["نور"], vocab["رؤية"]] = 2.8
    K[vocab["نور"], vocab["كشف"]] = 2.0
    K[vocab["الماء"], vocab["النبات"]] = 4.0
    K[vocab["النبات"], vocab["نمو"]] = 3.5
    K[vocab["نمو"], vocab["حياة"]] = 2.0
    K[vocab["الماء"], vocab["نمو"]] = 2.6

    for i in 1:size(K, 1), j in 1:size(K, 2)
        K[j, i] = max(K[j, i], K[i, j] * 0.75)
    end

    field = Physics.build_semantic_attention(vocab, id2word, K, K, ["كيف", "يكون", "العلم", "نور"])
    @test Physics.has_semantic_attention(field)
    @test "العلم" in field.anchors
    @test "نور" in field.anchors
    science_terms = Physics.attention_bias_terms(field)
    @test any(t -> t in science_terms, ["تعلم", "فهم", "كشف", "رؤية"])
    @test !("كيف" in field.anchors)

    water_field = Physics.build_semantic_attention(vocab, id2word, K, K, ["لماذا", "يساعد", "الماء", "النبات"])
    @test Physics.has_semantic_attention(water_field)
    @test "الماء" in water_field.anchors
    @test "النبات" in water_field.anchors
    water_terms = Physics.attention_bias_terms(water_field)
    @test any(t -> t in water_terms, ["نمو", "حياة"])

    empty_field = Physics.build_semantic_attention(vocab, id2word, K, K, ["كيف", "هل", "ما"])
    @test !Physics.has_semantic_attention(empty_field)
end
