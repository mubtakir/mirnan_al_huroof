include("../train.jl")
include("../src/preprocessing/Preprocessing.jl")
using Test

@testset "training preserves Arabic diacritics" begin
    sample = "عِلْمٌ ونوراً وقرأَ المُعلِّمُ كتاباً."

    @test _normalize_arabic_text(sample) == sample
    @test _strip_diacritics(sample) == sample
    @test Preprocessing.remove_diacritics(sample) == sample
    @test Preprocessing.preprocess_text(sample; preprocessor=Preprocessing.TextPreprocessor(
        normalize_alef=false,
        normalize_ta_marbuta=false,
        remove_tatweel=false,
    )) == sample

    vocab = build_vocab([sample]; min_count=1, max_vocab=100)
    @test haskey(vocab, "عِلْمٌ")
    @test haskey(vocab, "ونوراً")
    @test haskey(vocab, "وقرأَ")
    @test haskey(vocab, "المُعلِّمُ")
    @test !haskey(vocab, "علم")
    @test !haskey(vocab, "ونورا")
end
