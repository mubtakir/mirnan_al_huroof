using Test
using JSON
using SparseArrays

const MIRNAN_DIR = dirname(@__DIR__)
include(joinpath(MIRNAN_DIR, "src", "MirnanNew.jl"))
using .MirnanNew

function _load_sparse_dat(path::String, vocab_size::Int)
    isfile(path) || return spzeros(vocab_size, vocab_size)
    open(path, "r") do io
        header = readline(io)
        header == "SPARSE_CSC" || return spzeros(vocab_size, vocab_size)
        m = read(io, Int32)
        n = read(io, Int32)
        nnz = read(io, Int32)
        colptr = read!(io, Vector{Int32}(undef, n + 1))
        rows = read!(io, Vector{Int32}(undef, nnz))
        vals = read!(io, Vector{Float64}(undef, nnz))
        return SparseMatrixCSC(Int(m), Int(n), Vector{Int}(colptr), Vector{Int}(rows), vals)
    end
end

function _load_trained_generator()
    model_dir = joinpath(MIRNAN_DIR, "model")
    vocab_path = joinpath(model_dir, "vocab.json")
    isfile(vocab_path) || error("Missing trained model. Run models/mirnan/train.jl first.")
    raw_vocab = JSON.parsefile(vocab_path)
    vocab = Dict{String,Int}(String(k) => Int(v) for (k, v) in raw_vocab)
    V = length(vocab)
    K_sem = _load_sparse_dat(joinpath(model_dir, "K_sem.dat"), V)
    K_syn = _load_sparse_dat(joinpath(model_dir, "K_syn.dat"), V)
    K_causal = _load_sparse_dat(joinpath(model_dir, "K_causal.dat"), V)
    return MirnanGenerator(vocab, K_sem; K_syn=K_syn, K_causal=K_causal, model_dir=model_dir)
end

_norm(s) = replace(lowercase(strip(s)), r"[[:punct:]؟،؛]+" => "")
_has_any(s, words) = any(w -> occursin(w, s), words)
_starts_no(s) = startswith(_norm(s), "لا") || startswith(_norm(s), "كلا")

function _answer(gen, question)
    generate!(gen, question; mode="auto", max_words=48)
end

function _assert_positive(answer; required=String[], forbidden=String[])
    text = _norm(answer)
    @test !_starts_no(text)
    for w in required
        @test occursin(w, text)
    end
    for w in forbidden
        @test !occursin(w, text)
    end
end

function _assert_negative(answer; required=String[])
    text = _norm(answer)
    @test _starts_no(text) || _has_any(text, ["ليس", "لا "])
    for w in required
        @test occursin(w, text)
    end
end

function _assert_explains(answer)
    text = _norm(answer)
    @test _has_any(text, ["لأن", "حين", "عندما", "بسبب", "إذ", "في", "من خلال", "كلما", "فيؤدي", "فيزيد", "فيمنع", "فيحفظ", "فيزيل"])
end

@testset "Mirnan question paths from hiwar.txt" begin
    gen = _load_trained_generator()

    @testset "هل: relation truth and negation" begin
        _assert_positive(_answer(gen, "هل العلم يزيد الفهم؟"); required=["العلم", "الفهم"])
        _assert_negative(_answer(gen, "هل العلم لا يزيد الفهم؟"); required=["العلم", "الفهم"])
        _assert_negative(_answer(gen, "هل الجهل يزيد الفهم؟"); required=["الجهل", "الفهم"])
        _assert_positive(_answer(gen, "هل الجهل لا يزيد الفهم؟"); required=["الجهل", "الفهم"])
    end

    @testset "السلام داخل العلاقة ليس تحية" begin
        ans = _answer(gen, "هل العدل يحفظ السلام؟")
        _assert_positive(ans; required=["العدل", "السلام"], forbidden=["وعليكم"])
        rel = _answer(gen, "ما العلاقة بين العدل والسلام؟")
        @test occursin("العدل", _norm(rel))
        @test occursin("السلام", _norm(rel))
        @test !occursin("وعليكم", _norm(rel))
    end

    @testset "اتجاه السبب والنتيجة لا ينقلب" begin
        _assert_positive(_answer(gen, "هل الرحمة تبني الثقة؟"); required=["الرحمة", "الثقة"])
        _assert_negative(_answer(gen, "هل القسوة تبني الرحمة؟"); required=["القسوة", "الرحمة"])
        _assert_negative(_answer(gen, "لماذا لا يزيد الفهم العلم؟"); required=["الفهم", "العلم"])
        _assert_negative(_answer(gen, "لماذا لا يحفظ السلام العدل؟"); required=["السلام", "العدل"])
    end

    @testset "لماذا وكيف: تفسير لا جواب نعم/لا فقط" begin
        why_science = _answer(gen, "لماذا يزيد العلم الفهم؟")
        how_peace = _answer(gen, "كيف يزيل السلام الخوف؟")
        _assert_explains(why_science)
        _assert_explains(how_peace)
        @test !_starts_no(_norm(why_science))
        @test occursin("السلام", _norm(how_peace))
        @test occursin("الخوف", _norm(how_peace))
    end
end
