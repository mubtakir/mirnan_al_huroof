using Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "src", "MirnanNew.jl"))
using .MirnanNew, JSON, SparseArrays

model_dir = joinpath(@__DIR__, "model")
raw_vocab = JSON.parsefile(joinpath(model_dir, "vocab.json"))
vocab = Dict{String,Int}(k => Int(v) for (k,v) in raw_vocab)

function _load_sparse_dat(path, vocab_size)
    isfile(path) || return spzeros(vocab_size, vocab_size)
    open(path, "r") do io
        header = readline(io)
        header == "SPARSE_CSC" || return spzeros(vocab_size, vocab_size)
        m = read(io, Int32); n = read(io, Int32); nnz = read(io, Int32)
        colptr = read!(io, Vector{Int32}(undef, n+1))
        rows = read!(io, Vector{Int32}(undef, nnz))
        vals = read!(io, Vector{Float64}(undef, nnz))
        return SparseMatrixCSC(Int(m), Int(n), Vector{Int}(colptr), Vector{Int}(rows), vals)
    end
end

V = length(vocab)
K_sem = _load_sparse_dat(joinpath(model_dir, "K_sem.dat"), V)
K_syn = _load_sparse_dat(joinpath(model_dir, "K_syn.dat"), V)

gen = MirnanNew.Physics.Generator.MirnanGenerator(vocab, K_sem; K_syn=K_syn)

# Test with a word from vocab
test_word = first(keys(vocab))
println("Testing with: $test_word")
result = MirnanNew.Physics.Generator.generate!(gen, test_word; mode="resonant", max_words=5)
println("Result: $result")
