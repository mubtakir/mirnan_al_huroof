using Pkg; Pkg.activate(@__DIR__)
using JSON, SparseArrays, LinearAlgebra
include(joinpath(@__DIR__, "src", "MirnanNew.jl"))
using .MirnanNew

# Load vocab + kernels
model_dir = joinpath(@__DIR__, "model")
raw_vocab = JSON.parsefile(joinpath(model_dir, "vocab.json"))
vocab = Dict{String,Int}(k => Int(v) for (k,v) in raw_vocab)
V = length(vocab)

function _load_sparse_dat(path, vocab_size)
    isfile(path) || return spzeros(vocab_size, vocab_size)
    open(path, "r") do io
        header = readline(io)
        header == "SPARSE_CSC" || return spzeros(vocab_size, vocab_size)
        m = read(io, Int32)
        n = read(io, Int32)
        nnz = read(io, Int32)
        colptr = read!(io, Vector{Int32}(undef, n+1))
        rows   = read!(io, Vector{Int32}(undef, nnz))
        vals   = read!(io, Vector{Float64}(undef, nnz))
        return SparseMatrixCSC(Int(m), Int(n),
                               Vector{Int}(colptr),
                               Vector{Int}(rows),
                               vals)
    end
end

K_sem = _load_sparse_dat(joinpath(model_dir, "K_sem.dat"), V)
K_syn = _load_sparse_dat(joinpath(model_dir, "K_syn.dat"), V)

println("Vocab: $V, K_sem nnz: $(length(K_sem.nzval)), K_syn nnz: $(length(K_syn.nzval))")

# Check vocab contains prompt words
prompt = "العلم نور"
tokens = String[lowercase(strip(w)) for w in split(prompt) if length(strip(w)) >= 1]
println("Prompt tokens: $tokens")
for t in tokens
    wid = get(vocab, t, nothing)
    println("  '$t' => vocab_id=$wid")
end

# Create generator
gen = MirnanNew.Physics.Generator.MirnanGenerator(vocab, K_sem; K_syn=K_syn)

println("\n--- Testing resonant mode ---")
result = MirnanNew.Physics.Generator.generate!(gen, prompt; mode="resonant", max_words=8)
println("RESONANT: '$result'")

println("\n--- Testing standard mode ---")
result2 = MirnanNew.Physics.Generator.generate!(gen, prompt; mode="standard", max_words=8)
println("STANDARD: '$result2'")

println("\n--- Testing single word ---")
result3 = MirnanNew.Physics.Generator.generate!(gen, "العلم"; mode="resonant", max_words=5)
println("SINGLE RESONANT: '$result3'")
