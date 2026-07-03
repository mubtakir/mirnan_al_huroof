push!(LOAD_PATH, joinpath(dirname(@__DIR__), "src"))
include(joinpath(@__DIR__, "src", "MirnanNew.jl")); using .MirnanNew, JSON, SparseArrays, LinearAlgebra, Statistics, Printf

println("Loading saved model from model/...")
model_dir = joinpath(dirname(@__DIR__), "model")
vf = joinpath(model_dir, "vocab.json")
raw_vocab = JSON.parsefile(vf)
vocab = Dict{String,Int}(k => Int(v) for (k,v) in raw_vocab)
V = length(vocab)
println("Vocab size: ", V)

function load_sparse(path)
    open(path, "r") do io
        header = readline(io)
        m = read(io, Int32)
        n = read(io, Int32)
        nnz = read(io, Int32)
        colptr = read!(io, Vector{Int32}(undef, n+1))
        rows = read!(io, Vector{Int32}(undef, nnz))
        vals = read!(io, Vector{Float64}(undef, nnz))
        return SparseMatrixCSC(Int(m), Int(n), Vector{Int}(colptr), Vector{Int}(rows), vals)
    end
end

K_sem = load_sparse(joinpath(model_dir, "K_sem.dat"))
K_syn = load_sparse(joinpath(model_dir, "K_syn.dat"))
K_dial = load_sparse(joinpath(model_dir, "K_dialogue.dat"))

println("Model matrices loaded. Initializing generator...")
gen = MirnanGenerator(vocab, K_sem; K_syn=K_syn, K_dialogue=K_dial)

println("Running a test generation...")
t0 = time()
res = MirnanNew.Physics.Generator.generate!(gen, "العلم"; max_words=5)
t1 = time()
println("Generation completed in ", round(t1 - t0; digits=2), " seconds.")
println("Result: ", res)
