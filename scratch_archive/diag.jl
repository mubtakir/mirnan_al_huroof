push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using JSON, SparseArrays

vocab_raw = JSON.parsefile("model/vocab.json")
vocab = Dict{String,Int}(k => Int(v) for (k,v) in vocab_raw)
V = length(vocab)
println("Vocab size: $V")

function load_sparse_dat(path, V)
    isfile(path) || return spzeros(V, V)
    open(path, "r") do io
        header = readline(io)
        header == "SPARSE_CSC" || return spzeros(V, V)
        m = read(io, Int32); n = read(io, Int32); nnz = read(io, Int32)
        colptr = read!(io, Vector{Int32}(undef, n+1))
        rows   = read!(io, Vector{Int32}(undef, nnz))
        vals   = read!(io, Vector{Float64}(undef, nnz))
        return SparseMatrixCSC(Int(m), Int(n), Vector{Int}(colptr), Vector{Int}(rows), vals)
    end
end

K_sem = load_sparse_dat("model/K_sem.dat", V)
println("K_sem size: $(size(K_sem)), nnz=$(length(K_sem.nzval))")

id2word = Dict{Int,String}(v=>k for (k,v) in vocab)
test_word = "العلم"
if haskey(vocab, test_word)
    id = vocab[test_word]
    println("'$test_word' id=$id")
    row = Vector(K_sem[id,:])
    top_ids = sortperm(row; rev=true)[1:min(10, end)]
    println("Top neighbors:")
    for tid in top_ids
        row[tid] > 1e-6 || continue
        w = get(id2word, tid, "?")
        println("  $w (score=$(round(row[tid],digits=4)))")
    end
else
    println("'العلم' NOT IN VOCAB!")
    println("Sample: $(collect(keys(vocab))[1:5])")
end
