push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using JSON, SparseArrays

vocab_raw = JSON.parsefile("model/vocab.json")
vocab = Dict{String,Int}(k => Int(v) for (k,v) in vocab_raw)
V = length(vocab)
id2word = Dict{Int,String}(v=>k for (k,v) in vocab)

function load_sparse_dat(path, V)
    open(path, "r") do io
        readline(io)
        m = read(io, Int32); n = read(io, Int32); nnz = read(io, Int32)
        colptr = read!(io, Vector{Int32}(undef, n+1))
        rows   = read!(io, Vector{Int32}(undef, nnz))
        vals   = read!(io, Vector{Float64}(undef, nnz))
        return SparseMatrixCSC(Int(m), Int(n), Vector{Int}(colptr), Vector{Int}(rows), vals)
    end
end

K_sem = load_sparse_dat("model/K_sem.dat", V)

function get_candidates(word)
    cid = get(vocab, word, nothing)
    cid === nothing && return String[]
    row = Vector(K_sem[cid,:])
    sorted = sortperm(row; rev=true)[1:min(50, length(row))]
    result = String[]
    for tid in sorted
        row[tid] > 1e-6 || continue
        w = get(id2word, tid, nothing)
        w !== nothing && length(w) >= 2 && push!(result, w)
    end
    return result
end

cands = get_candidates("العلم")
println("Candidates for 'العلم' ($(length(cands)) total):")
for (i,w) in enumerate(cands[1:min(15,end)])
    println("  [$i] '$w'")
end
