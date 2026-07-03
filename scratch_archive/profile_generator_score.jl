push!(LOAD_PATH, joinpath(dirname(@__DIR__), "src"))
include(joinpath(@__DIR__, "src", "MirnanNew.jl")); using .MirnanNew, JSON, SparseArrays, LinearAlgebra

println("Loading generator...")
model_dir = joinpath(dirname(@__DIR__), "model")
vf = joinpath(model_dir, "vocab.json")
raw_vocab = JSON.parsefile(vf)
vocab = Dict{String,Int}(k => Int(v) for (k,v) in raw_vocab)
V = length(vocab)

function _load_sparse_dat(path::String, vocab_size::Int)
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
gen = MirnanGenerator(vocab, K_sem)

println("Getting candidates...")
prompt_tokens = ["غابة", "استوائية"]
us = Set(prompt_tokens)
prev = "استوائية"
cids = [get(gen.vocab, w, nothing) for w in prompt_tokens]
pvs = [MirnanNew.Physics.Generator._pv(gen, w) for w in prompt_tokens]
cands = MirnanNew.Physics.Generator._resonance_candidates(gen, cids, pvs, us; prev_word=prev)

println("Total candidates: $(length(cands))")

# Profile actual generator _score function
if length(cands) >= 3
    cand1 = cands[1]
    println("Profiling first candidate (with compilation): '$cand1'")
    t0 = time_ns()
    s1 = MirnanNew.Physics.Generator._score(gen, cand1, us, pvs, pvs;
                               gen_pos=2, total_pos=10, prev_word=prev,
                               context_ids=cids, context_words=prompt_tokens)
    t1 = time_ns()
    dt1 = (t1 - t0) / 1e6 # ms
    println("Candidate 1 score: $s1")
    println("Time for candidate 1: $dt1 ms")

    cand2 = cands[2]
    println("Profiling second candidate (pure runtime): '$cand2'")
    t2 = time_ns()
    s2 = MirnanNew.Physics.Generator._score(gen, cand2, us, pvs, pvs;
                               gen_pos=2, total_pos=10, prev_word=prev,
                               context_ids=cids, context_words=prompt_tokens)
    t3 = time_ns()
    dt2 = (t3 - t2) / 1e6 # ms
    println("Candidate 2 score: $s2")
    println("Time for candidate 2: $dt2 ms")

    cand3 = cands[3]
    println("Profiling third candidate (pure runtime): '$cand3'")
    t4 = time_ns()
    s3 = MirnanNew.Physics.Generator._score(gen, cand3, us, pvs, pvs;
                               gen_pos=2, total_pos=10, prev_word=prev,
                               context_ids=cids, context_words=prompt_tokens)
    t5 = time_ns()
    dt3 = (t5 - t4) / 1e6 # ms
    println("Candidate 3 score: $s3")
    println("Time for candidate 3: $dt3 ms")
end
