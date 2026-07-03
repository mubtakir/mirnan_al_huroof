using Pkg; Pkg.activate(".")
using JSON, SparseArrays, Mirnan

const MODEL_DIR = "model"

function load_sparse(path)
    isfile(path) || return spzeros(0, 0)
    open(path, "r") do io
        header = readline(io); @assert header == "SPARSE_CSC"
        m = read(io, Int32); n = read(io, Int32); nnz = read(io, Int32)
        colptr = read!(io, Vector{Int32}(undef, n+1))
        rows = read!(io, Vector{Int32}(undef, nnz))
        vals = read!(io, Vector{Float64}(undef, nnz))
        return SparseMatrixCSC(Int(m), Int(n), Vector{Int}(colptr), Vector{Int}(rows), vals)
    end
end

vocab = Dict{String,Int}(k => Int(v) for (k,v) in JSON.parsefile(joinpath(MODEL_DIR, "vocab.json")))
K_sem = load_sparse(joinpath(MODEL_DIR, "K_sem.dat"))
K_syn = load_sparse(joinpath(MODEL_DIR, "K_syn.dat"))
K_dial = load_sparse(joinpath(MODEL_DIR, "K_dialogue.dat"))
K_causal = load_sparse(joinpath(MODEL_DIR, "K_causal.dat"))

gen = MirnanGenerator(vocab, K_sem; K_syn=K_syn, K_dialogue=K_dial, K_causal=K_causal)

prompt = "الحياة"
println("Calling generate! for '$prompt'...")
res = generate!(gen, prompt; max_words=8, mode="resonant")
println("Result: '", res, "'")

# Let's inspect the prompt_tokens
prompt_tokens = String[MirnanNew.Physics.Generator._normalize_arabic_token(MirnanNew.Physics.Generator._strip_punct_edge(w)) for w in split(prompt) if length(strip(w)) >= 1]
println("Prompt tokens: ", prompt_tokens)
for w in prompt_tokens
    println("  Token: '$w', wid: ", get(gen.vocab, w, 0))
end
