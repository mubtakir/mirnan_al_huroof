push!(LOAD_PATH, joinpath(dirname(@__DIR__), "src"))
include(joinpath(@__DIR__, "src", "MirnanNew.jl")); using .MirnanNew, JSON, SparseArrays, FFTW

println("Loading generator...")
model_dir = joinpath(dirname(@__DIR__), "model")
vf = joinpath(model_dir, "vocab.json")
raw_vocab = JSON.parsefile(vf)
vocab = Dict{String,Int}(k => Int(v) for (k,v) in raw_vocab)
V = length(vocab)
println("Vocabulary size: $V")

K_sem = spzeros(V, V) # dummy for test
gen = MirnanGenerator(vocab, K_sem)

println("Precomputing FFT for all vocabulary words...")
t1 = time()
plan = plan_fft(zeros(ComplexF64, MirnanNew.Physics.Constants.PHASE_DIM))
for (word, id) in vocab
    w_pv = MirnanNew.Physics.Generator._pv(gen, word)
    w_phase = get!(gen.pv_cache, "FFT\0" * word) do
        angle.(plan * ComplexF64.(w_pv[1:MirnanNew.Physics.Constants.PHASE_DIM]))
    end
end
t2 = time()
println("Precomputation took $(t2 - t1) seconds")
