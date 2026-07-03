push!(LOAD_PATH, joinpath(dirname(@__DIR__), "src"))
include(joinpath(@__DIR__, "src", "MirnanNew.jl")); using .MirnanNew, JSON, SparseArrays

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

println("Warmup generator...")
generate!(gen, "العلم"; mode="standard", max_words=2)

function run_test()
    println("Running generation step-by-step for 'غابة'...")
    prompt_tokens = ["غابة"]
    max_words = 15

    prompt_pv = [MirnanNew.Physics.Generator._pv(gen,w) for w in prompt_tokens]
    MirnanNew.Physics.Generator.PromptConstraint.set_prompt!(gen.prompt_constraint, prompt_pv)
    MirnanNew.Physics.Generator.PPM.absorb!(gen.prompt_field, join(prompt_tokens," "))
    MirnanNew.Physics.Generator.DensityMatrix.build!(gen.density_matrix, prompt_pv)

    beams = [prompt_tokens[:]]; used_sets = [Set(prompt_tokens)]
    k_B_cur = gen.entropy.k_B; beta_cur = gen.beta; S_cur = 0.0

    for step in 1:max_words
        t_start = time()
        gen.current_tau = gen.tau_min + (gen.tau_max-gen.tau_min)*exp(-step/gen.tau_decay)
        
        # Trace beam step
        println("Step $step: Beams count = $(length(beams))")
        for (i, b) in enumerate(beams)
            println("  Beam $i: $b")
        end
        
        new = MirnanNew.Physics.Generator._beam_step(gen, beams, used_sets, prompt_pv; step=step, total=max_words, k_B_cur=k_B_cur, beta_cur=beta_cur, S=S_cur)
        
        t_end = time()
        println("  Step $step took $(t_end - t_start) seconds")
        
        if isempty(new)
            println("  No candidates, breaking.")
            break
        end
        
        beams = [b[2] for b in new]
        used_sets = [b[3] for b in new]
        
        if length(beams[1]) >= 3
            m_seq = [MirnanNew.Physics.Generator._mass(gen,w) for w in beams[1]]
            pv_seq = [MirnanNew.Physics.Generator._pv(gen,w) for w in beams[1]]
            coh = MirnanNew.Physics.Generator.ResonantChain.sentence_coherence(gen.resonant_chain, m_seq, pv_seq)
            println("  Sentence coherence = $coh")
            if coh < 0.3 && step >= 3
                println("  Coherence $coh < 0.3, breaking.")
                break
            end
        end
        MirnanNew.Physics.Generator.PPM.step!(gen.prompt_field)
    end

    output = beams[1][length(prompt_tokens)+1:end]
    println("Output: $(join(output, " "))")
end

run_test()
