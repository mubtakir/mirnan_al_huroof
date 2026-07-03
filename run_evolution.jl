"""
تشغيل عملية التحسين التطوري للأوزان — run_evolution.jl
تقوم بتشغيل الخوارزمية الجينية على عينات من الكوربس وحفظ الأوزان المتطورة في config.yaml.
"""

include(joinpath(@__DIR__, "src", "MirnanNew.jl"))
using .MirnanNew
using .MirnanNew.Physics: MirnanCfg, load_config, MirnanGenerator, evolve_weights!
using SparseArrays
using JSON

const PROJECT_DIR = @__DIR__

function _load_sparse_dat(path::String, vocab_size::Int)
    if !isfile(path)
        return spzeros(vocab_size, vocab_size)
    end
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

function _load_sparse_dat_verbose(path::String, vocab_size::Int, label::String)
    if isfile(path)
        mb = round(filesize(path) / 1024^2; digits=1)
        println("  -> loading $label ($mb MB)...")
    else
        println("  -> loading $label: missing, using empty matrix")
    end
    flush(stdout)
    t0 = time()
    K = _load_sparse_dat(path, vocab_size)
    println("     done $label: $(length(K.nzval)) links in $(round(time() - t0; digits=1))s")
    flush(stdout)
    return K
end

function save_evolved_weights(yaml_path::String, evolved_weights::Dict{String, Float64})
    if !isfile(yaml_path)
        println("Warning: config file not found at $yaml_path")
        return
    end
    lines = readlines(yaml_path)
    new_lines = String[]
    in_scoring = false
    for line in lines
        # التحقق من الخروج من قسم scoring_weights عند رصد قسم جديد غير محاذٍ بمسافات
        if in_scoring && !startswith(line, " ") && !isempty(strip(line)) && !startswith(strip(line), "#")
            in_scoring = false
        end
        if startswith(line, "scoring_weights:")
            in_scoring = true
            push!(new_lines, line)
            continue
        end
        if in_scoring
            # مطابقة نمط "  align: 5.0" مع الحفاظ على التعليقات والمسافات
            m = match(r"^(\s+)([a-zA-Z_0-9]+):\s*([0-9\.\-\+eE]+)(.*)$", line)
            if m !== nothing
                indent = m.captures[1]
                key = m.captures[2]
                comment = m.captures[4]
                if haskey(evolved_weights, key)
                    new_val = round(evolved_weights[key]; digits=4)
                    push!(new_lines, "$(indent)$(key): $(new_val)$(comment)")
                    continue
                end
            end
        end
        push!(new_lines, line)
    end
    open(yaml_path, "w") do io
        for l in new_lines
            println(io, l)
        end
    end
    println("✓ Evolved weights successfully saved to config.yaml")
end

function main()
    println("=== Starting Al-Tawweer Evolutionary Weight Optimizer ===")
    config_path = joinpath(PROJECT_DIR, "config.yaml")
    model_dir = joinpath(PROJECT_DIR, "model")
    vf = joinpath(model_dir, "vocab.json")
    
    if !isfile(vf)
        println("Error: vocab.json not found in model/ directory. Please run train.jl first.")
        return
    end
    
    raw_vocab = JSON.parsefile(vf)
    vocab = Dict{String,Int}(k => Int(v) for (k,v) in raw_vocab)
    V = length(vocab)
    println("  → Vocabulary loaded: $V words")
    
    K_sem = _load_sparse_dat_verbose(joinpath(model_dir, "K_sem.dat"), V, "K_sem")
    K_syn = _load_sparse_dat_verbose(joinpath(model_dir, "K_syn.dat"), V, "K_syn")
    K_causal = _load_sparse_dat_verbose(joinpath(model_dir, "K_causal.dat"), V, "K_causal")
    
    # تهيئة المولد بالكامل
    println("-> Constructing generator...")
    gen = MirnanGenerator(vocab, K_sem;
                           K_syn=K_syn,
                           K_causal=K_causal,
                           model_dir=model_dir)
    
    # أسئلة تحقق سريعة تغطي سيناريوهات الفهم المختلفة
    validation_prompts = [
        "ما هو العلم ؟",
        "كيف يعمل العقل ؟",
        "من هو الحكيم ؟",
        "هل النور مفيد ؟",
        "ما هو أثر النور ؟"
    ]
    
    println("-> Running evolution (5 generations, population size 6)...")
    evolved = evolve_weights!(gen, validation_prompts; generations=5, pop_size=6)
    
    println("-> Saving evolved weights back to config.yaml...")
    save_evolved_weights(config_path, evolved)
    println("=== Al-Tawweer evolution completed successfully ===")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
