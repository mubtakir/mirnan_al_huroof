#!/usr/bin/env julia
# اختبار مباشر لـ generate! بدون خادم HTTP
# يُظهر ما يحدث بالضبط عند إرسال "نور"
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

include(joinpath(@__DIR__, "src", "MirnanNew.jl")); using .MirnanNew, JSON, SparseArrays

println("═══════════════════════════════════════")
println("  اختبار مباشر لمحرك التوليد (diag5.jl)")
println("═══════════════════════════════════════")

# --- تحميل المعجم والمصفوفات ---
function load_sparse_dat(path, vocab_size)
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
        return SparseMatrixCSC(Int(m), Int(n), Vector{Int}(colptr), Vector{Int}(rows), vals)
    end
end

model_dir = joinpath(@__DIR__, "model")
raw_vocab = JSON.parsefile(joinpath(model_dir, "vocab.json"))
vocab = Dict{String,Int}(k => Int(v) for (k,v) in raw_vocab)
V = length(vocab)
K_sem  = load_sparse_dat(joinpath(model_dir, "K_sem.dat"),  V)
K_syn  = load_sparse_dat(joinpath(model_dir, "K_syn.dat"),  V)
K_dial = load_sparse_dat(joinpath(model_dir, "K_dialogue.dat"), V)

println("✓ معجم: $V كلمة | K_sem: $(length(K_sem.nzval)) اقتران")

# --- بناء المولّد ---
println("\n⟳ بناء MirnanGenerator...")
t0 = time()
gen = MirnanGenerator(vocab, K_sem; K_syn=K_syn, K_dialogue=K_dial)
println("✓ تم البناء في $(round(time()-t0; digits=1))s")

# --- الاختبار الأول (JIT compilation) ---
println("\n⟳ اختبار 1: توليد 'نور' (JIT سيُجمَّع الآن) ...")
t0 = time()
result = try
    generate!(gen, "نور"; mode="standard", max_words=8)
catch e
    println("❌ استثناء: $e")
    ""
end
elapsed = round(time()-t0; digits=2)
println("✓ النتيجة: '$result'")
println("  الزمن: $(elapsed)s")

# --- الاختبار الثاني (بعد JIT) ---
println("\n⟳ اختبار 2: توليد 'العلم' (بعد JIT)...")
t0 = time()
result2 = try
    generate!(gen, "العلم"; mode="standard", max_words=8)
catch e
    println("❌ استثناء: $e")
    ""
end
elapsed2 = round(time()-t0; digits=2)
println("✓ النتيجة: '$result2'")
println("  الزمن: $(elapsed2)s")

# --- اختبار وضع auto ---
println("\n⟳ اختبار 3: وضع auto مع 'السماء'...")
t0 = time()
result3 = try
    generate!(gen, "السماء"; mode="auto", max_words=6)
catch e
    println("❌ استثناء: $e")
    ""
end
elapsed3 = round(time()-t0; digits=2)
println("✓ النتيجة: '$result3'")
println("  الزمن: $(elapsed3)s")

println("\n═══ ملخص ═══")
println("1. نور → $result ($(elapsed)s)")
println("2. العلم → $result2 ($(elapsed2)s)")
println("3. السماء → $result3 ($(elapsed3)s)")
println("═══════════════════════════════════════")
