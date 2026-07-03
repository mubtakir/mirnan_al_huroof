#!/usr/bin/env julia
# تشخيص مباشر: اختبار خط الأنابيب بدون API server
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

println("جاري تحميل الحزم...")
using SparseArrays, JSON
println("✓ SparseArrays, JSON")

# تحميل المعجم
model_dir = joinpath(@__DIR__, "model")
vf = joinpath(model_dir, "vocab.json")

if !isfile(vf)
    println("❌ لم يوجد vocab.json في: $vf")
    exit(1)
end

raw_vocab = JSON.parsefile(vf)
vocab = Dict{String,Int}(k => Int(v) for (k,v) in raw_vocab)
id2word = Dict{Int,String}(v => k for (k,v) in vocab)
V = length(vocab)
println("✓ معجم: $V كلمة")

# تحميل K_sem
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
        return SparseMatrixCSC(Int(m), Int(n),
                               Vector{Int}(colptr),
                               Vector{Int}(rows),
                               vals)
    end
end

K_sem = load_sparse_dat(joinpath(model_dir, "K_sem.dat"), V)
println("✓ K_sem: $(length(K_sem.nzval)) اقتران غير صفري")
println("  قيمة min: $(minimum(K_sem.nzval)), max: $(maximum(K_sem.nzval))")

# هل كلمة "نور" موجودة في المعجم؟
test_word = "نور"
wid = get(vocab, test_word, nothing)
if wid === nothing
    println("❌ كلمة '$test_word' غير موجودة في المعجم!")
    # ابحث عن كلمات مشابهة
    matches = [k for k in keys(vocab) if contains(k, "نور") || contains(k, "نو")]
    println("   كلمات تحتوي على 'نور': $matches")
else
    println("✓ كلمة '$test_word' موجودة بمعرف: $wid")
    # ما هي اقترانات هذه الكلمة؟
    col = K_sem[:, wid]
    nz = col.nzind
    nz_vals = col.nzval
    println("  عدد الاقترانات: $(length(nz))")
    if !isempty(nz)
        sorted_idx = sortperm(nz_vals; rev=true)[1:min(10, length(nz))]
        println("  أعلى 10 اقترانات:")
        for i in sorted_idx
            tid = nz[i]
            w = get(id2word, tid, "???")
            println("    $w → $(round(nz_vals[i]; digits=4))")
        end
    else
        println("  ⚠ لا توجد اقترانات لهذه الكلمة!")
    end
end

# اختبار K_sem للكلمات الأولى
println("\n--- أعلى 5 اقترانات لكل كلمة من الكلمات الأولى ---")
for (word, wid) in collect(vocab)[1:min(10, end)]
    col = K_sem[:, wid]
    if !isempty(col.nzind)
        best_tid = col.nzind[argmax(col.nzval)]
        best_val = maximum(col.nzval)
        best_w = get(id2word, best_tid, "???")
        println("  $word → $best_w ($(round(best_val; digits=4)))")
    else
        println("  $word → لا اقترانات")
    end
end

# محاولة التوليد اليدوي البسيط
println("\n--- محاولة توليد يدوي من K_sem ---")
start_word = test_word
if wid !== nothing
    used = Set([start_word])
    output = String[]
    ctx = start_word
    cid = wid
    
    for step in 1:8
        col = K_sem[:, cid]
        if isempty(col.nzind)
            println("  خطوة $step: لا اقترانات - توقف")
            break
        end
        sorted_ids = col.nzind[sortperm(col.nzval; rev=true)]
        found = false
        for tid in sorted_ids
            w = get(id2word, tid, nothing)
            w === nothing && continue
            w in used && continue
            length(w) < 2 && continue
            col[tid] > 1e-6 || continue
            push!(output, w)
            push!(used, w)
            cid = tid
            println("  خطوة $step: $w ($(round(col[tid]; digits=4)))")
            found = true
            break
        end
        !found && break
    end
    
    if isempty(output)
        println("❌ فشل التوليد اليدوي - لا مرشحين صالحين")
    else
        println("✓ الناتج: $start_word " * join(output, " "))
    end
end
