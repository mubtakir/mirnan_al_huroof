#!/usr/bin/env julia
# benchmark.jl — تقييم صادق لجودة التوليد
# يقارن Mirnan مع:
# 1. نموذج عشوائي (baseline)
# 2. نموذج Markov من الدرجة الأولى

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "MirnanNew.jl"))
include(joinpath(@__DIR__, "..", "train.jl"))
using .MirnanNew
using JSON, SparseArrays, LinearAlgebra, Statistics

const PROJECT_DIR = joinpath(@__DIR__, "..")
const MODEL_DIR = joinpath(PROJECT_DIR, "model")

# ═══════════════════════════════════════════════════════
# 1. نموذج عشوائي (Baseline)
# ═══════════════════════════════════════════════════════
function random_generate(vocab::Dict{String,Int}, id2word::Dict{Int,String};
                         prompt_words::Vector{String}=String[], max_words::Int=10)
    output = String[]
    used = Set(prompt_words)
    for _ in 1:max_words
        candidates = filter(w -> !(w in used) && length(w) >= 2, keys(vocab))
        isempty(candidates) && break
        word = rand(collect(candidates))
        push!(output, word)
        push!(used, word)
    end
    return join(output, " ")
end

# ═══════════════════════════════════════════════════════
# 2. نموذج Markov من الدرجة الأولى
# ═══════════════════════════════════════════════════════
function build_markov_chain(texts::Vector{String}, vocab::Dict{String,Int})
    trans = Dict{Int, Dict{Int, Float64}}()
    for text in texts
        words = String[strip(w) for w in split(text) if length(strip(w)) >= 2]
        ids = [get(vocab, w, 0) for w in words]
        filter!(id -> id > 0, ids)
        for i in 1:length(ids)-1
            cur, nxt = ids[i], ids[i+1]
            if !haskey(trans, cur); trans[cur] = Dict{Int, Float64}(); end
            trans[cur][nxt] = get(trans[cur], nxt, 0.0) + 1.0
        end
    end
    for (cur, nexts) in trans
        total = sum(values(nexts))
        for (nxt, count) in nexts; trans[cur][nxt] = count / total; end
    end
    return trans
end

function markov_generate(trans, vocab, id2word; prompt_words=String[], max_words=10)
    output = String[]
    used = Set(prompt_words)
    current_id = !isempty(prompt_words) ? get(vocab, prompt_words[end], 0) : rand(keys(trans))
    current_id == 0 && (current_id = rand(keys(trans)))
    for _ in 1:max_words
        !haskey(trans, current_id) && break
        nexts = trans[current_id]
        probs = collect(values(nexts))
        candidates = collect(keys(nexts))
        valid_idx = findall(id -> haskey(id2word, id) && !(id2word[id] in used), candidates)
        isempty(valid_idx) && break
        probs, candidates = probs[valid_idx], candidates[valid_idx]
        cum_prob = cumsum(probs)
        r = rand() * sum(probs)
        sel = findfirst(p -> p >= r, cum_prob)
        sel === nothing && (sel = length(candidates))
        word = id2word[candidates[sel]]
        push!(output, word); push!(used, word); current_id = candidates[sel]
    end
    return join(output, " ")
end

# ═══════════════════════════════════════════════════════
# تقييم الجودة
# ═══════════════════════════════════════════════════════
function evaluate_quality(text::String; name::String="model")
    if isempty(strip(text))
        return Dict{String,Any}("model"=>name, "text"=>text, "length"=>0,
            "unique_ratio"=>0.0, "repetition_penalty"=>1.0, "coherence_score"=>0.0)
    end
    words = split(text)
    n = length(words)
    unique_ratio = n > 0 ? length(Set(words)) / n : 0.0
    word_counts = Dict{String,Int}()
    for w in words; word_counts[w] = get(word_counts, w, 0) + 1; end
    rep = n > 0 ? maximum(values(word_counts)) / n : 1.0
    return Dict{String,Any}("model"=>name, "text"=>text, "length"=>n,
        "unique_ratio"=>unique_ratio, "repetition_penalty"=>rep, "coherence_score"=>1.0-rep)
end

# ═══════════════════════════════════════════════════════
# الرئيسية
# ═══════════════════════════════════════════════════════
function main()
    println("=" ^ 60)
    println("📊 benchmark — تقييم صادق لجودة التوليد")
    println("=" ^ 60)
    println()

    println("1️⃣  تحميل النموذج...")
    model = load_model()
    if model === nothing
        println("❌ النموذج غير موجود! شغّل train.jl أولاً."); return
    end
    vocab = model["vocab"]
    K_sem = model["K_sem"]
    K_syn = model["K_syn"]
    id2word = Dict{Int,String}(v=>k for (k,v) in vocab)
    println("   ✓ حجم المعجم: $(length(vocab))")
    println("   ✓ K_sem: $(nnz(K_sem)) اقتران")

    println()
    println("2️⃣  بناء النماذج المرجعية...")
    texts = load_all_corpus()
    println("   ✓ $(length(texts)) وثيقة")

    println("   📊 بناء Markov...")
    markov_trans = build_markov_chain(texts, vocab)
    println("   ✓ $(length(markov_trans)) حالة انتقالية")

    println()
    println("3️⃣  اختبار التوليد...")

    test_prompts = [
        "السلام عليكم",
        "كيف حالك",
        "ماذا تعرف عن",
        "اكتب قصة قصيرة عن",
        "اشرح لي مفهوم",
    ]

    results = Dict{String, Vector{Dict{String,Any}}}()

    for prompt in test_prompts
        println()
        println("📝 \"$prompt\"")
        println("-" ^ 40)

        # Mirnan
        mirnan_result = try
            gen = MirnanNew.MirnanGenerator(vocab, K_sem)
            MirnanNew.generate!(gen, prompt; max_words=10)
        catch e
            println("   ⚠ Mirnan error: $e"); ""
        end

        # Random
        random_result = random_generate(vocab, id2word; max_words=10)

        # Markov
        markov_result = markov_generate(markov_trans, vocab, id2word; max_words=10)

        function _trunc(s, n=40)
            length(s) <= n && return s
            return s[1:nextind(s, 1, n)] * "..."
        end
        println("   Mirnan:  $(_trunc(mirnan_result))")
        println("   Random:  $(_trunc(random_result))")
        println("   Markov:  $(_trunc(markov_result))")

        for (text, name) in [(mirnan_result,"Mirnan"), (random_result,"Random"), (markov_result,"Markov")]
            if !haskey(results, name); results[name] = Dict{String,Any}[]; end
            push!(results[name], evaluate_quality(text; name=name))
        end
    end

    println()
    println("=" ^ 60)
    println("📊 ملخص النتائج")
    println("=" ^ 60)

    for (model_name, evals) in sort(collect(results); by=x->x[1])
        avg_len = mean(e["length"] for e in evals)
        avg_uniq = mean(e["unique_ratio"] for e in evals)
        avg_coh = mean(e["coherence_score"] for e in evals)
        println()
        println("🤖 $model_name:")
        println("   متوسط الطول: $(round(avg_len, digits=1)) كلمة")
        println("   نسبة الفريدة: $(round(avg_uniq * 100, digits=1))%")
        println("   درجة التماسك: $(round(avg_coh * 100, digits=1))%")
    end

    output_file = joinpath(@__DIR__, "..", "benchmark_results.json")
    open(output_file, "w") do io; JSON.print(io, results, 2); end
    println()
    println("💾 $output_file")
    println()
    println("✅ اكتمل التقييم!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
