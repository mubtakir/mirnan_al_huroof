#!/usr/bin/env julia
# مرنان — واجهة أوامر تفاعلية كاملة
using Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "src", "MirnanNew.jl")); using .MirnanNew, SparseArrays, JSON

# يجب أن يكون include على المستوى العلوي (top-level) لتجنب مشكلة world age في Julia 1.12
include(joinpath(@__DIR__, "train.jl"))

function _warmup_vocab!(gen, vocab, K_sem)
    V = length(vocab)
    V == 0 && return
    # Sort words by total K_sem coupling strength (most-connected first)
    id2word = Dict(v=>k for (k,v) in vocab)
    n_warm = min(V, 20000)
    println("⧳ تسخين المتجهات الطورية لأكثر $n_warm كلمة ارتباطاً...")
    if K_sem !== nothing && length(K_sem.nzval) > 0
        rowsums = sum(K_sem; dims=2)[:]
        sorted_ids = sortperm(vec(rowsums); rev=true)
        warm_ids = sorted_ids[1:n_warm]
        for i in 1:length(warm_ids)
            wid = warm_ids[i]
            w = get(id2word, wid, nothing)
            w === nothing && continue
            try
                MirnanNew.Physics.Generator._pv(gen, w)
            catch; end
            i % 2000 == 0 && print("."); flush(stdout)
        end
    else
        for (i, w) in enumerate(keys(vocab))
            i > n_warm && break
            try
                MirnanNew.Physics.Generator._pv(gen, w)
            catch; end
            i % 2000 == 0 && print("."); flush(stdout)
        end
    end
    println(" ✓")
end

function load_generator()
    model_dir = joinpath(@__DIR__, "model")
    vf = joinpath(model_dir, "vocab.json")
    
    # Try loading trained model
    if isfile(vf)
        println("جاري تحميل النموذج المدرب...")
        data = load_model(model_dir)
        if data !== nothing
            id2word = Dict(v=>k for (k,v) in data["vocab"])
            gen = MirnanNew.MirnanGenerator(data["vocab"], data["K_sem"];
                                  K_syn=data["K_syn"],
                                  K_causal=get(data, "K_causal", nothing),
                                  model_dir=model_dir)
            MirnanNew.Physics.Generator.gpu_init!(gen)   # فعّل GPU إن كان ممكّناً في config
            println("✓ $(length(data["vocab"])) كلمة + $(length(data["K_sem"].nzval)) اقتران")
            # Warm-up: precompute phase vectors for top N most-connected words
            _warmup_vocab!(gen, data["vocab"], data["K_sem"])
            return gen
        end
    end
    
    # Fallback: build demo vocab from data/ files (much richer than 105 words)
    println("⚠ لم يعثر على نموذج مدرب — بناء معجم تجريبي من data/")
    demo_texts = String[]
    data_dir = joinpath(@__DIR__, "data")
    corpus_dir = joinpath(data_dir, "corpus")
    demo_dirs = [corpus_dir, joinpath(data_dir, "toy_corpus")]
    for dd in demo_dirs
        isdir(dd) || continue
        for (root, dirs, files) in walkdir(dd)
            for f in files
                ext = lowercase(splitext(f)[2])
                ext in [".txt", ".md", ""] || continue
                try
                    content = read(joinpath(root, f), String)
                    lines = split(content, r"\n")
                    for l in lines[1:min(500, end)]
                        s = strip(l)
                        length(s) >= 10 && push!(demo_texts, s)
                    end
                catch; end
            end
        end
    end
    if isempty(demo_texts)
        demo_texts = [
            "العلم نور والجهل ظلام والسماء صافية والأرض خضراء والحياة جميلة والعالم كبير",
            "الله خالق كل شيء والكتاب مفيد والعلم نور والماء سر الحياة",
            "السلام عليكم ورحمة الله وبركاته القلب الكبير يعرف الحب والطريق الحق",
            "كان هناك رجل عالم يسعى دائما نحو الأفضل يعمل في النهار ويقرأ في الليل",
            "فقال الحكيم إن الصبر مفتاح الفرج ومن يسعى يجد ومن يطلب يحقق الأمل",
            "إن الإنسان لا يتعلم من الخطأ بل من التأمل في الخطأ والحكمة ضالة المؤمن",
        ]
    end
    vocab = Dict{String,Int}()
    for t in demo_texts[1:min(3000, end)]
        for w in split(t)
            wc = strip(w); length(wc) >= 2 || continue
            haskey(vocab, wc) || (vocab[wc] = length(vocab) + 1)
        end
    end
    V = min(length(vocab), 8000)
    final_vocab = Dict{String,Int}()
    for (i, (w, _)) in enumerate(collect(vocab)[1:V])
        final_vocab[w] = i
    end
    K_sem = spzeros(V, V)
    for t in demo_texts[1:min(2000, end)]
        words = [strip(w) for w in split(t) if length(strip(w)) >= 2]
        ids = [get(final_vocab, w, 0) for w in words]; filter!(x->x>0, ids)
        for i in 1:length(ids)
            for j in max(1, i-8):min(length(ids), i+8)
                j != i || continue
                K_sem[ids[i], ids[j]] += 1.0 / abs(j - i)
            end
        end
    end
    gen = MirnanNew.MirnanGenerator(final_vocab, K_sem)
    MirnanNew.Physics.Generator.gpu_init!(gen)
    println("✓ $(length(final_vocab)) كلمة (معجم تجريبي من data/)")
    return gen
end

function interactive_loop(gen)
    println("مرنان — الوضع التفاعلي (exit للخروج)")
    println("أوضاع: standard creative quantum multiverse poetic wave dialogue reason resonant prnn entangle")
    println("أوامر: /mode NAME  /beta N  /meter NAME  /rhyme حرف  /reset  /report")
    println("أوضاع حقل الجذر: /mode root | /mode root_poetic | /mode root_list")
    println("-"*50)
    mode = "auto"; meter = "كامل"; rhyme = nothing
    
    while true
        print("> "); flush(stdout)
        line = strip(readline())
        if isempty(line); continue
        elseif lowercase(line) == "exit"; break
        elseif startswith(line, "/mode")
            parts = split(line); length(parts)>=2 && (mode=parts[2]; println("الوضع = $mode"))
            continue
        elseif startswith(line, "/beta")
            parts = split(line); length(parts)>=2 && try gen.beta=parse(Float64,parts[2]); println("β=$(gen.beta)"); catch; end
            continue
        elseif startswith(line, "/meter"); parts=split(line); length(parts)>=2 && (meter=parts[2]; println("البحر = $meter")); continue
        elseif startswith(line, "/rhyme"); parts=split(line); length(parts)>=2 && (rhyme=parts[2]; println("القافية = $rhyme")); continue
        elseif line == "/reset"; MirnanNew.Physics.Generator.reset!(gen); println("إعادة تعيين ✓"); continue
        elseif line == "/report"
            r = MirnanNew.Physics.Generator.get_physics_report(gen, String[])
            fr = MirnanNew.Physics.LanguageFeedback.get_feedback_report(gen.lang_feedback)
            println("الإنتروبيا=$(round(r["entropy"],digits=3)) β=$(r["beta"]) كلمات=$(r["vocab_size"])")
            println("تغذية راجعة: جيد=$(fr["good_count"]) ضعيف=$(fr["poor_count"]) تماسك=$(fr["avg_coherence"]) نسبة=$(fr["good_ratio"])")
            continue
        end
        
        result = MirnanNew.Physics.Generator.generate!(gen, line; mode=mode, max_words=15,
                                   poetic_meter=meter, poetic_rhyme=rhyme)
        if !isempty(result); println("  ↳ $result"); else; println("  ↳ [توليد فارغ]"); end
    end
end

function main()
    gen = load_generator()
    args = ARGS; i = 1; mode = "auto"; interactive = false
    
    while i <= length(args)
        if args[i] == "-i" || args[i] == "--interactive"; interactive = true
        elseif args[i] == "--mode" && i < length(args); i+=1; mode = args[i]
        end; i += 1
    end
    
    if interactive; interactive_loop(gen); return; end
    
    prompt = join(filter(x->!startswith(x,"--"), args), " ")
    if isempty(prompt); println("استخدم: julia cli.jl -i للوضع التفاعلي"); return; end
    
    result = MirnanNew.Physics.Generator.generate!(gen, prompt; mode=mode)
    if !isempty(result); println(result); else; println("[توليد فارغ]"); end
end
main()
