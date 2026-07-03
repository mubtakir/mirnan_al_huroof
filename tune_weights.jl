using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "src", "MirnanNew.jl"))
using .MirnanNew
using .MirnanNew.Physics
using LinearAlgebra, Statistics

include(joinpath(@__DIR__, "train.jl"))

const PROMPTS = ["العلم نور", "السماء صافية", "الحياة جميلة", "الماء نبع"]

function fast_eval(text::String, prompt::String)
    st = strip(text)
    isempty(st) && return 0.0
    words = split(st)
    n = length(words)
    s = 0.0
    s += (n >= 2 && n <= 6) ? 1.0 : 0.3
    pws = split(prompt)
    s += sum(occursin(pw, st) ? 1.0 : 0.0 for pw in pws) / max(length(pws), 1)
    s += length(Set(words)) / max(n, 1)
    return s / 3.0
end

function main()
    println("=" ^ 50)
    println("  Weight Tuning - Mirnan Generator (v4)")
    println("=" ^ 50)
    println()

    texts = Base.invokelatest(Main.load_all_corpus; granularity=:document)
    if isempty(texts)
        texts = [
            "العلم نور والجهل ظلام والسماء صافية والأرض خضراء والحياة جميلة والعالم كبير",
            "الله خالق كل شيء والكتاب مفيد والعلم نور والماء سر الحياة والأرض خضراء",
            "السلام عليكم ورحمة الله وبركاته القلب الكبير يعرف الحب والطريق الحق",
            "كان هناك رجل عالم يسعى دائما نحو الأفضل يعمل في النهار ويقرأ في الليل",
        ]
    end
    vocab = Base.invokelatest(Main.build_vocab, texts; min_count=1)
    K_sem = Base.invokelatest(Main.build_K, vocab, texts; window=10)
    println("Vocab: $(length(vocab)), K_sem: $(length(K_sem.nzval))")
    println()

    base = Dict{String,Float64}(
        "align" => 5.0, "prompt_align" => 6.0, "diversity" => 3.0,
        "gravity" => 5.0, "syntax" => 4.0, "density_resonance" => 2.0,
        "dccf" => 3.0, "ppm" => 2.5, "root_affinity" => 4.0,
        "surface_affinity" => 1.5
    )

    configs = Dict{String,Dict{String,Float64}}()
    configs["current"] = copy(base)

    c = copy(base); c["syntax"] = 8.0; c["root_affinity"] = 8.0
    configs["syntax_root_hi"] = c

    c = copy(base); c["align"] = 10.0; c["prompt_align"] = 12.0
    configs["align_hi"] = c

    c = Dict(k => 4.0 for k in keys(base))
    configs["all_eq4"] = c

    c = copy(base); c["gravity"] = 8.0; c["diversity"] = 1.0
    configs["gravity_hi"] = c

    c = copy(base); c["root_affinity"] = 10.0; c["surface_affinity"] = 3.0
    configs["root_hi"] = c

    c = copy(base); c["prompt_align"] = 10.0; c["ppm"] = 5.0
    configs["prompt_ppm"] = c

    c = copy(base); c["gravity"] = 2.0; c["syntax"] = 8.0
    configs["syntax_hi_lowgrav"] = c

    c = copy(base); c["dccf"] = 8.0; c["density_resonance"] = 5.0
    configs["dccf_hi"] = c

    c = copy(base); c["diversity"] = 5.0; c["gravity"] = 2.0; c["prompt_align"] = 8.0
    configs["div_prompt"] = c

    println("Testing $(length(configs)) configs x $(length(PROMPTS)) prompts...")
    println()

    gen = Physics.Generator.MirnanGenerator(vocab, K_sem; beam_width=2, top_k=50)
    println("Pre-warming ALL vocab caches...")
    for (w, _) in gen.vocab
        Physics.Generator._pv(gen, w)
        Physics.Generator._mass(gen, w)
        Physics.Generator._syn(gen, w)
    end
    println("Caches warm (pv=$(length(gen.pv_cache)), syn=$(length(gen.syntax_cache)))")
    println()

    results = []
    orig_weights = gen.scoring_weights
    for (name, weights) in configs
        t0 = time()
        gen.scoring_weights = weights

        scores = Float64[]
        outputs = String[]
        for prompt in PROMPTS
            try
                r = Physics.Generator.generate!(gen, prompt; mode="standard", max_words=4)
                push!(scores, fast_eval(r, prompt))
                push!(outputs, r)
            catch e
                push!(scores, 0.0)
                push!(outputs, "(err: $e)")
            end
        end
        elapsed = round(time() - t0, digits=2)
        avg = mean(scores)
        push!(results, (name=name, score=avg))
        ex = strip(outputs[1])
        isempty(ex) && (ex = "(empty)")
        println("  $(rpad(name, 24)) avg=$(round(avg, digits=4))  $(elapsed)s  ex: '$ex'")
    end
    gen.scoring_weights = orig_weights

    println()
    sort!(results; by=r -> -r.score)
    println("RANKING:")
    for (i, r) in enumerate(results)
        tag = i == 1 ? " <== BEST" : ""
        println("  #$i $(rpad(r.name, 24)) score=$(round(r.score, digits=4))$tag")
    end
end

main()
