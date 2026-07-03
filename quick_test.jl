using Pkg
Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "src", "MirnanNew.jl"))
using .MirnanNew
using .MirnanNew.Physics
using LinearAlgebra

include(joinpath(@__DIR__, "train.jl"))

function main()
    # Test _parse_word_harakat with English
    println("=== _parse_word_harakat ===")
    for w in ["hello", "running", "the", "beautiful"]
        parsed = MirnanNew.Physics.WordPhysics._parse_word_harakat(w)
        println("  '$w' -> $(length(parsed)) chars: $([string(p[1]) for p in parsed])")
    end

    # Test _extract_root_light with English
    println("\n=== _extract_root_light ===")
    for w in ["running", "children", "unhappiness", "beautiful", "cats", "went"]
        root = MirnanNew.Physics.WordPhysics._extract_root_light(w)
        println("  '$w' -> '$(String(root))'")
    end

    # Test compute_syntax_vector with English
    println("\n=== compute_syntax_vector ===")
    for w in ["the", "run", "beautiful", "quickly", "and", "in", "is", "running"]
        sv = Physics.SyntaxField.compute_syntax_vector(w)
        println("  '$w' -> pos=$(Physics.SyntaxField._en_get_pos(w)), sv=$(round.(sv; digits=2))")
    end

    # Test compute_extended_phase_vector with English
    println("\n=== compute_extended_phase_vector ===")
    for w in ["hello", "running", "the", "beautiful", "knowledge"]
        pv = Float64.(MirnanNew.Physics.WordPhysics.compute_extended_phase_vector(w))
        println("  '$w' -> norm=$(round(norm(pv); digits=4)), nonzero=$(count(!=(0.0), pv))")
    end

    # Test phase_similarity between English words
    println("\n=== phase_similarity (English) ===")
    pairs = [("running","run"), ("cats","cat"), ("happiness","happy"), ("beautiful","beauty"), ("quick","quickly")]
    for (a,b) in pairs
        pv_a = Float64.(MirnanNew.Physics.WordPhysics.compute_extended_phase_vector(a))
        pv_b = Float64.(MirnanNew.Physics.WordPhysics.compute_extended_phase_vector(b))
        sim = MirnanNew.Physics.WordPhysics.phase_similarity(pv_a, pv_b)
        println("  '$a' <-> '$b' : sim=$(round(sim; digits=4))")
    end

    # Test full generation with English
    println("\n=== Generation (English) ===")
    texts = ["the sun is bright and the sky is blue",
             "running quickly through the beautiful garden",
             "knowledge is power and wisdom is understanding",
             "children play in the park every day"]
    vocab = Base.invokelatest(Main.build_vocab, texts; min_count=1)
    K_sem = Base.invokelatest(Main.build_K, vocab, texts; window=10)
    gen = Physics.Generator.MirnanGenerator(vocab, K_sem; beam_width=3, top_k=100)

    for (w, _) in gen.vocab
        Physics.Generator._pv(gen, w)
        Physics.Generator._mass(gen, w)
        Physics.Generator._syn(gen, w)
    end
    println("  Vocab: $(length(gen.vocab)) words")

    for p in ["the sun", "running quick", "knowledge is", "children play"]
        r = Physics.Generator.generate!(gen, p; mode="standard", max_words=4)
        println("  '$p' -> '$(strip(r))'")
    end
end

main()
