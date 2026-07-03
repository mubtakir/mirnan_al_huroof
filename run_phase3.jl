using LinearAlgebra
include("src/MirnanNew.jl")
using .MirnanNew
using .MirnanNew.Physics

println("=" ^ 60)
println("PHASE 3 TEST: Word & Sentence Level")
println("=" ^ 60)

function test_pv(word)
    randn(27)
end

# Test 1: Word Fusion
println("\n[1] Word Fusion...")
try
    using .Physics.WordFusion
    letter_pvs = [randn(27) for _ in 1:3]
    lin = linear_fusion(letter_pvs)
    println("  OK linear_fusion: norm=$(round(norm(lin); digits=4))")
    tensor = interaction_tensor(letter_pvs)
    println("  OK interaction_tensor: size=$(size(tensor))")
    eigen = eigen_fusion(letter_pvs)
    println("  OK eigen_fusion: eigenvalue=$(eigen["dominant_eigenvalue"])")
catch e
    println("  FAIL: $e")
end

# Test 2: Semantic Arithmetic
println("\n[2] Semantic Arithmetic...")
try
    using .Physics.SemanticArithmetic
    diff_result = semantic_difference("نور", "ظلام", test_pv)
    println("  OK semantic_difference: magnitude=$(diff_result["magnitude"])")
    antonym = predict_antonym("نور")
    println("  OK predict_antonym: $antonym")
    affinity = analyze_semantic_affinity("نور", "علم", test_pv)
    println("  OK analyze_semantic_affinity: sim=$(affinity["similarity"])")
catch e
    println("  FAIL: $e")
end

# Test 3: Syntax Field
println("\n[3] Syntax Field...")
try
    using .Physics.SyntaxField
    vec = compute_syntax_vector("كان")
    println("  OK compute_syntax_vector(كان): $vec")
    vec2 = compute_syntax_vector("الكتاب")
    println("  OK compute_syntax_vector(الكتاب): $vec2")
catch e
    println("  FAIL: $e")
end

# Test 4: Semantic Comprehension
println("\n[4] Semantic Comprehension...")
try
    using .Physics.SemanticComprehension
    topic = TopicDensityMatrix(dim=27, decay=0.85)
    println("  OK TopicDensityMatrix created")
    sv = compute_sentence_vector(["مرحبا", "بالعالم"], test_pv)
    println("  OK compute_sentence_vector: n_words=$(sv.n_words), coherence=$(round(sv.coherence; digits=4))")
    update_topic_with_word!(topic, "مرحبا", test_pv)
    update_topic_with_word!(topic, "بالعالم", test_pv)
    res = topic_resonance(topic, "سلام", test_pv)
    println("  OK topic_resonance: $res")
catch e
    println("  FAIL: $e")
end

# Test 5: MorphoTwistor
println("\n[5] MorphoTwistor...")
try
    using .Physics.MorphoTwistor
    engine = TwistorEngine()
    println("  OK TwistorEngine created")
    T = compute_twistor("نور", "نورا", test_pv)
    println("  OK compute_twistor: magnitude=$(round(T.magnitude; digits=4))")
    pairs = find_morphological_pairs(["نور", "نورا", "معلم", "معلمة"])
    println("  OK find_morphological_pairs: $(length(pairs)) pairs")
catch e
    println("  FAIL: $e")
end

println("\n" * "=" ^ 60)
println("PHASE 3 COMPLETE")
println("=" ^ 60)
