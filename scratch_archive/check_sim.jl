push!(LOAD_PATH, joinpath(@__DIR__, "src"))
include(joinpath(@__DIR__, "src", "MirnanNew.jl")); using .MirnanNew

function test_sim(w1, w2)
    v1 = compute_word_phase_vector(w1)
    v2 = compute_word_phase_vector(w2)
    sim = phase_similarity(v1, v2)
    println("Similarity between '$w1' and '$w2': $sim")
end

test_sim("علم", "معرفة")
test_sim("علم", "تعليم")
test_sim("سقراط", "سقراطه")
test_sim("نور", "ظلام")
