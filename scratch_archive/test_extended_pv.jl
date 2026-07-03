using Random
push!(LOAD_PATH, joinpath(dirname(@__DIR__), "src"))
include(joinpath(@__DIR__, "src", "MirnanNew.jl")); using .MirnanNew

function test_pv_speed()
    # Generate 1000 mock words (mix of Arabic-like strings)
    println("Generating 1000 mock words...")
    letters = collect("ابتدجسحخدرزسشصضطظعغفقلمنهوي")
    words = String[]
    for _ in 1:1000
        wlen = rand(3:7)
        push!(words, join(rand(letters, wlen)))
    end
    
    # Warm up
    MirnanNew.Physics.compute_extended_phase_vector(words[1])
    
    println("Timing 1000 calls to compute_extended_phase_vector...")
    t0 = time()
    for w in words
        MirnanNew.Physics.compute_extended_phase_vector(w)
    end
    t1 = time()
    println("Completed 1000 calls in ", round(t1 - t0; digits=4), " seconds.")
    println("Average time per word: ", round((t1 - t0) / 1000 * 1000; digits=4), " ms.")
end

test_pv_speed()
