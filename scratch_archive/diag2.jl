push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using JSON

vocab_raw = JSON.parsefile("model/vocab.json")
vocab = Dict{String,Int}(k => Int(v) for (k,v) in vocab_raw)

# Simulate what the server does with split(prompt)
prompt = "العلم نور"
words = [w for w in split(prompt) if length(w) >= 1]
println("Prompt tokens: $words")
for w in words
    id = get(vocab, w, nothing)
    println("  '$w' => $id")
end

# Check beam - cids will all be nothing!
println("\nSo cids = $([ get(vocab, w, nothing) for w in words ])")
println("All nothing => no candidates from K_sem => empty output!")

# What words DO exist without punctuation?
clean_words = filter(w -> !any(c -> c in ".,،:؛!?\"'", w), collect(keys(vocab)))
println("\nSample clean words ($(length(clean_words)) total):")
println(clean_words[1:min(20,end)])
