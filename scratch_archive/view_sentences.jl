using JSON

const MODEL_DIR = joinpath(@__DIR__, "..", "model")
vocab_file = joinpath(MODEL_DIR, "vocab.json")
vocab = Dict{String,Int}(k => Int(v) for (k,v) in JSON.parsefile(vocab_file))
id2word = Dict{Int,String}(v => k for (k,v) in vocab)

cs_path = joinpath(MODEL_DIR, "corpus_sentences.dat")
open(cs_path, "r") do io
    n = read(io, Int32)
    println("Total sentences in corpus: ", n)
    for i in 1:15
        len = read(io, Int32)
        s = read!(io, Vector{Int32}(undef, len))
        println("$i: ", join([get(id2word, Int(id), "") for id in s], " "))
    end
end
