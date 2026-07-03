using JSON
vocab = JSON.parsefile("model/vocab.json")
id2word = Dict(v=>k for (k,v) in vocab)
for wid in [106842, 108264]
    w = get(id2word, wid, "???")
    bytes_str = join(["\\x$(string(b, base=16, pad=2))" for b in Vector{UInt8}(w)], "")
    println("ID $wid: bytes=($(length(w))B) $bytes_str")
end
