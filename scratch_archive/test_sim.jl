using Pkg; Pkg.activate(".")
using JSON, LinearAlgebra, Mirnan

const MODEL_DIR = "model"
vocab = Dict{String,Int}(k => Int(v) for (k,v) in JSON.parsefile(joinpath(MODEL_DIR, "vocab.json")))
id2word = Dict{Int,String}(v => k for (k,v) in vocab)
V = length(vocab)

println("Loaded vocabulary of size: ", V)

# Precompute phase vectors
pvs = Vector{Vector{Float32}}(undef, V)
for i in 1:V
    pvs[i] = compute_extended_phase_vector(id2word[i])
end

const _AR_DIACRITICS = Set(['َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ً', 'ٌ', 'ٍ'])
function _strip_diacritics(word::String)
    return String(filter(c -> c ∉ _AR_DIACRITICS, word))
end

function find_duplicates_grouped(pvs, id2word, V)
    # Group word IDs by bare form
    groups = Dict{String, Vector{Int}}()
    for i in 1:V
        w = id2word[i]
        bare = _strip_diacritics(w)
        if !haskey(groups, bare)
            groups[bare] = Int[]
        end
        push!(groups[bare], i)
    end
    
    duplicates = Tuple{String, String, Float64}[]
    checked_pairs = 0
    
    for (bare, ids) in groups
        n = length(ids)
        n < 2 && continue
        for idx_i in 1:n
            i = ids[idx_i]
            w1 = id2word[i]
            pv1 = pvs[i]
            for idx_j in (idx_i + 1):n
                j = ids[idx_j]
                w2 = id2word[j]
                pv2 = pvs[j]
                
                checked_pairs += 1
                n1 = norm(pv1)
                n2 = norm(pv2)
                if n1 > 1e-10 && n2 > 1e-10
                    sim = dot(pv1, pv2) / (n1 * n2)
                    if sim >= 0.97
                        push!(duplicates, (w1, w2, sim))
                    end
                end
            end
        end
    end
    
    println("Checked pairs: ", checked_pairs)
    return duplicates
end

@time dups = find_duplicates_grouped(pvs, id2word, V)
println("Found ", length(dups), " duplicates.")
for i in 1:min(20, length(dups))
    println("  ", dups[i][1], " <-> ", dups[i][2], " (sim: ", dups[i][3], ")")
end
