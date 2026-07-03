"""AnatomicalField — فضاء تشريحي-ذوقي 7D مستقل لكل كلمة من حروفها."""
module AnatomicalField

using JSON

const _DATA_PATH = joinpath(@__DIR__, "..", "..", "..", "data", "letter_anatomical.json")
const _DATA_REF = Ref{Dict{String,Vector{Float64}}}()

function _ensure_loaded!()
    if !isassigned(_DATA_REF)
        raw = JSON.parsefile(_DATA_PATH)
        result = Dict{String,Vector{Float64}}()
        for (ch, vec) in raw
            result[ch] = Float64.(vec)
        end
        _DATA_REF[] = result
    end
    return _DATA_REF[]
end

function _strip_al(word::AbstractString)
    w = String(word)
    while startswith(w, "ال") && length(w) > 4
        w = w[nextind(w, firstindex(w), 2):end]
    end
    return w
end

function _letters_of(word::AbstractString)
    w = _strip_al(String(word))
    letters = Char[]
    for c in w
        if '\u0621' <= c <= '\u064A' || c == '\u0670'
            push!(letters, c)
        end
    end
    return letters
end

function word_anatomical_vector(word::AbstractString)
    data = _ensure_loaded!()
    letters = _letters_of(word)
    isempty(letters) && return zeros(Float64, 7)
    sum_vec = zeros(Float64, 7)
    count = 0
    for ch in letters
        s = string(ch)
        if haskey(data, s)
            v = data[s]
            for i in 1:7
                sum_vec[i] += v[i]
            end
            count += 1
        end
    end
    count > 0 || return zeros(Float64, 7)
    return [sum_vec[i]/count for i in 1:7]
end

function anatomical_similarity(w1::AbstractString, w2::AbstractString)
    a1 = word_anatomical_vector(w1)
    a2 = word_anatomical_vector(w2)
    n1 = sqrt(sum(a1[i]^2 for i in 1:7))
    n2 = sqrt(sum(a2[i]^2 for i in 1:7))
    (n1 < 1e-10 || n2 < 1e-10) && return 0.0
    return sum(a1[i]*a2[i] for i in 1:7) / (n1 * n2)
end

end
