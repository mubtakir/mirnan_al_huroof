"""PolarityField — فضاء قطبية مستقل 3D لكل كلمة من حروفها."""
module PolarityField

using JSON

const _DATA_PATH = joinpath(@__DIR__, "..", "..", "..", "data", "letter_polarity.json")
const _DATA_REF = Ref{Dict{String,Vector{Float64}}}()

function _ensure_loaded!()
    if !isassigned(_DATA_REF)
        raw = JSON.parsefile(_DATA_PATH)
        centered = raw["centered"]
        result = Dict{String,Vector{Float64}}()
        for (ch, vec) in centered
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

function word_polarity_vector(word::AbstractString)
    data = _ensure_loaded!()
    letters = _letters_of(word)
    isempty(letters) && return [0.0, 0.0, 0.0]
    sum_vec = [0.0, 0.0, 0.0]
    count = 0
    for ch in letters
        s = string(ch)
        if haskey(data, s)
            v = data[s]
            sum_vec[1] += v[1]; sum_vec[2] += v[2]; sum_vec[3] += v[3]
            count += 1
        end
    end
    count > 0 || return [0.0, 0.0, 0.0]
    return [sum_vec[1]/count, sum_vec[2]/count, sum_vec[3]/count]
end

function polarity_harmony(w1::AbstractString, w2::AbstractString)
    p1 = word_polarity_vector(w1)
    p2 = word_polarity_vector(w2)
    n1 = sqrt(p1[1]^2 + p1[2]^2 + p1[3]^2)
    n2 = sqrt(p2[1]^2 + p2[2]^2 + p2[3]^2)
    (n1 < 1e-10 || n2 < 1e-10) && return 0.0
    return (p1[1]*p2[1] + p1[2]*p2[2] + p1[3]*p2[3]) / (n1 * n2)
end

function polarity_net(word::AbstractString)
    p = word_polarity_vector(word)
    return p[1] - p[2]
end

end
