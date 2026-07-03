"""ObjectivityField — بُعد موضوعي مستقل 1D لكل حرف."""
module ObjectivityField

using JSON

const _DATA_PATH = joinpath(@__DIR__, "..", "..", "..", "data", "letter_objectivity.json")
const _DATA_REF = Ref{Dict{String,Float64}}()

function _ensure_loaded!()
    if !isassigned(_DATA_REF)
        raw = JSON.parsefile(_DATA_PATH)
        result = Dict{String,Float64}()
        for (ch, v) in raw
            result[ch] = Float64(v)
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

function word_objectivity(word::AbstractString)
    data = _ensure_loaded!()
    letters = _letters_of(word)
    isempty(letters) && return 0.5
    s = 0.0; n = 0
    for ch in letters
        c = string(ch)
        if haskey(data, c)
            s += data[c]
            n += 1
        end
    end
    return n > 0 ? s / n : 0.5
end

end
