"""WordSpectrum — طيف الكلمة (FFT على حروفها)."""
module WordSpectrum
using LinearAlgebra, FFTW, Statistics
using ..LetterDB: LetterDatabase, get_vector
using ..WordPhysics: _parse_word_harakat

export compute_word_spectrum, spectral_resonance, spectral_density

const _ws_db = Ref{Union{LetterDatabase,Nothing}}(nothing)
function _get_ws_db()
    if _ws_db[] === nothing; _ws_db[] = LetterDatabase(); end
    return _ws_db[]
end

function compute_word_spectrum(word::String)
    db = _get_ws_db()
    parsed = _parse_word_harakat(word)
    isempty(parsed) && return nothing
    letters = [string(l) for (l, _) in parsed]
    if length(letters) < 3; letters = vcat(letters, fill(letters[end], 3-length(letters))); end
    n = length(letters)
    pvs = [Float64.(get_vector(db, l)[1:22]) for l in letters]
    fft_vals = rfft(reduce(hcat, pvs)', 1)
    magnitudes = abs.(fft_vals[:,1])
    phases = angle.(fft_vals[:,1])
    freqs = rfftfreq(n)
    n_comp = min(6, length(freqs))
    top_idx = sortperm(magnitudes; rev=true)[1:n_comp]
    spec_vec = vcat(freqs[top_idx], magnitudes[top_idx], phases[top_idx])
    return Dict("frequencies"=>freqs[top_idx], "amplitudes"=>magnitudes[top_idx],
                "phases"=>phases[top_idx], "spectral_vector"=>spec_vec)
end

function spectral_resonance(word1::String, word2::String)
    s1 = compute_word_spectrum(word1); s1 === nothing && return 0.0
    s2 = compute_word_spectrum(word2); s2 === nothing && return 0.0
    v1, v2 = s1["spectral_vector"], s2["spectral_vector"]
    dim = min(length(v1), length(v2)); dim == 0 && return 0.0
    cs = dot(v1[1:dim], v2[1:dim]) / (norm(v1[1:dim])*norm(v2[1:dim]) + 1e-10)
    return clamp((cs+1.0)/2.0, 0.0, 1.0)
end

function spectral_density(words::Vector{String})
    length(words) < 2 && return 0.0
    spectra = [s["spectral_vector"] for w in words if (s=compute_word_spectrum(w); s !== nothing)]
    length(spectra) < 2 && return 0.0
    sims = Float64[]
    for i in 1:length(spectra)
        for j in (i+1):length(spectra)
            d = min(length(spectra[i]), length(spectra[j])); d==0 && continue
            cs = dot(spectra[i][1:d], spectra[j][1:d]) / (norm(spectra[i][1:d])*norm(spectra[j][1:d])+1e-10)
            push!(sims, cs)
        end
    end
    return 1.0 - mean(sims)
end
end
