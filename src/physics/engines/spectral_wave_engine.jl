"""
محرك الموجات الطيفية — Spectral Wave Engine.

كل كلمة = حزمة موجية:
   ψ_w(t) = A₀·cos(ω₀·t + φ₀) × [1 + Σ cᵢ·cos(Δωᵢ·t + θᵢ)]

يدعم: compute_word_wave, wave_at, wave_spectrum (FFT),
sidebands (توافقيات جانبية من السياق)، والتوليد عبر PGN
(PhysicsGenerativeNetwork).
"""
module SpectralWave

using LinearAlgebra, FFTW
using ..Constants: PHASE_DIM, TOTAL_DIM
using ..WordPhysics: compute_word_frequency, compute_word_mass,
                     compute_word_energy, compute_word_phase_vector

export compute_word_wave, wave_at, wave_spectrum,
       compute_sidebands, wave_signature,
       PhysicsGenerativeNetwork, adaptive_context_filter

"""
    compute_word_wave(word) -> Dict

حساب المعاملات الموجية الأساسية لكلمة:
omega_0, amplitude (A₀ ∝ √(mass·energy)),
phase_0 (من أول مركبتين طوريتين)، energy، mass.
"""
function compute_word_wave(word::String)
    freq = compute_word_frequency(word)
    mass = compute_word_mass(word)
    energy = compute_word_energy(word)
    base_pv = compute_word_phase_vector(word)

    omega_0 = max(freq, 0.1)
    amplitude = sqrt(mass * energy + 0.01)
    phase_0 = if norm(base_pv[1:2]) > 1e-10
        atan(base_pv[2], base_pv[1])
    else
        0.0
    end

    return Dict(
        "omega_0" => omega_0, "amplitude" => amplitude,
        "phase_0" => phase_0, "energy" => energy, "mass" => mass,
    )
end

"""
    wave_at(word_wave, t; num_harmonics=4) -> Float64

تقييم الموجة عند الزمن t:
ψ(t) = A₀·cos(ω₀·t + φ₀) + Σ (A₀/n)·cos(n·ω₀·t + n·φ₀)
"""
function wave_at(word_wave::Dict, t::Float64; num_harmonics::Int=4)
    omega = word_wave["omega_0"]
    amp = word_wave["amplitude"]
    phase = word_wave["phase_0"]

    val = amp * cos(omega * t + phase)
    for n in 2:num_harmonics
        val += (amp / n) * cos(n * omega * t + n * phase)
    end
    return val
end

"""
    wave_spectrum(word_wave; n_samples=32) -> Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}

طيف الكلمة — FFT على عينات موجتها.
تُرجع: (frequencies, magnitudes, phases)
"""
function wave_spectrum(word_wave::Dict; n_samples::Int=32)
    t = range(0, 2π; length=n_samples)
    signal = [wave_at(word_wave, ti) for ti in t]
    fft_vals = rfft(signal)
    freqs = rfftfreq(n_samples, 2π / n_samples)
    magnitudes = abs.(fft_vals)
    phases = angle.(fft_vals)
    return freqs, magnitudes, phases
end

"""
    wave_signature(word) -> Vector{Float64}

بصمة طيفية للكلمة: تجمع magnitudes و phases في متجه واحد.
"""
function wave_signature(word::String)
    ww = compute_word_wave(word)
    _, mags, phases = wave_spectrum(ww)
    return vcat(mags, phases)
end

"""
    compute_sidebands(word_wave, context_waves; coupling_threshold=0.15) -> Vector{Dict}

حساب التوافقيات الجانبية من ترافق الكلمة مع كلمات السياق:
Δω = ω₁ ± ω₂ (تداخل جمعي وطرحى)
"""
function compute_sidebands(word_wave::Dict, context_waves::Vector{Dict};
                           coupling_threshold::Float64=0.15)
    sidebands = Dict[]
    omega_0 = word_wave["omega_0"]
    if isempty(context_waves)
        return sidebands
    end

    # حساب قوى الاقتران لكل كلمة سياق
    raw_couplings = Float64[]
    for cw in context_waves
        delta_f = abs(omega_0 - cw["omega_0"])
        coupling = cw["amplitude"] / (delta_f + 1.0)
        push!(raw_couplings, coupling)
    end

    median_coupling = median(raw_couplings)
    threshold = max(coupling_threshold, median_coupling * 0.15)

    for (i, cw) in enumerate(context_waves)
        if raw_couplings[i] < threshold
            continue
        end
        omega_ctx = cw["omega_0"]
        push!(sidebands, Dict(
            "delta_omega_sum" => omega_0 + omega_ctx,
            "delta_omega_diff" => abs(omega_0 - omega_ctx),
            "coupling_strength" => round(raw_couplings[i]; digits=4),
            "phase_diff" => round(word_wave["phase_0"] - cw["phase_0"]; digits=4),
        ))
    end
    return sidebands
end

"""
    adaptive_context_filter(spectrum, context_spectra; bw=nothing) -> Vector{Float64}

مرشح سياق تكيفي: يُمرر المكونات الطيفية القريبة من السياق.
"""
function adaptive_context_filter(spectrum::AbstractVector,
                                 context_spectra::Vector{<:AbstractVector};
                                 bw::Union{Float64,Nothing}=nothing)
    if isempty(context_spectra)
        return zeros(Float64, length(spectrum))
    end

    # حساب عرض النطاق تلقائياً
    all_deltas = Float64[]
    for cs in context_spectra
        d = min(length(spectrum), length(cs))
        push!(all_deltas, abs.(spectrum[1:d] .- cs[1:d])...)
    end
    bandwidth = bw === nothing ? median(all_deltas) : bw

    filter_mask = zeros(Float64, length(spectrum))
    for cs in context_spectra
        d = min(length(spectrum), length(cs))
        delta = abs.(spectrum[1:d] .- cs[1:d])
        filter_mask[1:d] .+= 1.0 ./ (1.0 .+ (delta ./ bandwidth) .^ 2)
    end

    return filter_mask ./ max(maximum(filter_mask), 1e-10)
end

# ═══ PGN — Physics Generative Network ═══
"""
    PhysicsGenerativeNetwork

شبكة توليد فيزيائية — تبحث عن مسار الطاقة القصوى
في رسم بياني للرنين الطيفي.

الحقول:
- `K_spectral`: مصفوفة اقتران طيفي
- `vocab`: قاموس المفردات
- `cache`: تخزين مؤقت
"""
mutable struct PhysicsGenerativeNetwork
    K_spectral::Union{AbstractMatrix,Nothing}
    vocab::Dict{String,Int}
    id2word::Dict{Int,String}
    cache::Dict{String,Dict}

    function PhysicsGenerativeNetwork(vocab::Dict{String,Int};
                                      K_spectral=nothing)
        id2w = Dict{Int,String}(v => k for (k, v) in vocab)
        return new(K_spectral, vocab, id2w, Dict{String,Dict}())
    end
end

"""
    find_path(pgn, start_words; max_steps=10) -> Tuple{Vector{String},Float64}

البحث عن مسار الطاقة القصوى في الرسم البياني للرنين الطيفي.
يبدأ من start_words ويختار في كل خطوة الكلمة ذات أعلى طاقة رنين.
"""
function find_path(pgn::PhysicsGenerativeNetwork, start_words::Vector{String};
                   max_steps::Int=10)
    if isempty(start_words)
        return String[], 0.0
    end

    path = copy(start_words)
    total_energy = 0.0

    for step in 1:max_steps
        current = path[end]
        cid = get(pgn.vocab, current, nothing)
        if cid === nothing || pgn.K_spectral === nothing
            break
        end
        if cid > size(pgn.K_spectral, 1)
            break
        end

        # قراءة صف K_spectral الحالي
        row_data = if pgn.K_spectral isa AbstractSparseMatrix
            nz = rowvals(pgn.K_spectral)[nzrange(pgn.K_spectral, cid)]
            vals = nonzeros(pgn.K_spectral)[nzrange(pgn.K_spectral, cid)]
            collect(zip(nz, vals))
        else
            [(j, pgn.K_spectral[cid, j]) for j in 1:size(pgn.K_spectral, 2) if pgn.K_spectral[cid, j] > 1e-6]
        end

        # اختيار أفضل مرشح (غير موجود مسبقاً)
        best_word = ""
        best_score = -Inf
        for (tid, val) in row_data
            nxt = get(pgn.id2word, tid, nothing)
            if nxt !== nothing && nxt ∉ path && val > best_score
                best_score = val
                best_word = nxt
            end
        end

        if best_word == ""
            break
        end

        push!(path, best_word)
        total_energy += best_score
    end

    return path, total_energy
end

end # module SpectralWave
