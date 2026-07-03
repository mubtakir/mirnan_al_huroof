"""
PhaseEvolution — تطور المتجهات الطورية عبر المزامنة الهيبيانية.
Learns phase shifts from co-occurrence patterns.
"""
module PhaseEvolutionModule

using LinearAlgebra

export PhaseEvolution, observe_cooccurrence!, evolve!

mutable struct PhaseEvolution
    lr_cooc::Float64
    lr_contrast::Float64
    lr_spectral::Float64
    min_cooc::Int
    max_shift::Float64
    decay::Float64
    shifts::Dict{Int,Vector{Float64}}
    cooc_counts::Dict{Int,Dict{Int,Int}}
    epoch::Int
    dim::Int
end

function PhaseEvolution(; lr_cooc=0.03, lr_contrast=0.015, lr_spectral=0.02,
                          min_cooc=2, max_shift=0.25, decay=0.99, dim=27)
    return PhaseEvolution(lr_cooc, lr_contrast, lr_spectral, min_cooc, max_shift, decay,
                          Dict{Int,Vector{Float64}}(), Dict{Int,Dict{Int,Int}}(), 0, dim)
end

function observe_cooccurrence!(pe::PhaseEvolution, w1_pv, w2_pv;
                               w1_id=nothing, w2_id=nothing, distance=1)
    decay_factor = 1.0 / (1.0 + distance * 0.3)

    d = min(length(w1_pv), length(w2_pv), pe.dim)
    v1 = Float64.(w1_pv[1:d])
    v2 = Float64.(w2_pv[1:d])
    n1 = norm(v1); n2 = norm(v2)
    sim = (n1 > 1e-10 && n2 > 1e-10) ? max(0.0, dot(v1, v2) / (n1 * n2)) : 0.0

    shift_mag = pe.lr_cooc * (1.0 - sim) * decay_factor
    shift = shift_mag .* (v2 .- v1)
    shift_nrm = norm(shift)
    if shift_nrm > pe.max_shift
        shift .*= pe.max_shift / shift_nrm
    end
    if w1_id !== nothing
        if !haskey(pe.shifts, w1_id)
            pe.shifts[w1_id] = zeros(Float64, pe.dim)
        end
        pe.shifts[w1_id][1:d] .+= shift[1:d]
    end
end

function evolve!(pe::PhaseEvolution, all_pv::AbstractMatrix)
    isempty(pe.shifts) && return all_pv, Dict("words_shifted" => 0)
    new_pv = Float64.(all_pv)
    count = 0
    for (wid, shift) in pe.shifts
        wid > size(new_pv, 1) && continue
        conf = 0.3
        ns = Float64(norm(shift))
        if ns > 1e-12
            d = min(length(shift), size(new_pv, 2))
            new_pv[wid, 1:d] .+= shift[1:d] .* conf
            nrm2 = norm(view(new_pv, wid, :))
            if nrm2 > 1e-10; new_pv[wid, :] ./= nrm2; end
            count += 1
        end
    end
    empty!(pe.shifts); pe.epoch += 1
    return new_pv, Dict("words_shifted" => count)
end

end # module PhaseEvolutionModule
