"""
HierarchicalMemory — ذاكرة هرمية طيفية (4 مستويات: كلمة → عبارة → فقرة → محادثة).
"""
module HierarchicalMemoryModule
using LinearAlgebra
using ..Constants: TOTAL_DIM

export SpectralLevel, HierarchicalMemory

mutable struct SpectralLevel
    decay::Float64
    context::Vector{Float64}
    count::Int
    capacity::Int
end
SpectralLevel(decay=0.1, capacity=100) = SpectralLevel(decay, Float64[], 0, capacity)

mutable struct HierarchicalMemory
    word_level::SpectralLevel       # ~100 كلمة
    phrase_level::SpectralLevel     # ~40 عبارة
    paragraph_level::SpectralLevel  # ~15 فقرة
    conversation_level::SpectralLevel # ~5 محادثات
end

function HierarchicalMemory(; wc=100, pc=40, gc=15, cc=5)
    HierarchicalMemory(
        SpectralLevel(0.5, wc), SpectralLevel(0.3, pc),
        SpectralLevel(0.15, gc), SpectralLevel(0.05, cc),
    )
end

function add_word!(hm::HierarchicalMemory, pv::AbstractVector)
    pv_f = Float64.(pv)
    l = hm.word_level
    if l.count == 0; l.context = copy(pv_f); else; l.context = (1-l.decay).*l.context .+ l.decay.*pv_f; end
    l.count += 1
    if l.count >= hm.word_level.capacity
        # Promote to phrase
        pl = hm.phrase_level
        if pl.count == 0; pl.context = copy(l.context); else; pl.context = (1-pl.decay).*pl.context .+ pl.decay.*l.context; end
        pl.count += 1
        l.context = Float64[]; l.count = 0
        if pl.count >= hm.phrase_level.capacity
            # Promote to paragraph
            gl = hm.paragraph_level
            if gl.count == 0; gl.context = copy(pl.context); else; gl.context = (1-gl.decay).*gl.context .+ gl.decay.*pl.context; end
            gl.count += 1
            pl.context = Float64[]; pl.count = 0
            if gl.count >= hm.paragraph_level.capacity
                # Promote to conversation
                cl = hm.conversation_level
                if cl.count == 0; cl.context = copy(gl.context); else; cl.context = (1-cl.decay).*cl.context .+ cl.decay.*gl.context; end
                cl.count += 1
                gl.context = Float64[]; gl.count = 0
            end
        end
    end
end

function get_hierarchical_context(hm::HierarchicalMemory)
    levels = [hm.word_level, hm.phrase_level, hm.paragraph_level, hm.conversation_level]
    dim = 0
    for l in levels
        if l.count > 0 && !isempty(l.context)
            dim = length(l.context)
            break
        end
    end
    dim == 0 && return zeros(TOTAL_DIM)
    total = zeros(Float64, dim)
    weight = 1.0
    for l in levels
        if l.count > 0 && !isempty(l.context) && length(l.context) == dim
            total .+= weight .* l.context
        end
        weight *= 0.5
    end
    nrm = norm(total)
    return nrm > 1e-10 ? total ./ nrm : total
end
end

