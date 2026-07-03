"""PhaseVectorCache — مخبأ موحد للمتجهات الطورية."""
module PhaseCache
using ..WordPhysics: compute_extended_phase_vector
export PhaseVectorCache

mutable struct PhaseVectorCache
    cache::Dict{String,Vector{Float64}}
end
PhaseVectorCache() = PhaseVectorCache(Dict())
get(pvc::PhaseVectorCache, word::String) = get!(pvc.cache, word) do; Float64.(compute_extended_phase_vector(word)); end
clear!(pvc::PhaseVectorCache) = empty!(pvc.cache)
Base.in(word, pvc::PhaseVectorCache) = haskey(pvc.cache, word)
Base.length(pvc::PhaseVectorCache) = length(pvc.cache)
end
