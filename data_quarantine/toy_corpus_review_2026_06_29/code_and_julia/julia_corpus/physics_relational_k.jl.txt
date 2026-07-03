"""RelationalK — مصفوفة K علاقية بنظائر توافقية."""
module RelationalKModule
using LinearAlgebra, SparseArrays

export RelationalK, RELATION_FREQUENCY

const RELATION_FREQUENCY = Dict("syn"=>0,"sem"=>1,"verb_obj"=>2,"adj_noun"=>3,"causal"=>4,"dialogue"=>5)

mutable struct RelationalK
    base_freq::Float64; harmonic_amplitude::Float64
end
RelationalK(; base_freq=2.0, harmonic_amplitude=0.15) = RelationalK(base_freq, harmonic_amplitude)

function encode(rk::RelationalK, amplitude::Float64, relation_type::String="sem")
    amplitude <= 0 && return 0.0
    freq = get(RELATION_FREQUENCY, relation_type, 1)
    harmonic = rk.harmonic_amplitude * sin(rk.base_freq * freq)
    return amplitude * (1.0 + harmonic)
end

function extract_relation(rk::RelationalK, value::Float64, relation_type::String="sem")
    value <= 0 && return 0.0
    freq = get(RELATION_FREQUENCY, relation_type, 1)
    harmonic = sin(rk.base_freq * freq)
    return abs(value) * (0.7 + 0.3 * max(0, harmonic))
end

function build_from_syntax(rk::RelationalK, K_sem, K_syn, K_dial=nothing)
    K_sem === nothing && return nothing
    V = size(K_sem, 1)
    # Simple: multiply sem by its coefficient
    coeff_sem = 1.0 + rk.harmonic_amplitude * sin(rk.base_freq * RELATION_FREQUENCY["sem"])
    result = K_sem .* coeff_sem
    return result isa SparseMatrixCSC ? result : sparse(result)
end
end

