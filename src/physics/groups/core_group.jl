# Core Group
# Part of Phase 4 — Modularity Refactoring

include("../core/config.jl")
include("../core/constants.jl")
include("../core/letter_db_original.jl")
include("../core/letter_db.jl")
include("../core/word_physics.jl")
include("../core/clifford_math.jl")
include("../core/wave_field.jl")

using .MirnanConfig: MirnanCfg, load_config, get_val, get_section
using .WordPhysics: compute_word_phase_vector, compute_word_mass,
                    compute_word_frequency, compute_word_frequency_with_irab,
                    phase_similarity, compute_extended_phase_vector,
                    compute_compact_phase_vector, compact_phase_similarity,
                    get_irab_omega_bias, extract_irab,
                    IRAB_OMEGA_BIAS, IRAB_MAP
using .WaveField: WaveContribution, wave_superposition, born_rule, interference_strength

export MirnanCfg, load_config, get_val, get_section
export compute_word_phase_vector, compute_word_mass, compute_word_frequency,
       compute_word_frequency_with_irab, phase_similarity, compute_extended_phase_vector,
       compute_compact_phase_vector, compact_phase_similarity,
       get_irab_omega_bias, extract_irab
export WaveContribution, wave_superposition, born_rule, interference_strength
