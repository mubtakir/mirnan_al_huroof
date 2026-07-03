# Fusion & Morphology Group
# Part of Phase 4 — Modularity Refactoring

include("../engines/word_fusion.jl")
include("../engines/semantic_arithmetic.jl")
include("../engines/lexical_oracle.jl")
include("../engines/weight_resonance.jl")
include("../engines/syntax_field.jl")
include("../engines/morpho_phasic.jl")
include("../engines/morpho_twistor.jl")
include("../engines/english_morpheme.jl")

using .MorphoTwistor: TwistorEngine, TwistorOperator, MorphPattern,
                      compute_twistor, apply_twistor, twistor_similarity,
                      learn_patterns!, predict_derivation, find_morphological_pairs,
                      score_twistor_candidate
using .LexicalOracleModule: WordPhysicsProfile, LexicalNeighbor, CoinedWord,
                             word_physics_profile, phonosemantic_quality,
                             nearest_phase_words, analyze_unknown_word,
                             coin_word_for_concept

export TwistorEngine, TwistorOperator, MorphPattern, compute_twistor, apply_twistor,
       twistor_similarity, learn_patterns!, predict_derivation, find_morphological_pairs,
       score_twistor_candidate
export WordPhysicsProfile, LexicalNeighbor, CoinedWord,
       word_physics_profile, phonosemantic_quality,
       nearest_phase_words, analyze_unknown_word,
       coin_word_for_concept
