# PRNN & Advanced Group
# Part of Phase 4 — Modularity Refactoring

include("../engines/prnn_core.jl")
include("../engines/prnn_learner.jl")
include("../engines/prnn_generator.jl")
include("../engines/path_integral_reasoner.jl")
include("../engines/beam_reasoner.jl")
include("../engines/advanced_engines.jl")

using .PRNNCore: simulate_stuart_landau!, bind_phase, unbind_phase,
                 complete_pattern!, prnn_noise_sample
using .PRNNGenerator: prnn_generate_standalone, PRNNSession,
                      prnn_complete_pattern_standalone, prnn_noise_sample_standalone
using .BeamReasonerModule: BeamReasoner, ReasonStep, ReasonChain,
                            reason!, reason_steps!, explain_reasoning

export complete_pattern!, prnn_noise_sample,
       prnn_generate_standalone, prnn_complete_pattern_standalone, prnn_noise_sample_standalone
export BeamReasoner, ReasonStep, ReasonChain, reason!, reason_steps!, explain_reasoning
