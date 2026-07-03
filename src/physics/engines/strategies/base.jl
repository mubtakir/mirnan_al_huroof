abstract type GenerationStrategy end

struct RootLexicalStrategy <: GenerationStrategy end
struct LexicalOracleStrategy <: GenerationStrategy end
struct SenseSuperpositionStrategy <: GenerationStrategy end
struct MathStrategy <: GenerationStrategy end
struct CodeStrategy <: GenerationStrategy end
struct TadbirStrategy <: GenerationStrategy end
struct DialogueStrategy <: GenerationStrategy end
struct DefinitionStrategy <: GenerationStrategy end
struct AqlStrategy <: GenerationStrategy end
struct RelationStrategy <: GenerationStrategy end
struct RelationFrameStrategy <: GenerationStrategy end
struct ConditionalFrameStrategy <: GenerationStrategy end
struct TemporalFrameStrategy <: GenerationStrategy end
struct SpatialFrameStrategy <: GenerationStrategy end
struct StateFrameStrategy <: GenerationStrategy end
struct QuantityFrameStrategy <: GenerationStrategy end
struct SemanticSceneStrategy <: GenerationStrategy end
struct ScenePurposeStrategy <: GenerationStrategy end
struct ResonantStrategy <: GenerationStrategy end



function try_generate(strategy::GenerationStrategy, gen::MirnanGenerator, prompt::String, prompt_tokens::Vector{String}, mode::String, cereb_obs, cereb_policy, response_plan, active_paras, max_words::Int)
    return try_generate(strategy, gen, prompt, prompt_tokens, mode, cereb_obs, cereb_policy, response_plan, active_paras)
end
