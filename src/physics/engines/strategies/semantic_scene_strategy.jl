"""
    SemanticSceneStrategy

Experimental semantic-imagination strategy. It is only inserted into
`generate!` when `MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY=1`.
"""

function try_generate(::SemanticSceneStrategy, gen::MirnanGenerator, prompt::String,
                      prompt_tokens::Vector{String}, mode::String, cereb_obs, cereb_policy,
                      response_plan, active_paras)
    mem = _LEARNED_SEMANTIC_SCENE_MEMORY[]
    (mem === nothing || !has_semantic_scenes(mem)) && (mem = gen.semantic_scenes)
    has_semantic_scenes(mem) || return nothing

    ans = semantic_scene_answer(mem, gen.hisban, prompt)
    isempty(strip(ans)) && return nothing

    return _finish_generation!(gen, prompt, prompt_tokens, ans,
                               cereb_obs, cereb_policy;
                               sanitize_output=false,
                               apply_templates=false)
end
