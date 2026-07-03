"""
    ScenePurposeStrategy

Experimental bridge strategy between semantic imagination and purpose frames.
It is inserted into `generate!` only when
`MIRNAN_ENABLE_SCENE_PURPOSE_STRATEGY=1`.
"""

function try_generate(::ScenePurposeStrategy, gen::MirnanGenerator, prompt::String,
                      prompt_tokens::Vector{String}, mode::String, cereb_obs, cereb_policy,
                      response_plan, active_paras)
    scene_mem = _LEARNED_SEMANTIC_SCENE_MEMORY[]
    (scene_mem === nothing || !has_semantic_scenes(scene_mem)) && (scene_mem = gen.semantic_scenes)
    has_semantic_scenes(scene_mem) || return nothing

    istinbat_mem = _LEARNED_ISTINBAT_MEMORY[]
    istinbat_mem === nothing && return nothing

    ans = scene_purpose_answer(scene_mem, gen.hisban, istinbat_mem, prompt)
    isempty(strip(ans)) && return nothing

    return _finish_generation!(gen, prompt, prompt_tokens, ans,
                               cereb_obs, cereb_policy;
                               sanitize_output=false,
                               apply_templates=false)
end

