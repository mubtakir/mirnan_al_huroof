"""
    ConditionalFrameStrategy

Experimental RelationFrame strategy for conditional answers. It is inserted into
`generate!` only when `MIRNAN_ENABLE_CONDITIONAL_FRAME_STRATEGY=1`.
"""

function try_generate(::ConditionalFrameStrategy, gen::MirnanGenerator, prompt::String,
                      prompt_tokens::Vector{String}, mode::String, cereb_obs, cereb_policy,
                      response_plan, active_paras)
    mem = _LEARNED_ISTINBAT_MEMORY[]
    mem === nothing && return nothing

    ans = conditional_answer(mem, prompt)
    isempty(strip(ans)) && return nothing

    return _finish_generation!(gen, prompt, prompt_tokens, ans,
                               cereb_obs, cereb_policy;
                               sanitize_output=false,
                               apply_templates=false)
end
