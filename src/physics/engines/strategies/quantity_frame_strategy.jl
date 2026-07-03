"""
    QuantityFrameStrategy

Experimental QuantityFrame strategy. It is inserted into `generate!` only when
`MIRNAN_ENABLE_QUANTITY_FRAME_STRATEGY=1`, and it reads only the explicit
`_LEARNED_QUANTITY_MEMORY` reference.
"""

function try_generate(::QuantityFrameStrategy, gen::MirnanGenerator, prompt::String,
                      prompt_tokens::Vector{String}, mode::String, cereb_obs, cereb_policy,
                      response_plan, active_paras)
    mem = _LEARNED_QUANTITY_MEMORY[]
    mem === nothing && return nothing

    ans = quantity_answer(mem, prompt)
    isempty(strip(ans)) && return nothing

    return _finish_generation!(gen, prompt, prompt_tokens, ans,
                               cereb_obs, cereb_policy;
                               sanitize_output=false,
                               apply_templates=false)
end
