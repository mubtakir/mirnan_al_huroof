"""
    SpatialFrameStrategy

RelationFrame strategy for spatial answers. It is inserted into `generate!` by
default; set `MIRNAN_ENABLE_SPATIAL_FRAME_STRATEGY=0` to disable it.
"""

function _spatial_strategy_question(prompt::AbstractString)
    s = lowercase(strip(String(prompt)))
    return startswith(s, "\u0623\u064a\u0646") || startswith(s, "\u0627\u064a\u0646") || startswith(s, "where")
end

function _spatial_strategy_unknown(prompt::AbstractString)
    s = lowercase(strip(String(prompt)))
    startswith(s, "where") && return "I do not find a specific place in memory."
    return "\u0644\u0627 \u0623\u062c\u062f \u0645\u0643\u0627\u0646\u0627\u064b \u0645\u062d\u062f\u062f\u0627\u064b \u0641\u064a \u0627\u0644\u0630\u0627\u0643\u0631\u0629."
end

function try_generate(::SpatialFrameStrategy, gen::MirnanGenerator, prompt::String,
                      prompt_tokens::Vector{String}, mode::String, cereb_obs, cereb_policy,
                      response_plan, active_paras)
    mem = _LEARNED_ISTINBAT_MEMORY[]
    if mem === nothing
        _spatial_strategy_question(prompt) || return nothing
        ans = _spatial_strategy_unknown(prompt)
        return _finish_generation!(gen, prompt, prompt_tokens, ans,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    ans = spatial_answer(mem, prompt)
    if isempty(strip(ans))
        _spatial_strategy_question(prompt) || return nothing
        ans = _spatial_strategy_unknown(prompt)
    end

    return _finish_generation!(gen, prompt, prompt_tokens, ans,
                               cereb_obs, cereb_policy;
                               sanitize_output=false,
                               apply_templates=false)
end
