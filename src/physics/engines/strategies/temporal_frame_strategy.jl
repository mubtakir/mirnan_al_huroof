"""
    TemporalFrameStrategy

RelationFrame strategy for temporal answers. It is inserted into `generate!` by
default; set `MIRNAN_ENABLE_TEMPORAL_FRAME_STRATEGY=0` to disable it.
"""

function _temporal_strategy_question(prompt::AbstractString)
    s = lowercase(strip(String(prompt)))
    return startswith(s, "\u0645\u062a\u0649") || startswith(s, "when")
end

function _temporal_strategy_unknown(prompt::AbstractString)
    s = lowercase(strip(String(prompt)))
    startswith(s, "when") && return "I do not find a specific time in memory."
    return "\u0644\u0627 \u0623\u062c\u062f \u0632\u0645\u0646\u0627\u064b \u0645\u062d\u062f\u062f\u0627\u064b \u0641\u064a \u0627\u0644\u0630\u0627\u0643\u0631\u0629."
end

function try_generate(::TemporalFrameStrategy, gen::MirnanGenerator, prompt::String,
                      prompt_tokens::Vector{String}, mode::String, cereb_obs, cereb_policy,
                      response_plan, active_paras)
    mem = _LEARNED_ISTINBAT_MEMORY[]
    if mem === nothing
        _temporal_strategy_question(prompt) || return nothing
        ans = _temporal_strategy_unknown(prompt)
        return _finish_generation!(gen, prompt, prompt_tokens, ans,
                                   cereb_obs, cereb_policy;
                                   sanitize_output=false,
                                   apply_templates=false)
    end

    ans = temporal_answer(mem, prompt)
    if isempty(strip(ans))
        _temporal_strategy_question(prompt) || return nothing
        ans = _temporal_strategy_unknown(prompt)
    end

    return _finish_generation!(gen, prompt, prompt_tokens, ans,
                               cereb_obs, cereb_policy;
                               sanitize_output=false,
                               apply_templates=false)
end
