"""
    RelationFrameStrategy — استراتيجية توليد مستقلة للعلاقات الغائية

تستخرج جواباً من RelationFrame من نوع purpose عبر `purpose_answer`.
لا تُدرج في ترتيب generate! العام بعد — تُختبر مباشرة.
"""

function try_generate(::RelationFrameStrategy, gen::MirnanGenerator, prompt::String,
                      prompt_tokens::Vector{String}, mode::String, cereb_obs, cereb_policy,
                      response_plan, active_paras)
    mem = _LEARNED_ISTINBAT_MEMORY[]
    mem === nothing && return nothing

    ans = purpose_answer(mem, prompt)
    isempty(strip(ans)) && return nothing

    return _finish_generation!(gen, prompt, prompt_tokens, ans,
                               cereb_obs, cereb_policy;
                               sanitize_output=false,
                               apply_templates=false)
end
