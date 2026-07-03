function _is_tadbir_prompt(prompt::AbstractString)
    p = lowercase(String(prompt))
    return occursin("plan", p) || occursin("steps", p) ||
           occursin("workflow", p) || occursin("roadmap", p) ||
           occursin("خطة", p) ||
           occursin("خطوات", p) ||
           occursin("كيف أنفذ", p) ||
           occursin("كيف انفذ", p)
end

function _generate_tadbir(gen::MirnanGenerator, prompt::String)
    has_tadbir_patterns(gen.tadbir) || return ""
    steps = render_tadbir_plan(gen.tadbir, prompt)
    isempty(steps) && return ""
    lines = String[]
    for (i, step) in enumerate(steps)
        push!(lines, string(i, ". ", step))
    end
    return join(lines, "\n")
end

function try_generate(::TadbirStrategy, gen::MirnanGenerator, prompt::String, prompt_tokens::Vector{String}, mode::String, cereb_obs, cereb_policy, response_plan, active_paras)
    if _is_tadbir_prompt(prompt)
        tadbir_answer = _generate_tadbir(gen, prompt)
        if !isempty(strip(tadbir_answer))
            return _finish_generation!(gen, prompt, prompt_tokens, tadbir_answer,
                                       cereb_obs, cereb_policy;
                                       sanitize_output=false)
        end
    end
    return nothing
end
