function _generate_code(gen::MirnanGenerator, prompt::String)
    try
        result = generate_code_from_pattern(gen.code_patterns, prompt)
        if !isempty(strip(result))
            ok, _ = compile_check(result)
            if ok
                return result
            end
        end
    catch e
        @debug "al_code pattern generation failed" exception=(typeof(e), catch_backtrace())
    end
    try
        result = generate_code_physics(gen.code_phase, prompt; lang="python")
        if !isempty(result) && result != "# no concepts"
            ok, _ = compile_check(result)
            if ok
                return result
            end
        end
    catch e
        @debug "Code phase generation failed" exception=(typeof(e), catch_backtrace())
    end
    try
        result = generate_code(gen.code_engine, prompt; max_tokens=30)
        if !isempty(result)
            return result
        end
    catch e
        @debug "Code engine generation failed" exception=(typeof(e), catch_backtrace())
    end
    return ""
end

function _strong_code_request(prompt::AbstractString)
    p = lowercase(String(prompt))
    code_words = Set(String[
        "كود", "الكود", "برنامج",
        "برمجة", "دالة", "دالتي",
        "حلقة", "متغير", "متغيرا",
        "متغيراً", "بايثون",
        "جوليا", "python", "julia", "function", "loop",
    ])
    words = Set(String[
        strip(w, [' ', '\t', '\n', '\r', '.', ',', '،', ';', '؛', ':', '?', '؟', '!'])
        for w in split(p)
    ])
    return occursin("```", p) ||
           occursin("python", p) || occursin("julia", p) ||
           !isempty(intersect(words, code_words)) ||
           occursin("برمج", p) ||
           (occursin("اكتب", p) && (occursin("شرط", p) || occursin("أكبر", p) || occursin("اكبر", p))) ||
           occursin("function", p) || occursin("def ", p) || occursin("loop", p)
end

function try_generate(::CodeStrategy, gen::MirnanGenerator, prompt::String, prompt_tokens::Vector{String}, mode::String, cereb_obs, cereb_policy, response_plan, active_paras)
    is_code_req = (mode == "code") || 
                  (mode != "math" && _strong_code_request(prompt)) ||
                  (mode == "auto" && (detect_mode(prompt) == "code" || cereb_policy.mode == "code"))
                  
    if is_code_req
        result = _generate_code(gen, prompt)
        return _finish_generation!(gen, prompt, prompt_tokens, result,
                                   cereb_obs, cereb_policy; observe_ram=false,
                                   sanitize_output=false)
    end
    return nothing
end
