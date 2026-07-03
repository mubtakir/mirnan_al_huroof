function _strong_math_request(prompt::AbstractString)
    p = lowercase(String(prompt))
    has_number = occursin(r"\d", p) || occursin(r"[\u0660-\u0669\u06f0-\u06f9]", p)
    has_number || return false
    return occursin("+", p) || occursin("-", p) || occursin("*", p) ||
           occursin("/", p) || occursin("\u00d7", p) || occursin("\u00f7", p) ||
           occursin("\u0627\u062d\u0633\u0628", p) || occursin("\u0645\u062c\u0645\u0648\u0639", p) ||
           occursin("\u062d\u0627\u0635\u0644", p) || occursin("\u0636\u0631\u0628", p) ||
           occursin("\u0637\u0631\u062d", p) || occursin("\u062c\u0630\u0631", p) ||
           occursin("\u0645\u0633\u0627\u062d\u0629", p) || occursin("\u0643\u0645 \u062a\u0628\u0642", p)
end

function _generate_math(gen::MirnanGenerator, prompt::String)
    hisab_solution = solve_hisab(gen.hisab, prompt)
    if hisab_solution !== nothing && hisab_solution.verified
        return _has_arabic_letter(prompt) ?
               render_hisab_solution_ar(hisab_solution) :
               render_hisab_solution(hisab_solution)
    end
    direct = evaluate_math(gen.math_bridge, prompt)
    if direct !== nothing
        return "الناتج: $direct"
    end
    words = split(prompt)
    numbers = Int[]
    op_word = ""
    ar_digits = Dict('٠'=>'0','١'=>'1','٢'=>'2','٣'=>'3','٤'=>'4','٥'=>'5','٦'=>'6','٧'=>'7','٨'=>'8','٩'=>'9')
    for w in words
        w_clean = replace(w, r"[^\d٠-٩]" => "")
        if !isempty(w_clean)
            en_clean = map(c -> get(ar_digits, c, c), w_clean)
            try push!(numbers, parse(Int, en_clean)); catch e; @debug "Math number parse failed: $e"; end
        else
            op_word = string(w)
        end
    end
    if length(numbers) >= 2
        try
            result, confidence = solve_arithmetic(gen.symbolic_math, numbers[1], op_word, numbers[2])
            if confidence > 0.1
                return generate_math_explanation(gen.symbolic_math, numbers[1], op_word, numbers[2], result)
            end
        catch e
            @warn "Symbolic math failed: $e"
        end
    end
    return ""
end

function try_generate(::MathStrategy, gen::MirnanGenerator, prompt::String, prompt_tokens::Vector{String}, mode::String, cereb_obs, cereb_policy, response_plan, active_paras)
    is_math_req = (mode == "math") || 
                  (mode != "code" && _strong_math_request(prompt)) || 
                  (mode == "auto" && (detect_mode(prompt) == "math" || cereb_policy.mode == "math"))
    
    if is_math_req
        result = _generate_math(gen, prompt)
        return _finish_generation!(gen, prompt, prompt_tokens, result,
                                   cereb_obs, cereb_policy; observe_ram=false,
                                   sanitize_output=false)
    end
    return nothing
end
