"""
al_code - Code pattern memory for Mirnan.

This layer remembers program shapes separately from K_code. It stores structural
templates such as functions, branches, loops, returns, assignments, and imports.
"""
module AlCode

using JSON

export CodePatternRecord, CodePatternMemory,
       learn_code_patterns_from_text!, train_code_patterns_from_texts!,
       select_code_pattern, preferred_code_slot_values,
       generate_code_from_pattern,
       save_al_code, load_al_code, al_code_to_dict,
       has_code_patterns, detect_code_language

const AL_CODE_VERSION = 1

mutable struct CodePatternRecord
    language::String
    shape::String
    roles::Vector{String}
    count::Int
    examples::Vector{String}
    slots::Dict{String,Dict{String,Int}}
end

mutable struct CodePatternMemory
    patterns::Dict{String,CodePatternRecord}
    max_examples::Int
    max_slot_values::Int
end

function CodePatternMemory(; max_examples::Int=5, max_slot_values::Int=80)
    return CodePatternMemory(Dict{String,CodePatternRecord}(), max_examples, max_slot_values)
end

function _strip_code_comment(line::AbstractString)
    s = String(line)
    idx = findfirst('#', s)
    idx === nothing && return strip(s)
    return strip(s[begin:prevind(s, idx)])
end

function _code_lines(source::AbstractString)
    out = String[]
    for raw in split(String(source), '\n')
        line = _strip_code_comment(raw)
        isempty(line) && continue
        push!(out, line)
    end
    return out
end

function detect_code_language(source::AbstractString)
    s = lowercase(String(source))
    if occursin("\u0627\u0637\u0628\u0639", s) ||
       occursin("\u062f\u0627\u0644\u0629", s) ||
       occursin("\u0625\u0630\u0627", s) ||
       occursin("\u0646\u0647\u0627\u064a\u0629", s)
        return "bayan"
    elseif occursin(r"\bfunction\s+\w+", s) || occursin(r"\bend\b", s) ||
           occursin(r"\bstruct\s+\w+", s) || occursin(r"\bmodule\s+\w+", s) ||
           occursin("@testset", s)
        return "julia"
    elseif occursin(r"\bdef\s+\w+\s*\(", s) || occursin(":", s)
        return "python"
    elseif occursin(r"\bfunction\s+\w+\s*\(", s) || occursin("=>", s) || occursin("{", s)
        return "javascript"
    end
    return "generic"
end

function _first_match(pattern::Regex, s::AbstractString)
    m = match(pattern, String(s))
    m === nothing && return ""
    return String(m.captures[1])
end

function _classify_line(line::String, language::String)
    s = strip(line)
    lower = lowercase(s)

    if language == "python" && startswith(lower, "def ")
        return "function_def", "FUNCTION_DEF", Dict(
            "name" => _first_match(r"def\s+([A-Za-z_]\w*)", s),
            "params" => _first_match(r"def\s+[A-Za-z_]\w*\s*\(([^)]*)\)", s),
        )
    elseif language == "julia" && startswith(lower, "function ")
        return "function_def", "FUNCTION_DEF", Dict(
            "name" => _first_match(r"function\s+([A-Za-z_]\w*)", s),
            "params" => _first_match(r"function\s+[A-Za-z_]\w*\s*\(([^)]*)\)", s),
        )
    elseif startswith(lower, "module ")
        return "module_def", "MODULE_DEF", Dict(
            "name" => _first_match(r"module\s+([A-Za-z_]\w*)", s),
        )
    elseif startswith(lower, "mutable struct ")
        return "struct_def", "STRUCT_DEF", Dict(
            "name" => _first_match(r"mutable\s+struct\s+([A-Za-z_]\w*)", s),
        )
    elseif startswith(lower, "struct ")
        return "struct_def", "STRUCT_DEF", Dict(
            "name" => _first_match(r"struct\s+([A-Za-z_]\w*)", s),
        )
    elseif startswith(lower, "@testset")
        return "testset", "TESTSET", Dict(
            "name" => _first_match(r"@testset\s+\"([^\"]+)\"", s),
        )
    elseif lower == "try"
        return "try", "TRY", Dict{String,String}()
    elseif startswith(lower, "catch")
        return "catch", "CATCH", Dict(
            "name" => strip(replace(s, r"^catch\s*" => "")),
        )
    elseif lower == "finally" || lower == "finally:"
        return "finally", "FINALLY", Dict{String,String}()
    elseif startswith(lower, "class ")
        return "class_def", "CLASS_DEF", Dict(
            "name" => _first_match(r"class\s+([A-Za-z_]\w*)", s),
        )
    elseif startswith(s, "\u062f\u0627\u0644\u0629 ")
        return "function_def", "FUNCTION_DEF", Dict(
            "name" => _first_match(Regex("\u062f\u0627\u0644\u0629\\s+(\\S+)"), s),
        )
    elseif startswith(s, "\u0625\u0630\u0627 ") || startswith(s, "\u0627\u0630\u0627 ")
        return "condition", "IF", Dict("condition" => replace(s, Regex("^\u0625\u0630\u0627\\s+|^\u0627\u0630\u0627\\s+") => ""))
    elseif startswith(s, "\u0644\u0643\u0644 ")
        return "loop", "FOR", Dict("iterator" => replace(s, Regex("^\u0644\u0643\u0644\\s+") => ""))
    elseif startswith(s, "\u0627\u0637\u0628\u0639")
        return "call", "CALL", Dict("call" => s)
    elseif s == "\u0646\u0647\u0627\u064a\u0629"
        return "end", "END", Dict{String,String}()
    elseif startswith(lower, "if ")
        return "condition", "IF", Dict("condition" => replace(s, r"^if\s+|:\s*$" => ""))
    elseif startswith(lower, "elseif ") || startswith(lower, "elif ")
        return "elseif", "ELSEIF", Dict("condition" => replace(s, r"^(elseif|elif)\s+|:\s*$" => ""))
    elseif lower == "else" || lower == "else:"
        return "else", "ELSE", Dict{String,String}()
    elseif startswith(lower, "for ")
        return "loop", "FOR", Dict("iterator" => replace(s, r"^for\s+|:\s*$" => ""))
    elseif startswith(lower, "while ")
        return "loop", "WHILE", Dict("condition" => replace(s, r"^while\s+|:\s*$" => ""))
    elseif startswith(lower, "return")
        return "return", "RETURN", Dict("value" => strip(replace(s, r"^return\s*" => "")))
    elseif startswith(lower, "import ") || startswith(lower, "using ") || startswith(lower, "from ")
        return "import", "IMPORT", Dict("module" => strip(replace(s, r"^(import|using|from)\s+" => "")))
    elseif lower == "end"
        return "end", "END", Dict{String,String}()
    elseif lower == "pass"
        return "pass", "PASS", Dict{String,String}()
    elseif occursin("=", s) && !occursin("==", s)
        name = strip(first(split(s, "=")))
        value = strip(join(split(s, "=")[2:end], "="))
        return "assignment", "ASSIGN", Dict("name" => name, "value" => value)
    elseif occursin(r"[A-Za-z_]\w*\s*\(", s)
        return "call", "CALL", Dict("call" => s)
    end
    return "statement", "STMT", Dict("statement" => s)
end

function _push_slot_value!(bucket::Dict{String,Int}, value::String, max_values::Int)
    isempty(strip(value)) && return
    bucket[value] = get(bucket, value, 0) + 1
    if length(bucket) > max_values
        ordered = sort(collect(bucket); by=x -> (x[2], x[1]))
        delete!(bucket, ordered[1][1])
    end
end

function _extract_code_pattern(source::AbstractString)
    lines = _code_lines(source)
    1 <= length(lines) <= 80 || return nothing
    language = detect_code_language(source)
    roles = String[]
    labels = String[]
    slots = Vector{Dict{String,String}}()
    for line in lines
        role, label, line_slots = _classify_line(line, language)
        push!(roles, role)
        push!(labels, label)
        push!(slots, line_slots)
    end
    shape = join(labels, " ")
    return language, shape, roles, lines, slots
end

function learn_code_patterns_from_text!(mem::CodePatternMemory, source::AbstractString)
    extracted = _extract_code_pattern(source)
    extracted === nothing && return 0
    language, shape, roles, lines, line_slots = extracted
    key = string(language, "\t", shape)
    rec = get!(mem.patterns, key) do
        CodePatternRecord(language, shape, copy(roles), 0, String[], Dict{String,Dict{String,Int}}())
    end
    rec.count += 1
    example = join(lines, "\n")
    if length(rec.examples) < mem.max_examples && !(example in rec.examples)
        push!(rec.examples, example)
    end
    for (role, slots) in zip(roles, line_slots)
        bucket = get!(rec.slots, role, Dict{String,Int}())
        for value in values(slots)
            _push_slot_value!(bucket, value, mem.max_slot_values)
        end
    end
    for slots in line_slots
        for (slot, value) in slots
            bucket = get!(rec.slots, slot, Dict{String,Int}())
            _push_slot_value!(bucket, value, mem.max_slot_values)
        end
    end
    return 1
end

function train_code_patterns_from_texts!(mem::CodePatternMemory, texts::Vector{String};
                                         max_blocks::Int=50_000)
    total = 0
    for text in texts
        total >= max_blocks && break
        total += learn_code_patterns_from_text!(mem, text)
    end
    return total
end

has_code_patterns(mem::CodePatternMemory) = !isempty(mem.patterns)

function _prompt_key(s::AbstractString)
    return lowercase(strip(String(s)))
end

function _prompt_tokens(prompt::AbstractString)
    return [_prompt_key(t) for t in split(String(prompt)) if !isempty(strip(t))]
end

function _prompt_language(prompt::AbstractString)
    p = lowercase(String(prompt))
    occursin("julia", p) && return "julia"
    occursin("python", p) && return "python"
    occursin("javascript", p) && return "javascript"
    (occursin("bayan", p) || occursin("\u0628\u064a\u0627\u0646", p)) && return "bayan"
    return ""
end

function _pattern_intent_score(rec::CodePatternRecord, prompt::AbstractString)
    p = lowercase(String(prompt))
    score = 0.0
    if occursin("function", p) || occursin("def", p) || occursin("دالة", p)
        "function_def" in rec.roles && (score += 2.0)
    end
    if occursin("loop", p) || occursin("for", p) || occursin("while", p) || occursin("حلقة", p)
        any(r -> r in ("loop",), rec.roles) && (score += 2.0)
    end
    if occursin("if", p) || occursin("condition", p) || occursin("شرط", p)
        "condition" in rec.roles && (score += 2.0)
    end
    if occursin("class", p) || occursin("صنف", p) || occursin("كلاس", p)
        "class_def" in rec.roles && (score += 2.0)
    end
    if occursin("struct", p) || occursin("mutable", p)
        "struct_def" in rec.roles && (score += 2.0)
        !isempty(rec.roles) && rec.roles[1] == "struct_def" && (score += 1.0)
    end
    if occursin("module", p)
        "module_def" in rec.roles && (score += 2.0)
        !isempty(rec.roles) && rec.roles[1] == "module_def" && (score += 1.0)
    end
    if occursin("testset", p) || occursin("test set", p)
        "testset" in rec.roles && (score += 2.0)
        !isempty(rec.roles) && rec.roles[1] == "testset" && (score += 1.0)
    end
    if occursin("try", p) || occursin("catch", p) || occursin("finally", p)
        "try" in rec.roles && (score += 2.0)
        "catch" in rec.roles && (score += 1.0)
        !isempty(rec.roles) && rec.roles[1] == "try" && (score += 1.0)
    end
    if occursin("bayan", p) || occursin("\u0628\u064a\u0627\u0646", p)
        rec.language == "bayan" && (score += 2.0)
    end
    return score
end

function _prompt_overlap(rec::CodePatternRecord, prompt::AbstractString)
    pts = Set(_prompt_tokens(prompt))
    isempty(pts) && return 0.0
    hits = 0
    total = 0
    for bucket in values(rec.slots)
        for value in keys(bucket)
            for word in _prompt_tokens(value)
                total += 1
                word in pts && (hits += 1)
            end
        end
    end
    total == 0 && return 0.0
    return hits / min(total, length(pts))
end

function select_code_pattern(mem::CodePatternMemory, prompt::AbstractString; language::String="")
    isempty(mem.patterns) && return nothing
    wanted_language = isempty(language) ? _prompt_language(prompt) : language
    scored = Tuple{Float64,CodePatternRecord}[]
    for rec in values(mem.patterns)
        if !isempty(wanted_language) && rec.language != wanted_language
            continue
        end
        score = log(1 + rec.count)
        score += _pattern_intent_score(rec, prompt)
        score += 1.5 * _prompt_overlap(rec, prompt)
        length(rec.roles) > 30 && (score -= 0.25)
        push!(scored, (score, rec))
    end
    isempty(scored) && return nothing
    sort!(scored; by=x -> -x[1])
    return scored[1][2]
end

function preferred_code_slot_values(rec::CodePatternRecord, role::String; limit::Int=20)
    bucket = get(rec.slots, role, Dict{String,Int}())
    vals = sort(collect(bucket); by=x -> (-x[2], x[1]))
    isempty(vals) && return String[]
    return [v[1] for v in vals[1:min(limit, length(vals))]]
end

function _extract_requested_name(prompt::AbstractString)
    p = String(prompt)
    for rx in (r"`([A-Za-z_]\w*)`", r"(?:named|called|name)\s+([A-Za-z_]\w*)", r"دالة\s+([A-Za-z_]\w*)")
        m = match(rx, p)
        m === nothing || return String(m.captures[1])
    end
    return ""
end

function _prompt_has_any(prompt::AbstractString, words)
    p = lowercase(String(prompt))
    return any(w -> occursin(w, p), words)
end

function _requested_operation(prompt::AbstractString)
    if _prompt_has_any(prompt, ("add", "sum", "plus", "addition", "+",
                                "\u0627\u062c\u0645\u0639", "\u062c\u0645\u0639", "\u0645\u062c\u0645\u0648\u0639"))
        return "add"
    elseif _prompt_has_any(prompt, ("subtract", "minus", "difference", "-",
                                    "\u0627\u0637\u0631\u062d", "\u0637\u0631\u062d", "\u0641\u0631\u0642"))
        return "subtract"
    elseif _prompt_has_any(prompt, ("multiply", "product", "times", "*",
                                    "\u0627\u0636\u0631\u0628", "\u0636\u0631\u0628", "\u062d\u0627\u0635\u0644"))
        return "multiply"
    elseif _prompt_has_any(prompt, ("divide", "division", "quotient", "/",
                                    "\u0627\u0642\u0633\u0645", "\u0642\u0633\u0645\u0629"))
        return "divide"
    end
    return ""
end

function _operation_name(op::String)
    op == "add" && return "add"
    op == "subtract" && return "subtract"
    op == "multiply" && return "multiply"
    op == "divide" && return "divide"
    return "generated_function"
end

function _operation_expr(op::String)
    op == "add" && return "a + b"
    op == "subtract" && return "a - b"
    op == "multiply" && return "a * b"
    op == "divide" && return "a / b"
    return ""
end

function _render_operation_function(lang::String, name::String, op::String)
    expr = _operation_expr(op)
    isempty(expr) && return ""
    params = "a, b"
    if lang == "julia"
        return "function $(name)($(params))\n    return $(expr)\nend\n"
    elseif lang == "bayan"
        return "\u062f\u0627\u0644\u0629 $(name)(a, b)\n    \u0623\u0631\u062c\u0639 $(expr)\n\u0646\u0647\u0627\u064a\u0629\n"
    else
        return "def $(name)($(params)):\n    return $(expr)\n"
    end
end

function _normalize_prompt_digits(s::AbstractString)
    map = Dict(
        '٠' => '0', '١' => '1', '٢' => '2', '٣' => '3', '٤' => '4',
        '٥' => '5', '٦' => '6', '٧' => '7', '٨' => '8', '٩' => '9',
        '۰' => '0', '۱' => '1', '۲' => '2', '۳' => '3', '۴' => '4',
        '۵' => '5', '۶' => '6', '۷' => '7', '۸' => '8', '۹' => '9',
    )
    out = IOBuffer()
    for ch in String(s)
        print(out, get(map, ch, ch))
    end
    return String(take!(out))
end

function _loop_bounds_from_prompt(prompt::AbstractString)
    p = _normalize_prompt_digits(prompt)
    nums = [parse(Int, m.match) for m in eachmatch(r"\d+", p)]
    if length(nums) >= 2
        return nums[end-1], nums[end]
    elseif length(nums) == 1
        return 1, nums[1]
    end
    return 1, 10
end

function _direct_code_for_prompt(prompt::AbstractString, lang::String)
    p = lowercase(String(prompt))
    if occursin("fibonacci", p) || occursin("\u0641\u064a\u0628\u0648\u0646\u0627\u062a\u0634\u064a", p)
        if lang == "julia"
            return "function fibonacci(n)\n    n <= 0 && return Int[]\n    n == 1 && return [0]\n    fib = [0, 1]\n    while length(fib) < n\n        push!(fib, fib[end] + fib[end-1])\n    end\n    return fib\nend\n"
        end
        return "def fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    fib = [0, 1]\n    while len(fib) < n:\n        fib.append(fib[-1] + fib[-2])\n    return fib\n"
    elseif occursin("factorial", p) || occursin("\u0645\u0636\u0631\u0648\u0628", p)
        if lang == "julia"
            return "function factorial_num(n)\n    n < 0 && return nothing\n    res = 1\n    for i in 1:n\n        res *= i\n    end\n    return res\nend\n"
        end
        return "def factorial(n):\n    if n < 0:\n        return None\n    res = 1\n    for i in range(1, n + 1):\n        res *= i\n    return res\n"
    elseif occursin("sort", p) || occursin("\u062a\u0631\u062a\u064a\u0628", p) || occursin("\u0641\u0631\u0632", p)
        if lang == "julia"
            return "function bubble_sort!(arr)\n    n = length(arr)\n    for i in 1:n\n        for j in 1:n-i\n            if arr[j] > arr[j+1]\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n            end\n        end\n    end\n    return arr\nend\n"
        end
        return "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\n"
    elseif occursin("prime", p) || occursin("\u0623\u0648\u0644\u064a", p) || occursin("\u0627\u0648\u0644\u064a", p)
        if lang == "julia"
            return "function is_prime(n)\n    n <= 1 && return false\n    for i in 2:Int(floor(sqrt(n)))\n        n % i == 0 && return false\n    end\n    return true\nend\n"
        end
        return "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n"
    elseif occursin("\u062d\u0644\u0642\u0629", p) || occursin("loop", p) || occursin("for ", p)
        start_n, end_n = _loop_bounds_from_prompt(prompt)
        if lang == "julia"
            return "for i in $(start_n):$(end_n)\n    println(i)\nend\n"
        end
        return "for i in range($(start_n), $(end_n + 1)):\n    print(i)\n"
    elseif occursin("\u0634\u0631\u0637", p) || occursin("condition", p) || occursin("if ", p)
        if lang == "julia"
            return "if number > 10\n    println(number)\nend\n"
        end
        return "if number > 10:\n    print(number)\n"
    elseif occursin("\u0645\u062a\u063a\u064a\u0631", p) || occursin("variable", p)
        if lang == "julia"
            return "x = 1\n"
        end
        return "x = 1\n"
    elseif occursin("\u0623\u0643\u0628\u0631", p) || occursin("\u0627\u0643\u0628\u0631", p) || occursin("maximum", p) || occursin("max", p)
        if lang == "julia"
            return "function max_two(a, b)\n    return a > b ? a : b\nend\n"
        end
        return "def max_two(a, b):\n    return a if a > b else b\n"
    end
    return ""
end

function _simple_body(rec::CodePatternRecord)
    if "return" in rec.roles
        vals = preferred_code_slot_values(rec, "value"; limit=1)
        !isempty(vals) && return "    return $(vals[1])"
    end
    return "    pass"
end

function generate_code_from_pattern(mem::CodePatternMemory, prompt::AbstractString;
                                    language::String="")
    wanted_language = isempty(language) ? _prompt_language(prompt) : language
    isempty(wanted_language) && (wanted_language = "python")
    direct = _direct_code_for_prompt(prompt, wanted_language)
    isempty(strip(direct)) || return direct
    op = _requested_operation(prompt)
    if !isempty(op) && (occursin("function", lowercase(String(prompt))) ||
                        occursin("def", lowercase(String(prompt))) ||
                        occursin("\u062f\u0627\u0644\u0629", String(prompt)))
        requested = _extract_requested_name(prompt)
        name = isempty(requested) ? _operation_name(op) : requested
        rendered = _render_operation_function(wanted_language, name, op)
        isempty(rendered) || return rendered
    end
    rec = select_code_pattern(mem, prompt; language=language)
    rec === nothing && return ""
    lang = rec.language
    name = _extract_requested_name(prompt)
    isempty(name) && begin
        names = preferred_code_slot_values(rec, "name"; limit=1)
        name = isempty(names) ? "generated_function" : names[1]
    end
    if "function_def" in rec.roles && !isempty(op)
        if isempty(_extract_requested_name(prompt))
            name = _operation_name(op)
        end
        rendered = _render_operation_function(lang, name, op)
        isempty(rendered) || return rendered
    end
    params_vals = preferred_code_slot_values(rec, "params"; limit=1)
    params = isempty(params_vals) ? "" : params_vals[1]
    body = _simple_body(rec)
    if "function_def" in rec.roles
        if lang == "julia"
            return "function $(name)($(params))\n$(body)\nend\n"
        elseif lang == "bayan"
            return "\u062f\u0627\u0644\u0629 $(name)\n    \u0627\u0637\u0628\u0639 \"\"\n\u0646\u0647\u0627\u064a\u0629\n"
        else
            return "def $(name)($(params)):\n$(body)\n"
        end
    elseif "module_def" in rec.roles
        return "module $(name)\nend\n"
    elseif "struct_def" in rec.roles
        return "struct $(name)\nend\n"
    elseif "testset" in rec.roles
        title_vals = preferred_code_slot_values(rec, "name"; limit=1)
        title = isempty(title_vals) ? "generated" : title_vals[1]
        return "@testset \"$(title)\" begin\n    @test true\nend\n"
    elseif "try" in rec.roles
        return lang == "python" ?
               "try:\n    pass\nexcept Exception:\n    pass\n" :
               "try\n    nothing\ncatch e\n    nothing\nend\n"
    elseif "condition" in rec.roles
        return lang == "julia" ? "if condition\n    nothing\nend\n" : "if condition:\n    pass\n"
    elseif any(==("loop"), rec.roles)
        return lang == "julia" ? "for item in items\n    nothing\nend\n" : "for item in items:\n    pass\n"
    elseif "class_def" in rec.roles
        return lang == "julia" ? "struct $(name)\nend\n" : "class $(name):\n    pass\n"
    end
    isempty(rec.examples) && return ""
    return rec.examples[1]
end

function al_code_to_dict(mem::CodePatternMemory)
    records = sort(collect(values(mem.patterns)); by=r -> (-r.count, r.language, r.shape))
    return Dict{String,Any}(
        "version" => AL_CODE_VERSION,
        "n_patterns" => length(records),
        "patterns" => [Dict{String,Any}(
            "language" => rec.language,
            "shape" => rec.shape,
            "roles" => rec.roles,
            "count" => rec.count,
            "examples" => rec.examples,
            "slots" => rec.slots,
        ) for rec in records],
    )
end

function save_al_code(mem::CodePatternMemory, path::String)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, al_code_to_dict(mem))
    end
    return path
end

function load_al_code(path::String)
    mem = CodePatternMemory()
    isfile(path) || return mem
    data = JSON.parsefile(path)
    for item in get(data, "patterns", Any[])
        item isa AbstractDict || continue
        language = String(get(item, "language", "generic"))
        shape = String(get(item, "shape", ""))
        isempty(shape) && continue
        roles = String[String(r) for r in get(item, "roles", String[])]
        examples = String[String(e) for e in get(item, "examples", String[])]
        raw_slots = get(item, "slots", Dict{String,Any}())
        slots = Dict{String,Dict{String,Int}}()
        for (role, bucket) in raw_slots
            bucket isa AbstractDict || continue
            slots[String(role)] = Dict{String,Int}(String(k) => Int(v) for (k, v) in bucket)
        end
        key = string(language, "\t", shape)
        mem.patterns[key] = CodePatternRecord(
            language, shape, roles, Int(get(item, "count", 0)), examples, slots)
    end
    return mem
end

end # module AlCode
