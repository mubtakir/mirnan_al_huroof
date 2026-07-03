"""
al_hisab - verified math problem pattern memory for Mirnan.

This layer remembers problem types and solves small, checkable problems with
explicit verification. It complements SymbolicMathEngine instead of replacing it.
"""
module AlHisab

using JSON

export HisabPatternRecord, HisabMemory, HisabSolution,
       learn_hisab_from_text!, train_hisab_from_texts!,
       solve_hisab, render_hisab_solution, render_hisab_solution_ar,
       save_hisab, load_hisab, hisab_to_dict,
       has_hisab_patterns

const AL_HISAB_VERSION = 1

mutable struct HisabPatternRecord
    problem_type::String
    shape::String
    count::Int
    examples::Vector{String}
    slots::Dict{String,Dict{String,Int}}
    checks::Vector{String}
end

mutable struct HisabMemory
    patterns::Dict{String,HisabPatternRecord}
    max_examples::Int
    max_slot_values::Int
end

struct HisabSolution
    problem_type::String
    expression::String
    result::String
    steps::Vector{String}
    verified::Bool
    check::String
end

function HisabMemory(; max_examples::Int=5, max_slot_values::Int=80)
    return HisabMemory(Dict{String,HisabPatternRecord}(), max_examples, max_slot_values)
end

const _NUMERAL_MAP = Dict(
    '٠' => '0', '١' => '1', '٢' => '2', '٣' => '3', '٤' => '4',
    '٥' => '5', '٦' => '6', '٧' => '7', '٨' => '8', '٩' => '9',
    '۰' => '0', '۱' => '1', '۲' => '2', '۳' => '3', '۴' => '4',
    '۵' => '5', '۶' => '6', '۷' => '7', '۸' => '8', '۹' => '9',
    '٫' => '.', '٬' => ',',
    '−' => '-', '﹣' => '-', '－' => '-',
)

function _normalize_number_text(text::AbstractString)
    out = IOBuffer()
    for ch in String(text)
        write(out, get(_NUMERAL_MAP, ch, ch))
    end
    return String(take!(out))
end

function _numbers(text::AbstractString)
    normalized = _normalize_number_text(text)
    values = Float64[]
    for m in eachmatch(r"-?\d+(?:\.\d+)?", normalized)
        value = try
            parse(Float64, replace(m.match, "," => ""))
        catch
            NaN
        end
        isfinite(value) && push!(values, value)
    end
    return values
end

function _fmt(x::Real)
    xf = Float64(x)
    isfinite(xf) || return string(xf)
    if isinteger(xf) && abs(xf) <= typemax(Int)
        return string(Int(round(xf)))
    end
    abs(xf) >= 1e12 && return string(xf)
    return string(round(xf; digits=6))
end

function _push_slot_value!(bucket::Dict{String,Int}, value::String, max_values::Int)
    isempty(strip(value)) && return
    bucket[value] = get(bucket, value, 0) + 1
    if length(bucket) > max_values
        ordered = sort(collect(bucket); by=x -> (x[2], x[1]))
        delete!(bucket, ordered[1][1])
    end
end

function _op_from_text(text::AbstractString)
    s = lowercase(_normalize_number_text(text))
    if occursin("+", s) || occursin("plus", s) || occursin("add", s) ||
       occursin("\u0632\u0627\u0626\u062f", s) || occursin("\u062c\u0645\u0639", s)
        return "add", "+"
    elseif occursin("-", s) || occursin("minus", s) || occursin("subtract", s) ||
           occursin("\u0646\u0627\u0642\u0635", s) || occursin("\u0637\u0631\u062d", s)
        return "subtract", "-"
    elseif occursin("*", s) || occursin("×", s) || occursin("multiply", s) ||
           occursin("\u0636\u0631\u0628", s)
        return "multiply", "*"
    elseif occursin("/", s) || occursin("÷", s) || occursin("divide", s) ||
           occursin("\u0642\u0633\u0645\u0629", s)
        return "divide", "/"
    end
    return "", ""
end

function _binary_result(a::Float64, op::String, b::Float64)
    (isfinite(a) && isfinite(b)) || return nothing
    op == "add" && return a + b
    op == "subtract" && return a - b
    op == "multiply" && return a * b
    op == "divide" && return b == 0 ? nothing : a / b
    return nothing
end

function _detect_binary_arithmetic(text::AbstractString)
    op, symbol = _op_from_text(text)
    isempty(op) && return nothing
    nums = _numbers(text)
    length(nums) >= 2 || return nothing
    return nums[1], op, symbol, nums[2]
end

function _detect_square_area(text::AbstractString)
    s = lowercase(String(text))
    (occursin("\u0645\u0633\u0627\u062d\u0629", s) && occursin("\u0645\u0631\u0628\u0639", s)) || return nothing
    nums = _numbers(text)
    isempty(nums) && return nothing
    side = nums[end]
    isfinite(side) || return nothing
    return side
end

function _detect_square_root(text::AbstractString)
    s = lowercase(String(text))
    (occursin("\u062c\u0630\u0631", s) || occursin("sqrt", s) || occursin("\u221a", s)) || return nothing
    nums = _numbers(text)
    isempty(nums) && return nothing
    value = nums[end]
    (isfinite(value) && value >= 0) || return nothing
    return value
end

function _detect_remainder_problem(text::AbstractString)
    s = lowercase(String(text))
    (occursin("\u0643\u0645", s) && (occursin("\u062a\u0628\u0642\u0649", s) || occursin("\u0628\u0642\u064a", s) ||
     occursin("\u0623\u0643\u0644", s) || occursin("\u0627\u0643\u0644", s))) || return nothing
    nums = _numbers(text)
    length(nums) >= 2 || return nothing
    return nums[1], nums[2]
end

function _detect_linear_equation(text::AbstractString)
    s = replace(_normalize_number_text(text), r"\s+" => "")
    m = match(r"^(-?\d+(?:\.\d+)?)\*?x([+\-]\d+(?:\.\d+)?)=(-?\d+(?:\.\d+)?)$", s)
    if m !== nothing
        return parse(Float64, m.captures[1]), parse(Float64, m.captures[2]), parse(Float64, m.captures[3])
    end
    m = match(r"^x([+\-]\d+(?:\.\d+)?)=(-?\d+(?:\.\d+)?)$", s)
    if m !== nothing
        return 1.0, parse(Float64, m.captures[1]), parse(Float64, m.captures[2])
    end
    m = match(r"^(-?\d+(?:\.\d+)?)\*?x=(-?\d+(?:\.\d+)?)$", s)
    if m !== nothing
        return parse(Float64, m.captures[1]), 0.0, parse(Float64, m.captures[2])
    end
    return nothing
end

function _record!(mem::HisabMemory, problem_type::String, shape::String,
                  example::AbstractString, slots::Dict{String,String}, checks::Vector{String})
    rec = get!(mem.patterns, problem_type) do
        HisabPatternRecord(problem_type, shape, 0, String[], Dict{String,Dict{String,Int}}(), checks)
    end
    rec.count += 1
    example_s = String(example)
    if length(rec.examples) < mem.max_examples && !(example_s in rec.examples)
        push!(rec.examples, example_s)
    end
    for (slot, value) in slots
        bucket = get!(rec.slots, slot, Dict{String,Int}())
        _push_slot_value!(bucket, value, mem.max_slot_values)
    end
    return rec
end

function learn_hisab_from_text!(mem::HisabMemory, text::AbstractString)
    s = strip(String(text))
    lin = _detect_linear_equation(s)
    if lin !== nothing
        a, b, c = lin
        all(isfinite, (a, b, c)) || return 0
        _record!(mem, "linear_equation", "a*x + b = c", s,
                 Dict("a" => _fmt(a), "b" => _fmt(b), "c" => _fmt(c), "variable" => "x"),
                 ["substitution"])
        return 1
    end
    bin = _detect_binary_arithmetic(s)
    if bin !== nothing
        a, op, symbol, b = bin
        result = _binary_result(a, op, b)
        result === nothing && return 0
        isfinite(result) || return 0
        _record!(mem, "binary_arithmetic", "a OP b", s,
                 Dict("a" => _fmt(a), "b" => _fmt(b), "operator" => symbol),
                 ["recompute"])
        return 1
    end
    return 0
end

function train_hisab_from_texts!(mem::HisabMemory, texts::Vector{String}; max_problems::Int=50_000)
    total = 0
    for text in texts
        total >= max_problems && break
        total += learn_hisab_from_text!(mem, text)
    end
    return total
end

has_hisab_patterns(mem::HisabMemory) = !isempty(mem.patterns)

function solve_hisab(mem::HisabMemory, text::AbstractString)
    s = strip(String(text))
    remainder = _detect_remainder_problem(s)
    if remainder !== nothing
        a, b = remainder
        result = a - b
        verified = isfinite(result) && result + b == a
        steps = [
            "identify remaining quantity",
            "subtract used amount from original amount",
            "verify by recomputation",
        ]
        check = "$(_fmt(a)) - $(_fmt(b)) = $(_fmt(result))"
        return HisabSolution("remainder_word_problem", s, _fmt(result), steps, verified, check)
    end
    side = _detect_square_area(s)
    if side !== nothing
        result = side * side
        verified = isfinite(result)
        steps = [
            "identify square side",
            "multiply side by itself",
            "verify by recomputation",
        ]
        check = "$(_fmt(side)) * $(_fmt(side)) = $(_fmt(result))"
        return HisabSolution("square_area", s, _fmt(result), steps, verified, check)
    end
    root_value = _detect_square_root(s)
    if root_value !== nothing
        result = sqrt(root_value)
        verified = isfinite(result) && abs(result * result - root_value) < 1e-8
        steps = [
            "identify square root",
            "find the number whose square gives the input",
            "verify by squaring",
        ]
        check = "$(_fmt(result)) * $(_fmt(result)) = $(_fmt(root_value))"
        return HisabSolution("square_root", s, _fmt(result), steps, verified, check)
    end
    lin = _detect_linear_equation(s)
    if lin !== nothing
        a, b, c = lin
        a == 0 && return nothing
        x = (c - b) / a
        verified = abs(a * x + b - c) < 1e-8
        steps = [
            "subtract $(_fmt(b)) from both sides",
            "divide by $(_fmt(a))",
            "verify by substitution",
        ]
        check = "$(_fmt(a)) * $(_fmt(x)) + $(_fmt(b)) = $(_fmt(a * x + b))"
        return HisabSolution("linear_equation", s, "x = $(_fmt(x))", steps, verified, check)
    end
    bin = _detect_binary_arithmetic(s)
    if bin !== nothing
        a, op, symbol, b = bin
        result = _binary_result(a, op, b)
        result === nothing && return nothing
        isfinite(result) || return nothing
        verified = _binary_result(a, op, b) == result
        steps = [
            "identify operation: $(op)",
            "compute $(_fmt(a)) $(symbol) $(_fmt(b))",
            "verify by recomputation",
        ]
        check = "$(_fmt(a)) $(symbol) $(_fmt(b)) = $(_fmt(result))"
        return HisabSolution("binary_arithmetic", s, _fmt(result), steps, verified, check)
    end
    return nothing
end

function render_hisab_solution(sol::HisabSolution)
    lines = String[]
    push!(lines, "$(sol.expression) => $(sol.result)")
    for (i, step) in enumerate(sol.steps)
        push!(lines, string(i, ". ", step))
    end
    push!(lines, "check: $(sol.check)")
    push!(lines, "verified: $(sol.verified)")
    return join(lines, "\n")
end

function _arabic_op_name(op::AbstractString)
    op == "add" && return "\u0645\u062c\u0645\u0648\u0639"
    op == "subtract" && return "\u0646\u0627\u062a\u062c \u0627\u0644\u0637\u0631\u062d"
    op == "multiply" && return "\u0646\u0627\u062a\u062c \u0627\u0644\u0636\u0631\u0628"
    op == "divide" && return "\u0646\u0627\u062a\u062c \u0627\u0644\u0642\u0633\u0645\u0629"
    return "\u0627\u0644\u0646\u0627\u062a\u062c"
end

function _arabic_op_verb(op::AbstractString)
    op == "add" && return "\u062c\u0645\u0639"
    op == "subtract" && return "\u0637\u0631\u062d"
    op == "multiply" && return "\u0636\u0631\u0628"
    op == "divide" && return "\u0642\u0633\u0645\u0629"
    return "\u062d\u0633\u0627\u0628"
end

function _parse_binary_check(check::AbstractString)
    m = match(r"^\s*(.+?)\s*([+\-*/])\s*(.+?)\s*=\s*(.+?)\s*$", String(check))
    m === nothing && return nothing
    return strip.(String.(m.captures))
end

function render_hisab_solution_ar(sol::HisabSolution)
    if sol.problem_type == "binary_arithmetic"
        parsed = _parse_binary_check(sol.check)
        if parsed !== nothing
            a, symbol, b, result = parsed
            op, _ = _op_from_text(symbol)
            isempty(op) && (op, _ = _op_from_text(sol.expression))
            op_name = _arabic_op_name(op)
            op_verb = _arabic_op_verb(op)
            lines = String[]
            push!(lines, "$op_name $a $(symbol) $b هو $result.")
            push!(lines, "1. \u062a\u062d\u062f\u064a\u062f \u0627\u0644\u0639\u0645\u0644\u064a\u0629: $op_verb.")
            push!(lines, "2. \u062d\u0633\u0627\u0628 $a $(symbol) $b.")
            push!(lines, "3. \u0627\u0644\u062a\u062d\u0642\u0642 \u0628\u0625\u0639\u0627\u062f\u0629 \u0627\u0644\u062d\u0633\u0627\u0628.")
            push!(lines, "\u0627\u0644\u062a\u062d\u0642\u0642: $(sol.check).")
            push!(lines, "\u0645\u0624\u0643\u062f: $(sol.verified ? "\u0646\u0639\u0645" : "\u0644\u0627").")
            return join(lines, "\n")
        end
    elseif sol.problem_type == "linear_equation"
        lines = String[]
        push!(lines, "\u062d\u0644 \u0627\u0644\u0645\u0639\u0627\u062f\u0644\u0629 $(sol.expression) \u0647\u0648 $(sol.result).")
        push!(lines, "1. \u0646\u0646\u0642\u0644 \u0627\u0644\u062d\u062f \u0627\u0644\u062b\u0627\u0628\u062a \u0625\u0644\u0649 \u0627\u0644\u0637\u0631\u0641 \u0627\u0644\u0622\u062e\u0631.")
        push!(lines, "2. \u0646\u0642\u0633\u0645 \u0639\u0644\u0649 \u0645\u0639\u0627\u0645\u0644 x.")
        push!(lines, "3. \u0646\u062a\u062d\u0642\u0642 \u0628\u0627\u0644\u062a\u0639\u0648\u064a\u0636.")
        push!(lines, "\u0627\u0644\u062a\u062d\u0642\u0642: $(sol.check).")
        push!(lines, "\u0645\u0624\u0643\u062f: $(sol.verified ? "\u0646\u0639\u0645" : "\u0644\u0627").")
        return join(lines, "\n")
    elseif sol.problem_type == "square_area"
        return join([
            "\u0645\u0633\u0627\u062d\u0629 \u0627\u0644\u0645\u0631\u0628\u0639 \u0647\u064a $(sol.result).",
            "1. \u0646\u062d\u062f\u062f \u0637\u0648\u0644 \u0627\u0644\u0636\u0644\u0639.",
            "2. \u0646\u0636\u0631\u0628 \u0627\u0644\u0636\u0644\u0639 \u0641\u064a \u0646\u0641\u0633\u0647.",
            "\u0627\u0644\u062a\u062d\u0642\u0642: $(sol.check).",
            "\u0645\u0624\u0643\u062f: $(sol.verified ? "\u0646\u0639\u0645" : "\u0644\u0627").",
        ], "\n")
    elseif sol.problem_type == "square_root"
        return join([
            "\u062c\u0630\u0631 \u0627\u0644\u0639\u062f\u062f \u0647\u0648 $(sol.result).",
            "1. \u0646\u062d\u062f\u062f \u0627\u0644\u0639\u062f\u062f \u062a\u062d\u062a \u0627\u0644\u062c\u0630\u0631.",
            "2. \u0646\u0628\u062d\u062b \u0639\u0646 \u0639\u062f\u062f \u0645\u0631\u0628\u0639\u0647 \u064a\u0633\u0627\u0648\u064a\u0647.",
            "\u0627\u0644\u062a\u062d\u0642\u0642: $(sol.check).",
            "\u0645\u0624\u0643\u062f: $(sol.verified ? "\u0646\u0639\u0645" : "\u0644\u0627").",
        ], "\n")
    elseif sol.problem_type == "remainder_word_problem"
        return join([
            "\u0627\u0644\u0628\u0627\u0642\u064a \u0647\u0648 $(sol.result).",
            "1. \u0646\u062d\u062f\u062f \u0627\u0644\u0643\u0645\u064a\u0629 \u0627\u0644\u0623\u0635\u0644\u064a\u0629.",
            "2. \u0646\u0637\u0631\u062d \u0627\u0644\u0643\u0645\u064a\u0629 \u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645\u0629.",
            "\u0627\u0644\u062a\u062d\u0642\u0642: $(sol.check).",
            "\u0645\u0624\u0643\u062f: $(sol.verified ? "\u0646\u0639\u0645" : "\u0644\u0627").",
        ], "\n")
    end
    return render_hisab_solution(sol)
end

function hisab_to_dict(mem::HisabMemory)
    records = sort(collect(values(mem.patterns)); by=r -> (-r.count, r.problem_type))
    return Dict{String,Any}(
        "version" => AL_HISAB_VERSION,
        "n_patterns" => length(records),
        "patterns" => [Dict{String,Any}(
            "problem_type" => rec.problem_type,
            "shape" => rec.shape,
            "count" => rec.count,
            "examples" => rec.examples,
            "slots" => rec.slots,
            "checks" => rec.checks,
        ) for rec in records],
    )
end

function save_hisab(mem::HisabMemory, path::String)
    mkpath(dirname(path))
    tmp = string(path, ".tmp")
    open(tmp, "w") do io
        JSON.print(io, hisab_to_dict(mem))
    end
    mv(tmp, path; force=true)
    return path
end

function load_hisab(path::String)
    mem = HisabMemory()
    isfile(path) || return mem
    filesize(path) == 0 && return mem
    data = try
        JSON.parsefile(path)
    catch e
        @warn "تعذر تحميل al_hisab: $path — $e"
        return mem
    end
    for item in get(data, "patterns", Any[])
        item isa AbstractDict || continue
        problem_type = String(get(item, "problem_type", ""))
        isempty(problem_type) && continue
        raw_slots = get(item, "slots", Dict{String,Any}())
        slots = Dict{String,Dict{String,Int}}()
        for (slot, bucket) in raw_slots
            bucket isa AbstractDict || continue
            values = Dict{String,Int}()
            for (k, v) in bucket
                count = try
                    Int(v)
                catch
                    0
                end
                count > 0 && (values[String(k)] = count)
            end
            slots[String(slot)] = values
        end
        mem.patterns[problem_type] = HisabPatternRecord(
            problem_type,
            String(get(item, "shape", "")),
            Int(get(item, "count", 0)),
            String[String(e) for e in get(item, "examples", String[])],
            slots,
            String[String(c) for c in get(item, "checks", String[])],
        )
    end
    return mem
end

end # module AlHisab
