"""MathBridge — جسر رياضي وبرمجي (كشف الرياضيات والكود وتقييم الصحة النحوية)."""
module MathBridgeModule
export MathBridge, is_math_expression, evaluate_math,
       is_code_prompt, detect_mode, validate_code_syntax

const MATH_KEYWORDS = Set(["زائد","ناقص","مضروب","مقسوم","جذر","مربع","مكعب","قوة","لوغاريتم","جيب","جيب تمام","ظل","مشتق","تكامل","نهاية","مجموع","حاصل ضرب","متوسط","وسيط","قيمة مطلقة","باقي قسمة"])
const MATH_SYMBOLS = Dict("+"=>"+","-"=>"-","×"=>"*","÷"=>"/","√"=>"sqrt","^"=>"^","π"=>"3.14159","∞"=>"Inf","∑"=>"sum","∏"=>"prod")

const CODE_KEYWORDS_AR = Set([
    "كود","برنامج","برمجة","دالة","دوال","حلقة","شرط","مصفوفة","متغير","متغيرات",
    "كلاس","صنف","استيراد","تصدير","إرجاع","خطأ","تصحيح","خوارزمية","خوارزميات",
    "بايثون","جوليا","جافا","سي++","جافاسكريبت","أخطاء","تركيب","نحوي",
    "اكتب","أنشئ","صمم","برمج","نفذ","نفّذ","compile","debug","build","run",
])
const CODE_KEYWORDS_EN = Set([
    "def","class","import","from","return","yield","async","await",
    "function","end","module","using","export","struct","mutable",
    "print","for","while","if","else","elif","switch","case",
    "try","catch","except","finally","throw","raise",
    "lambda","lambda","typeof","instanceof",
    "var","let","const","int","float","str","bool","list","dict","set","tuple",
    "array","object","new","this","self","super",
    "code","program","function","loop","array","debug","compile","syntax",
    "error","bug","fix","patch","commit","push","pull","merge",
    "api","http","json","xml","sql","query","database",
    "algorithm","binary","search","sort","filter","map","reduce","recursion",
    "react","vue","angular","node","express","flask","django","fastapi",
    "import","export","require","include",
])

struct MathBridge end

function is_math_expression(ce::MathBridge, text::String)
    return any(w->w in MATH_KEYWORDS, split(text)) || any(k->occursin(k, text), keys(MATH_SYMBOLS))
end

const AR_OP_MAP = Dict(
    "زائد" => "+", "ناقص" => "-", "مضروب" => "*", "مقسوم" => "/",
    "جذر" => "sqrt", "مربع" => "^2", "مكعب" => "^3",
    "حاصل ضرب" => "*", "مطلق" => "abs", "قيمة مطلقة" => "abs",
)

function evaluate_math(ce::MathBridge, expr::String)
    try
        math_expr = _extract_math_expr(_normalize_math_expr(expr))
        isempty(math_expr) && return nothing
        return _safe_eval_arithmetic(math_expr)
    catch e
        @warn "MathBridge: evaluate_math failed: $e"
        return nothing
    end
end

const MAX_MATH_EXPR_LEN = 200
const MAX_MATH_EXPONENT = 16.0

function _normalize_math_expr(expr::String)
    ar_digits = Dict('٠'=>'0', '١'=>'1', '٢'=>'2', '٣'=>'3', '٤'=>'4', '٥'=>'5', '٦'=>'6', '٧'=>'7', '٨'=>'8', '٩'=>'9')
    e = map(c -> get(ar_digits, c, c), expr)
    for (ar, en) in AR_OP_MAP
        e = replace(e, ar => en)
    end
    for (k, v) in MATH_SYMBOLS
        e = replace(e, k => v)
    end
    return e
end

function _extract_math_expr(text::AbstractString)
    math_matches = collect(eachmatch(r"[0-9.+\-*/^%()a-z]+", text))
    isempty(math_matches) && return ""
    return strip(join((m.match for m in math_matches), ""))
end

function _safe_eval_arithmetic(raw::AbstractString)
    s = replace(String(raw), r"\s+" => "")
    (isempty(s) || length(s) > MAX_MATH_EXPR_LEN) && return nothing
    occursin(r"[^0-9.+\-*/^%()a-z]", s) && return nothing

    value, pos = _parse_math_expression(s, firstindex(s))
    (value === nothing || pos <= lastindex(s)) && return nothing
    return value
end

function _parse_math_expression(s::String, pos::Int)
    value, pos = _parse_math_term(s, pos)
    value === nothing && return nothing, pos
    while pos <= lastindex(s) && s[pos] in ('+', '-')
        op = s[pos]
        rhs, next_pos = _parse_math_term(s, pos + 1)
        rhs === nothing && return nothing, pos
        value = op == '+' ? value + rhs : value - rhs
        isfinite(value) || return nothing, pos
        pos = next_pos
    end
    return value, pos
end

function _parse_math_term(s::String, pos::Int)
    value, pos = _parse_math_power(s, pos)
    value === nothing && return nothing, pos
    while pos <= lastindex(s) && s[pos] in ('*', '/', '%')
        op = s[pos]
        rhs, next_pos = _parse_math_power(s, pos + 1)
        rhs === nothing && return nothing, pos
        if op == '*'
            value *= rhs
        elseif op == '/'
            rhs == 0.0 && return nothing, pos
            value /= rhs
        else
            rhs == 0.0 && return nothing, pos
            value = value % rhs
        end
        isfinite(value) || return nothing, pos
        pos = next_pos
    end
    return value, pos
end

function _parse_math_power(s::String, pos::Int)
    left, pos = _parse_math_unary(s, pos)
    left === nothing && return nothing, pos
    if pos <= lastindex(s) && s[pos] == '^'
        right, next_pos = _parse_math_power(s, pos + 1)
        right === nothing && return nothing, pos
        abs(right) <= MAX_MATH_EXPONENT || return nothing, pos
        try
            value = left ^ right
            (value isa Real && isfinite(value)) || return nothing, pos
            return Float64(value), next_pos
        catch
            return nothing, pos
        end
    end
    return left, pos
end

function _parse_math_unary(s::String, pos::Int)
    pos > lastindex(s) && return nothing, pos
    if s[pos] == '+'
        return _parse_math_unary(s, pos + 1)
    elseif s[pos] == '-'
        value, next_pos = _parse_math_unary(s, pos + 1)
        value === nothing && return nothing, pos
        return -value, next_pos
    end
    return _parse_math_primary(s, pos)
end

function _parse_math_primary(s::String, pos::Int)
    pos > lastindex(s) && return nothing, pos
    if s[pos] == '('
        value, next_pos = _parse_math_expression(s, pos + 1)
        value === nothing && return nothing, pos
        (next_pos <= lastindex(s) && s[next_pos] == ')') || return nothing, pos
        return value, next_pos + 1
    elseif startswith(view(s, pos:lastindex(s)), "sqrt")
        if startswith(view(s, pos:lastindex(s)), "sqrt(")
            inner_pos = pos + 5
            value, next_pos = _parse_math_expression(s, inner_pos)
            value === nothing && return nothing, pos
            (next_pos <= lastindex(s) && s[next_pos] == ')') || return nothing, pos
            value < 0.0 && return nothing, pos
            return sqrt(value), next_pos + 1
        else
            inner_pos = pos + 4
            value, next_pos = _parse_math_unary(s, inner_pos)
            value === nothing && return nothing, pos
            value < 0.0 && return nothing, pos
            return sqrt(value), next_pos
        end
    elseif startswith(view(s, pos:lastindex(s)), "abs")
        if startswith(view(s, pos:lastindex(s)), "abs(")
            inner_pos = pos + 4
            value, next_pos = _parse_math_expression(s, inner_pos)
            value === nothing && return nothing, pos
            (next_pos <= lastindex(s) && s[next_pos] == ')') || return nothing, pos
            return abs(value), next_pos + 1
        else
            inner_pos = pos + 3
            value, next_pos = _parse_math_unary(s, inner_pos)
            value === nothing && return nothing, pos
            return abs(value), next_pos
        end
    end
    return _parse_math_number(s, pos)
end

function _parse_math_number(s::String, pos::Int)
    start_pos = pos
    has_digit = false
    has_dot = false
    while pos <= lastindex(s)
        c = s[pos]
        if isdigit(c)
            has_digit = true
            pos += 1
        elseif c == '.' && !has_dot
            has_dot = true
            pos += 1
        else
            break
        end
    end
    has_digit || return nothing, start_pos
    try
        value = parse(Float64, s[start_pos:pos - 1])
        return isfinite(value) ? value : nothing, pos
    catch
        return nothing, start_pos
    end
end

function is_code_prompt(text::String)
    lower_text = lowercase(text)
    words = Set(split(lower_text))
    command_only_ar = Set(["اكتب", "أكتب", "أنشئ", "صمم", "نفذ", "نفّذ", "اعمل"])
    strong_code_ar = Set(["كود", "برنامج", "برمجة", "دالة", "دوال", "كلاس", "صنف",
                          "بايثون", "جوليا", "جافا", "جافاسكريبت", "خوارزمية",
                          "خوارزميات", "متغير", "مصفوفة", "حلقة", "شرط"])
    any(w -> w in strong_code_ar, words) && return true
    for w in words
        w in command_only_ar && continue
        w in CODE_KEYWORDS_AR && return true
        w in CODE_KEYWORDS_EN && return true
    end
    startswith(lower_text, "def ") && return true
    startswith(lower_text, "class ") && return true
    startswith(lower_text, "import ") && return true
    startswith(lower_text, "from ") && occursin("import", lower_text) && return true
    return false
end

function detect_mode(text::String)
    is_code_prompt(text) && return "code"
    is_math_expression(MathBridge(), text) && return "math"
    return "text"
end

function validate_code_syntax(code::String)
    try
        parsed = Meta.parse(code, 1; raise=true)
        if parsed === nothing
            return (true, "")
        end
        return (true, "")
    catch e
        return (false, string(e))
    end
end

function extract_code_blocks(text::String)
    blocks = String[]
    pattern = r"```(\w*)\n(.*?)```"s
    for m in eachmatch(pattern, text)
        push!(blocks, strip(m.captures[2]))
    end
    return blocks
end

function strip_code_blocks(text::String)
    return replace(text, r"```[\s\S]*?```"s => " ")
end

end
