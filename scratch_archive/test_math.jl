# test_math.jl
using Test

const MATH_SYMBOLS = Dict("+"=>"+","-"=>"-","×"=>"*","÷"=>"/","√"=>"sqrt","^"=>"^","π"=>"3.14159","∞"=>"Inf","∑"=>"sum","∏"=>"prod")

function evaluate_math(expr::String)
    try
        ar_digits = Dict('٠'=>'0', '١'=>'1', '٢'=>'2', '٣'=>'3', '٤'=>'4', '٥'=>'5', '٦'=>'6', '٧'=>'7', '٨'=>'8', '٩'=>'9')
        e = map(c -> get(ar_digits, c, c), expr)
        
        # Replace symbols
        for (k, v) in MATH_SYMBOLS
            e = replace(e, k => v)
        end
        
        # Extract math characters
        math_matches = collect(eachmatch(r"[0-9.+\-*/^%()]+", e))
        if !isempty(math_matches)
            math_expr = join([m.match for m in math_matches], " ")
            math_expr = strip(math_expr)
            println("Cleaned math expression: '", math_expr, "'")
            parsed = Meta.parse(math_expr, 1; raise=false)
            if parsed !== nothing && (parsed[1] isa Expr || parsed[1] isa Number)
                return eval(parsed[1])
            end
        end
        return nothing
    catch e
        println("Error during math evaluation: ", e)
        return nothing
    end
end

println("Result: ", evaluate_math("ما مجموع 1 + 2"))
println("Result 2: ", evaluate_math("ما حاصل ضرب 3 × 5"))
println("Result 3: ", evaluate_math("ما قيمة ٢ + ٤"))
