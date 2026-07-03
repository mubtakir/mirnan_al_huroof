"""
CodeEngine كامل — توليد كود Python/Julia عبر تحليل نحوي + مصفوفة اقتران K_code.
TokenType، VALID_NEXT، tokenize، build_K_code، توليد بالقوالب، تحقق نحوي، حفظ/تحميل.
"""
module CodeEngineModule
using SparseArrays, Random, LinearAlgebra, JSON

export CodeEngine, CodeVocabulary, CodePhaseVector, TokenType, VALID_NEXT,
       tokenize_code, build_K_code, validate_syntax, compile_check,
       save_code_model, load_code_model, generate_code, code_phase_vector,
       extract_code_name, strip_inline_code

@enum TokenType::Int32 begin
    C_KEYWORD; C_IDENTIFIER; C_OPERATOR; C_LITERAL_NUM; C_LITERAL_STR
    C_BRACKET_OPEN; C_BRACKET_CLOSE; C_PAREN_OPEN; C_PAREN_CLOSE
    C_INDENT; C_DEDENT; C_NEWLINE; C_COLON; C_DOT; C_COMMA; C_EQUALS; C_ANNOTATION
end

const PY_KEYWORDS = Set(["False","None","True","and","as","assert","async","await","break","class","continue",
    "def","del","elif","else","except","finally","for","from","global","if","import","in","is",
    "lambda","nonlocal","not","or","pass","raise","return","try","while","with","yield"])
const JL_KEYWORDS = Set(["function","end","module","using","export","if","else","elseif","for","while",
    "return","break","continue","struct","mutable","abstract","type","const","global","local","macro",
    "do","begin","let","quote","try","catch","finally","throw"])

const VALID_NEXT = Dict{TokenType,Set{TokenType}}(
    C_KEYWORD=>Set([C_IDENTIFIER,C_KEYWORD,C_PAREN_OPEN,C_LITERAL_NUM,C_LITERAL_STR,C_COLON,C_OPERATOR,C_NEWLINE]),
    C_IDENTIFIER=>Set([C_OPERATOR,C_PAREN_OPEN,C_PAREN_CLOSE,C_NEWLINE,C_COMMA,C_COLON,C_EQUALS,C_DOT,C_BRACKET_OPEN,C_KEYWORD,C_IDENTIFIER]),
    C_OPERATOR=>Set([C_IDENTIFIER,C_LITERAL_NUM,C_LITERAL_STR,C_PAREN_OPEN,C_OPERATOR]),
    C_LITERAL_NUM=>Set([C_OPERATOR,C_NEWLINE,C_COMMA,C_BRACKET_CLOSE,C_PAREN_CLOSE,C_COLON]),
    C_LITERAL_STR=>Set([C_OPERATOR,C_NEWLINE,C_COMMA,C_BRACKET_CLOSE,C_PAREN_CLOSE]),
    C_PAREN_OPEN=>Set([C_IDENTIFIER,C_LITERAL_NUM,C_LITERAL_STR,C_KEYWORD,C_PAREN_CLOSE,C_OPERATOR]),
    C_PAREN_CLOSE=>Set([C_OPERATOR,C_NEWLINE,C_COMMA,C_COLON,C_PAREN_CLOSE,C_BRACKET_CLOSE,C_KEYWORD]),
    C_COLON=>Set([C_NEWLINE,C_INDENT]),
    C_NEWLINE=>Set([C_KEYWORD,C_IDENTIFIER,C_DEDENT,C_INDENT,C_LITERAL_NUM,C_LITERAL_STR]),
    C_INDENT=>Set([C_KEYWORD,C_IDENTIFIER,C_DEDENT,C_LITERAL_NUM,C_LITERAL_STR]),
    C_DEDENT=>Set([C_KEYWORD,C_IDENTIFIER,C_DEDENT,C_NEWLINE]),
    C_EQUALS=>Set([C_IDENTIFIER,C_LITERAL_NUM,C_LITERAL_STR,C_PAREN_OPEN,C_BRACKET_OPEN,C_OPERATOR]),
    C_COMMA=>Set([C_IDENTIFIER,C_LITERAL_NUM,C_LITERAL_STR,C_PAREN_OPEN,C_BRACKET_OPEN]),
    C_DOT=>Set([C_IDENTIFIER]),
    C_BRACKET_OPEN=>Set([C_IDENTIFIER,C_LITERAL_NUM,C_LITERAL_STR,C_BRACKET_CLOSE,C_KEYWORD,C_OPERATOR]),
    C_BRACKET_CLOSE=>Set([C_OPERATOR,C_NEWLINE,C_COMMA,C_BRACKET_CLOSE,C_PAREN_CLOSE,C_COLON]),
    C_ANNOTATION=>Set([C_IDENTIFIER,C_KEYWORD,C_NEWLINE]),
)

function tokenize_code(source::String)
    tokens = Tuple{String,TokenType}[]
    s = replace(source, ";"=>"\n")
    for line in split(s, '\n')
        l = strip(line); isempty(l) && continue
        if startswith(l, "#"); push!(tokens,(l,C_ANNOTATION)); push!(tokens,("\n",C_NEWLINE)); continue; end
        words = split(l)
        for w in words
            w in PY_KEYWORDS && (push!(tokens,(w,C_KEYWORD)); continue)
            w in JL_KEYWORDS && (push!(tokens,(w,C_KEYWORD)); continue)
            w == "(" && (push!(tokens,(w,C_PAREN_OPEN)); continue)
            w == ")" && (push!(tokens,(w,C_PAREN_CLOSE)); continue)
            w == "[" && (push!(tokens,(w,C_BRACKET_OPEN)); continue)
            w == "]" && (push!(tokens,(w,C_BRACKET_CLOSE)); continue)
            w == ":" && (push!(tokens,(w,C_COLON)); continue)
            w == "=" && (push!(tokens,(w,C_EQUALS)); continue)
            w == "," && (push!(tokens,(w,C_COMMA)); continue)
            w == "." && (push!(tokens,(w,C_DOT)); continue)
            w in Set(["+","-","*","/","%","**","//","==","!=","<",">","<=",">=","+=","-=","*=","/="]) && (push!(tokens,(w,C_OPERATOR)); continue)
            try; parse(Float64,w); push!(tokens,(w,C_LITERAL_NUM)); catch e; @debug "Code engine: parse float failed for '$w': $e"; push!(tokens,(w,C_IDENTIFIER)); end
        end
        push!(tokens,("\n",C_NEWLINE))
    end
    return tokens
end

mutable struct CodeVocabulary
    t2id::Dict{Tuple{String,TokenType},Int}
    id2t::Dict{Int,Tuple{String,TokenType}}
    next::Int
end
CodeVocabulary() = CodeVocabulary(Dict(), Dict(), 1)
function Base.get!(cv::CodeVocabulary, k::Tuple{String,TokenType})
    haskey(cv.t2id, k) && return cv.t2id[k]
    cv.t2id[k] = cv.next
    cv.id2t[cv.next] = k
    cv.next += 1
    return cv.next - 1
end

function save_code_vocab(cv::CodeVocabulary, path::String)
    data = Dict{String,Any}()
    for ((text, tt), id) in cv.t2id
        data[string(tt)*":"*text] = id
    end
    open(path, "w") do io; JSON.print(io, data); end
end

function load_code_vocab(path::String)
    cv = CodeVocabulary()
    isfile(path) || return cv
    data = JSON.parsefile(path)
    for (k, v) in data
        colon_idx = findfirst(':', k)
        if colon_idx !== nothing
            tt_str = k[1:colon_idx-1]
            text = k[colon_idx+1:end]
            tt = try
                getproperty(CodeEngineModule, Symbol(tt_str))
            catch e
                @debug "Code engine: failed to get TokenType for '$tt_str': $e"
                nothing
            end
            if tt !== nothing && tt isa TokenType
                cv.t2id[(text, tt)] = v
                cv.id2t[v] = (text, tt)
                if v >= cv.next; cv.next = v + 1; end
            end
        end
    end
    return cv
end

mutable struct CodePhaseVector
    tpdim::Int
    tpvec::Dict{TokenType,Vector{Float64}}
end
function CodePhaseVector()
    tpvec = Dict{TokenType,Vector{Float64}}()
    D = 64
    for tt in instances(TokenType)
        seed = UInt64(Int(tt) * 42)
        rng = MersenneTwister(seed)
        v = randn(rng, D)
        v ./= norm(v)
        tpvec[tt] = v
    end
    return CodePhaseVector(D, tpvec)
end

mutable struct CodeEngine
    code_vocab::Union{CodeVocabulary,Nothing}
    K_code::Any
    cpv::Union{CodePhaseVector,Nothing}
    feedback_log::Vector{Tuple{String,Bool}}
end
CodeEngine(;code_vocab=nothing, K_code=nothing, cpv=nothing) =
    CodeEngine(code_vocab, K_code, cpv, Tuple{String,Bool}[])

function code_phase_vector(ce::CodeEngine, token_text::String)
    tokens = tokenize_code(token_text)
    isempty(tokens) && return zeros(Float64, 64)
    ce.cpv === nothing && (ce.cpv = CodePhaseVector())
    cpv = ce.cpv
    result = zeros(Float64, cpv.tpdim)
    for (text, tt) in tokens
        haskey(cpv.tpvec, tt) || continue
        v = cpv.tpvec[tt]
        if tt == C_IDENTIFIER
            scale = max(length(text) * 0.3, 0.5)
            result .+= scale .* v
        else
            result .+= v
        end
    end
    nrm = norm(result)
    nrm > 1e-10 && (result ./= nrm)
    return result
end

function build_K_code(texts::Vector{String})
    cv = CodeVocabulary()
    cpv = CodePhaseVector()
    tokens = Tuple{String,TokenType}[]
    for (idx, t) in enumerate(texts)
        for tok in tokenize_code(t)
            get!(cv, tok)
            push!(tokens, tok)
        end
        if idx % 500 == 0
            println("      K_code tokenize: $idx / $(length(texts)) blocks, $(length(tokens)) tokens")
            flush(stdout)
        end
    end
    V = cv.next - 1
    V == 0 && return cv, spzeros(0, 0), cpv

    counts = Dict{Tuple{Int,Int},Float64}()
    for i in 1:length(tokens)-1
        a = cv.t2id[tokens[i]]
        b = cv.t2id[tokens[i+1]]
        if haskey(VALID_NEXT, tokens[i][2]) && tokens[i+1][2] in VALID_NEXT[tokens[i][2]]
            key = (a, b)
            counts[key] = get(counts, key, 0.0) + 1.0
        end
        if i % 500_000 == 0
            println("      K_code pairs: $i / $(length(tokens)-1), $(length(counts)) links")
            flush(stdout)
        end
    end
    if isempty(counts)
        return cv, spzeros(V, V), cpv
    end
    rows = Vector{Int}(undef, length(counts))
    cols = Vector{Int}(undef, length(counts))
    vals = Vector{Float64}(undef, length(counts))
    for (idx, (key, val)) in enumerate(counts)
        rows[idx] = key[1]
        cols[idx] = key[2]
        vals[idx] = val
    end
    K = sparse(rows, cols, vals, V, V)
    return cv, K, cpv
end

function validate_syntax(tokens::Vector{Tuple{String,TokenType}})
    isempty(tokens) && return true
    filtered = [(t, tt) for (t, tt) in tokens if tt != C_INDENT && tt != C_DEDENT]
    for i in 1:length(filtered)-1
        curr_type = filtered[i][2]
        next_type = filtered[i+1][2]
        allowed = get(VALID_NEXT, curr_type, Set{TokenType}())
        next_type in allowed || return false
    end
    return true
end

function compile_check(source::String)
    s = strip(source)
    isempty(s) && return (false, "empty")
    lines = split(s, '\n')
    has_structure = false
    for line in lines
        l = strip(line)
        isempty(l) && continue
        if occursin("def ", l) || occursin("class ", l) || occursin("if ", l) ||
           occursin("for ", l) || occursin("while ", l) || occursin("import ", l) ||
           occursin("return ", l) || occursin("print(", l)
            has_structure = true
            break
        end
        if occursin('=', l) && !startswith(l, "import") && !startswith(l, "from")
            has_structure = true
            break
        end
    end
    if !has_structure
        return (false, "no recognizable code structure")
    end
    bad_chars = ['{', '}', ';']
    for c in bad_chars
        if c in s
            return (false, "unexpected character: $c")
        end
    end
    return (true, "")
end

function infer_type(text::String)
    text in PY_KEYWORDS && return C_KEYWORD
    text in ("(", "[") && return text == "(" ? C_PAREN_OPEN : C_BRACKET_OPEN
    text in (")", "]") && return text == ")" ? C_PAREN_CLOSE : C_BRACKET_CLOSE
    text == ":" && return C_COLON
    text == "=" && return C_EQUALS
    text == "," && return C_COMMA
    text == "." && return C_DOT
    text == "\n" && return C_NEWLINE
    text in ("+","-","*","/","%","==","!=","<",">","<=",">=","and","or","not") && return C_OPERATOR
    try; parse(Float64, text); return C_LITERAL_NUM; catch e; @debug "Type inference: not a number: $text"; end
    return C_IDENTIFIER
end

function suggest_next(ce::CodeEngine, tokens::Vector{Tuple{String,TokenType}}; top_k=15)
    isempty(tokens) && return Tuple{String,Float64}[]
    last_tok = tokens[end]
    last_type = last_tok[2]
    candidates = Tuple{String,Float64}[]

    if ce.K_code !== nothing && ce.code_vocab !== nothing
        wid = get(ce.code_vocab.t2id, last_tok, 0)
        if wid > 0 && wid <= size(ce.K_code, 1)
            row = ce.K_code[wid, :]
            p = sortperm(row; rev=true)
            for tid in p
                row[tid] <= 1e-6 && break
                nt = ce.code_vocab.id2t[tid]
                next_type = nt[2]
                allowed = get(VALID_NEXT, last_type, Set{TokenType}())
                if next_type in allowed
                    push!(candidates, (nt[1], row[tid]))
                end
            end
        end
    end

    fallback = Dict{TokenType,Vector{String}}(
        C_KEYWORD => ["def", "class", "return", "if", "for", "import", "print"],
        C_COLON => ["\n"],
        C_NEWLINE => ["def", "class", "if", "for", "return", "print", "x"],
        C_EQUALS => ["x", "0", "\"\"", "[]", "{}"],
        C_PAREN_OPEN => ["x", "0", "\"\"", "self"],
    )
    fallback_scores = Dict{TokenType,Float64}(
        C_KEYWORD => 0.1, C_COLON => 0.2, C_NEWLINE => 0.15, C_EQUALS => 0.1,
    )

    if isempty(candidates) && haskey(fallback, last_type)
        for txt in fallback[last_type]
            push!(candidates, (txt, get(fallback_scores, last_type, 0.05)))
        end
    end

    sort!(candidates; by=x -> -x[2])
    return candidates[1:min(top_k, end)]
end

function generate_code(ce::CodeEngine, prompt::String; max_tokens::Int=30)
    ce.code_vocab === nothing && return "# [CodeEngine: no code vocab loaded]\npass\n"

    lower_prompt = lowercase(prompt)

    if occursin("fibonacci", lower_prompt) || occursin("فيبوناتشي", lower_prompt)
        return "def fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    fib = [0, 1]\n    while len(fib) < n:\n        fib.append(fib[-1] + fib[-2])\n    return fib\n"
    elseif occursin("factorial", lower_prompt) || occursin("مضروب", lower_prompt)
        return "def factorial(n):\n    if n < 0:\n        return None\n    res = 1\n    for i in range(1, n + 1):\n        res *= i\n    return res\n"
    elseif occursin("sort", lower_prompt) || occursin("ترتيب", lower_prompt) || occursin("فرز", lower_prompt)
        return "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\n"
    elseif occursin("prime", lower_prompt) || occursin("أولي", lower_prompt) || occursin("اولي", lower_prompt)
        return "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n"
    elseif occursin("function", lower_prompt) || occursin("دالة", lower_prompt) || occursin("def", lower_prompt)
        return _gen_function(prompt)
    elseif occursin("loop", lower_prompt) || occursin("حلقة", lower_prompt) || occursin("for", lower_prompt) || occursin("while", lower_prompt)
        return _gen_loop(prompt)
    elseif occursin("class", lower_prompt) || occursin("صنف", lower_prompt) || occursin("كلاس", lower_prompt)
        return _gen_class(prompt)
    elseif occursin("condition", lower_prompt) || occursin("شرط", lower_prompt) || occursin("if", lower_prompt)
        return _gen_condition(prompt)
    elseif occursin("import", lower_prompt) || occursin("from", lower_prompt) || occursin("استيراد", lower_prompt)
        return _gen_import(prompt)
    else
        return _gen_resonant(ce, prompt, max_tokens)
    end
end

function extract_code_name(prompt::String)
    m = match(r"(?:اسمه|يسمى|named|called|اسم)\s+(\w+)", prompt)
    m !== nothing && return m.captures[1]
    m = match(r"`(\w+)`", prompt)
    m !== nothing && return m.captures[1]
    m = match(r"دالة\s+(\w+)", prompt)
    m !== nothing && return m.captures[1]
    return nothing
end

function _gen_function(prompt::String)
    name = something(extract_code_name(prompt), "my_func")
    return "def $(name)():\n    pass\n"
end

function _gen_loop(prompt::String)
    if occursin("for", lowercase(prompt))
        return "for i in range(10):\n    pass\n"
    end
    return "while True:\n    pass\n"
end

function _gen_class(prompt::String)
    name = something(extract_code_name(prompt), "MyClass")
    return "class $(name):\n    pass\n"
end

function _gen_condition(prompt::String)
    return "if x > 0:\n    pass\nelse:\n    pass\n"
end

function _gen_import(prompt::String)
    return "import math\n"
end

function _gen_resonant(ce::CodeEngine, prompt::String, max_tokens::Int)
    pl = lowercase(prompt)
    starter = "x"
    starter_map = Dict(
        "print" => "print", "hello" => "print",
        "import" => "import", "from" => "from",
        "return" => "return", "yield" => "yield",
        "binary" => "def", "search" => "def", "sort" => "def",
        "function" => "def", "class" => "class",
        "factorial" => "x", "fib" => "x", "sum" => "x",
        "max" => "x", "min" => "x", "average" => "x",
    )
    for (kw, st) in starter_map
        if occursin(kw, pl)
            starter = st; break
        end
    end

    if starter == "def"
        name = "f"
        skip_words = Set(["a","an","the","that","function","def","define","new","create","write","make","sort","search","find","binary","calculate","compute"])
        for w in split(prompt)
            if !(lowercase(w) in skip_words)
                name = w; break
            end
        end
        return "def $(name)():\n    pass\n"
    end

    if starter == "class"
        name = "MyClass"
        skip_words = Set(["a","an","the","that","class","define","new","create","write","make"])
        for w in split(prompt)
            if !(lowercase(w) in skip_words)
                name = uppercasefirst(w); break
            end
        end
        return "class $(name):\n    pass\n"
    end

    if starter == "print"
        return "print(\"hello\")\n"
    end

    if starter in ("import", "from")
        return "import math\n"
    end

    if starter == "return"
        return "return x + 1\n"
    end

    result_tokens = Tuple{String,TokenType}[(starter, infer_type(starter))]
    prev_type = result_tokens[end][2]
    repeat_count = 0
    for _ in 1:max_tokens-1
        cands = suggest_next(ce, result_tokens)
        isempty(cands) && break
        chosen = nothing
        for i in 1:min(5, length(cands))
            text = cands[i][1]
            ttype = infer_type(text)
            trial = vcat(result_tokens, [(text, ttype)])
            if validate_syntax(trial)
                chosen = (text, ttype); break
            end
        end
        chosen === nothing && break
        text, ttype = chosen
        if ttype == prev_type
            repeat_count += 1
        else
            repeat_count = 0
            prev_type = ttype
        end
        repeat_count > 2 && break
        push!(result_tokens, chosen)
        ttype == C_NEWLINE && break
    end

    source = _tokens_to_source(result_tokens)
    ok, _ = compile_check(source)
    return ok ? source : ""
end

function _tokens_to_source(tokens::Vector{Tuple{String,TokenType}})
    parts = String[]
    indent_level = 0
    for (tstr, tt) in tokens
        if tt == C_NEWLINE
            push!(parts, "\n")
            for _ in 1:indent_level; push!(parts, "    "); end
        elseif tt == C_INDENT
            indent_level += 1
            push!(parts, "\n")
            for _ in 1:indent_level; push!(parts, "    "); end
        elseif tt == C_DEDENT
            indent_level = max(0, indent_level - 1)
        elseif tt in (C_OPERATOR, C_COMMA, C_DOT)
            push!(parts, tstr)
        elseif tt == C_COLON
            push!(parts, ":")
        elseif tt in (C_PAREN_OPEN, C_BRACKET_OPEN)
            push!(parts, tstr)
        elseif tt in (C_PAREN_CLOSE, C_BRACKET_CLOSE)
            push!(parts, tstr)
        elseif tt == C_EQUALS
            push!(parts, " = ")
        else
            push!(parts, tstr)
        end
    end
    source = join(parts, " ")
    source = replace(source, r"\s*\n\s*" => "\n")
    source = replace(source, r"\n +" => "\n")
    source = replace(source, r"(\S):" => s"\1:")
    source = replace(source, r"\s*\(\s*" => "(")
    source = replace(source, r"\s*\)" => ")")
    source = replace(source, r" ," => ",")
    return source
end

function code_feedback!(ce::CodeEngine, generated::String)
    isempty(strip(generated)) && return
    ok, msg = compile_check(generated)
    push!(ce.feedback_log, (generated, ok))
    if length(ce.feedback_log) > 100
        popfirst!(ce.feedback_log)
    end
    return ok
end

function apply_code_feedback!(ce::CodeEngine, gen::Any)
    if isempty(ce.feedback_log) || ce.K_code === nothing
        return
    end
    last_gen, last_ok = ce.feedback_log[end]
    tokens = tokenize_code(last_gen)
    isempty(tokens) && return

    for i in 1:length(tokens)-1
        a = get(ce.code_vocab === nothing ? Dict() : ce.code_vocab.t2id, tokens[i], 0)
        b = get(ce.code_vocab === nothing ? Dict() : ce.code_vocab.t2id, tokens[i+1], 0)
        if a > 0 && b > 0 && a <= size(ce.K_code, 1) && b <= size(ce.K_code, 2)
            if last_ok
                ce.K_code[a, b] = min(ce.K_code[a, b] + 0.5, 100.0)
            else
                ce.K_code[a, b] = max(ce.K_code[a, b] * 0.5, 0.0)
            end
        end
    end
end

function save_code_model(ce::CodeEngine, out_dir::String)
    mkpath(out_dir)
    if ce.code_vocab !== nothing
        save_code_vocab(ce.code_vocab, joinpath(out_dir, "code_vocab.json"))
    end
    if ce.K_code !== nothing && ce.K_code isa AbstractSparseMatrix && length(ce.K_code.nzval) > 0
        open(joinpath(out_dir, "K_code.dat"), "w") do io
            I, J, V = findnz(ce.K_code)
            write(io, "SPARSE_COO_V1\n")
            write(io, Int32(size(ce.K_code, 1)), Int32(size(ce.K_code, 2)), Int32(length(V)))
            for k in eachindex(V)
                write(io, Int32(I[k]))
                write(io, Int32(J[k]))
                write(io, Float64(V[k]))
            end
        end
    end
    return true
end

function load_code_model(in_dir::String)
    cv = load_code_vocab(joinpath(in_dir, "code_vocab.json"))
    n_vocab = max(cv.next - 1, 1)
    K_code = spzeros(n_vocab, n_vocab)
    kf = joinpath(in_dir, "K_code.dat")
    if isfile(kf)
        try
            open(kf, "r") do io
                header = readline(io)
                if header == "SPARSE_COO_V1"
                    m = Int(read(io, Int32))
                    n = Int(read(io, Int32))
                    nnz = Int(read(io, Int32))
                    rows = Vector{Int}(undef, nnz)
                    cols = Vector{Int}(undef, nnz)
                    vals = Vector{Float64}(undef, nnz)
                    for k in 1:nnz
                        rows[k] = Int(read(io, Int32))
                        cols[k] = Int(read(io, Int32))
                        vals[k] = read(io, Float64)
                    end
                    K_code = sparse(rows, cols, vals, max(m, n_vocab), max(n, n_vocab))
                else
                    seekstart(io)
                    n = Int(read(io, Int32))
                    nz_ind = Int[]
                    nz_val = Float64[]
                    for _ in 1:n
                        push!(nz_ind, Int(read(io, Int32)))
                        push!(nz_val, read(io, Float64))
                    end
                    # Legacy fallback: old files did not store both row and column.
                    K_code = sparse(nz_ind, ones(Int, length(nz_ind)), nz_val, n_vocab, n_vocab)
                end
            end
        catch e
            @warn "Failed to load K_code: $e"
        end
    end
    return CodeEngine(code_vocab=cv, K_code=K_code, cpv=CodePhaseVector())
end

function strip_inline_code(text::String)
    text = replace(text, r"```[\s\S]*?```"s => " ")
    text = replace(text, r"`([^`]+)`" => s"\1")
    return text
end

end
