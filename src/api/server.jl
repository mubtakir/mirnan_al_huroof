"""
خادم API - خادم HTTP لمشروع مرنان الجديد
"""

module APIServer

using HTTP
using JSON
using SparseArrays
using ..Physics: MirnanGenerator, generate!

export start_server, stop_server, handle_request, set_generator!

# ═══════════════════════════════════════════════════════
# إعدادات الخادم
# ═══════════════════════════════════════════════════════

mutable struct ServerConfig
    host::String
    port::Int
    max_connections::Int
    timeout::Int
end

function ServerConfig(;
    host::String = "127.0.0.1",
    port::Int = 8000,
    max_connections::Int = 100,
    timeout::Int = 60
)
    ServerConfig(host, port, max_connections, timeout)
end

mutable struct ServerState
    running::Bool
    server::Any
    request_count::Int
    error_count::Int
    generator::Any
    generator_lock::ReentrantLock
end

function ServerState()
    ServerState(false, nothing, 0, 0, nothing, ReentrantLock())
end

# ═══════════════════════════════════════════════════════
# الخادم
# ═══════════════════════════════════════════════════════

const GLOBAL_STATE = ServerState()

function start_server(config::ServerConfig = ServerConfig(); generator=nothing)
    println("Starting Mirnan API Server on $(config.host):$(config.port)...")
    generator !== nothing && set_generator!(generator)
    _ensure_generator!()

    router = HTTP.Router()

    HTTP.register!(router, "GET", "/", handle_root)
    HTTP.register!(router, "GET", "/health", handle_health)
    HTTP.register!(router, "POST", "/analyze", handle_analyze)
    HTTP.register!(router, "POST", "/generate", handle_generate)
    HTTP.register!(router, "POST", "/semantics", handle_semantics)
    HTTP.register!(router, "POST", "/grammar", handle_grammar)
    HTTP.register!(router, "POST", "/preprocess", handle_preprocess)

    server = HTTP.serve!(router, config.host, config.port;
        verbose = false,
        on_listen = () -> println("Server listening on http://$(config.host):$(config.port)")
    )

    GLOBAL_STATE.running = true
    GLOBAL_STATE.server = server

    println("✅ Server started successfully!")
    println("   Endpoints:")
    println("   - GET  /           - API Info")
    println("   - GET  /health     - Health Check")
    println("   - POST /analyze    - Full Analysis")
    println("   - POST /generate   - Text Generation")
    println("   - POST /semantics  - Semantic Analysis")
    println("   - POST /grammar    - Grammar Analysis")
    println("   - POST /preprocess - Text Preprocessing")

    return server
end

function start_server(generator; host::String = "127.0.0.1", port::Int = 8000,
                      max_connections::Int = 100, timeout::Int = 60)
    config = ServerConfig(; host=host, port=port, max_connections=max_connections, timeout=timeout)
    return start_server(config; generator=generator)
end

function stop_server()
    if GLOBAL_STATE.server !== nothing
        HTTP.close(GLOBAL_STATE.server)
        GLOBAL_STATE.running = false
        GLOBAL_STATE.server = nothing
        println("Server stopped.")
    end
end

function set_generator!(generator)
    lock(GLOBAL_STATE.generator_lock) do
        GLOBAL_STATE.generator = generator
    end
    return generator
end

function handle_request(req::HTTP.Request)
    method = String(req.method)
    path = String(HTTP.URI(req.target).path)

    if method == "GET" && path == "/"
        return handle_root(req)
    elseif method == "GET" && path == "/health"
        return handle_health(req)
    elseif method == "POST" && path == "/analyze"
        return handle_analyze(req)
    elseif method == "POST" && path == "/generate"
        return handle_generate(req)
    elseif method == "POST" && path == "/semantics"
        return handle_semantics(req)
    elseif method == "POST" && path == "/grammar"
        return handle_grammar(req)
    elseif method == "POST" && path == "/preprocess"
        return handle_preprocess(req)
    end

    return HTTP.Response(404, ["Content-Type" => "application/json"],
        body = JSON.json(Dict("error" => "Not found")))
end

# ═══════════════════════════════════════════════════════
# معالجات الطلبات
# ═══════════════════════════════════════════════════════

function handle_root(req::HTTP.Request)
    response = Dict(
        "name" => "Mirnan New API",
        "version" => "0.8.0",
        "description" => "Physical-dynamical language model",
        "endpoints" => [
            "GET /",
            "GET /health",
            "POST /analyze",
            "POST /generate",
            "POST /semantics",
            "POST /grammar",
            "POST /preprocess"
        ]
    )
    return HTTP.Response(200, ["Content-Type" => "application/json"],
        body = JSON.json(response))
end

function handle_health(req::HTTP.Request)
    response = Dict(
        "status" => "healthy",
        "running" => GLOBAL_STATE.running,
        "requests" => GLOBAL_STATE.request_count,
        "errors" => GLOBAL_STATE.error_count
    )
    return HTTP.Response(200, ["Content-Type" => "application/json"],
        body = JSON.json(response))
end

function handle_analyze(req::HTTP.Request)
    try
        body = JSON.parse(String(req.body))
        text = get(body, "text", "")

        if isempty(text)
            return HTTP.Response(400, ["Content-Type" => "application/json"],
                body = JSON.json(Dict("error" => "Text is required")))
        end

        GLOBAL_STATE.request_count += 1

        response = Dict(
            "text" => text,
            "preprocessed" => _preprocess_text(text),
            "semantics" => _analyze_semantics(text),
            "grammar" => _analyze_grammar(text),
            "length" => length(text)
        )

        return HTTP.Response(200, ["Content-Type" => "application/json"],
            body = JSON.json(response))
    catch e
        GLOBAL_STATE.error_count += 1
        @warn "API: analyze failed: $e"
        return HTTP.Response(500, ["Content-Type" => "application/json"],
            body = JSON.json(Dict("error" => string(e))))
    end
end

function handle_generate(req::HTTP.Request)
    try
        body = JSON.parse(String(req.body))
        prompt = get(body, "prompt", "")
        max_tokens = get(body, "max_tokens", 100)

        GLOBAL_STATE.request_count += 1

        response = Dict(
            "prompt" => prompt,
            "generated" => _generate_text(prompt, max_tokens),
            "max_tokens" => max_tokens
        )

        return HTTP.Response(200, ["Content-Type" => "application/json"],
            body = JSON.json(response))
    catch e
        GLOBAL_STATE.error_count += 1
        @warn "API: generate failed: $e"
        return HTTP.Response(500, ["Content-Type" => "application/json"],
            body = JSON.json(Dict("error" => string(e))))
    end
end

function handle_semantics(req::HTTP.Request)
    try
        body = JSON.parse(String(req.body))
        text = get(body, "text", "")

        GLOBAL_STATE.request_count += 1

        response = Dict(
            "text" => text,
            "semantics" => _analyze_semantics(text)
        )

        return HTTP.Response(200, ["Content-Type" => "application/json"],
            body = JSON.json(response))
    catch e
        GLOBAL_STATE.error_count += 1
        @warn "API: semantics failed: $e"
        return HTTP.Response(500, ["Content-Type" => "application/json"],
            body = JSON.json(Dict("error" => string(e))))
    end
end

function handle_grammar(req::HTTP.Request)
    try
        body = JSON.parse(String(req.body))
        text = get(body, "text", "")

        GLOBAL_STATE.request_count += 1

        response = Dict(
            "text" => text,
            "grammar" => _analyze_grammar(text)
        )

        return HTTP.Response(200, ["Content-Type" => "application/json"],
            body = JSON.json(response))
    catch e
        GLOBAL_STATE.error_count += 1
        @warn "API: grammar failed: $e"
        return HTTP.Response(500, ["Content-Type" => "application/json"],
            body = JSON.json(Dict("error" => string(e))))
    end
end

function handle_preprocess(req::HTTP.Request)
    try
        body = JSON.parse(String(req.body))
        text = get(body, "text", "")

        GLOBAL_STATE.request_count += 1

        response = Dict(
            "original" => text,
            "preprocessed" => _preprocess_text(text)
        )

        return HTTP.Response(200, ["Content-Type" => "application/json"],
            body = JSON.json(response))
    catch e
        GLOBAL_STATE.error_count += 1
        @warn "API: preprocess failed: $e"
        return HTTP.Response(500, ["Content-Type" => "application/json"],
            body = JSON.json(Dict("error" => string(e))))
    end
end

# ═══════════════════════════════════════════════════════
# دوال المعالجة
# ═══════════════════════════════════════════════════════

function _preprocess_text(text::AbstractString)::String
    result = String(text)
    result = replace(result, r"\s+" => " ")
    return strip(result)
end

function _analyze_semantics(text::AbstractString)::Dict{String,Any}
    words = split(text)
    return Dict(
        "word_count" => length(words),
        "unique_words" => length(unique(words)),
        "avg_word_length" => isempty(words) ? 0 : sum(length.(words)) / length(words)
    )
end

function _analyze_grammar(text::AbstractString)::Dict{String,Any}
    sentences = split(text, ['.', '!', '؟', '?', '؛', ';'])
    sentences = filter(!isempty, strip.(sentences))

    return Dict(
        "sentence_count" => length(sentences),
        "avg_sentence_length" => isempty(sentences) ? 0 : sum(length.(sentences)) / length(sentences)
    )
end

function _generate_text(prompt::AbstractString, max_tokens::Integer)::String
    gen = _ensure_generator!()
    max_words = clamp(Int(max_tokens), 1, 200)
    return lock(GLOBAL_STATE.generator_lock) do
        generate!(gen, String(prompt); max_words=max_words)
    end
end

function _ensure_generator!()
    if GLOBAL_STATE.generator === nothing
        set_generator!(_default_generator())
    end
    return GLOBAL_STATE.generator
end

function _default_generator()
    trained = _load_trained_generator()
    trained !== nothing && return trained

    vocab_words = [
        "مرنان", "يفهم", "الفكرة", "الشيء", "الحدث", "النتيجة",
        "العلاقة", "السياق", "المعنى", "التخطيط",
        "mirnan", "understands", "meaning", "idea", "event", "result",
    ]
    vocab = Dict{String,Int}(w => i for (i, w) in enumerate(vocab_words))
    return MirnanGenerator(vocab)
end

function _project_model_dir()
    return normpath(joinpath(@__DIR__, "..", "..", "model"))
end

function _load_sparse_dat(path::String, vocab_size::Int)
    isfile(path) || return spzeros(vocab_size, vocab_size)
    open(path, "r") do io
        header = readline(io)
        header == "SPARSE_CSC" || return spzeros(vocab_size, vocab_size)
        m = Int(read(io, Int32))
        n = Int(read(io, Int32))
        nnz = Int(read(io, Int32))
        colptr = read!(io, Vector{Int32}(undef, n + 1))
        rows = read!(io, Vector{Int32}(undef, nnz))
        vals = read!(io, Vector{Float64}(undef, nnz))
        return SparseMatrixCSC(m, n, Vector{Int}(colptr), Vector{Int}(rows), vals)
    end
end

function _load_trained_generator()
    model_dir = _project_model_dir()
    vocab_path = joinpath(model_dir, "vocab.json")
    isfile(vocab_path) || return nothing
    try
        raw_vocab = JSON.parsefile(vocab_path)
        vocab = Dict{String,Int}(k => Int(v) for (k, v) in raw_vocab)
        V = length(vocab)
        K_sem = _load_sparse_dat(joinpath(model_dir, "K_sem.dat"), V)
        K_syn = _load_sparse_dat(joinpath(model_dir, "K_syn.dat"), V)
        K_causal = _load_sparse_dat(joinpath(model_dir, "K_causal.dat"), V)
        return MirnanGenerator(vocab, K_sem; K_syn=K_syn,
                               K_causal=K_causal,
                               model_dir=model_dir)
    catch e
        @warn "API: failed to load trained model: $e"
        return nothing
    end
end

end # module APIServer
