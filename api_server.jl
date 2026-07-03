#!/usr/bin/env julia
# مرنان V8 — خادم واجهة المستخدم الرسومية والـ API
using Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "src", "MirnanNew.jl")); using .MirnanNew, JSON, SparseArrays, Mmap, Sockets

# ═══ دالة تحميل المصفوفة المتفرقة من ملف .dat ═══
function _load_sparse_dat(path::String, vocab_size::Int)
    isfile(path) || return spzeros(vocab_size, vocab_size)
    open(path, "r") do io
        header = readline(io)
        header == "SPARSE_CSC" || return spzeros(vocab_size, vocab_size)
        m = read(io, Int32)
        n = read(io, Int32)
        nnz = read(io, Int32)
        colptr = read!(io, Vector{Int32}(undef, n+1))
        rows   = read!(io, Vector{Int32}(undef, nnz))
        vals   = read!(io, Vector{Float64}(undef, nnz))
        return SparseMatrixCSC(Int(m), Int(n),
                               Vector{Int}(colptr),
                               Vector{Int}(rows),
                               vals)
    end
end

function _load_sparse_dat_verbose(path::String, vocab_size::Int, label::String)
    if isfile(path)
        mb = round(filesize(path) / 1024^2; digits=1)
        println("  -> loading $label ($mb MB)...")
    else
        println("  -> loading $label: missing, using empty matrix")
    end
    flush(stdout)
    t0 = time()
    K = _load_sparse_dat(path, vocab_size)
    println("     done $label: $(length(K.nzval)) links in $(round(time() - t0; digits=1))s")
    flush(stdout)
    return K
end

function load_generator()
    model_dir = joinpath(@__DIR__, "model")
    vf = joinpath(model_dir, "vocab.json")
    println("Mirnan server dir: $(@__DIR__)")
    println("Mirnan model dir: $model_dir")
    flush(stdout)

    # Try loading trained model — WITHOUT calling train.jl main()
    if isfile(vf)
        println("جاري تحميل النموذج المدرب..."); flush(stdout)
        try
            raw_vocab = JSON.parsefile(vf)
            vocab = Dict{String,Int}(k => Int(v) for (k,v) in raw_vocab)
            V = length(vocab)
            println("  → معجم: $V كلمة")

            K_sem  = _load_sparse_dat_verbose(joinpath(model_dir, "K_sem.dat"),  V, "K_sem")
            K_syn  = _load_sparse_dat_verbose(joinpath(model_dir, "K_syn.dat"),  V, "K_syn")
            K_dial = _load_sparse_dat_verbose(joinpath(model_dir, "K_dialogue.dat"), V, "K_dialogue")
            K_causal = _load_sparse_dat_verbose(joinpath(model_dir, "K_causal.dat"), V, "K_causal")

            gen = MirnanGenerator(vocab, K_sem;
                                  K_syn=K_syn,
                                  K_causal=K_causal,
                                  model_dir=model_dir)
            if get(ENV, "MIRNAN_API_GPU", "0") in ("1", "true", "yes", "on")
                println("  -> GPU init...")
                flush(stdout)
                t_gpu = time()
                MirnanNew.Physics.Generator.gpu_init!(gen)
                println("     done GPU init in $(round(time() - t_gpu; digits=1))s")
                flush(stdout)
            else
                println("  -> GPU init skipped for API startup (set MIRNAN_API_GPU=1 to enable)")
                flush(stdout)
            end
            println("✓ تم تحميل النموذج بنجاح. (معجم: $V كلمة، K_sem: $(length(K_sem.nzval)) اقتران)")
            flush(stdout)
            return gen
        catch e
            println("⚠ خطأ في تحميل النموذج: $e")
            flush(stdout)
        end
    else
        println("⚠ لم يتم العثور على ملف المعجم: $vf")
        flush(stdout)
    end

# Fallback: demo vocab with more connections for meaningful generation
    println("⚠ التشغيل بالوضع التجريبي (نموذج صغير)...")
    flush(stdout)
    vocab = Dict(
        "العلم"=>1,"نور"=>2,"الجهل"=>3,"ظلام"=>4,
        "السماء"=>5,"صافية"=>6,"الأرض"=>7,"خضراء"=>8,
        "الحياة"=>9,"جميلة"=>10,"العالم"=>11,"كبير"=>12,
        "الله"=>13,"خالق"=>14,"الكتاب"=>15,"مفيد"=>16,
        "الماء"=>17,"سر"=>18,"السلام"=>19,"عليكم"=>20,
        "ورحمة"=>21,"القلب"=>22,"يعرف"=>23,"الحب"=>24,
        "الطريق"=>25,"الحق"=>26,"الإنسان"=>27,"يسعى"=>28,
        "دائماً"=>29,"نحو"=>30,"الأفضل"=>31,"مع"=>32,
        "في"=>33,"من"=>34,"على"=>35,"إلى"=>36,"هو"=>37,"هي"=>38,
        "كان"=>39,"يكون"=>40,"ذلك"=>41,"هذا"=>42,"هذه"=>43,
        "الذي"=>44,"التي"=>45,"لا"=>46,"لم"=>47,"لن"=>48,
        "و"=>49,"ف"=>50,"ثم"=>51,"أو"=>52,"بل"=>53,"لكن"=>54,
        "إن"=>55,"أن"=>56,"إذا"=>57,"حتى"=>58,"بعد"=>59,"قبل"=>60,
        "كل"=>61,"بعض"=>62,"كثير"=>63,"قليل"=>64,"أول"=>65,"آخر"=>66,
        "يوم"=>67,"ليل"=>68,"شمس"=>69,"قمر"=>70,"نجم"=>71,
        "بحر"=>72,"نهر"=>73,"جبل"=>74,"سهل"=>75,"صحراء"=>76,
        "شجرة"=>77,"زهرة"=>78,"طير"=>79,"سمك"=>80,"أسد"=>81,
        "حكمة"=>82,"عقل"=>83,"فكر"=>84,"عمل"=>85,"صبر"=>86,
        "أمل"=>87,"حرية"=>88,"عدالة"=>89,"سلام"=>90,"حرب"=>91,
        "علم"=>92,"فن"=>93,"تاريخ"=>94,"مستقبل"=>95,"ماضي"=>96,
        "جديد"=>97,"قديم"=>98,"صغير"=>99,"شديد"=>100,
        "مرحبا"=>101,"أهلا"=>102,"معك"=>103,"شكرا"=>105,
    )
    V = length(vocab)
    K_sem = spzeros(V, V)
    pairs = [
        (1,2),(2,1),(3,4),(4,3),(5,6),(6,5),(7,8),(8,7),
        (9,10),(10,9),(11,12),(12,11),(13,14),(14,13),
        (15,16),(16,15),(17,18),(18,17),(19,20),(20,19),
        (22,24),(24,22),(25,26),(26,25),
        (1,92),(92,1),(2,69),(69,2),(4,68),(68,4),
        (9,87),(87,9),(10,78),(78,10),(12,74),(74,12),
        (23,82),(82,23),(83,84),(84,83),(85,86),(86,85),
        (33,34),(34,33),(35,36),(36,35),
        (39,40),(40,39),(49,50),(50,49),
        (46,47),(47,46),(55,56),(56,55),
        (61,62),(62,61),(67,68),(68,67),(69,70),(70,69),
        (73,74),(74,73),(82,83),(83,82),(90,91),(91,90),
        (1,85),(85,1),(9,95),(95,9),(14,82),(82,14),
        (101,102),(102,101),(101,19),(19,101),(103,20),(20,103),
        (1,4),(4,1),(2,92),(92,2),(16,84),(84,16),
    ]
    for (i,j) in pairs
        if i <= V && j <= V; K_sem[i,j]=0.7; K_sem[j,i]=0.7; end
    end
    gen = MirnanGenerator(vocab, K_sem)
    MirnanNew.Physics.Generator.gpu_init!(gen)   # فعّل GPU إن كان ممكّناً في config
    return gen
end

function start_server(gen; port=8000)
    server = listen(port)
    gen_lock = ReentrantLock()
    println("==================================================")
    println("  خادم مرنان V8 يعمل الآن على: http://127.0.0.1:$port")
    println("==================================================")
    while true
        sock = accept(server)
        @async handle_request(sock, gen, gen_lock)
    end
end

function HTTP_UNESCAPE(s::String)
    s = replace(s, "%20" => " ")
    s = replace(s, "+" => " ")
    return s
end

function _generate_locked(gen_lock, gen, prompt; mode::String="auto", max_words::Int=10)
    if Threads.nthreads() > 1
        task = Threads.@spawn lock(gen_lock) do
            MirnanNew.Physics.Generator.generate!(gen, prompt; mode=mode, max_words=max_words)
        end
        return fetch(task)
    end
    return lock(gen_lock) do
        MirnanNew.Physics.Generator.generate!(gen, prompt; mode=mode, max_words=max_words)
    end
end

function handle_request(sock, gen, gen_lock)
    try
        request_line = readline(sock)
        if request_line === nothing || isempty(request_line)
            close(sock); return
        end
        parts = split(request_line, " ")
        method = String(parts[1])
        path_raw = String(parts[2])
        path = occursin("?", path_raw) ? split(path_raw, "?")[1] : path_raw

        headers = Dict{String,String}()
        while true
            line = readline(sock)
            line === nothing && break
            line == "" && break
            if contains(line, ":")
                k, v = strip.(split(line, ":", limit=2))
                headers[k] = v
            end
        end

        content_length = parse(Int, get(headers, "Content-Length", "0"))
        body = content_length > 0 ? read(sock, content_length) : UInt8[]
        body_str = String(body)

        if path == "/" || path == "/index.html"
            html = read(joinpath(@__DIR__, "ui", "index.html"), String)
            write(sock, "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\r\n")
            write(sock, html)
        elseif path == "/static/style.css" || path == "/style.css"
            css = read(joinpath(@__DIR__, "ui", "style.css"), String)
            write(sock, "HTTP/1.1 200 OK\r\nContent-Type: text/css; charset=utf-8\r\n\r\n")
            write(sock, css)
        elseif path == "/static/script.js" || path == "/script.js"
            js = read(joinpath(@__DIR__, "ui", "script.js"), String)
            write(sock, "HTTP/1.1 200 OK\r\nContent-Type: application/javascript; charset=utf-8\r\n\r\n")
            write(sock, js)
        elseif path == "/api/generate" && method == "POST"
            data = JSON.parse(body_str)
            prompt = get(data, "prompt", "")
            max_words = get(data, "max_words", 15)
            api_max_words = try
                parse(Int, get(ENV, "MIRNAN_API_MAX_WORDS", "10"))
            catch
                10
            end
            max_words = min(Int(max_words), max(1, api_max_words))
            mode = get(data, "mode", "auto")
            if get(data, "dialogue", false) == true && mode == "auto"
                mode = "dialogue"
            elseif mode == "lexical"
                data["dialogue"] = false
            end
            try
                println("DEBUG generate: prompt='$prompt' mode='$mode' max_words=$max_words"); flush(stdout)
                result = _generate_locked(gen_lock, gen, prompt; mode=mode, max_words=max_words)
                println("DEBUG generate result: '$(result)' (len=$(length(result)))"); flush(stdout)
                response = Dict("result" => result)
                write(sock, "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n\r\n")
                write(sock, JSON.json(response))
            catch gen_err
                @warn "Generate error" exception=(gen_err, catch_backtrace())
                println("DEBUG generate ERROR: $gen_err"); flush(stdout)
                response = Dict("result" => "", "error" => string(gen_err))
                write(sock, "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n\r\n")
                write(sock, JSON.json(response))
            end
        elseif path == "/api/chat" && method == "POST"
            data = JSON.parse(body_str)
            prompt = get(data, "message", get(data, "prompt", ""))
            max_words = get(data, "max_words", 15)
            api_max_words = try
                parse(Int, get(ENV, "MIRNAN_API_MAX_WORDS", "10"))
            catch
                10
            end
            max_words = min(Int(max_words), max(1, api_max_words))
            mode = get(data, "mode", "auto")
            if get(data, "dialogue", false) == true && mode == "auto"
                mode = "dialogue"
            elseif mode == "lexical"
                data["dialogue"] = false
            end
            try
                result = _generate_locked(gen_lock, gen, prompt; mode=mode, max_words=max_words)
                response = Dict("reply" => result, "result" => result)
                write(sock, "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n\r\n")
                write(sock, JSON.json(response))
            catch gen_err
                @warn "Generate error" exception=(gen_err, catch_backtrace())
                response = Dict("reply" => "", "result" => "", "detail" => string(gen_err))
                write(sock, "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n\r\n")
                write(sock, JSON.json(response))
            end
        elseif path == "/api/letters" && method == "GET"
            json_path = joinpath(@__DIR__, "data", "letter_physics_matrix.json")
            if isfile(json_path)
                write(sock, "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n\r\n")
                write(sock, read(json_path, String))
            else
                response = Dict("letters" => Dict(), "dim_names" => String[])
                write(sock, "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n\r\n")
                write(sock, JSON.json(response))
            end
        elseif startswith(path, "/api/letters/rich/") && method == "GET"
            letter = split(path, "/api/letters/rich/")[2]
            letter = HTTP_UNESCAPE(letter)
            pv = try Float64.(MirnanNew.Physics.WordPhysics.compute_enhanced_vector(Char(first(letter)))) catch e Float64[] end
            mass = try MirnanNew.Physics.WordPhysics.compute_word_mass(string(Char(first(letter)))) catch e 0.0 end
            response = Dict("letter" => letter, "phase_vector" => pv, "mass" => mass)
            write(sock, "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n\r\n")
            write(sock, JSON.json(response))
        elseif path == "/api/letters/update" && method == "POST"
            write(sock, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"ok\":true}")
        elseif path == "/api/benchmark" && method == "GET"
            response = Dict("status" => "ok", "vocab_size" => length(gen.vocab))
            write(sock, "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n\r\n")
            write(sock, JSON.json(response))
        elseif path == "/api/benchmark/update" && method == "POST"
            write(sock, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"ok\":true}")
        elseif path == "/api/field"
            target = occursin("?", path_raw) ? split(path_raw, "?")[2] : ""
            for param in split(target, "&")
                kv = split(param, "="; limit=2)
                if length(kv) == 2 && kv[1] == "target"
                    target = HTTP_UNESCAPE(String(kv[2]))
                end
            end
            result = lock(gen_lock) do
                if !isempty(target) && haskey(gen.vocab, target)
                    candidates = Dict{String,Float64}()
                    for (w, id) in gen.vocab
                        w == target && continue
                        pv_w = MirnanNew.Physics.Generator._pv(gen, w)
                        pv_t = MirnanNew.Physics.Generator._pv(gen, target)
                        s = MirnanNew.Physics.WordPhysics.phase_similarity(pv_w, pv_t)
                        if s > 0.1
                            candidates[w] = s
                        end
                    end
                    sorted = sort(collect(candidates); by=x->x[2], rev=true)
                    Dict("target" => target, "attracted" => [Dict("word"=>k,"score"=>v) for (k,v) in sorted[1:min(20,end)]])
                else
                    Dict("target" => target, "attracted" => Any[])
                end
            end
            write(sock, "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n\r\n")
            write(sock, JSON.json(result))
        elseif path == "/api/synthesize" && method == "POST"
            data = JSON.parse(body_str)
            prompt = get(data, "prompt", get(data, "input", ""))
            result = lock(gen_lock) do
                MirnanNew.Physics.Generator.generate!(gen, prompt; mode="standard", max_words=20)
            end
            response = Dict("result" => result)
            write(sock, "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n\r\n")
            write(sock, JSON.json(response))
        elseif path == "/api/status" && method == "GET"
            response = Dict("status" => "ok", "vocab_size" => length(gen.vocab))
            write(sock, "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n\r\n")
            write(sock, JSON.json(response))
        elseif path == "/api/word" && method == "POST"
            data = JSON.parse(body_str)
            word = get(data, "word", "")
            pv = lock(gen_lock) do
                Float64.(MirnanNew.Physics.WordPhysics.compute_extended_phase_vector(word))
            end
            response = Dict("word" => word, "phase_vector" => pv)
            write(sock, "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n\r\n")
            write(sock, JSON.json(response))
        else
            notfound = Dict("error" => "Not Found", "path" => path)
            write(sock, "HTTP/1.1 404 Not Found\r\nContent-Type: application/json; charset=utf-8\r\n\r\n")
            write(sock, JSON.json(notfound))
        end
    catch e
        try
            err_resp = Dict("error" => string(e), "detail" => string(e), "message" => string(e))
            write(sock, "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json; charset=utf-8\r\n\r\n")
            write(sock, JSON.json(err_resp))
        catch; end
    finally
        try close(sock) catch; end
    end
end

function main()
    gen = load_generator()

    model_dir = joinpath(@__DIR__, "model")
    pv_cache_file = joinpath(model_dir, "pv_cache.bin")
    mass_cache_file = joinpath(model_dir, "mass_cache.bin")
    V = length(gen.vocab)
    dim = MirnanNew.Physics.Constants.TOTAL_DIM
    large_threshold = try
        parse(Int, get(ENV, "MIRNAN_LARGE_MODEL_THRESHOLD", "100000"))
    catch
        100000
    end
    large_model = V >= large_threshold

    env_on(name::String, default::String) =
        lowercase(get(ENV, name, default)) in ("1", "true", "yes", "on")

    function cache_matches()
        isfile(pv_cache_file) && isfile(mass_cache_file) || return false
        try
            pv_ok = open(pv_cache_file, "r") do io
                file_v = Int(read(io, Int32))
                file_dim = Int(read(io, Int32))
                file_v == V && file_dim == dim
            end
            mass_ok = open(mass_cache_file, "r") do io
                file_v = Int(read(io, Int64))
                file_v == V
            end
            return pv_ok && mass_ok
        catch
            return false
        end
    end

    # ═══ المرحلة 1: JIT Warm-up ═══
    warmup_enabled = env_on("MIRNAN_API_WARMUP", large_model ? "0" : "1")
    if warmup_enabled
        println("⟳ [1/3] JIT warm-up — تجميع محرك التوليد الفيزيائي...")
        flush(stdout)
        try
            t_wu = time()
            test_word = first(keys(gen.vocab))
            _ = MirnanNew.Physics.Generator.generate!(gen, test_word; mode="resonant", max_words=3)
            println("  ✓ JIT اكتمل في $(round(time()-t_wu; digits=1))s")
            flush(stdout)
        catch e
            println("  ⚠ تحذير JIT: $e")
            flush(stdout)
        end
    else
        println("⟳ [1/3] JIT warm-up — متخطى للنموذج الكبير (set MIRNAN_API_WARMUP=1 to enable)")
        flush(stdout)
    end

    # ═══ المرحلة 2: بناء/تحميل التخزين المؤقت المستمر على القرص ═══
    build_cache_enabled = env_on("MIRNAN_API_BUILD_CACHE", large_model ? "0" : "1")
    cache_ready = cache_matches()
    if build_cache_enabled && !cache_ready
        println("⟳ [2/3] بناء cache على القرص ($V كلمة — مرة واحدة فقط)...")
        pv_size_gb = round(V * dim * 4 / 1_073_741_824; digits=1)
        println("  الحجم المتوقع: ~$(pv_size_gb) GB")
        println("  (سيُحفظ على القرص ولن يُحسب مجدداً في المرات القادمة)")
        flush(stdout)
        try
            t_build = time()
            n_cached = 0
            n_skipped = 0
            # Save masses: Int64(V) + V * Float64 (8-byte aligned)
            open(mass_cache_file, "w") do io
                write(io, Int64(V))
                for wid in 1:V
                    word = get(gen.id2word, wid, nothing)
                    if word !== nothing
                        try
                            write(io, Float64(MirnanNew.Physics.WordPhysics.compute_word_mass(word)))
                        catch e
                            write(io, Float64(0.0))
                            n_skipped += 1
                            n_skipped <= 5 && @warn "تخطي كلمة #$wid في mass: $e"
                        end
                    else
                        write(io, Float64(0.0))
                    end
                end
            end
            # Save phase vectors: Int32(V) + Int32(dim) + V * dim * Float32
            open(pv_cache_file, "w") do io
                write(io, Int32(V))
                write(io, Int32(dim))
                buf = Vector{Float32}(undef, dim)
                for wid in 1:V
                    word = get(gen.id2word, wid, nothing)
                    if word !== nothing
                        try
                            pv = MirnanNew.Physics.WordPhysics.compute_extended_phase_vector(word)
                            buf .= Float32.(pv)
                        catch e
                            fill!(buf, 0f0)
                            n_skipped += 1
                            n_skipped <= 5 && @warn "تخطي كلمة #$wid في pv: $e"
                        end
                    else
                        fill!(buf, 0f0)
                    end
                    write(io, buf)
                    n_cached += 1
                    (n_cached % 5000 == 0 || n_cached == V) && (println("    $(round(n_cached/V*100; digits=1))%..."); flush(stdout))
                end
            end
            dt = round(time()-t_build; digits=1)
            println("  ✓ تم بناء وحفظ $n_cached كلمة في $(dt)s ($(pv_size_gb) GB)")
            if n_skipped > 0
                println("  ⚠ تم تخطي $n_skipped كلمة (أحرف غير صالحة)")
            end
            flush(stdout)
        catch e
            println("  ⚠ خطأ في بناء cache: $e")
            rm(pv_cache_file; force=true)
            rm(mass_cache_file; force=true)
        end
        cache_ready = cache_matches()
    elseif cache_ready
        println("⟳ [2/3] cache موجود مسبقاً على القرص — جاهز ✓")
        flush(stdout)
    else
        println("⟳ [2/3] بناء cache — متخطى للنموذج الكبير (set MIRNAN_API_BUILD_CACHE=1 to enable)")
        flush(stdout)
    end

    # ═══ توصيل التخزين المؤقت المستمر بمحرك التوليد عبر Mmap ═══
    if cache_ready
        MirnanNew.Physics.Generator._disk_pv_dim[] = dim
        MirnanNew.Physics.Generator._disk_pv_data[] = Mmap.mmap(pv_cache_file, Vector{Float32}, V * dim, 8)
        MirnanNew.Physics.Generator._disk_mass_data[] = Mmap.mmap(mass_cache_file, Vector{Float64}, V, 8)
        println("  ✓ cache مُحمّل في الذاكرة الافتراضية (Mmap)")
        flush(stdout)
    end

    # ═══ المرحلة 3: توليد اختباري (Resonant — سريع) ═══
    speed_test_enabled = env_on("MIRNAN_API_SPEED_TEST", large_model ? "0" : "1")
    if speed_test_enabled
        println("⟳ [3/3] اختبار السرعة...")
        flush(stdout)
        try
            t_test = time()
            test_words = collect(keys(gen.vocab))[1:min(2, length(gen.vocab))]
            for tw in test_words
                r = MirnanNew.Physics.Generator.generate!(gen, tw; mode="resonant", max_words=5)
                dt_t = round(time()-t_test; digits=1)
                println("  '$tw' → '$r' ($(dt_t)s)")
                flush(stdout)
                t_test = time()
            end
        catch e
            println("  ⚠ تحذير اختبار: $e")
            flush(stdout)
        end
    else
        println("⟳ [3/3] اختبار السرعة — متخطى للنموذج الكبير (set MIRNAN_API_SPEED_TEST=1 to enable)")
        flush(stdout)
    end

    println("==================================================")
    println("  ✓ النموذج جاهز للاستخدام!")
    println("==================================================")
    flush(stdout)

    # Start the APIServer
    start_server(gen; port=8000)
end

main()
