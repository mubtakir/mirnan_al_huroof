using Pkg; Pkg.activate(@__DIR__)
using HTTP, JSON

server_url = "http://127.0.0.1:8000/api/generate"
question_file = isempty(ARGS) ? joinpath(@__DIR__, "hiwar.txt") : abspath(ARGS[1])

println("==================================================")
println("      مرحباً بك في فاحص أسئلة مرنان من ملف hiwar     ")
println("==================================================")
println("ملف الأسئلة: $question_file")
println("==================================================")
flush(stdout)

if !isfile(question_file)
    println("خطأ: لم أجد ملف الأسئلة.")
    println("ضع الأسئلة في: $(joinpath(@__DIR__, "hiwar.txt"))")
    exit(1)
end

questions = String[]
for line in eachline(question_file)
    q = strip(line)
    (isempty(q) || startswith(q, "#")) && continue
    push!(questions, q)
end

if isempty(questions)
    println("لا توجد أسئلة في الملف.")
    exit(0)
end

for (i, q) in enumerate(questions)
    println("\n[$i/$(length(questions))] السؤال: $q")
    t0 = time()
    payload = JSON.json(Dict("prompt" => q, "max_words" => 25, "mode" => "auto"))
    try
        res = HTTP.post(server_url, ["Content-Type" => "application/json"], payload; connect_timeout=5, read_timeout=15)
        elapsed = time() - t0
        data = JSON.parse(String(res.body))
        ans = get(data, "result", "")
        println("الجواب: $ans")
        println("الزمن المستغرق: ", round(elapsed * 1000; digits=1), " ملي ثانية")
    catch e
        println("خطأ: تعذر الاتصال بالخادم. يرجى التأكد من تشغيل خادم مرنان أولاً.")
    end
    flush(stdout)
end
