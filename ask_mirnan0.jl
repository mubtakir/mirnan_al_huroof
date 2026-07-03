using Pkg; Pkg.activate(@__DIR__)
using HTTP, JSON

server_url = "http://127.0.0.1:8000/api/generate"

println("==================================================")
println("      مرحباً بك في واجهة أسئلة مرنان التفاعلية     ")
println("==================================================")
println("اكتب سؤالك ثم اضغط Enter للحصول على الإجابة.")
println("اكتب 'خروج' أو 'exit' لإنهاء البرنامج.")
println("==================================================")
flush(stdout)

while true
    print("\nالسؤال: ")
    flush(stdout)
    q = readline()
    q = strip(q)
    if isempty(q)
        continue
    end
    if q in ("خروج", "exit", "quit", "q")
        println("مع السلامة!")
        break
    end
    
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
