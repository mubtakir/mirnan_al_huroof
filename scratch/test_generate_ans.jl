# Enable all debug logs
ENV["JULIA_DEBUG"] = "all"

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "MirnanNew.jl"))
using .MirnanNew
using .MirnanNew.Physics
using .MirnanNew.Physics.RAPGModule: RAPGKnowledgeBase, init_rapg_db!, store_passage!, load_rapg_kb
using .MirnanNew.Physics.AlIstinbat: IstinbatAttentionMemory
using .MirnanNew.Physics: MirnanGenerator, generate!
using .MirnanNew.Physics.Generator: _LEARNED_ISTINBAT_MEMORY
using SparseArrays

# 1. Setup mock RAPG DB
db_path = joinpath(mktempdir(), "rapg_test_p7_script.db")
init_rapg_db!(db_path)
vec1 = zeros(Float32, 10000)
vec1[1] = 1.0f0
store_passage!(db_path, "العلم هو النور الكاشف للعقل", vec1, "definitions")
kb = load_rapg_kb(db_path)

# 2. Setup mock istinbat memory
istinbat = IstinbatAttentionMemory()
istinbat.discovered_markers["هل"] = "question"
istinbat.discovered_confidences["هل"] = 0.6

# 3. Setup generator
vocab = Dict("ما" => 1, "هو" => 2, "العلم" => 3, "؟" => 4, "النور" => 5, "الكاشف" => 6, "العقل" => 7)
K_sem = spzeros(7, 7)
K_sem[3, 5] = 0.8  # العلم -> النور
K_sem[5, 3] = 0.8  # النور -> العلم
K_sem[5, 6] = 0.8  # النور -> الكاشف
K_sem[6, 7] = 0.8  # الكاشف -> العقل

gen = MirnanGenerator(vocab, K_sem; model_dir=mktempdir())
gen.rapg_kb = kb

push!(gen.retrieved_passages, "العلم هو النور الكاشف للعقل")
push!(gen.retrieved_pvs, vec1)
push!(gen.retrieved_similarities, 0.8)

_LEARNED_ISTINBAT_MEMORY[] = istinbat

empty!(gen.cerebellum.integration_log)
println("PID Enabled: ", gen.cerebellum.pid_enabled)
ans = generate!(gen, "ما هو العلم ؟"; mode="resonant", max_words=3)

println("Result: '$ans'")
println("Log length: ", length(gen.cerebellum.integration_log))
