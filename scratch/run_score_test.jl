using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "MirnanNew.jl"))
using .MirnanNew
using .MirnanNew.Physics
using .MirnanNew.Physics.RAPGModule: RAPGKnowledgeBase, init_rapg_db!, store_passage!, load_rapg_kb
using .MirnanNew.Physics: MirnanGenerator, generate!
using .MirnanNew.Physics.Generator: _score, _pv
using SparseArrays
using LinearAlgebra

vocab = Dict("ما" => 1, "هو" => 2, "العلم" => 3, "؟" => 4, "النور" => 5, "الكاشف" => 6, "العقل" => 7)
K_sem = spzeros(7, 7)
gen = MirnanGenerator(vocab, K_sem; model_dir=mktempdir())

db_path = joinpath(mktempdir(), "rapg_test.db")
init_rapg_db!(db_path)
store_passage!(db_path, "العلم هو النور الكاشف للعقل", zeros(Float32, 10000), "definitions")
gen.rapg_kb = load_rapg_kb(db_path)

push!(gen.retrieved_passages, "العلم هو النور الكاشف للعقل")
push!(gen.retrieved_pvs, zeros(Float32, 10000))
push!(gen.retrieved_similarities, 0.8)

# Try running _score directly
best_w = "النور"
used = Set(["ما", "هو", "العلم", "؟"])
output = String[]
prompt_pv = [_pv(gen, w) for w in ["ما", "هو", "العلم", "؟"]]
context_words = ["ما", "هو", "العلم", "؟"]

try
    println("Running _score...")
    res = _score(gen, best_w, used, [_pv(gen, ow) for ow in output], prompt_pv;
                 prev_word="؟", context_words=context_words)
    println("Success! Result: ", res)
catch e
    println("Failed with exception: ", e)
    Base.showerror(stdout, e, catch_backtrace())
end
