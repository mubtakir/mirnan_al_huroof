include("../src/MirnanNew.jl")
using .MirnanNew
using Test, SparseArrays

const Physics = MirnanNew.Physics

@testset "al_muradif learned semantic equivalence memory" begin
    vocab = Dict(
        "علم" => 1,
        "معرفه" => 2,
        "فهم" => 3,
        "نور" => 4,
        "عمل" => 5,
        "مدرسه" => 6,
        "كتب" => 7,
        "ذهب" => 8,
    )

    n = length(vocab)
    K_sem = spzeros(Float64, n, n)
    K_syn = spzeros(Float64, n, n)
    K_causal = spzeros(Float64, n, n)

    # "علم" و"معرفه" لا يتطابقان حرفياً، لكنهما يشتركان في الجوار الدلالي والسببي.
    for w in ("فهم", "نور", "عمل")
        K_sem[vocab[w], vocab["علم"]] = 1.0
        K_sem[vocab[w], vocab["معرفه"]] = 0.95
    end
    K_sem[vocab["معرفه"], vocab["علم"]] = 0.12
    K_sem[vocab["علم"], vocab["معرفه"]] = 0.12

    K_syn[vocab["مدرسه"], vocab["علم"]] = 0.8
    K_syn[vocab["مدرسه"], vocab["معرفه"]] = 0.75
    K_causal[vocab["عمل"], vocab["علم"]] = 0.9
    K_causal[vocab["عمل"], vocab["معرفه"]] = 0.85

    mem = Physics.build_muradif_memory(vocab, K_sem;
                                       K_syn=K_syn,
                                       K_causal=K_causal,
                                       max_words=8,
                                       top_neighbors=4,
                                       max_candidates=4,
                                       min_score=0.10)

    @test Physics.has_muradif_records(mem)
    terms = Physics.muradif_terms(mem, "العلم"; min_score=0.10, limit=4)
    @test "معرفه" in terms

    dir = mktempdir()
    path = Physics.save_muradif(mem, joinpath(dir, "semantic_equivalence.json"))
    @test isfile(path)
    loaded = Physics.load_muradif(path)
    @test "معرفه" in Physics.muradif_terms(loaded, "علم"; min_score=0.10, limit=4)

    persistent = Physics.MuradifMemory(max_candidates=4)
    persistent.entries["علم"] = [
        Physics.MuradifCandidate("تعلم", 0.80, "thematic", 0.8, 0.5, 0.5, 0.0)
    ]
    Physics.merge_muradif!(loaded, persistent)
    @test "تعلم" in Physics.muradif_terms(loaded, "علم"; min_score=0.10, limit=4)
end
