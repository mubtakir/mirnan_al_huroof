using Test
using JSON
using SparseArrays

const MIRNAN_DIR = dirname(@__DIR__)
include(joinpath(MIRNAN_DIR, "src", "MirnanNew.jl"))
using .MirnanNew

const IRP = MirnanNew.Physics.IntentResponsePlanner

function _load_sparse_dat(path::String, vocab_size::Int)
    isfile(path) || return spzeros(vocab_size, vocab_size)
    open(path, "r") do io
        header = readline(io)
        header == "SPARSE_CSC" || return spzeros(vocab_size, vocab_size)
        m = read(io, Int32)
        n = read(io, Int32)
        nnz = read(io, Int32)
        colptr = read!(io, Vector{Int32}(undef, n + 1))
        rows = read!(io, Vector{Int32}(undef, nnz))
        vals = read!(io, Vector{Float64}(undef, nnz))
        return SparseMatrixCSC(Int(m), Int(n), Vector{Int}(colptr), Vector{Int}(rows), vals)
    end
end

function _load_trained_generator()
    model_dir = joinpath(MIRNAN_DIR, "model")
    vocab_path = joinpath(model_dir, "vocab.json")
    isfile(vocab_path) || error("Missing trained model. Run models/mirnan/train.jl first.")
    raw_vocab = JSON.parsefile(vocab_path)
    vocab = Dict{String,Int}(String(k) => Int(v) for (k, v) in raw_vocab)
    V = length(vocab)
    K_sem = _load_sparse_dat(joinpath(model_dir, "K_sem.dat"), V)
    K_syn = _load_sparse_dat(joinpath(model_dir, "K_syn.dat"), V)
    K_causal = _load_sparse_dat(joinpath(model_dir, "K_causal.dat"), V)
    return MirnanGenerator(vocab, K_sem; K_syn=K_syn, K_causal=K_causal, model_dir=model_dir)
end

_norm(s) = replace(lowercase(strip(String(s))), r"[[:punct:]\u061F\u060C\u061B]+" => "")
_starts_no(s) = startswith(_norm(s), "لا") || startswith(_norm(s), "كلا")
_is_negative_or_corrective(s) = _starts_no(s) ||
    occursin("ليس", _norm(s)) ||
    occursin("لا ", String(s)) ||
    startswith(_norm(s), "بل") ||
    occursin("بل ", String(s)) ||
    occursin("العكس", _norm(s))
_has_explanation_marker(s) = any(w -> occursin(w, _norm(s)),
    ["لأن", "لان", "بسبب", "حين", "عندما", "كلما", "من خلال", "فيؤدي", "فيزيد", "فيزيل", "فيحفظ", "إذ"])
_has_definition_shape(s) = any(w -> occursin(w, String(s)), [" هو ", " هي ", " يعني ", " تعني "])
_has_contrast_shape(s) = any(w -> occursin(w, String(s)), ["أما", "بينما", "في المقابل", "والفرق بينهما"])

function _answer(gen, prompt)
    return generate!(gen, String(prompt); mode="auto", max_words=56)
end

function _assert_terms(answer, required, forbidden)
    text = _norm(answer)
    for w in required
        @test occursin(String(w), text)
    end
    for w in forbidden
        @test !occursin(_norm(String(w)), text)
    end
end

@testset "Mirnan organized question type matrix" begin
    probes = JSON.parsefile(joinpath(@__DIR__, "fixtures", "question_type_probes.json"))

    @testset "planner recognizes question type without answering from templates" begin
        for probe in probes["planner_probes"]
            plan = IRP.detect_response_intent(probe["prompt"])
            @test plan.intent == probe["intent"]
            subject_text = _norm(join([plan.subject, plan.action, plan.cause, plan.result], " "))
            for term in get(probe, "contains", Any[])
                @test occursin(_norm(String(term)), subject_text)
            end
        end
    end

    @testset "trained generation respects question behavior" begin
        gen = _load_trained_generator()
        for probe in probes["generation_probes"]
            ans = _answer(gen, probe["prompt"])
            _assert_terms(ans, get(probe, "required", Any[]), get(probe, "forbidden", Any[]))
            if get(probe, "polarity", "") == "positive"
                @test !_starts_no(ans)
            elseif get(probe, "polarity", "") == "negative"
                @test _is_negative_or_corrective(ans)
            elseif get(probe, "mode", "") == "explain"
                @test _has_explanation_marker(ans)
                @test !startswith(strip(String(ans)), "نعم")
            elseif get(probe, "mode", "") == "contrast"
                @test _has_contrast_shape(ans)
                @test length(split(ans)) >= 12
                if occursin("العلم", String(probe["prompt"])) && occursin("الفهم", String(probe["prompt"]))
                    @test !occursin("ضياء يكشف", ans)
                    @test !occursin("النور", ans)
                end
                if occursin("العدل", String(probe["prompt"]))
                    @test !startswith(_norm(ans), "العدل من غير زيادة")
                    @test !occursin("العدل يدور حول من غير زيادة", _norm(ans))
                end
            elseif get(probe, "mode", "") == "definition"
                @test !_starts_no(ans)
                @test _has_definition_shape(ans)
                @test length(split(ans)) >= 6
            end
        end
    end
end
