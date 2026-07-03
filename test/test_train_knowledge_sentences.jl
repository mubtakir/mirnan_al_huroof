using Test
using JSON

include(joinpath(@__DIR__, "..", "train.jl"))

@testset "training knowledge sentences for letter topics" begin
    mktempdir() do dir
        open(joinpath(dir, "definitions.json"), "w") do io
            JSON.print(io, Dict(
                "السلام" => Dict(
                    "definition" => "السلام أمن واستقرار.",
                    "examples" => ["يحفظ العدل السلام.", ""],
                ),
                "العلم" => Dict(
                    "definition" => "العلم إدراك الحقائق.",
                    "examples" => ["يزيد العلم الفهم."],
                ),
            ))
        end

        open(joinpath(dir, "istinbat_attention.json"), "w") do io
            JSON.print(io, Dict(
                "records" => [
                    Dict("examples" => ["سافر الرجل قبل الفجر.", "يحفظ العدل السلام."]),
                    Dict("examples" => ["يدرس الطالب لكي ينجح."]),
                ],
            ))
        end

        open(joinpath(dir, "dialogue_facts.json"), "w") do io
            JSON.print(io, Dict(
                "speech_acts" => [
                    Dict("prompt" => "كيف حالك؟", "response" => "الحمد لله."),
                    Dict("prompt" => "", "response" => "السلام عليكم."),
                ],
            ))
        end

        sentences = extract_knowledge_sentences(dir)

        @test "السلام أمن واستقرار." in sentences
        @test "يحفظ العدل السلام." in sentences
        @test count(==("يحفظ العدل السلام."), sentences) == 1
        @test "العلم إدراك الحقائق." in sentences
        @test "يزيد العلم الفهم." in sentences
        @test "سافر الرجل قبل الفجر." in sentences
        @test "يدرس الطالب لكي ينجح." in sentences
        @test "كيف حالك؟" in sentences
        @test "الحمد لله." in sentences
        @test "السلام عليكم." in sentences
        @test !("" in sentences)
    end
end
