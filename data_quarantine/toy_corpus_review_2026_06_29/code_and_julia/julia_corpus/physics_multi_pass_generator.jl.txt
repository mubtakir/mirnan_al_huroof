"""MultiPassGenerator — توليد متعدد الممرات (relaxation)."""
module MultiPassGeneratorModule
using LinearAlgebra, Random, Statistics
export MultiPassGenerator, RelaxationPass

struct RelaxationPass
    words::Vector{String}; score::Float64; entropy::Float64
end

mutable struct MultiPassGenerator
    passes::Vector{RelaxationPass}; max_passes::Int
end
MultiPassGenerator(; max_passes=3) = MultiPassGenerator(RelaxationPass[], max_passes)

function relax!(mpg::MultiPassGenerator, gen, prompt::String; max_words=15)
    empty!(mpg.passes)
    best_result = ""; best_score = -Inf
    for p in 1:mpg.max_passes
        temperature = 0.5 + 0.5*(p-1)/(mpg.max_passes-1)
        result = generate!(gen, prompt; max_words=max_words, temperature=temperature)
        words = split(result)
        if !isempty(words)
            pvs = [Float64.(gen.pv_cache[w]) for w in words if haskey(gen.pv_cache, w)]
            score = isempty(pvs) ? 0.0 : mean([norm(pv) for pv in pvs])
            push!(mpg.passes, RelaxationPass(words, score, 0.0))
            if score > best_score; best_score = score; best_result = result; end
        end
    end
    return best_result
end
end

