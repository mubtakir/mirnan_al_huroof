lines = readlines("src/physics/generator.jl")
for (i, line) in enumerate(lines)
    if occursin("function _resonance_candidates", line) || (occursin("_resonance_candidates(", line) && occursin("function", line))
        println("Line $i: ", line)
    end
end
