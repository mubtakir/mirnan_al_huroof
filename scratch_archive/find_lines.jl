lines = readlines("src/physics/generator.jl")
for (i, line) in enumerate(lines)
    if occursin("semantic_factor", line) || occursin("positional_factor", line) || occursin("gravity", line)
        println("Line $i: ", line)
    end
end
