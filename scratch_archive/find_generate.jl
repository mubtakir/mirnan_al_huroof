lines = readlines("src/physics/generator.jl")
for (i, line) in enumerate(lines)
    if occursin("function generate!", line) || (occursin("generate!(", line) && occursin("function", line))
        println("Line $i: ", line)
    end
end
