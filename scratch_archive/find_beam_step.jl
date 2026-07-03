lines = readlines("src/physics/generator.jl")
for (i, line) in enumerate(lines)
    if occursin("function _beam_step", line) || (occursin("_beam_step(", line) && occursin("function", line))
        println("Line $i: ", line)
    end
end
