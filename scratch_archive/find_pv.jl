lines = readlines("src/physics/generator.jl")
for (i, line) in enumerate(lines)
    if occursin("function _pv", line) || occursin("_pv(gen", line)
        println("Line $i: ", line)
    end
end
