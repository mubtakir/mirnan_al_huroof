content = read("src/physics/engines/strategies/resonant_strategy.jl", String)
for (i, line) in enumerate(split(content, "\n"))
    if occursin("_relation_or_difference_prompt", line)
        println("L$i: $line")
    end
end
