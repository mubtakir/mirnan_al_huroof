content = read("src/physics/engines/strategies/resonant_strategy.jl", String)
for (i, line) in enumerate(split(content, "\n"))
    if occursin("_fallback_generate", line)
        println("L$i: $line")
    end
end
