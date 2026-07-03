content = read("src/physics/engines/strategies/definition_strategy.jl", String)
lines = split(content, "\n")
for i in 121:min(170, length(lines))
    println("$i: $(lines[i])")
end
