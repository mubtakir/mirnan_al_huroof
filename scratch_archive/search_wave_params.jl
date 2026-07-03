lines = readlines("config.yaml")
for (i, line) in enumerate(lines)
    line_clean = lowercase(line)
    if occursin("carrier", line_clean) || occursin("amplitude", line_clean) || occursin("frequency", line_clean) || occursin("wave", line_clean) || occursin("حامل", line_clean) || occursin("سعة", line_clean) || occursin("موج", line_clean)
        println("Line $i: ", line)
    end
end
