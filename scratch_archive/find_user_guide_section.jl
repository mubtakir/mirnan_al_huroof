lines = readlines("docs/USER_GUIDE.md")
for (i, line) in enumerate(lines)
    if occursin("## 6.", line) || occursin("gravity controls", line)
        println("Line $i: ", line)
    end
end
