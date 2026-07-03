# Search for _relation_or_difference_prompt in the codebase
for (root, dirs, files) in walkdir("src")
    for file in files
        if endswith(file, ".jl")
            path = joinpath(root, file)
            content = read(path, String)
            if occursin("_relation_or_difference_prompt", content)
                println("Found in $path:")
                for (i, line) in enumerate(split(content, "\n"))
                    if occursin("_relation_or_difference_prompt", line)
                        println("  L$i: $line")
                    end
                end
            end
        end
    end
end
