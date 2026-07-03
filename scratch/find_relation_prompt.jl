# Search for _relationship_prompt in the codebase
for (root, dirs, files) in walkdir("src")
    for file in files
        if endswith(file, ".jl")
            path = joinpath(root, file)
            content = read(path, String)
            if occursin("_relationship_prompt", content)
                println("Found in $path:")
                for (i, line) in enumerate(split(content, "\n"))
                    if occursin("_relationship_prompt", line)
                        println("  L$i: $line")
                    end
                end
            end
        end
    end
end
