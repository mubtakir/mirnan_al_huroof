# Search for _is_generation_candidate in the codebase
for (root, dirs, files) in walkdir("src")
    for file in files
        if endswith(file, ".jl")
            path = joinpath(root, file)
            content = read(path, String)
            if occursin("_is_generation_candidate", content)
                println("Found in $path:")
                for (i, line) in enumerate(split(content, "\n"))
                    if occursin("_is_generation_candidate", line)
                        println("  L$i: $line")
                    end
                end
            end
        end
    end
end
