for (root, dirs, fs) in walkdir("src")
    for f in fs
        if endswith(f, ".jl")
            path = joinpath(root, f)
            txt = read(path, String)
            if occursin("semantic_factor", txt) || occursin("positional_factor", txt)
                println("FOUND: ", path)
            end
        end
    end
end
