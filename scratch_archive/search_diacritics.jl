for (root, dirs, files) in walkdir("src")
    for f in files
        if endswith(f, ".jl")
            path = joinpath(root, f)
            txt = read(path, String)
            if occursin("strip", txt) && (occursin("diacritic", txt) || occursin("تشكيل", txt) || occursin("حركات", txt))
                println("FOUND: ", path)
            end
        end
    end
end
