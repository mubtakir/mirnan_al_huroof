for (root, dirs, fs) in walkdir(".")
    if occursin(".git", root) || occursin(".julia", root) || occursin("model", root)
        continue
    end
    for f in fs
        if endswith(f, ".jl") || endswith(f, ".json") || endswith(f, ".txt") || endswith(f, ".html") || endswith(f, ".js")
            path = joinpath(root, f)
            try
                txt = read(path, String)
                if occursin("اتحدث", txt) || occursin("الملف بنجاح", txt) || occursin("تم النافذه", txt) || occursin("بطلاقة", txt) || occursin("بطلاقه", txt)
                    println("MATCH: ", path)
                end
            catch e
            end
        end
    end
end
