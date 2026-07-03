# prepare_training.jl
# Scans all Julia codebase files (Mirnan.jl and BasilAgent.jl) and prepares them in the training corpus directory.

using Pkg
Pkg.activate(@__DIR__)

const CORPUS_DIR = joinpath(@__DIR__, "data", "toy_corpus", "julia_corpus")
mkpath(CORPUS_DIR)

println("📂 Preparing Julia and Bayan corpus for training...")

# 1. Scan mirnan_julia/src
mirnan_src = joinpath(@__DIR__, "src")
for (root, dirs, files) in walkdir(mirnan_src)
    for file in files
        if endswith(file, ".jl")
            src_path = joinpath(root, file)
            rel = relpath(src_path, mirnan_src)
            dest_name = replace(rel, '/' => '_', '\\' => '_') * ".txt"
            dest_path = joinpath(CORPUS_DIR, dest_name)
            
            content = read(src_path, String)
            write(dest_path, content)
            println("  + Added Mirnan file: $rel")
        end
    end
end

# 2. Scan mirnan_julia/basil_agent/src
basil_src = joinpath(@__DIR__, "basil_agent", "src")
if isdir(basil_src)
    for (root, dirs, files) in walkdir(basil_src)
        for file in files
            if endswith(file, ".jl")
                src_path = joinpath(root, file)
                rel = relpath(src_path, basil_src)
                dest_name = "basil_" * replace(rel, '/' => '_', '\\' => '_') * ".txt"
                dest_path = joinpath(CORPUS_DIR, dest_name)
                
                content = read(src_path, String)
                write(dest_path, content)
                println("  + Added BasilAgent file: $rel")
            end
        end
    end
end

println("✅ Corpus preparation complete! Ready for PGN / K-matrix training.")
