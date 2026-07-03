"""ModelIO — إدخال/إخراج النموذج (حفظ/تحميل K matrices, vocab, spectra)."""
module ModelIO
using JSON, SparseArrays, NPZ, LinearAlgebra
export save_model, load_model

function save_model(out_dir::String, vocab::Dict{String,Int}, K_sem, K_syn=nothing, K_dial=nothing; id2word=nothing)
    mkpath(out_dir)
    open(joinpath(out_dir, "vocab.json"), "w") do io; JSON.print(io, vocab); end
    if K_sem !== nothing && K_sem isa AbstractSparseMatrix
        NPZ.npzwrite(joinpath(out_dir, "K_sem.npz"), Dict("data"=>Matrix(K_sem)))
    end
    if K_syn !== nothing && K_syn isa AbstractSparseMatrix
        NPZ.npzwrite(joinpath(out_dir, "K_syn.npz"), Dict("data"=>Matrix(K_syn)))
    end
    if K_dial !== nothing && K_dial isa AbstractSparseMatrix
        NPZ.npzwrite(joinpath(out_dir, "K_dialogue.npz"), Dict("data"=>Matrix(K_dial)))
    end
    return true
end

function load_model(in_dir::String)
    vocab_file = joinpath(in_dir, "vocab.json")
    vocab = isfile(vocab_file) ? JSON.parsefile(vocab_file) : Dict{String,Int}()
    K_sem = nothing; K_syn = nothing; K_dial = nothing
    for (name, var_sym) in [("K_sem", :K_sem), ("K_syn", :K_syn), ("K_dialogue", :K_dial)]
        f = joinpath(in_dir, name*".npz")
        if isfile(f)
            data = NPZ.npzread(f)
            if haskey(data, "data")
                mat = data["data"]
                if var_sym == :K_sem; K_sem = SparseMatrixCSC(sparse(mat))
                elseif var_sym == :K_syn; K_syn = SparseMatrixCSC(sparse(mat))
                elseif var_sym == :K_dial; K_dial = SparseMatrixCSC(sparse(mat))
                end
            end
        end
    end
    return Dict("vocab"=>vocab, "K_sem"=>K_sem, "K_syn"=>K_syn, "K_dial"=>K_dial)
end
end
