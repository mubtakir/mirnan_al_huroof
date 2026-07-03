using SparseArrays, Random

function benchmark_build_k()
    println("Simulating corpus of 3.7 million words...")
    V = 495307
    total_words = 3700000
    window = 10
    
    # Generate mock vocab and ids
    println("Generating mock word IDs...")
    ids = rand(Int32(1):Int32(V), total_words)
    
    println("Starting build_K logic...")
    t0 = time()
    
    # We estimate coordinate count
    est_coords = min(total_words * window * 2, 2_000_000_000)
    println("Estimated coordinates: ", est_coords)
    
    I_coords = Vector{Int32}()
    J_coords = Vector{Int32}()
    V_coords = Vector{Float64}()
    
    sizehint!(I_coords, est_coords)
    sizehint!(J_coords, est_coords)
    sizehint!(V_coords, est_coords)
    
    for i in 1:length(ids)
        id_i = ids[i]
        for j in max(1, i-window):min(length(ids), i+window)
            j == i && continue
            push!(I_coords, id_i)
            push!(J_coords, ids[j])
            push!(V_coords, 1.0 / abs(j - i))
        end
    end
    
    t1 = time()
    println("Vector push completed in ", round(t1 - t0; digits=2), " seconds.")
    println("Vector sizes: ", length(I_coords))
    
    println("Building sparse matrix...")
    t2 = time()
    K = sparse(I_coords, J_coords, V_coords, V, V)
    t3 = time()
    println("Sparse matrix built in ", round(t3 - t2; digits=2), " seconds.")
    println("Matrix nonzeros: ", length(K.nzval))
    println("Total time: ", round(t3 - t0; digits=2), " seconds.")
end

benchmark_build_k()
