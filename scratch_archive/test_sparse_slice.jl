using SparseArrays, Random

function test_slice()
    V = 497991
    nnz_target = 29356341
    println("Creating a mock SparseMatrixCSC of size ", V, "x", V, " with ", nnz_target, " nonzeros...")
    
    # We populate the matrix
    colptr = Vector{Int}(undef, V + 1)
    colptr[1] = 1
    elements_per_col = div(nnz_target, V)
    for i in 2:V
        colptr[i] = colptr[i-1] + elements_per_col
    end
    colptr[V+1] = nnz_target + 1
    
    rowval = Vector{Int}(undef, nnz_target)
    nzval = rand(nnz_target)
    for col in 1:V
        start_idx = colptr[col]
        end_idx = colptr[col+1] - 1
        # Random row indices for this column
        rowval[start_idx:end_idx] = sort(rand(1:V, end_idx - start_idx + 1))
    end
    
    K_sem = SparseMatrixCSC(V, V, colptr, rowval, nzval)
    println("Matrix created. Size in memory: ", Base.summarysize(K_sem) / 1024^2, " MB")
    
    # Measure row slice
    cid = 10000
    println("Timing row slice: K_sem[cid, :]...")
    t0 = time()
    row = K_sem[cid, :]
    t1 = time()
    println("Row slice completed in ", round(t1 - t0; digits=4), " seconds.")
    
    # Measure conversion to Vector
    t2 = time()
    v_row = Vector(row)
    t3 = time()
    println("Conversion to Vector completed in ", round(t3 - t2; digits=4), " seconds.")
    
    # Measure sortperm
    t4 = time()
    sorted_ids = sortperm(v_row; rev=true)[1:30]
    t5 = time()
    println("Sortperm completed in ", round(t5 - t4; digits=4), " seconds.")
end

test_slice()
