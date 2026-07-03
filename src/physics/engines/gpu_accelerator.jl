"""
مسرّع كرت الشاشة (GPU Accelerator) — لعمليات التوليد فقط (وليس التدريب).
يدعم تجميع عمليات الضرب النقطي وجيب التمام للمرشحين دفعة واحدة.
"""
module GpuAcceleratorModule

using LinearAlgebra
using ..Constants

export GpuContext, gpu_available, gpu_init!, gpu_batch_fast_score,
       gpu_release!, gpu_batch_gravity, gpu_batch_dot

const CUDA_AVAIL = Ref{Bool}(false)
const CUDA_MODULE = Ref{Any}(nothing)

function _short_error(e)
    msg = sprint(showerror, e)
    msg = replace(msg, r"\s+" => " ")
    length(msg) > 240 && (msg = msg[1:240] * "...")
    return "$(typeof(e)): $msg"
end

function _try_load_cuda()
    if CUDA_MODULE[] !== nothing
        return true
    end
    try
        CUDA_MODULE[] = Base.require(Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))
        CUDA_AVAIL[] = true
        return true
    catch e
        CUDA_AVAIL[] = false
        @warn "CUDA.jl غير متوفر — سيعمل التوليد على المعالج (CPU) فقط. للتثبيت: ] add CUDA" error=_short_error(e)
        return false
    end
end

gpu_available() = _try_load_cuda() && CUDA_AVAIL[]

mutable struct GpuContext
    active::Bool
    pv_matrix::Any            # CuMatrix{Float32} [dim × vocab_size] أو nothing
    pv_norms::Any             # CuVector{Float32} [vocab_size] أو nothing
    word_to_col::Dict{String,Int}  # كلمة → رقم العمود في المصفوفة
    dim::Int                  # البعد الفعلي المستخدم (PHASE_DIM)
    vocab_size::Int           # عدد الكلمات في المصفوفة
end

function GpuContext()
    return GpuContext(false, nothing, nothing, Dict{String,Int}(), 0, 0)
end

function gpu_init!(ctx::GpuContext, pv_lookup::Function, vocab::Dict{String,Int},
                   scoring_weights::Dict{String,Float64}; dim::Int=PHASE_DIM)
    if !gpu_available()
        ctx.active = false
        return ctx
    end
    CUDA = CUDA_MODULE[]
    try
        words = collect(keys(vocab))
        n = length(words)
        n == 0 && (ctx.active = false; return ctx)

        cpu_mat = zeros(Float32, dim, n)
        valid_cols = trues(n)
        for (col, w) in enumerate(words)
            pv = try pv_lookup(w) catch; nothing end
            if pv !== nothing && length(pv) >= dim
                cpu_mat[:, col] .= Float32.(view(pv, 1:dim))
            else
                valid_cols[col] = false
            end
        end

        if count(valid_cols) < 10
            ctx.active = false
            return ctx
        end

        gpu_mat = CUDA.CuArray(cpu_mat)
        gpu_norms = CUDA.sqrt.(CUDA.sum(gpu_mat .^ 2; dims=1))[:]
        mapping = Dict{String,Int}(w => i for (i, w) in enumerate(words))

        ctx.active = true
        ctx.pv_matrix = gpu_mat
        ctx.pv_norms = gpu_norms
        ctx.word_to_col = mapping
        ctx.dim = dim
        ctx.vocab_size = n
    catch e
        @warn "فشل تهيئة GPU — سيعمل التوليد على CPU" error=_short_error(e)
        ctx.active = false
    end
    return ctx
end

function gpu_release!(ctx::GpuContext)
    ctx.active = false
    ctx.pv_matrix = nothing
    ctx.pv_norms = nothing
    ctx.word_to_col = Dict{String,Int}()
    ctx.dim = 0
    ctx.vocab_size = 0
    GC.gc()
    return nothing
end

function _get_candidate_indices(ctx::GpuContext, candidates::Vector{String})
    indices = Int[]
    for w in candidates
        col = get(ctx.word_to_col, w, 0)
        if col > 0
            push!(indices, col)
        else
            push!(indices, -1)
        end
    end
    return indices
end

function gpu_batch_cosine(ctx::GpuContext, candidates::Vector{String},
                          target_pv::Vector{Float64}, pv_lookup::Function)
    CUDA = CUDA_MODULE[]
    n = length(candidates)
    dim = ctx.dim
    col_indices = _get_candidate_indices(ctx, candidates)

    # Fallback to CPU for words not in GPU cache
    results = zeros(Float64, n)
    gpu_cands = Int[]; gpu_positions = Int[]
    for (i, col) in enumerate(col_indices)
        if col > 0
            push!(gpu_cands, col)
            push!(gpu_positions, i)
        else
            # Need to compute on CPU for this word
            w = candidates[i]
            w_pv = try pv_lookup(w) catch; nothing end
            if w_pv !== nothing && length(w_pv) >= dim
                nw = norm(view(w_pv, 1:dim))
                nt = norm(view(target_pv, 1:dim))
                if nw > 1e-10 && nt > 1e-10
                    results[i] = max(0.0, dot(view(w_pv, 1:dim), view(target_pv, 1:dim)) / (nw * nt))
                end
            end
        end
    end

    if !isempty(gpu_cands)
        target_gpu = CUDA.CuArray(Float32.(view(target_pv, 1:dim)))
        target_norm = CUDA.norm(target_gpu)

        cand_mat = ctx.pv_matrix[:, gpu_cands]        # [dim × k]
        cand_norms = ctx.pv_norms[gpu_cands]           # [k]
        dots_k = cand_mat' * target_gpu               # [k]
        safe_norms = cand_norms .* target_norm .+ 1e-10f0
        sims_gpu = dots_k ./ safe_norms
        sims_cpu = Vector{Float64}(sims_gpu)

        for (j, pos) in enumerate(gpu_positions)
            results[pos] = max(0.0, sims_cpu[j])
        end
    end

    return results
end

function gpu_batch_prompt_align(ctx::GpuContext, candidates::Vector{String},
                                 prompt_pvs::Vector{<:AbstractVector},
                                 pv_lookup::Function)
    CUDA = CUDA_MODULE[]
    n = length(candidates)
    m = length(prompt_pvs)
    dim = ctx.dim
    (m == 0) && return zeros(Float64, n)

    col_indices = _get_candidate_indices(ctx, candidates)

    # Build prompt matrix on GPU
    prompt_cpu = zeros(Float32, dim, m)
    prompt_norms_cpu = zeros(Float32, m)
    for j in 1:m
        pp = prompt_pvs[j]
        L = min(length(pp), dim)
        prompt_cpu[1:L, j] .= Float32.(view(pp, 1:L))
        prompt_norms_cpu[j] = Float32(norm(view(pp, 1:L)))
    end
    prompt_gpu = CUDA.CuArray(prompt_cpu)
    prompt_norms_gpu = CUDA.CuArray(prompt_norms_cpu)

    results = zeros(Float64, n)
    gpu_cands = Int[]; gpu_positions = Int[]
    for (i, col) in enumerate(col_indices)
        if col > 0
            push!(gpu_cands, col)
            push!(gpu_positions, i)
        else
            w = candidates[i]
            w_pv = try pv_lookup(w) catch; nothing end
            if w_pv !== nothing && length(w_pv) >= dim
                nw = norm(view(w_pv, 1:dim))
                if nw > 1e-10
                    s = 0.0
                    for j in 1:m
                        pn = prompt_norms_cpu[j]
                        if pn > 1e-10
                            s += max(0.0, dot(view(w_pv, 1:dim), view(prompt_pvs[j], 1:min(length(prompt_pvs[j]), dim))) / (nw * pn))
                        end
                    end
                    results[i] = s / m
                end
            end
        end
    end

    if !isempty(gpu_cands)
        cand_mat = ctx.pv_matrix[:, gpu_cands]             # [dim × k]
        cand_norms = ctx.pv_norms[gpu_cands]               # [k]
        dots_gpu = cand_mat' * prompt_gpu                  # [k × m]
        norm_ij = cand_norms * prompt_norms_gpu'           # [k × m]
        safe_norms = norm_ij .+ 1e-10f0
        sims_ij = dots_gpu ./ safe_norms
        avg_sims = CUDA.mean(sims_ij; dims=2)[:]           # [k]
        avg_cpu = Vector{Float64}(avg_sims)

        for (j, pos) in enumerate(gpu_positions)
            results[pos] = max(0.0, avg_cpu[j])
        end
    end

    return results
end

function gpu_batch_fast_score(ctx::GpuContext, candidates::Vector{String},
                               target_pv::Vector{Float64},
                               prompt_pvs::Vector{<:AbstractVector},
                               used_set::Set{String},
                               weights::Dict{String,Float64},
                               pv_lookup::Function)
    n = length(candidates)
    n == 0 && return Tuple{Float64,String}[]

    w_align = get(weights, "align", 0.0)
    w_palign = get(weights, "prompt_align", 0.0)
    w_div = get(weights, "diversity", 0.0)

    # Batch cosine similarity with target
    aligns = gpu_batch_cosine(ctx, candidates, target_pv, pv_lookup)

    # Batch prompt alignment
    prompt_aligns = gpu_batch_prompt_align(ctx, candidates, prompt_pvs, pv_lookup)

    # Diversity (CPU — per-word, very cheap)
    used_list = collect(used_set)
    n_used = max(1, length(used_list))

    scored = Tuple{Float64,String}[]
    for (i, w) in enumerate(candidates)
        wc = count(x -> x == w, used_list)
        diversity = 1.0 - wc / n_used

        s = w_align * aligns[i] + w_palign * prompt_aligns[i] + w_div * diversity
        push!(scored, (clamp(s, -5.0, 5.0), w))
    end

    sort!(scored; by=x -> -x[1])
    return scored
end

function gpu_batch_dot(ctx::GpuContext, candidates::Vector{String},
                       target_pv::Vector{Float64}, pv_lookup::Function)
    n = length(candidates)
    dim = ctx.dim
    col_indices = _get_candidate_indices(ctx, candidates)
    CUDA = CUDA_MODULE[]

    results = zeros(Float64, n)
    gpu_cands = Int[]; gpu_positions = Int[]
    for (i, col) in enumerate(col_indices)
        if col > 0
            push!(gpu_cands, col)
            push!(gpu_positions, i)
        else
            w = candidates[i]
            w_pv = try pv_lookup(w) catch; nothing end
            if w_pv !== nothing && length(w_pv) >= dim
                results[i] = dot(view(w_pv, 1:dim), view(target_pv, 1:dim))
            end
        end
    end

    if !isempty(gpu_cands)
        target_gpu = CUDA.CuArray(Float32.(view(target_pv, 1:dim)))
        cand_mat = ctx.pv_matrix[:, gpu_cands]           # [dim × k]
        dots_gpu = cand_mat' * target_gpu                # [k]
        dots_cpu = Vector{Float64}(dots_gpu)
        for (j, pos) in enumerate(gpu_positions)
            results[pos] = dots_cpu[j]
        end
    end

    return results
end

function gpu_batch_gravity(ctx::GpuContext, candidates::Vector{String},
                            context_pvs::Vector{<:AbstractVector},
                            context_masses::Vector{Float64},
                            candidate_masses::Vector{Float64},
                            pv_lookup::Function)
    n = length(candidates)
    c_len = length(context_pvs)
    dim = ctx.dim
    CUDA = CUDA_MODULE[]
    (n == 0 || c_len == 0) && return zeros(Float64, n)

    col_indices = _get_candidate_indices(ctx, candidates)

    # Compute context norms and build GPU context matrix
    ctx_norms_sq = zeros(Float64, c_len)
    ctx_gpu = CUDA.CuArray(zeros(Float32, dim, c_len))
    for j in 1:c_len
        cp = context_pvs[j]
        L = min(length(cp), dim)
        ctx_norms_sq[j] = sum(abs2, view(cp, 1:L))
        ctx_gpu[:, j] .= Float32.(view(cp, 1:L))
    end

    results = zeros(Float64, n)
    gpu_cands = Int[]; gpu_positions = Int[]
    for (i, col) in enumerate(col_indices)
        if col > 0
            push!(gpu_cands, col)
            push!(gpu_positions, i)
        else
            w = candidates[i]
            w_pv = try pv_lookup(w) catch; nothing end
            if w_pv !== nothing && length(w_pv) >= dim
                w_norm_sq = sum(abs2, view(w_pv, 1:dim))
                m_word = Float32(candidate_masses[i])
                g_score = 0.0
                for j in 1:c_len
                    dist_sq = max(0.0, w_norm_sq + ctx_norms_sq[j] - 2.0 * dot(view(w_pv, 1:dim), view(context_pvs[j], 1:min(length(context_pvs[j]), dim))))
                    r = sqrt(dist_sq) + 1e-6
                    g_score += m_word * Float32(context_masses[j]) / (r^2 + 0.01f0) * exp(-0.3f0 * (j - 1))
                end
                results[i] = g_score
            end
        end
    end

    if !isempty(gpu_cands)
        cand_mat = ctx.pv_matrix[:, gpu_cands]                         # [dim × k]
        cand_norms_sq = ctx.pv_norms[gpu_cands] .^ 2                  # [k]
        k = length(gpu_cands)

        # Compute all dot products at once: [k × c_len]
        dots_ij = cand_mat' * ctx_gpu                                   # [k × c_len]

        # Compute distances: dist^2 = ||cand||^2 + ||ctx||^2 - 2<cand,ctx>
        w_norms_sq_mat = reshape(Float32.(collect(cand_norms_sq)), :, 1)  # [k × 1]
        ctx_norms_sq_mat = reshape(Float32.(collect(ctx_norms_sq)), 1, :)  # [1 × c_len]
        dist_sq_mat = w_norms_sq_mat .+ ctx_norms_sq_mat .- 2.0f0 .* dots_ij
        dist_sq_mat = CUDA.max.(dist_sq_mat, 0.0f0)
        r_mat = CUDA.sqrt.(dist_sq_mat) .+ 1e-6f0

        # Gravity: F = m₁m₂ / (r² + 0.01) * exp(-0.3*j)
        cand_m = Float32.(candidate_masses[gpu_positions])
        ctx_m = Float32.(context_masses)
        grav_mat = (cand_m * ctx_m') ./ (r_mat .^ 2 .+ 0.01f0)   # [k × c_len]
        decay = exp.(-0.3f0 .* Float32.(0:(c_len-1)))'             # [1 × c_len]
        grav_weighted = grav_mat .* decay                          # [k × c_len]
        grav_scores = CUDA.sum(grav_weighted; dims=2)[:]           # [k]
        grav_cpu = Vector{Float64}(grav_scores)

        for (j, pos) in enumerate(gpu_positions)
            results[pos] = grav_cpu[j]
        end
    end

    return results
end

end # module GpuAcceleratorModule
