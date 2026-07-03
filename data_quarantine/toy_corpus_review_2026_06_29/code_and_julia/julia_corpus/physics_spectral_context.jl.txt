"""SpectralContext — طبقة طيفية سياقية."""
module SpectralContext
using LinearAlgebra, SparseArrays
export build_contextual_spectra, context_spectral_score

function build_contextual_spectra(K, all_pv; normalize=true)
    V = size(K,1); D = size(all_pv,2)
    row_sums = vec(sum(K; dims=2)); row_sums[row_sums .== 0] .= 1
    K_norm = K .* (1.0 ./ row_sums)
    spectra = K_norm * all_pv
    if normalize
        for i in 1:V
            nrm = norm(view(spectra,i,:)); nrm > 1e-10 && (spectra[i,:] ./= nrm)
        end
    end
    return spectra
end

function context_spectral_score(word_id::Int, context_ids::Vector{Int}, spectra::AbstractMatrix)
    isempty(context_ids) && return 0.0
    ctx_sum = zeros(Float64, size(spectra,2)); n=0
    for cid in context_ids[max(1, end-min(3, length(context_ids))):end]
        if 1 <= cid <= size(spectra,1); ctx_sum .+= spectra[cid,:]; n+=1; end
    end
    n == 0 && return 0.0
    ctx_spec = ctx_sum ./ n; ctx_norm = norm(ctx_spec); ctx_norm < 1e-10 && return 0.0
    ctx_spec ./= ctx_norm
    1 <= word_id <= size(spectra,1) || return 0.0
    cand_spec = spectra[word_id,:]; cand_norm = norm(cand_spec); cand_norm < 1e-10 && return 0.0
    return dot(ctx_spec, cand_spec) / cand_norm
end
end
