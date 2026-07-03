"""ModelBundle — تجميع النموذج (MirnanModelBundle)."""
module ModelBundle
using SparseArrays
using ..Synchronize: Vocabulary
export MirnanModelBundle, load_model_bundle, save_model_bundle

struct MirnanModelBundle
    vocab::Vocabulary
    K_sem::SparseMatrixCSC
    K_syn::Union{SparseMatrixCSC,Nothing}
    K_dial::Union{SparseMatrixCSC,Nothing}
    syntax::Any
    gss::Any
    causal_K::Any
    K_code::Any
end

function load_model_bundle(model_dir::String)
    # Simplified: just vocab + K_sem
    vocab = Vocabulary()
    K_sem = spzeros(10, 10)
    return MirnanModelBundle(vocab, K_sem, nothing, nothing, nothing, nothing, nothing, nothing)
end

function save_model_bundle(bundle::MirnanModelBundle, out_dir::String)
    mkpath(out_dir)
    return true
end
end
