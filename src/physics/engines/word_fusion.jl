"""
Word Fusion Engine — الاندماج الهندسي للكلمة.
Simplified version using 27D phase vectors (no Clifford algebra).
Word = weighted linear fusion of letter phase vectors.
"""
module WordFusion

using LinearAlgebra
using ..LetterDB: LetterDatabase, get_vector
using ..CliffordMath

export _get_letter_vectors, linear_fusion, interaction_tensor, eigen_fusion,
       geometric_word_fusion, decompose_fusion, full_word_analysis

const DIM_NAMES = [
    "concentration", "internal_external", "stability_motion", "density",
    "temperature", "time_accumulation", "time_peak", "time_discharge",
    "motion_linear", "motion_rotary", "motion_pulse", "motion_stretch",
    "motion_slip", "motion_air", "axis_v", "mass",
    "hardness_solid", "penetration", "charge", "reference_self",
    "space_extensionality", "time_causality",
    "dim_23", "dim_24", "dim_25", "dim_26", "dim_27",
]

function _get_letter_vectors(word::String, db::LetterDatabase)
    return [Float64.(get_vector(db, string(ch))) for ch in word if !isspace(ch)]
end

function linear_fusion(letter_vectors::Vector{<:AbstractVector}; position_weights=nothing)
    if isempty(letter_vectors)
        return zeros(Float64, 27)
    end
    lvs = [Float64.(v[1:min(27, end)]) for v in letter_vectors]

    if position_weights === nothing
        n = length(lvs)
        pw = Float64[3.5, 2.5, 2.0]
        for i in 1:(n - 3)
            push!(pw, 1.5 / sqrt(i))
        end
        if n > 0
            pw = pw[1:n]
        end
    else
        pw = Float64.(position_weights)
    end

    weighted = zeros(Float64, 27)
    for (i, v) in enumerate(lvs)
        w = i <= length(pw) ? pw[i] : 1.0
        weighted .+= v .* w
    end
    total_w = sum(pw[1:min(end, length(lvs))])
    if total_w > 0
        weighted ./= total_w
    end
    nrm = LinearAlgebra.norm(weighted)
    return nrm > 1e-10 ? weighted ./ nrm : weighted
end

function interaction_tensor(letter_vectors::Vector{<:AbstractVector})
    if isempty(letter_vectors)
        return zeros(Float64, 27, 27)
    end
    lvs = [Float64.(v[1:min(27, end)]) for v in letter_vectors]
    if length(lvs) < 2
        return lvs[1] * lvs[1]'
    end
    tensor = zeros(Float64, 27, 27)
    count = 0
    for i in 1:(length(lvs)-1)
        tensor .+= lvs[i] * lvs[i+1]'
        count += 1
    end
    return count > 0 ? tensor ./ count : tensor
end

function eigen_fusion(letter_vectors::Vector{<:AbstractVector})
    tensor = interaction_tensor(letter_vectors)
    F = svd(tensor)
    s = F.S
    U = F.U
    dominant_mode = U[:, 1] .* s[1]

    top_dims_idx = sortperm(abs.(U[:, 1]); rev=true)[1:min(5, end)]
    top_contrib = Tuple{String,Float64}[]
    for i in top_dims_idx
        push!(top_contrib, (DIM_NAMES[i], round(U[i, 1]; digits=4)))
    end

    emergent = dominant_mode ./ (LinearAlgebra.norm(dominant_mode) + 1e-10)
    energy_conc = s[1]^2 / (sum(s .^ 2) + 1e-10)

    return Dict(
        "dominant_eigenvalue" => round(s[1]; digits=4),
        "eigenvalue_spectrum" => [round(x; digits=4) for x in s[1:min(5, end)]],
        "top_dimensions_eigen" => top_contrib,
        "emergent_vector_eigen" => emergent,
        "energy_concentration" => round(energy_conc; digits=4),
    )
end

function geometric_word_fusion(letter_vectors::Vector{<:AbstractVector})
    isempty(letter_vectors) && return CliffordMath.from_vector(zeros(22))
    acc = CliffordMath.from_vector(Float64.(letter_vectors[1][1:min(22, end)]))
    for v in letter_vectors[2:end]
        acc = acc * CliffordMath.from_vector(Float64.(v[1:min(22, end)]))
    end
    return acc
end

function decompose_fusion(mv)
    return Dict(
        "shared_essence" => getfield(mv, :s),
        "vector_norm" => norm(getfield(mv, :v)),
        "bivector_norm" => norm(getfield(mv, :b)),
    )
end

function full_word_analysis(word::String, letter_pv_fn::Function)
    letter_pvs = letter_pv_fn(word)
    if isempty(letter_pvs)
        return Dict("word" => word, "linear" => Dict("vector" => zeros(27), "norm" => 0.0))
    end
    lin_vec = linear_fusion(letter_pvs)
    eigen_r = eigen_fusion(letter_pvs)
    return Dict(
        "word" => word, "num_letters" => length(letter_pvs),
        "linear" => Dict("vector" => lin_vec, "norm" => round(LinearAlgebra.norm(lin_vec); digits=4)),
        "eigen" => eigen_r,
    )
end

end # module WordFusion
