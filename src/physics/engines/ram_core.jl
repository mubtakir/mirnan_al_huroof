"""
الذاكرة الزمنية المتلاشية — AttractorMemory (RAM Core) (محسّنة).

تخزن المخرجات السابقة كجواذب طورية وتسترجعها كسياق.
تستخدم اضمحلالاً أسياً exp(-decay × age) لتقليل تأثير الماضي.
"""

module RAMCore

using LinearAlgebra, Statistics

using ..Constants: ENHANCED_DIM
using ..WordPhysics: compute_word_enhanced_vector, phase_similarity_enhanced,
       phase_distance_enhanced

export AttractorMemory, observe!, resonate, retrieve_context, tick!, clear!

"""
    AttractorMemory

ذاكرة زمنية متلاشية — تخزين جواذب طورية.

الحقول:
- `centers`: مراكز الجذب (متجهات ENHANCED_DIM)
- `sigmas`: عرض النطاق σ لكل جاذب
- `word_seqs`: تسلسلات الكلمات المخزنة
- `coupling_masks`: أوزان الاقتران
- `ages`: أعمار الجواذب للاضمحلال
- `decay`: ثابت الاضمحلال
- `merge_cos`: عتبة الدمج (تشابه جيب التمام)
"""
mutable struct AttractorMemory
    centers::Vector{Vector{Float64}}
    sigmas::Vector{Float64}
    word_seqs::Vector{Vector{String}}
    coupling_masks::Vector{Dict{String,Float64}}
    ages::Vector{Float64}
    decay::Float64
    merge_cos::Float64
    pv_cache::Dict{String,Vector{Float64}}

    function AttractorMemory(; decay::Float64=0.01, merge_cos::Float64=0.92)
        return new(
            Vector{Float64}[],
            Float64[],
            Vector{String}[],
            Dict{String,Float64}[],
            Float64[],
            decay, merge_cos,
            Dict{String,Vector{Float64}}(),
        )
    end
end

function _get_pv(ram::AttractorMemory, word::String)
    if !haskey(ram.pv_cache, word)
        ram.pv_cache[word] = Float64.(compute_word_enhanced_vector(word))
    end
    return ram.pv_cache[word]
end

"""
    tick!(ram::AttractorMemory)

تقديم العمر لجميع الجواذب.
"""
function tick!(ram::AttractorMemory)
    ram.ages .+= 1.0
end

"""
    observe!(ram::AttractorMemory, words::Vector{String}; sigma=nothing) -> Int

تخزين (أو دمج) تسلسل كلمات جديد كجاذب طوري.
"""
function observe!(ram::AttractorMemory, words::Vector{String}; sigma::Union{Float64,Nothing}=nothing)
    if isempty(words)
        return -1
    end

    pvs = [_get_pv(ram, w) for w in words]
    pv_mat = reduce(hcat, pvs)'
    phi_center = vec(mean(pv_mat; dims=1))
    nrm = norm(phi_center)
    if nrm > 1e-10
        phi_center ./= nrm
    end

    sigma_val = if sigma === nothing
        dists = [norm(pv - phi_center) for pv in pvs]
        mean(dists) + 0.01
    else
        sigma
    end

    mask = Dict{String,Float64}(w => 1.0 for w in Set(words))

    # دمج مع جاذب موجود إن كان التشابه عالياً
    for i in 1:length(ram.centers)
        sim = dot(ram.centers[i], phi_center)
        if sim > ram.merge_cos
            n1 = length(ram.word_seqs[i])
            n2 = length(words)
            ram.centers[i] .= (n1 .* ram.centers[i] .+ n2 .* phi_center) ./ (n1 + n2)
            nrm2 = norm(ram.centers[i])
            if nrm2 > 1e-10
                ram.centers[i] ./= nrm2
            end
            ram.sigmas[i] = min(ram.sigmas[i], sigma_val)
            for (w, wt) in mask
                ram.coupling_masks[i][w] = get(ram.coupling_masks[i], w, 0.0) + wt
            end
            append!(ram.word_seqs[i], words)
            return i
        end
    end

    # إضافة جاذب جديد
    push!(ram.centers, phi_center)
    push!(ram.sigmas, sigma_val)
    push!(ram.word_seqs, copy(words))
    push!(ram.coupling_masks, mask)
    push!(ram.ages, 0.0)
    return length(ram.centers)
end

"""
    resonate(ram::AttractorMemory, phi_current::AbstractVector; top_k=3) -> Vector{Tuple{Float64,Int}}

استرجاع أقوى الجواذب المتجاوبة مع المتجه الحالي.
"""
function resonate(ram::AttractorMemory, phi_current::AbstractVector; top_k::Int=3)
    if isempty(ram.centers)
        return Tuple{Float64,Int}[]
    end

    scores = Tuple{Float64,Int}[]
    for i in 1:length(ram.centers)
        c = ram.centers[i]
        s = ram.sigmas[i]
        age = ram.ages[i]
        d = min(length(phi_current), length(c))
        diff = phi_current[1:d] .- c[1:d]
        dist2 = sum(diff .^ 2)
        score = exp(-dist2 / (2.0 * s^2 + 1e-10))
        decay_factor = exp(-ram.decay * age)
        score *= decay_factor
        push!(scores, (score, i))
    end

    sort!(scores; by=x -> -x[1])
    return scores[1:min(top_k, end)]
end

"""
    retrieve_context(ram::AttractorMemory, phi_current; max_words=10) -> Vector{String}

استرجاع كلمات السياق من أقوى الجواذب.
"""
function retrieve_context(ram::AttractorMemory, phi_current::AbstractVector; max_words::Int=10)
    hits = resonate(ram, phi_current; top_k=2)
    if isempty(hits)
        return String[]
    end

    words = String[]
    seen = Set{String}()
    for (score, idx) in hits
        score < 0.1 && continue
        for w in ram.word_seqs[idx]
            if w ∉ seen
                push!(words, w)
                push!(seen, w)
                length(words) >= max_words && return words
            end
        end
    end
    return words
end

"""
    clear!(ram::AttractorMemory)

مسح كامل للذاكرة.
"""
function clear!(ram::AttractorMemory)
    empty!(ram.centers)
    empty!(ram.sigmas)
    empty!(ram.word_seqs)
    empty!(ram.coupling_masks)
    empty!(ram.ages)
end

"""
    size(ram::AttractorMemory) -> Int

عدد الجواذب المخزنة.
"""
Base.size(ram::AttractorMemory) = length(ram.centers)

end # module RAMCore
