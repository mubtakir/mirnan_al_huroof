module LexicalOracleModule

using LinearAlgebra, Statistics

using ..WordPhysics: compute_word_phase_vector, compute_extended_phase_vector,
    compute_word_enhanced_vector, compute_enhanced_vector,
    compute_word_frequency, compute_word_energy, compute_word_mass,
    phase_similarity, phase_similarity_enhanced, _extract_root_light
using ..CliffordMath: word_to_multivector, clifford_similarity

export WordPhysicsProfile, LexicalNeighbor, CoinedWord,
       word_physics_profile, phonosemantic_quality,
       nearest_phase_words, analyze_unknown_word,
       coin_word_for_concept

struct WordPhysicsProfile
    word::String
    letters::Vector{Char}
    root::String
    frequency::Float64
    energy::Float64
    mass::Float64
    phase_norm::Float64
    enhanced_norm::Float64
    clifford_norm::Float64
    harmony::Float64
    stability::Float64
    heaviness::Float64
end

struct LexicalNeighbor
    word::String
    phase_similarity::Float64
    clifford_similarity::Float64
    hybrid_score::Float64
end

struct CoinedWord
    word::String
    score::Float64
    phase_alignment::Float64
    aesthetic_score::Float64
    source_root::String
end

const _DIACRITICS = Set(['ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ـ'])
const _CLIFFORD_CHAR_MAP = Dict(
    'ا' => 'أ', 'إ' => 'أ', 'ٱ' => 'أ',
    'ة' => 'ه', 'ى' => 'ي',
)

const _ARABIC_PATTERNS = [
    "فعل", "فعيل", "فعول", "فعال", "فعلان",
    "فاعل", "مفعل", "مفعلة", "مفعال",
    "تفعيل", "تفاعل", "انفعال", "استفعال",
]

function _clean_letters(word::AbstractString; clifford::Bool=false)
    letters = Char[]
    for ch in String(word)
        ch in _DIACRITICS && continue
        isspace(ch) && continue
        isletter(ch) || continue
        c = lowercase(ch)
        if clifford
            c = get(_CLIFFORD_CHAR_MAP, c, c)
        end
        push!(letters, c)
    end
    return letters
end

_clifford_word(word::AbstractString) = String(_clean_letters(word; clifford=true))

function _safe_extended(word::AbstractString)
    try
        return Float64.(compute_extended_phase_vector(String(word)))
    catch e
        @debug "LexicalOracle: extended vector failed for '$word': $e"
        return Float64[]
    end
end

function _safe_phase(word::AbstractString)
    try
        return Float64.(compute_word_phase_vector(String(word)))
    catch e
        @debug "LexicalOracle: phase vector failed for '$word': $e"
        return Float64[]
    end
end

function _safe_clifford_norm(word::AbstractString)
    try
        return norm(word_to_multivector(_clifford_word(word)))
    catch e
        @debug "LexicalOracle: Clifford norm failed for '$word': $e"
        return 0.0
    end
end

function _safe_clifford_similarity(a::AbstractString, b::AbstractString)
    try
        ma = word_to_multivector(_clifford_word(a))
        mb = word_to_multivector(_clifford_word(b))
        (norm(ma) < 1e-10 || norm(mb) < 1e-10) && return 0.0
        return clifford_similarity(ma, mb)
    catch e
        @debug "LexicalOracle: Clifford similarity failed for '$a'/'$b': $e"
        return 0.0
    end
end

function _cosine(a::AbstractVector, b::AbstractVector)
    (isempty(a) || isempty(b) || length(a) != length(b)) && return 0.0
    na, nb = norm(a), norm(b)
    (na < 1e-10 || nb < 1e-10) && return 0.0
    return clamp(dot(a, b) / (na * nb), -1.0, 1.0)
end

function _root_string(word::AbstractString)
    try
        return join(_extract_root_light(String(word)))
    catch e
        @debug "LexicalOracle: root extraction failed for '$word': $e"
        return ""
    end
end

function _letter_vectors(word::AbstractString)
    vecs = Vector{Float64}[]
    for ch in _clean_letters(word; clifford=true)
        try
            push!(vecs, Float64.(compute_enhanced_vector(ch)))
        catch e
            @debug "LexicalOracle: letter vector failed for '$ch': $e"
        end
    end
    return vecs
end

function _adjacent_similarities(word::AbstractString)
    vecs = _letter_vectors(word)
    length(vecs) < 2 && return Float64[1.0]
    sims = Float64[]
    for i in 1:length(vecs)-1
        push!(sims, phase_similarity_enhanced(vecs[i], vecs[i + 1]))
    end
    return sims
end

function phonosemantic_quality(word::AbstractString)
    sims = _adjacent_similarities(word)
    positive = clamp.(sims, 0.0, 1.0)
    harmony = isempty(positive) ? 0.0 : mean(positive)
    tension = length(sims) <= 1 ? 0.0 : std(sims)
    stability = clamp(exp(-2.0 * tension), 0.0, 1.0)
    frequency = compute_word_frequency(String(word))
    heaviness = clamp(log1p(frequency) / log1p(40.0), 0.0, 1.0)
    balance = 1.0 - min(1.0, abs(heaviness - 0.5) * 2.0)
    aesthetic = clamp(0.45 * harmony + 0.35 * stability + 0.20 * balance, 0.0, 1.0)
    return Dict{String,Any}(
        "word" => String(word),
        "harmony" => round(harmony; digits=4),
        "stability" => round(stability; digits=4),
        "heaviness" => round(heaviness; digits=4),
        "tension" => round(tension; digits=4),
        "aesthetic_score" => round(aesthetic; digits=4),
    )
end

function word_physics_profile(word::AbstractString)
    w = String(word)
    phase = _safe_phase(w)
    enhanced = try Float64.(compute_word_enhanced_vector(w)) catch e
        @debug "LexicalOracle: enhanced vector failed for '$w': $e"
        Float64[]
    end
    q = phonosemantic_quality(w)
    return WordPhysicsProfile(
        w,
        _clean_letters(w),
        _root_string(w),
        compute_word_frequency(w),
        compute_word_energy(w),
        compute_word_mass(w),
        isempty(phase) ? 0.0 : norm(phase),
        isempty(enhanced) ? 0.0 : norm(enhanced),
        _safe_clifford_norm(w),
        Float64(q["harmony"]),
        Float64(q["stability"]),
        Float64(q["heaviness"]),
    )
end

function _valid_vocab_word(word::AbstractString)
    w = strip(String(word))
    length(w) >= 2 || return false
    any(isletter, w) || return false
    occursin(r"[#<>{}\[\]\(\)=+*/\\|@%^~`$]", w) && return false
    return true
end

function nearest_phase_words(word::AbstractString, vocab::Dict{String,Int};
                             top_k::Int=10, min_score::Float64=-1.0)
    target = _safe_extended(word)
    isempty(target) && return LexicalNeighbor[]
    results = LexicalNeighbor[]
    for candidate in keys(vocab)
        candidate == String(word) && continue
        _valid_vocab_word(candidate) || continue
        cv = _safe_extended(candidate)
        isempty(cv) && continue
        phase_sim = _cosine(target, cv)
        phase_sim >= min_score || continue
        cliff_sim = _safe_clifford_similarity(word, candidate)
        hybrid = clamp(0.75 * max(phase_sim, 0.0) + 0.25 * max(cliff_sim, 0.0), 0.0, 1.0)
        push!(results, LexicalNeighbor(candidate, phase_sim, cliff_sim, hybrid))
    end
    sort!(results; by=x -> x.hybrid_score, rev=true)
    return results[1:min(top_k, length(results))]
end

function _neighbor_dict(n::LexicalNeighbor)
    return Dict{String,Any}(
        "word" => n.word,
        "phase_similarity" => round(n.phase_similarity; digits=4),
        "clifford_similarity" => round(n.clifford_similarity; digits=4),
        "hybrid_score" => round(n.hybrid_score; digits=4),
    )
end

function _profile_dict(p::WordPhysicsProfile)
    return Dict{String,Any}(
        "word" => p.word,
        "letters" => [string(ch) for ch in p.letters],
        "root" => p.root,
        "frequency" => round(p.frequency; digits=4),
        "energy" => round(p.energy; digits=4),
        "mass" => round(p.mass; digits=4),
        "phase_norm" => round(p.phase_norm; digits=4),
        "enhanced_norm" => round(p.enhanced_norm; digits=4),
        "clifford_norm" => round(p.clifford_norm; digits=4),
        "harmony" => round(p.harmony; digits=4),
        "stability" => round(p.stability; digits=4),
        "heaviness" => round(p.heaviness; digits=4),
    )
end

function analyze_unknown_word(word::AbstractString, vocab::Dict{String,Int}; top_k::Int=8)
    neighbors = nearest_phase_words(word, vocab; top_k=top_k)
    best = isempty(neighbors) ? 0.0 : neighbors[1].hybrid_score
    known = haskey(vocab, String(word))
    novelty = known ? 0.0 : clamp(1.0 - best, 0.0, 1.0)
    return Dict{String,Any}(
        "word" => String(word),
        "known_in_vocab" => known,
        "profile" => _profile_dict(word_physics_profile(word)),
        "quality" => phonosemantic_quality(word),
        "nearest_words" => [_neighbor_dict(n) for n in neighbors],
        "interpretation_confidence" => round(best; digits=4),
        "novelty_score" => round(novelty; digits=4),
    )
end

function _is_arabic_word(word::AbstractString)
    return any(c -> '\u0600' <= c <= '\u06FF', String(word))
end

function _apply_root_pattern(pattern::AbstractString, root::Vector{Char})
    length(root) >= 3 || return ""
    r1, r2, r3 = root[1], root[2], root[3]
    out = IOBuffer()
    for ch in String(pattern)
        if ch == 'ف'
            print(out, r1)
        elseif ch == 'ع'
            print(out, r2)
        elseif ch == 'ل'
            print(out, r3)
        else
            print(out, ch)
        end
    end
    return String(take!(out))
end

function _candidate_roots(seed_words::Vector{String})
    roots = Vector{Vector{Char}}()
    for word in seed_words
        try
            root = _extract_root_light(word)
            length(root) >= 3 && push!(roots, root[1:3])
        catch e
            @debug "LexicalOracle: coin root failed for '$word': $e"
        end
    end
    return roots
end

function _english_candidates(seed_words::Vector{String})
    clean = [lowercase(filter(isletter, w)) for w in seed_words]
    filter!(!isempty, clean)
    isempty(clean) && return String[]
    first_word = clean[1]
    last_word = clean[end]
    a = first_word[1:max(1, cld(length(first_word), 2))]
    b = last_word[max(1, fld(length(last_word), 2)):end]
    base = a * b
    return unique([base, base * "ic", base * "ion", "meta" * base])
end

function _target_vector(seed_words::Vector{String})
    vectors = [_safe_extended(w) for w in seed_words]
    filter!(!isempty, vectors)
    isempty(vectors) && return Float64[]
    target = zeros(Float64, length(vectors[1]))
    n = 0
    for v in vectors
        length(v) == length(target) || continue
        target .+= v ./ (norm(v) + 1e-10)
        n += 1
    end
    n == 0 && return Float64[]
    target ./= n
    return target
end

function coin_word_for_concept(seed_words::Vector{String};
                               vocab::Union{Nothing,Dict{String,Int}}=nothing,
                               max_candidates::Int=20,
                               allow_existing::Bool=false)
    isempty(seed_words) && return CoinedWord[]
    target = _target_vector(seed_words)
    isempty(target) && return CoinedWord[]

    candidates = String[]
    if any(_is_arabic_word, seed_words)
        for root in _candidate_roots(seed_words)
            for pattern in _ARABIC_PATTERNS
                candidate = _apply_root_pattern(pattern, root)
                !isempty(candidate) && push!(candidates, candidate)
            end
        end
    else
        append!(candidates, _english_candidates(seed_words))
    end
    unique!(candidates)

    scored = CoinedWord[]
    for candidate in candidates
        _valid_vocab_word(candidate) || continue
        if vocab !== nothing && !allow_existing && haskey(vocab, candidate)
            continue
        end
        cv = _safe_extended(candidate)
        isempty(cv) && continue
        alignment = clamp(_cosine(target, cv), -1.0, 1.0)
        quality = phonosemantic_quality(candidate)
        aesthetic = Float64(quality["aesthetic_score"])
        novelty_bonus = vocab === nothing || !haskey(vocab, candidate) ? 0.08 : 0.0
        score = clamp(0.70 * max(alignment, 0.0) + 0.22 * aesthetic + novelty_bonus, 0.0, 1.0)
        push!(scored, CoinedWord(candidate, score, alignment, aesthetic, _root_string(candidate)))
    end
    sort!(scored; by=x -> x.score, rev=true)
    return scored[1:min(max_candidates, length(scored))]
end

end # module LexicalOracleModule
