"""
الدلالة العربية - تضمين الحروف واستخراج الجذور والمعاني
"""

module Semantics

export LetterEmbedding, WordEmbedding, SemanticVector,
       embed_letter, embed_word, compute_semantic_similarity,
       extract_root, compute_word_semantics, SemanticAnalyzer

# ═══════════════════════════════════════════════════════
# الثوابت
# ═══════════════════════════════════════════════════════

const ARABIC_LETTERS = [
    'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر',
    'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف',
    'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي'
]

const LETTER_FEATURES = Dict(
    'ا' => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # حرف مد
    'ب' => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # صامت شفوي
    'ت' => [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # صامت أسناني
    'ث' => [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # صامت أسناني رفيع
    'ج' => [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # صامت شجري
    'ح' => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # حرف حلقية
    'خ' => [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # حرف حلقية رفيعة
    'د' => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # صامت لثوي
    'ذ' => [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # صامت لثوي رفيع
    'ر' => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],  # صامت شبه حركي
    'ز' => [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # صامت سibilant
    'س' => [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # صامت أسناني
    'ش' => [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],  # صامت شجري
    'ص' => [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],  # صامت م強化
    'ض' => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],  # صامت لثوي م強化
    'ط' => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],  # صامت أسناني م强
    'ظ' => [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],  # صامت أسناني رفيع م强
    'ع' => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # حرف حلقية
    'غ' => [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # حرف حلقية رفيعة
    'ف' => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # صامت شفوي أسناني
    'ق' => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # صامت طرفي
    'ك' => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # صامت أسناني
    'ل' => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],  # صامت شبه حركي
    'م' => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # صامت شفوي أنفي
    'ن' => [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # صامت أنفي
    'ه' => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # حرف حلقية
    'و' => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # حرف مد
    'ي' => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # حرف مد شبه حركي
)

const COMMON_ROOTS = Dict(
    "كتب" => [0.8, 0.2, 0.1, 0.3, 0.5, 0.1, 0.2, 0.4, 0.1],
    "قرأ" => [0.7, 0.3, 0.2, 0.4, 0.6, 0.2, 0.3, 0.5, 0.2],
    "علم" => [0.9, 0.1, 0.3, 0.5, 0.7, 0.3, 0.4, 0.6, 0.3],
    "فهم" => [0.85, 0.15, 0.25, 0.45, 0.65, 0.25, 0.35, 0.55, 0.25],
    "ذكر" => [0.6, 0.4, 0.3, 0.2, 0.4, 0.1, 0.5, 0.3, 0.1],
    "نسى" => [0.5, 0.5, 0.4, 0.3, 0.3, 0.2, 0.4, 0.2, 0.2],
    "جاء" => [0.7, 0.3, 0.1, 0.2, 0.5, 0.1, 0.3, 0.4, 0.1],
    "ذهب" => [0.65, 0.35, 0.2, 0.35, 0.45, 0.15, 0.35, 0.35, 0.15],
    "أكل" => [0.6, 0.4, 0.15, 0.25, 0.4, 0.1, 0.3, 0.3, 0.1],
    "شرب" => [0.55, 0.45, 0.2, 0.3, 0.35, 0.15, 0.35, 0.25, 0.15],
)

# ═══════════════════════════════════════════════════════
# التضمينات
# ═══════════════════════════════════════════════════════

struct LetterEmbedding
    letter::Char
    features::Vector{Float64}
    semantic_vector::Vector{Float64}
end

struct WordEmbedding
    word::String
    root::String
    letters::Vector{Char}
    letter_embeddings::Vector{LetterEmbedding}
    semantic_vector::Vector{Float64}
end

struct SemanticVector
    dimensions::Int
    values::Vector{Float64}
end

# ═══════════════════════════════════════════════════════
# محلل الدلالة
# ═══════════════════════════════════════════════════════

struct SemanticAnalyzer
    letter_embeddings::Dict{Char,LetterEmbedding}
    root_embeddings::Dict{String,Vector{Float64}}
    dimensions::Int
end

function SemanticAnalyzer(; dimensions::Int = 9)
    letter_embeddings = Dict{Char,LetterEmbedding}()

    for letter in ARABIC_LETTERS
        features = get(LETTER_FEATURES, letter, zeros(dimensions))
        semantic_vec = _compute_letter_semantic(letter, features)
        letter_embeddings[letter] = LetterEmbedding(letter, features, semantic_vec)
    end

    SemanticAnalyzer(letter_embeddings, COMMON_ROOTS, dimensions)
end

# ═══════════════════════════════════════════════════════
# دوال التضمين
# ═══════════════════════════════════════════════════════

function embed_letter(analyzer::SemanticAnalyzer, letter::Char)::LetterEmbedding
    if haskey(analyzer.letter_embeddings, letter)
        return analyzer.letter_embeddings[letter]
    end
    return LetterEmbedding(letter, zeros(analyzer.dimensions), zeros(analyzer.dimensions))
end

function embed_word(analyzer::SemanticAnalyzer, word::AbstractString)::WordEmbedding
    letters = collect(String(word))
    letter_embeddings = [embed_letter(analyzer, l) for l in letters]

    semantic_vector = zeros(analyzer.dimensions)
    for le in letter_embeddings
        semantic_vector .+= le.semantic_vector
    end
    if !isempty(letter_embeddings)
        semantic_vector ./= length(letter_embeddings)
    end

    root = extract_root(analyzer, word)

    WordEmbedding(String(word), root, letters, letter_embeddings, semantic_vector)
end

function compute_semantic_similarity(embedding1::WordEmbedding, embedding2::WordEmbedding)::Float64
    v1 = embedding1.semantic_vector
    v2 = embedding2.semantic_vector

    dot_product = dot(v1, v2)
    norm1 = norm(v1)
    norm2 = norm(v2)

    if norm1 == 0.0 || norm2 == 0.0
        return 0.0
    end

    return dot_product / (norm1 * norm2)
end

# ═══════════════════════════════════════════════════════
# استخراج الجذور
# ═══════════════════════════════════════════════════════

function extract_root(analyzer::SemanticAnalyzer, word::AbstractString)::String
    word_str = String(word)

    if haskey(analyzer.root_embeddings, word_str)
        return word_str
    end

    consonants = filter(c -> c in ARABIC_LETTERS && c ∉ ['ا', 'و', 'ي', 'ه'], collect(word_str))

    if length(consonants) >= 3
        return String(consonants[1:3])
    elseif length(consonants) == 2
        return String(consonants)
    else
        return word_str
    end
end

function compute_word_semantics(analyzer::SemanticAnalyzer, word::AbstractString)::SemanticVector
    embedding = embed_word(analyzer, word)
    return SemanticVector(analyzer.dimensions, embedding.semantic_vector)
end

# ═══════════════════════════════════════════════════════
# دوال مساعدة
# ═══════════════════════════════════════════════════════

function _compute_letter_semantic(letter::Char, features::Vector{Float64})::Vector{Float64}
    semantic = copy(features)

    if letter in ['ا', 'و', 'ي']
        semantic[1] += 0.3
    end

    if letter in ['ع', 'غ', 'ح', 'خ', 'ه']
        semantic[5] += 0.2
    end

    if letter in ['ص', 'ض', 'ط', 'ظ']
        semantic[8] += 0.3
    end

    return semantic
end

function dot(v1::Vector{Float64}, v2::Vector{Float64})::Float64
    return sum(v1[i] * v2[i] for i in 1:length(v1))
end

function norm(v::Vector{Float64})::Float64
    return sqrt(sum(x^2 for x in v))
end

function cosine_similarity(v1::Vector{Float64}, v2::Vector{Float64})::Float64
    d = dot(v1, v2)
    n1 = norm(v1)
    n2 = norm(v2)
    if n1 == 0.0 || n2 == 0.0
        return 0.0
    end
    return d / (n1 * n2)
end

end # module Semantics
