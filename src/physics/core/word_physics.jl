"""
enhanced_phase_vector.jl - حساب المتجه الطوري المحسّن لكل حرف
يستخرج أبعاداً متعددة من معادلة الشكل العام:

المتجه النهائي = [إشارة(100) + مشتقات(30) + طيف(30) + إحصائي(30) + شكل(15) + معاملات(27)]
= 100 + 30 + 30 + 30 + 15 + 27 = 232 بُعد

ملاحظة: الإشارة y(x) هي البصمة الفريدة للحرف.
لا نستخدم L2 normalization على المتجه الكامل.
"""

module WordPhysics

using LinearAlgebra
using Statistics
using Random
using JSON

using ..Constants: PLANCK_H, LIGHT_SPEED_C, PHASE_DIM, ROOT_DIMS, EXTRA_DIMS,
                   SYNTAX_DIMS, SEMANTIC_DIMS, PRAGMATIC_DIMS, TOTAL_DIM
using ..LetterDB: LetterDatabase, get_vector, get_omega_0, has, get_operator

using ..LetterEquations

export compute_letter_signal, compute_enhanced_vector, compute_word_enhanced_vector,
       ENHANCED_DIM, signal_term, phase_similarity_enhanced, phase_distance_enhanced,
       letter_mass, gravitational_force_enhanced, compute_raw_param_vector,
       COMPACT_PHASE_DIM, artificial_letter_vector, compute_compact_phase_vector,
       compact_phase_similarity,
       apply_exponential_factor,
       compute_extended_phase_vector, phase_similarity, dress_phase_vector,
       dress_extended_phase_vector, get_letter_db, _normalize_letters,
       _parse_word_harakat, _build_modulated_vectors,
       compute_word_phase_vector,
       compute_word_frequency, compute_word_frequency_with_irab,
       compute_word_energy, compute_word_mass,
       IRAB_MAP, IRAB_OMEGA_BIAS, extract_irab, get_irab_omega_bias,
       load_letter_topic_embeddings!, train_letter_topic_embeddings!,
       _letter_pos_topic_embeddings

const _letter_db = Ref{Union{LetterDatabase,Nothing}}(nothing)

function get_letter_db()
    if _letter_db[] === nothing
        _letter_db[] = LetterDatabase()
    end
    return _letter_db[]
end

const _NORMALIZE_MAP = Dict(
    'آ' => 'ا', 'أ' => 'ا', 'إ' => 'ا',
    'ؤ' => 'و', 'ئ' => 'ي', 'ة' => 'ه', 'ى' => 'ي',
)

const IRAB_MAP = Dict(
    0x064F => :n,
    0x064C => :n,
    0x064E => :a,
    0x064B => :a,
    0x0650 => :g,
    0x064D => :g,
    0x0652 => :j,
)

const IRAB_OMEGA_BIAS = Dict(
    :n =>  0.30,
    :a => -0.20,
    :g => -0.30,
    :j => -0.10,
)

function _normalize_letters(word::String)
    return lowercase(map(c -> get(_NORMALIZE_MAP, c, c), word))
end

function extract_irab(word::String)
    isempty(word) && return nothing
    chars = collect(word)
    last_chars = chars[max(1, length(chars)-1):end]
    for ch in reverse(last_chars)
        state = get(IRAB_MAP, Int(ch), nothing)
        state !== nothing && return state
    end
    return nothing
end

function get_irab_omega_bias(word::String)
    irab = extract_irab(word)
    irab === nothing && return 0.0
    return get(IRAB_OMEGA_BIAS, irab, 0.0)
end

function _parse_word_harakat(word::String)
    parsed = Tuple{Char, Vector{Int}}[]
    current_letter = nothing
    current_harakat = Int[]

    for ch in word
        cp = Int(ch)
        norm_ch = get(_NORMALIZE_MAP, ch, ch)
        is_arabic_letter = (0x0621 <= cp <= 0x064A && cp != 0x0640) || haskey(_NORMALIZE_MAP, ch)
        is_english_letter = isletter(ch) && cp <= 0x007A
        is_harakat = 0x064B <= cp <= 0x0652

        if is_arabic_letter || is_english_letter
            if current_letter !== nothing
                push!(parsed, (current_letter, current_harakat))
            end
            current_letter = is_english_letter ? lowercase(ch) : lowercase(norm_ch)
            current_harakat = Int[]
        elseif is_harakat
            if current_letter !== nothing
                push!(current_harakat, cp)
            end
        end
    end
    if current_letter !== nothing
        push!(parsed, (current_letter, current_harakat))
    end
    return parsed
end

function _is_english_word(word::String)
    return all(c -> isletter(c) && c <= 'z' || isspace(c), word) && any(c -> isletter(c) && c <= 'z', word)
end

const _EN_IRREGULARS = Dict(
    "went"=>"go","goes"=>"go","gone"=>"go","going"=>"go",
    "children"=>"child","mice"=>"mouse","feet"=>"foot",
    "better"=>"good","best"=>"good","worse"=>"bad","worst"=>"bad",
    "was"=>"is","were"=>"is","been"=>"be","being"=>"be",
    "has"=>"have","had"=>"have","having"=>"have",
    "did"=>"do","does"=>"do","done"=>"do","doing"=>"do",
    "said"=>"say","says"=>"say","saw"=>"see","seen"=>"see",
    "took"=>"take","taken"=>"take","took"=>"take",
    "ran"=>"run","run"=>"run","running"=>"run",
    "made"=>"make","made"=>"make",
    "got"=>"get","gotten"=>"get",
    "knew"=>"know","known"=>"know",
    "thought"=>"think","brought"=>"bring","bought"=>"buy",
    "taught"=>"teach","caught"=>"catch","found"=>"find",
    "held"=>"hold","told"=>"tell","left"=>"leave",
    "kept"=>"keep","sat"=>"sit","stood"=>"stand",
    "felt"=>"feel","led"=>"lead","read"=>"read",
    "wrote"=>"write","written"=>"write",
    "gave"=>"give","given"=>"give",
    "spoke"=>"speak","spoken"=>"speak",
    "drove"=>"drive","driven"=>"drive",
    "chose"=>"choose","chosen"=>"choose",
    "broke"=>"break","broken"=>"break",
    "wore"=>"wear","worn"=>"wear",
    "threw"=>"throw","thrown"=>"throw",
    "flew"=>"fly","flown"=>"fly",
    "grew"=>"grow","grown"=>"grow",
    "knew"=>"know","known"=>"know",
    "drew"=>"draw","drawn"=>"draw",
    "blew"=>"blow","blown"=>"blow",
    "froze"=>"freeze","frozen"=>"freeze",
    "shook"=>"shake","shaken"=>"shake",
    "woke"=>"wake","woken"=>"wake",
    "bit"=>"bite","bitten"=>"bite",
    "began"=>"begin","begun"=>"begin",
    "drank"=>"drink","drunk"=>"drink",
    "sang"=>"sing","sung"=>"sing",
    "swam"=>"swim","swum"=>"swim",
    "rang"=>"ring","rung"=>"ring",
    "sprang"=>"spring","sprung"=>"spring",
    "clung"=>"cling","hung"=>"hang",
    "dug"=>"dig","spun"=>"spin",
    "stuck"=>"stick","struck"=>"strike",
    "lit"=>"light","quit"=>"quit",
    "spread"=>"spread","shed"=>"shed",
    "hurt"=>"hurt","cut"=>"cut","put"=>"put",
    "set"=>"set","let"=>"let","shut"=>"shut",
    "split"=>"split","cost"=>"cost","hit"=>"hit",
)

function _english_stem(word::String)
    w = lowercase(word)
    haskey(_EN_IRREGULARS, w) && return _EN_IRREGULARS[w]
    length(w) <= 3 && return w
    endswith(w, "ing") && length(w) > 4 && return w[1:end-3]
    endswith(w, "ed") && length(w) > 4 && return w[1:end-2]
    endswith(w, "ly") && length(w) > 4 && return w[1:end-2]
    endswith(w, "tion") && length(w) > 5 && return w[1:end-4]
    endswith(w, "sion") && length(w) > 5 && return w[1:end-4]
    endswith(w, "ment") && length(w) > 5 && return w[1:end-4]
    endswith(w, "ness") && length(w) > 5 && return w[1:end-4]
    endswith(w, "ful") && length(w) > 4 && return w[1:end-3]
    endswith(w, "less") && length(w) > 5 && return w[1:end-4]
    endswith(w, "able") && length(w) > 5 && return w[1:end-4]
    endswith(w, "ible") && length(w) > 5 && return w[1:end-4]
    endswith(w, "er") && length(w) > 4 && return w[1:end-2]
    endswith(w, "est") && length(w) > 5 && return w[1:end-3]
    endswith(w, "ies") && length(w) > 4 && return w[1:end-3] * "y"
    endswith(w, "es") && length(w) > 4 && return w[1:end-2]
    endswith(w, "s") && !endswith(w, "ss") && length(w) > 3 && return w[1:end-1]
    return w
end

function _build_modulated_vectors(parsed::Vector{Tuple{Char, Vector{Int}}}, db)
    vectors = Vector{Float64}[]
    for (letter, harakat) in parsed
        v = Float64.(get_vector(db, string(letter)))
        for h in harakat
            if h in (0x064E, 0x064B)
                v .+= 0.5 .* Float64.(get_vector(db, "ا"))
            elseif h in (0x064F, 0x064C)
                v .+= 0.5 .* Float64.(get_vector(db, "و"))
            elseif h in (0x0650, 0x064D)
                v .+= 0.5 .* Float64.(get_vector(db, "ي"))
            elseif h == 0x0651
                v .*= 2.0
            end
        end
        push!(vectors, v)
    end
    return vectors
end

# ─────────────────────────────────────────────────────────
# حساب الإشارة من المعادلة
# ─────────────────────────────────────────────────────────

"""
    signal_term(x, params) -> Float64
حساب قيمة حد واحد من المعادلة عند النقطة x
"""
function signal_term(x::Float64, params::Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64})
    C, A, β, γ, B, W, k, x0, n = params
    
    linear = A * (β * x + γ)
    
    dx = x - x0
    if abs(dx) < 1e-10
        powered = 0.0
    else
        powered = sign(dx) * (abs(dx))^n
    end
    
    exponent = k * powered
    exponent = clamp(exponent, -50.0, 50.0)
    
    sigmoid = B * W / (1.0 + exp(-exponent))
    
    return C * (linear + sigmoid)
end

"""
    compute_letter_signal(letter::Char; n_samples=200, x_range=(-1.0, 1.0)) -> Vector{Float64}
حساب الإشارة الكاملة للحرف عبر نطاق x
"""
function compute_letter_signal(letter::Char; n_samples::Int=200, x_range::Tuple{Float64,Float64}=(-1.0, 1.0))
    params = get_letter_params(letter)
    if params === nothing
        return zeros(n_samples)
    end
    
    x_min, x_max = x_range
    xs = range(x_min, x_max, length=n_samples)
    
    y = zeros(n_samples)
    for (i, x) in enumerate(xs)
        for term_params in params
            y[i] += signal_term(x, term_params)
        end
    end
    
    return y
end

# ─────────────────────────────────────────────────────────
# استخراج الأبعاد الممتدة (غير مستخدمة حالياً ومحفوظة للتوافق)
# ─────────────────────────────────────────────────────────



"""
    _extract_raw_params(letter::Char) -> Vector{Float64}
استخراج المعاملات الخام كمتجه (3 حدود × 9 معاملات = 27 بُعد)
"""
function _extract_raw_params(letter::Char)
    params = get_letter_params(letter)
    if params === nothing
        return zeros(27)
    end
    
    result = zeros(27)
    for (i, p) in enumerate(params)
        for (j, val) in enumerate(p)
            result[(i-1)*9 + j] = val
        end
    end
    
    return result
end

# ─────────────────────────────────────────────────────────
# الدالة الرئيسية: حساب المتجه الطوري المحسّن
# ─────────────────────────────────────────────────────────

const SIG_DIMS = 100
const DERIV_DIMS = 30
const SPECTRAL_DIMS = 30
const STAT_DIMS = 30
const SHAPE_DIMS = 15
const RAW_PARAM_DIMS = 27
const EXTENDED_FEATURE_DIM = SIG_DIMS + DERIV_DIMS + SPECTRAL_DIMS + STAT_DIMS + SHAPE_DIMS + RAW_PARAM_DIMS  # = 232
# Reserved for future visual/wave rendering; currently projected to ENHANCED_DIM = 27 (RAW_PARAM_DIMS)
const ENHANCED_DIM = RAW_PARAM_DIMS  # المتجه النهائي = المعاملات بعد الأسي فقط
# بعد تطبيق العامل الأسي، المعاملات الخام كافية للتمييز

# ─────────────────────────────────────────────────────────
# العامل الأسي لتكبير التباعد بين الحروف
# ─────────────────────────────────────────────────────────

"""
    apply_exponential_factor(v::Vector{Float64}; alpha::Float64=1.0) -> Vector{Float64}
تطبيق العامل الأسي على المتجه لتكبير التباعد بين الحروف.
العامل الأسي يعمل على عكس اللوغارتمي:
- اللوغارitmي: يقارب بين القيم الكبيرة والصغيرة (compress)
- الأسي: يُبعد بين القيم (amplify)
"""
function apply_exponential_factor(v::AbstractVector; alpha::Float64=1.0)
    return exp.(alpha .* v)
end

"""
    compute_raw_param_vector(letter::Char) -> Vector{Float64}
حساب المتجه الخام للحرف (27 بُعد) مع العامل الأسي
"""
function compute_raw_param_vector(letter::Char; alpha::Float64=1.0)
    params = get_letter_params(letter)
    if params === nothing
        return zeros(27)
    end
    
    vec = zeros(27)
    for (i, p) in enumerate(params)
        for (j, val) in enumerate(p)
            vec[(i-1)*9 + j] = val
        end
    end
    
    # تطبيق العامل الأسي لتكبير التباعد
    return apply_exponential_factor(vec, alpha=alpha)
end

# ─────────────────────────────────────────────────────────
# الدالة الرئيسية: حساب المتجه الطوري المحسّن
# ─────────────────────────────────────────────────────────

"""
    compute_enhanced_vector(letter::Char; n_samples=200, alpha=1.0) -> Vector{Float64}
حساب المتجه الطوري المحسّن للحرف (27 بُعد بعد العامل الأسي)
المتجه النهائي = exp(alpha × raw_params)
"""
function compute_enhanced_vector(letter::Char; n_samples::Int=200, alpha::Float64=1.0)
    # استخدام المعاملات الخام مع العامل الأسي فقط
    # هذا يعطي أفضل تمييز بين الحروف
    return compute_raw_param_vector(letter, alpha=alpha)
end

"""
    compute_word_enhanced_vector(word::String; n_samples=200, alpha=1.0) -> Vector{Float64}
حساب المتجه الطوري المحسّن للكلمة (متوسط مرجّح لحروفها بعد العامل الأسي)
"""
function compute_word_enhanced_vector(word::String; n_samples::Int=200, alpha::Float64=1.0)
    letters = collect(filter(c -> !isspace(c), word))
    if isempty(letters)
        return zeros(ENHANCED_DIM)
    end
    
    # أوزان مبنية على الموقع (البداية أثقل)
    n = length(letters)
    weights = Float64[1.0 / (0.5 + 0.5 * i / n) for i in 1:n]
    weights ./= sum(weights)
    
    result = zeros(ENHANCED_DIM)
    for (i, letter) in enumerate(letters)
        v = compute_enhanced_vector(letter, n_samples=n_samples, alpha=alpha)
        result .+= weights[i] .* v
    end
    
    return result
end

const ARTIFICIAL_LETTER_DIMS = 5
const COMPACT_PHASE_DIM = ENHANCED_DIM + ARTIFICIAL_LETTER_DIMS

function _letter_code_value(letter::Char)
    return Float64(Int(letter)) / 65535.0
end

function artificial_letter_vector(letter::Char)::Vector{Float64}
    x = _letter_code_value(letter)
    seed = max(1.0, Float64(Int(letter)))
    return Float64[
        x,
        sin(seed * 0.017453292519943295),
        cos(seed * 0.017453292519943295),
        sin(seed * 1.618033988749895),
        cos(seed * 2.718281828459045),
    ]
end

function compute_compact_phase_vector(word::String; n_samples::Int=200, alpha::Float64=1.0)
    parsed = _parse_word_harakat(word)
    letters = [letter for (letter, _) in parsed]
    if isempty(letters)
        return zeros(Float64, COMPACT_PHASE_DIM)
    end

    n = length(letters)
    weights = Float64[1.0 / (0.5 + 0.5 * i / n) for i in 1:n]
    weights ./= sum(weights)

    result = zeros(Float64, COMPACT_PHASE_DIM)
    for (i, letter) in enumerate(letters)
        result[1:ENHANCED_DIM] .+= weights[i] .* compute_enhanced_vector(letter; n_samples=n_samples, alpha=alpha)
        result[ENHANCED_DIM+1:COMPACT_PHASE_DIM] .+= weights[i] .* artificial_letter_vector(letter)
    end

    nrm = norm(result)
    nrm > 1e-10 && (result ./= nrm)
    return result
end

compact_phase_similarity(v1::AbstractVector, v2::AbstractVector) = phase_similarity_enhanced(v1, v2)

"""
    phase_similarity_enhanced(v1::Vector{Float64}, v2::Vector{Float64}) -> Float64
حساب التشابه الطوري المحسّن ( cosine similarity )
"""
function phase_similarity_enhanced(v1::AbstractVector, v2::AbstractVector)
    n1 = norm(v1)
    n2 = norm(v2)
    if n1 < 1e-10 || n2 < 1e-10
        return 0.0
    end
    return clamp(dot(v1, v2) / (n1 * n2), -1.0, 1.0)
end

"""
    phase_distance_enhanced(v1::AbstractVector, v2::AbstractVector) -> Float64
حساب المسافة الطورية المحسّنة (Euclidean distance) — متسق مع phase_similarity_enhanced
"""
function phase_distance_enhanced(v1::AbstractVector, v2::AbstractVector)
    return norm(v1 - v2)
end

"""
    letter_mass(letter::Char) -> Float64
كتلة الحرف = طاقة الإشارة = ∫y²dx
"""
function letter_mass(letter::Char; n_samples::Int=200)
    y = compute_letter_signal(letter, n_samples=n_samples)
    dx = 2.0 / n_samples
    return sum(y .^ 2) * dx
end

"""
    gravitational_force_enhanced(letter1::Char, letter2::Char; eps=0.01) -> Float64
قوة الجاذبية الطورية بين حرفين
"""
function gravitational_force_enhanced(letter1::Char, letter2::Char; eps::Float64=0.01)
    v1 = compute_enhanced_vector(letter1)
    v2 = compute_enhanced_vector(letter2)
    sim = phase_similarity_enhanced(v1, v2)
    m1 = letter_mass(letter1)
    m2 = letter_mass(letter2)
    r = 1.0 - sim
    return (m1 * m2) / (r^2 + eps)
end

function compute_word_frequency(word::String)
    db = get_letter_db()
    freq = 0.0
    for (letter, _) in _parse_word_harakat(word)
        freq += get_omega_0(db, string(letter))
    end
    return freq == 0.0 ? 1.0 : freq
end

function compute_word_frequency_with_irab(word::String)
    return compute_word_frequency(word) + get_irab_omega_bias(word)
end

function compute_word_energy(word::String)
    return PLANCK_H * compute_word_frequency(word)
end

function compute_word_mass(word::String)
    return compute_word_energy(word) / (LIGHT_SPEED_C^2)
end

function compute_word_phase_vector(word::String; widen::Float64=1.0,
                                   method::String="linear",
                                   vowel_modulation::Bool=true)
    db = get_letter_db()
    parsed = _parse_word_harakat(word)
    vectors = if vowel_modulation
        _build_modulated_vectors(parsed, db)
    else
        [Float64.(get_vector(db, string(letter))) for (letter, _) in parsed]
    end

    isempty(vectors) && return zeros(Float32, PHASE_DIM)

    result = zeros(Float32, PHASE_DIM)
    for (i, v) in enumerate(vectors)
        shift = i - 1
        for j in 1:length(v)
            dest = mod1(j + shift, PHASE_DIM)
            result[dest] += Float32(v[j])
        end
    end

    nrm = norm(result)
    nrm > 1e-10 && (result ./= nrm)

    if widen != 1.0
        result .= sign.(result) .* abs.(result) .^ widen
    end
    return result
end

function _compute_extra_dims(word::String)
    db = get_letter_db()
    parsed = _parse_word_harakat(word)

    ops = Float64[]
    bare_letters = Char[]
    for (letter, _) in parsed
        op = get_operator(db, string(letter))
        if op == "+1"
            push!(ops, 1.0)
        elseif op == "-1"
            push!(ops, -1.0)
        else
            push!(ops, 0.0)
        end
        push!(bare_letters, letter)
    end

    op_score = isempty(ops) ? 0.0 : sum(ops) / length(ops)
    wlen = min(length(bare_letters) / 12.0, 1.0)
    omega = compute_word_frequency(word)
    freq_norm = min(omega / 15.0, 1.0)
    energy = compute_word_energy(word)
    energy_norm = min(energy / 15.0, 1.0)
    op_var = length(ops) > 1 ? var(ops) : 0.0
    diversity = length(Set(bare_letters)) / max(length(bare_letters), 1)

    return Float64[op_score, wlen, freq_norm, energy_norm, op_var, diversity]
end

const _AUGMENTATION = Set("سألتمونيها")
const _PREFIXES = ["و", "ف", "ب", "ل", "ك"]
const _ATTACHED = ["هم", "هن", "هما", "ها", "ه", "ون", "ين", "ان", "ات",
                   "كم", "كن", "كما", "ك", "تم", "تن", "تما", "وا",
                   "نا", "ني", "ي", "ت"]

const _ROOT_PATTERNS = [
    ("استفعال", [4, 5, 7]), ("مستفعل", [4, 5, 6]), ("استفعل", [4, 5, 6]),
    ("انفعال", [3, 4, 6]), ("افتعال", [2, 4, 6]), ("مفاعيل", [2, 4, 6]),
    ("مفاعلة", [2, 4, 5]), ("مفاعل", [2, 4, 5]), ("تفعيل", [2, 3, 5]),
    ("تفاعل", [2, 4, 5]), ("تفعّل", [2, 3, 5]), ("مفعول", [2, 3, 5]),
    ("فعائل", [1, 2, 5]), ("فواعل", [1, 4, 5]), ("أفعال", [2, 3, 5]),
    ("إفعال", [2, 3, 5]), ("أفعل", [2, 3, 4]), ("مفعل", [2, 3, 4]),
    ("فاعل", [1, 3, 4]), ("فعال", [1, 2, 4]), ("فعول", [1, 2, 4]),
    ("فعيل", [1, 2, 4]), ("فعلة", [1, 2, 3]), ("يفعل", [2, 3, 4]),
    ("تفعل", [2, 3, 4]), ("نفعل", [2, 3, 4]),
]

function _extract_root_light(word::String)
    if _is_english_word(word)
        stem = _english_stem(word)
        chars = Char[]
        seen = Set{Char}()
        for c in stem
            if !(c in seen)
                push!(chars, c)
                push!(seen, c)
            end
            length(chars) >= 4 && break
        end
        return length(chars) >= 2 ? chars : (isempty(stem) ? Char[] : [stem[1]])
    end

    diacritics = Set(['ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ـ'])
    w = collect(filter(c -> !(c in diacritics), _normalize_letters(word)))

    if length(w) > 4 && w[1] == 'ا' && w[2] == 'ل'
        w = w[3:end]
    end

    prefix_chars = Set{Char}("وفبلك")
    for _ in 1:2
        if !isempty(w) && w[1] in prefix_chars && length(w) > 4
            w = w[2:end]
        end
    end

    for suffix in sort(_ATTACHED; by=length, rev=true)
        chars = collect(suffix)
        n = length(chars)
        if length(w) > n + 2 && w[end-n+1:end] == chars
            w = w[1:end-n]
            break
        end
    end

    if !isempty(w) && w[end] == 'ة' && length(w) > 3
        w = w[1:end-1]
    end

    for (pat, indices) in _ROOT_PATTERNS
        pat_chars = collect(pat)
        length(w) == length(pat_chars) || continue
        chars = Char[]
        matched = true
        for i in 1:length(w)
            if i in indices
                push!(chars, w[i])
            elseif w[i] != pat_chars[i] && !(w[i] in _AUGMENTATION)
                matched = false
                break
            end
        end
        matched && length(chars) >= 3 && return chars[1:min(4, end)]
    end

    chars = Char[]
    seen = Set{Char}()
    for c in w
        if !(c in _AUGMENTATION) || length(chars) < 3
            if !(c in seen) || length(chars) < 3
                push!(chars, c)
                push!(seen, c)
            end
        end
        length(chars) >= 4 && break
    end
    return length(chars) >= 2 ? chars[1:min(4, end)] : (isempty(w) ? Char[] : [w[1]])
end

function _compute_root_dims(word::String)
    root_chars = _extract_root_light(word)
    isempty(root_chars) && return zeros(Float32, ROOT_DIMS)

    db = get_letter_db()
    vecs = Float32[]
    for c in root_chars
        v_raw = get_vector(db, string(c))
        for k in 1:min(2, length(v_raw))
            push!(vecs, Float32(v_raw[k]))
        end
    end

    if length(vecs) < ROOT_DIMS
        append!(vecs, zeros(Float32, ROOT_DIMS - length(vecs)))
    else
        vecs = vecs[1:ROOT_DIMS]
    end

    nrm = norm(vecs)
    nrm > 1e-10 && (vecs ./= nrm)
    return vecs
end

function compute_extended_phase_vector(word::String; widen::Float64=1.0,
                                       current_topic::Union{Vector{Float32}, Nothing}=nothing)
    base = compute_word_phase_vector(word; widen=widen)
    root = _compute_root_dims(word)
    extra = Float32.(_compute_extra_dims(word))
    syn = zeros(Float32, SYNTAX_DIMS)
    sem = zeros(Float32, SEMANTIC_DIMS)
    prag = zeros(Float32, PRAGMATIC_DIMS)
    vec = vcat(base, root, extra, syn, sem, prag)

    if current_topic !== nothing && !isempty(_letter_pos_topic_embeddings)
        parsed = _parse_word_harakat(word)
        blend_vec = zeros(Float32, length(vec))
        has_blend = false
        for (pos, (letter, _)) in enumerate(parsed)
            key = (letter, pos)
            if haskey(_letter_pos_topic_embeddings, key)
                emb = _letter_pos_topic_embeddings[key]
                sim = phase_similarity(emb, current_topic)
                if sim > 0.3f0
                    decay = Float32(0.85^pos)
                    blend_vec .+= 0.2f0 * Float32(sim) * decay .* emb
                    has_blend = true
                end
            end
        end
        if has_blend
            vec .+= blend_vec
            nrm = norm(vec)
            nrm > 1e-10 && (vec ./= nrm)
        end
    end
    return vec
end

function phase_similarity(v1::AbstractVector, v2::AbstractVector)
    n1 = norm(v1)
    n2 = norm(v2)
    if n1 < 1e-10 || n2 < 1e-10
        return 0.0
    end
    return max(-1.0, min(1.0, dot(v1, v2) / (n1 * n2)))
end

function dress_phase_vector(word::String, context_words::Vector{String}; alpha::Float64=0.3, widen::Float64=1.0)
    base = compute_word_phase_vector(word; widen=widen)
    if isempty(context_words)
        return base
    end
    # باز Float32 — ctx_avg بنفس النوع لتجنب التحويل الضمني Float32→Float64→Float32
    ctx_avg = zeros(Float32, length(base))
    n_ctx = 0
    for cw in context_words
        cv = compute_word_phase_vector(cw; widen=widen)
        if length(cv) == length(base)
            s = phase_similarity(base, cv)
            if s > 0.0
                ctx_avg .+= Float32(s) .* cv
                n_ctx += 1
            end
        end
    end
    if n_ctx > 0
        ctx_avg ./= n_ctx
    end
    return base .+ Float32(alpha) .* ctx_avg
end

function dress_extended_phase_vector(word::String, context_words::Vector{String};
                                     alpha::Float64=0.3, widen::Float64=1.0)
    base = compute_extended_phase_vector(word; widen=widen)
    isempty(context_words) && return base

    dress = zeros(Float32, length(base))
    n = 0
    for cw in context_words[max(1, end-4):end]
        cpv = compute_extended_phase_vector(cw; widen=widen)
        align = phase_similarity(base, cpv)
        if align > 0.3
            dress .+= Float32(align) .* cpv ./ Float32(norm(cpv) + 1e-10)
            n += 1
        end
    end

    n > 0 && (dress .= Float32(alpha) .* dress ./ Float32(n))
    return base .+ dress
end

# ─────────────────────────────────────────────────────────
# الأبعاد الموضوعية للحرف (Letter Topic Embeddings)
# ─────────────────────────────────────────────────────────

const _PUNCT_EDGE_LOCAL = Set(['.', ',', '،', ':', '؛', '?', '؟', '!', ')', '(', '"', '\'',
                               '«', '»', '-', '…', '\u200F', '\u200E', '\u00AD'])

function _strip_punct_boundary_local(word::AbstractString)
    chars = collect(word)
    while !isempty(chars) && first(chars) in _PUNCT_EDGE_LOCAL
        popfirst!(chars)
    end
    while !isempty(chars) && last(chars) in _PUNCT_EDGE_LOCAL
        pop!(chars)
    end
    return String(chars)
end

const _letter_pos_topic_embeddings = Dict{Tuple{Char, Int}, Vector{Float32}}()

const _PARTICLES_TO_IGNORE = Set([
    "من", "إلى", "عن", "على", "في", "ب", "ل", "ك", "حتى", "منذ", "مذ", "رب", "واو", "تالله", "عدا", "خلا", "حاشا",
    "لن", "كي", "لكي", "إذن", "أن", "أنّ", "كأن", "لأن", "لكن", "لكنّ",
    "لم", "لما", "لام", "لا", "ما", "ليس", "غير", "إن", "إذما", "مهما", "متى", "أيان", "أين", "أينما", "حيثما", "أنى", "كيفما", "أي",
    "هل", "أ", "ماذا", "كيف", "كم",
    "بينما", "لطالما", "طالما", "كأنما", "إنما", "كيما", "لكيما", "مما", "عما", "فيم", "بم", "ألا", "لئلا", "كيلا",
    "ربما", "ريثما", "حين", "حينما", "عند", "عندما"
])

function load_letter_topic_embeddings!(model_dir::String)
    path = joinpath(model_dir, "letter_topic_embeddings.json")
    empty!(_letter_pos_topic_embeddings)
    if isfile(path)
        try
            data = JSON.parsefile(path)
            for (key_str, val) in data
                parts = split(key_str, ',')
                if length(parts) == 2
                    char = parts[1][1]
                    pos = parse(Int, parts[2])
                    _letter_pos_topic_embeddings[(char, pos)] = Vector{Float32}(val)
                end
            end
            println("✓ Loaded $(length(_letter_pos_topic_embeddings)) letter-position topic embeddings.")
        catch e
            @warn "Failed to load letter topic embeddings: $e"
        end
    end
    return _letter_pos_topic_embeddings
end

function train_letter_topic_embeddings!(texts::Vector{String}, vocab::Dict{String,Int}, model_dir::String; min_occurrences::Int=10)
    println("⏳ جاري حساب الأبعاد الموضوعية للحروف...")
    
    accumulated_sums = Dict{Tuple{Char, Int}, Vector{Float32}}()
    accumulated_counts = Dict{Tuple{Char, Int}, Int}()
    
    processed_sentences = 0
    for text in texts
        # نقسم المستند إلى جمل أو فقرات فرعية
        lines = split(text, r"[.!\n؟]")
        for line in lines
            trimmed = strip(line)
            isempty(trimmed) && continue
            
            # تقطيع الجملة إلى كلمات ونزع علامات الترقيم
            words = String[_strip_punct_boundary_local(strip(w)) for w in split(trimmed) if length(strip(w)) >= 2]
            # الاحتفاظ بالكلمات التي في المعجم والغير موجودة في قائمة التجاهل
            filter!(w -> haskey(vocab, w) && !(w in _PARTICLES_TO_IGNORE), words)
            
            length(words) < 2 && continue
            
            # 1. حساب البعد الموضوعي للجملة = متوسط متجهات كلماتها (10000D)
            sentence_topic = zeros(Float32, TOTAL_DIM)
            n_words = 0
            for w in words
                pv = try Float32.(compute_extended_phase_vector(w)) catch; nothing end
                if pv !== nothing
                    sentence_topic .+= pv
                    n_words += 1
                end
            end
            
            n_words == 0 && continue
            sentence_topic ./= n_words
            nrm = norm(sentence_topic)
            nrm > 1e-10 && (sentence_topic ./= nrm)
            
            # 2. إضافة هذا السياق لكل (حرف + موضع) في كلمات الجملة
            for w in words
                parsed = _parse_word_harakat(w)
                for (pos, (letter, _)) in enumerate(parsed)
                    key = (letter, pos)
                    if !haskey(accumulated_sums, key)
                        accumulated_sums[key] = zeros(Float32, TOTAL_DIM)
                        accumulated_counts[key] = 0
                    end
                    accumulated_sums[key] .+= sentence_topic
                    accumulated_counts[key] += 1
                end
            end
            processed_sentences += 1
        end
    end
    
    # 3. حساب المتوسط النهائي والترشيح والترميز
    output_data = Dict{String, Vector{Float32}}()
    updated = 0
    for (key, count) in accumulated_counts
        if count >= min_occurrences
            avg_vec = accumulated_sums[key] ./ count
            nrm = norm(avg_vec)
            nrm > 1e-10 && (avg_vec ./= nrm)
            
            # التخزين في الذاكرة
            _letter_pos_topic_embeddings[key] = avg_vec
            
            # تجهيز الحفظ بصيغة JSON
            char, pos = key
            output_data["$(char),$(pos)"] = avg_vec
            updated += 1
        end
    end
    
    # 4. الكتابة إلى ملف JSON
    mkpath(model_dir)
    path = joinpath(model_dir, "letter_topic_embeddings.json")
    open(path, "w") do io
        JSON.print(io, output_data)
    end
    
    println("✓ تم تدريب وحفظ $updated بُعد موضوعي للحروف/المواضع في $path (إجمالي الجمل الممررة: $processed_sentences)")
end

end # module WordPhysics
