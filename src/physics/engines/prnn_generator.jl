"""
محرك التوليد الطوري الهولوغرافي للـ PRNN — PRNN Generator.

واجهة برمجية مستقلة (Standalone API) لمحرك الشبكة العصبية الرنينية الطورية.
تعمل بمعاملات صريحة بمعزل عن MirnanGenerator لتسهيل الاختبار والتطوير.

الاستخدام المدمج في المولّد الرئيسي: انظر Generator._prnn_generate في generator.jl
"""
module PRNNGenerator

using LinearAlgebra, Random
using ..PRNNCore: PRNNState, LowRankPRNN, simulate_stuart_landau!, bind_phase, unbind_phase,
                  complete_pattern!
using ..PRNNLearner: build_dense_phase_vector, train_hebbian_transitions!
using ..Constants: TOTAL_DIM

export prnn_generate_standalone, PRNNSession,
       prnn_complete_pattern_standalone, prnn_noise_sample_standalone

# ═══════════════════════════════════════════════════════════════════
# جلسة PRNN المستقلة — تحمل كل ما يلزم لتوليد النص
# ═══════════════════════════════════════════════════════════════════

"""
    PRNNSession

بنية جلسة توليد PRNN مستقلة تحمل المعجم وروابط الانتقال
الهيبية المُدرَّبة مسبقاً لإعادة الاستخدام بين طلبات التوليد.
"""
struct PRNNSession
    vocab::Dict{String,Int}
    id2word::Dict{Int,String}
    base_vectors::Dict{String, Vector{ComplexF64}}
    transitions::Vector{Tuple{Vector{ComplexF64}, Vector{ComplexF64}, Float64}}
    active_vocab::Vector{String}
    beta_coupling::Float64
    N::Int
end

"""
    PRNNSession(vocab, id2word, corpus_sentences, seed_words; 
                beta=3.0, N=TOTAL_DIM, window=5, max_sentences=100, base_cache)

بناء جلسة PRNN: يُعيِّن المفردات النشطة حول الكلمات البذرية،
ويُدرِّب الانتقالات الهيبية الهولوغرافية من جمل الكوربوس ذات الصلة.
"""
function PRNNSession(vocab::Dict{String,Int}, id2word::Dict{Int,String},
                     corpus_sentences::Vector{Vector{Int32}},
                     seed_words::Vector{String};
                     beta::Float64=3.0, N::Int=TOTAL_DIM,
                     window::Int=5, max_sentences::Int=100,
                     base_cache::Dict{String, Vector{ComplexF64}}=Dict{String, Vector{ComplexF64}}())
    # 1. تحديد معرّفات البذور
    seed_ids = Int32[get(vocab, w, Int32(-1)) for w in seed_words]
    filter!(x -> x > 0, seed_ids)
    active_ids = Set{Int32}(seed_ids)

    # 2. توسيع السياق بنافذة ±window حول البذور في الجمل ذات الصلة
    relevant = Vector{Int32}[]
    for s in corpus_sentences
        positions = findall(id -> id in seed_ids, s)
        if !isempty(positions)
            push!(relevant, s)
            for pos in positions
                for j in max(1, pos-window):min(length(s), pos+window)
                    push!(active_ids, s[j])
                end
            end
            length(relevant) >= max_sentences && break
        end
    end

    # 3. بناء المفردات النشطة (عربية أبجدية فقط)
    active_vocab = String[]
    for id in active_ids
        w = get(id2word, Int(id), nothing)
        if w !== nothing && any(ch -> '\u0621' <= ch <= '\u064A', w)
            push!(active_vocab, w)
        end
    end

    isempty(active_vocab) && @warn "PRNNSession: لم تُوجد مفردات نشطة للبذور: $seed_words"

    # 4. بناء متجهات الطور الكثيف (Geometric Phase Expansion)
    base_vectors = Dict{String, Vector{ComplexF64}}()
    for word in active_vocab
        if haskey(base_cache, word)
            base_vectors[word] = base_cache[word]
        else
            base_vectors[word] = build_dense_phase_vector(word, N)
            base_cache[word] = base_vectors[word]
        end
    end

    # 5. التدريب الهيبي الهولوغرافي
    transitions = train_hebbian_transitions!(vocab, id2word, relevant, active_vocab, seed_ids, beta, N, base_vectors; window=window)

    return PRNNSession(vocab, id2word, base_vectors, transitions, active_vocab, beta, N)
end

# ═══════════════════════════════════════════════════════════════════
# التوليد المستقل
# ═══════════════════════════════════════════════════════════════════

"""
    prnn_generate_standalone(session, prompt_tokens; max_words=15,
                             mu=1.0, g_inh=0.5, gamma=2.0, tau_a=1.5,
                             steps=40, dt=0.02, overlap_threshold=0.10,
                             noise_amp=0.0, anneal_rate=0.0) -> String

توليد نص من جلسة PRNN مستقلة بالانزلاق الحتمي على حقل الجهد الفيزيائي.
noise_amp > 0.0 يُضيف استكشافاً إبداعياً عبر ضوضاء لانجفان.
anneal_rate > 0.0 يُبرّد الضوضاء تدريجياً نحو الجاذب.
لا يعتمد على MirnanGenerator — يُستخدم للاختبار والتطوير والبحث.
"""
function prnn_generate_standalone(session::PRNNSession, prompt_tokens::Vector{String};
                                   max_words::Int=15,
                                   mu::Float64=1.0, g_inh::Float64=0.5,
                                   gamma::Float64=2.0, tau_a::Float64=1.5,
                                   steps::Int=40, dt::Float64=0.02,
                                   overlap_threshold::Float64=0.10,
                                   noise_amp::Float64=0.0, anneal_rate::Float64=0.0)
    bv = session.base_vectors
    N  = session.N
    isempty(session.transitions) && return ""

    # دالة فك التشفير الطوري
    function decode(z_state, v_ctx)
        z_u = unbind_phase(z_state, v_ctx)
        best_w, best_ov = "", -Inf
        for w in session.active_vocab
            ov = real(dot(z_u, bv[w])) / N
            if ov > best_ov
                best_ov = ov; best_w = w
            end
        end
        return best_w, best_ov
    end

    # تهيئة حالة المذبذبات
    state = PRNNState(N)
    model = LowRankPRNN(session.transitions, session.beta_coupling)
    current_word = prompt_tokens[end]

    if haskey(bv, current_word)
        if length(prompt_tokens) >= 2 && haskey(bv, prompt_tokens[end-1])
            state.z .= bind_phase(bv[current_word], bv[prompt_tokens[end-1]])
        else
            state.z .= bv[current_word]
        end
    end

    output = String[]
    used   = Set{String}(prompt_tokens)

    for _ in 1:max_words
        simulate_stuart_landau!(state, model;
            mu=mu, g_inh=g_inh, gamma=gamma, tau_a=tau_a, steps=steps, dt=dt,
            noise_amp=noise_amp, anneal_rate=anneal_rate)

        !haskey(bv, current_word) && break
        next_w, ov = decode(state.z, bv[current_word])

        (isempty(next_w) || next_w in used || ov < overlap_threshold) && break

        push!(output, next_w)
        push!(used, next_w)
        state.z .= bind_phase(bv[next_w], bv[current_word])
        current_word = next_w
    end

    return join(output, " ")
end

# ═══════════════════════════════════════════════════════════════════
# إكمال الأنماط المستقل
# ═══════════════════════════════════════════════════════════════════

"""
    prnn_complete_pattern_standalone(session, damaged_tokens; tol=1e-6, max_stable=20,
                                     noise_amp=0.2, anneal_rate=0.05, kwargs...)

إكمال نمط ناقص أو تالف عبر جاذبات PRNN الطورية.
- damaged_tokens: كلمات البذور (قد تكون ناقصة أو بها أخطاء)
- تُعيد الكلمة الأقرب للنمط المستقر، أو "" إذا فشل الإكمال

الاستخدام:
    completed = prnn_complete_pattern_standalone(session, ["كلم", "ناقص"])
"""
function prnn_complete_pattern_standalone(session::PRNNSession, damaged_tokens::Vector{String};
                                          tol::Float64=1e-6, max_stable::Int=20,
                                          noise_amp::Float64=0.2, anneal_rate::Float64=0.05,
                                          mu::Float64=1.0, g_inh::Float64=0.5,
                                          gamma::Float64=2.0, tau_a::Float64=1.5,
                                          steps::Int=40, dt::Float64=0.02,
                                          overlap_threshold::Float64=0.10)
    bv = session.base_vectors
    N  = session.N
    isempty(session.transitions) && return ""
    isempty(damaged_tokens) && return ""

    # دالة فك التشفير الطوري
    function decode_to_word(z_state)
        best_w, best_ov = "", -Inf
        for w in session.active_vocab
            ov_real = real(dot(z_state, bv[w])) / N
            ov_imag = abs(imag(dot(z_state, bv[w]))) / N
            ov = ov_real - ov_imag * 0.3
            if ov > best_ov
                best_ov = ov; best_w = w
            end
        end
        return best_w, best_ov
    end

    # تهيئة الحالة من الكلمات التالفة
    state = PRNNState(N, rand(1:10000))
    current_word = damaged_tokens[end]
    if haskey(bv, current_word)
        state.z .= bv[current_word]
    else
        # الكلمة غير موجودة — نبحث عن أقرب كلمة
        sim_w, _ = decode_to_word(zeros(ComplexF64, N))
        if haskey(bv, sim_w)
            state.z .= bv[sim_w]
        else
            # تهيئة عشوائية مع الضوضاء
            state.z .= randn(ComplexF64, N) ./ sqrt(N)
        end
    end

    model = LowRankPRNN(session.transitions, session.beta_coupling)

    # إكمال النمط
    stabilized = complete_pattern!(state, model;
        steps=steps*5, tol=tol, max_stable=max_stable,
        mu=mu, g_inh=g_inh, gamma=gamma, tau_a=tau_a, dt=dt,
        noise_amp=noise_amp, anneal_rate=anneal_rate)

    if !stabilized
        # إذا لم يستقر، نأخذ آخر حالة
        @warn "PRNN pattern completion did not fully stabilize"
    end

    # فك التشفير
    completed_w, ov = decode_to_word(state.z)
    return ov >= overlap_threshold ? completed_w : ""
end

"""
    prnn_noise_sample_standalone(session, seed_tokens; steps=100, noise_amp=0.5, 
                                 anneal_rate=0.02, kwargs...)

توليد عيّنة إبداعية من فضاء الطور مع ضوضاء لانجفان.
تُعيد كلمة عشوائية من الجاذب الذي استقرت عنده المحاكاة.
"""
function prnn_noise_sample_standalone(session::PRNNSession, seed_tokens::Vector{String};
                                       steps::Int=100, noise_amp::Float64=0.5,
                                       anneal_rate::Float64=0.02, kwargs...)
    isempty(session.transitions) && return ""
    bv = session.base_vectors
    N = session.N

    state = PRNNState(N, rand(1:10000))
    if !isempty(seed_tokens) && haskey(bv, seed_tokens[end])
        state.z .= bv[seed_tokens[end]]
    else
        state.z .= randn(ComplexF64, N) ./ sqrt(N)
    end

    model = LowRankPRNN(session.transitions, session.beta_coupling)

    simulate_stuart_landau!(state, model;
        steps=steps, noise_amp=noise_amp, anneal_rate=anneal_rate, kwargs...)

    # فك التشفير
    best_w, best_ov = "", -Inf
    for w in session.active_vocab
        ov = real(dot(state.z, bv[w])) / N
        if ov > best_ov
            best_ov = ov; best_w = w
        end
    end

    return best_ov > 0.05 ? best_w : ""
end

end # module PRNNGenerator
