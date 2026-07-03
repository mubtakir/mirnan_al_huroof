"""
التعلم الهيبياني الهولوغرافي للـ PRNN — PRNN Learner.
يتكفل بتحويل المتجهات الدلالية المتفرقة لطور كثيف وتدريب الانتقالات السببية هيبياً.
"""
module PRNNLearner

using LinearAlgebra, Random
using ..WordPhysics: compute_extended_phase_vector
using ..PRNNCore: bind_phase

export build_dense_phase_vector, train_hebbian_transitions!

"""
    build_dense_phase_vector(word::String, N::Int) -> Vector{ComplexF64}

التوسيع الهندسي للطور (Geometric Phase Expansion):
يقوم بتحويل المتجه الحقيقي المتفرق (Sparse) إلى متجه طوري كثيف (Dense Complex Phase Vector)
على دائرة الوحدة عن طريق ضرب المكونات بـ √N·π لتجنب مشكلة تكدس الطور وانهيار المحاكاة.
"""
function build_dense_phase_vector(v_real::AbstractVector, N::Int)
    phases = Float64.(v_real) .* (sqrt(N) * pi)
    return exp.(im .* phases)
end

const _dense_vector_cache = Dict{Tuple{String, Int}, Vector{ComplexF64}}()
const _dense_vector_cache_lock = ReentrantLock()

function build_dense_phase_vector(word::String, N::Int)
    lock(_dense_vector_cache_lock) do
        key = (word, N)
        if haskey(_dense_vector_cache, key)
            return _dense_vector_cache[key]
        end
        v_real = Float64.(compute_extended_phase_vector(word))
        v_dense = build_dense_phase_vector(v_real, N)
        _dense_vector_cache[key] = v_dense
        return v_dense
    end
end

"""
    train_hebbian_transitions!(vocab::Dict{String,Int}, id2word::Dict{Int,String}, 
                                corpus_sentences::Vector{Vector{Int32}}, active_vocab::Vector{String}, 
                                prompt_ids::Vector{Int32}, beta::Float64, N::Int,
                                base_vectors::Dict{String, Vector{ComplexF64}}; window::Int=5) -> Vector{Tuple{Vector{ComplexF64}, Vector{ComplexF64}, Float64}}

التدريب الهيبي المتسلسل الهولوغرافي من الجمل ذات الصلة:
يربط الكلمات المتجاورة سياقياً هولوغرافياً في النوافذ القريبة من المحفز، ويجمعها بأوزان فريدة.
"""
function train_hebbian_transitions!(vocab::Dict{String,Int}, id2word::Dict{Int,String}, 
                                    corpus_sentences::Vector{Vector{Int32}}, active_vocab::Vector{String}, 
                                    prompt_ids::Vector{Int32}, beta::Float64, N::Int,
                                    base_vectors::Dict{String, Vector{ComplexF64}}=Dict{String, Vector{ComplexF64}}(); 
                                    window::Int=5)
    # بناء قاعدة متجهات الطور للكلمات النشطة إن لم تكن موجودة
    for word in active_vocab
        if !haskey(base_vectors, word)
            base_vectors[word] = build_dense_phase_vector(word, N)
        end
    end
    
    # تحديد المعرفات النشطة لتسهيل الترشيح
    active_set = Set{Int32}([Int32(vocab[w]) for w in active_vocab if haskey(vocab, w)])
    
    # تجميع الانتقالات الفريدة وحساب أوزانها وتكرارها
    transition_weights = Dict{Tuple{String, String}, Float64}()
    
    for sentence in corpus_sentences
        # إيجاد مواضع الكلمات البذرية (المحفزة)
        indices = findall(id -> id in prompt_ids, sentence)
        for idx in indices
            # نقتصر على النافذة المحلية حول المحفز فقط
            start_idx = max(1, idx - window)
            end_idx = min(length(sentence), idx + window)
            for t in start_idx:(end_idx - 1)
                id_curr = sentence[t]
                id_next = sentence[t+1]
                if id_curr in active_set && id_next in active_set
                    w_curr = get(id2word, Int(id_curr), nothing)
                    w_next = get(id2word, Int(id_next), nothing)
                    if w_curr !== nothing && w_next !== nothing
                        pair = (w_curr, w_next)
                        transition_weights[pair] = get(transition_weights, pair, 0.0) + 1.0
                    end
                end
            end
        end
    end
    
    # بناء روابط الانتقال الموزونة هولوغرافياً
    transitions = Tuple{Vector{ComplexF64}, Vector{ComplexF64}, Float64}[]
    for ((w_curr, w_next), weight) in transition_weights
        if haskey(base_vectors, w_curr) && haskey(base_vectors, w_next)
            v_curr = base_vectors[w_curr]
            v_next = base_vectors[w_next]
            # ربط الكلمة التالية بالسابقة سياقياً: v_next_adapted = v_next ⊙ v_curr
            v_next_adapted = bind_phase(v_next, v_curr)
            push!(transitions, (v_curr, v_next_adapted, weight))
        end
    end
    
    return transitions
end

end # module PRNNLearner
