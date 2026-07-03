"""
محرك الفكرة الدلالي — Idea / Concept Engine.

عناصر الفكرة الثلاثة:
1. أشياء (الأسماء) - حوامل طورية (In-phase).
2. حدث (الفعل) - طور متعامد بزاوية π/2 (Quadrature).
3. نتيجة (الفعل أو الفكرة الجديدة) - طور رد الفعل المترتب.

يدعم تمثيل الفكرة كموجة مركبة متعامدة الأطوار (QAM-like Semantic Wave) 
والبحث عنها في الكوربس أو تقييم مدى مطابقة الجملة لها.
"""
module IdeaEngine

using LinearAlgebra
using ..Constants: TOTAL_DIM, PHASE_DIM
using ..CarrierWave: rotate_phase

export Idea, compute_idea_vector, match_sentence_to_idea, score_candidate_for_idea

"""
    Idea

بنية تمثل فكرة متكاملة من ثلاثة عناصر:
- `objects`: قائمة بالأسماء المشاركة (الأشياء)
- `event`: الفعل الأساسي (الحدث)
- `result`: النتيجة أو رد الفعل (فعل آخر أو فكرة)
"""
struct Idea
    objects::Vector{String}
    event::String
    result::String
end

"""
    compute_idea_vector(idea::Idea, pv_fn) -> Vector{Float64}

بناء المتجه الطوري المركب للفكرة:
v_idea = sum(v_objects) + i * v_event + v_result
حيث يمثل الضرب في i إزاحة طورية متعامدة (π/2) لمتجه الحدث.
"""
function compute_idea_vector(idea::Idea, pv_fn)
    v_idea = zeros(Float64, TOTAL_DIM)
    
    # 1. جمع متجهات الأشياء (الأسماء الحاملة)
    v_objs = zeros(Float64, TOTAL_DIM)
    n_objs = 0
    for obj in idea.objects
        v_raw = pv_fn(obj)
        if norm(v_raw[1:PHASE_DIM]) > 1e-10
            v_objs .+= v_raw ./ norm(v_raw[1:PHASE_DIM])
            n_objs += 1
        end
    end
    if n_objs > 0
        v_objs ./= n_objs
    end
    
    # 2. متجه الحدث (الفعل) متعامد الطور (بزاوية π/2 لجعلها متعامدة تماماً)
    v_ev_raw = pv_fn(idea.event)
    v_event = zeros(Float64, TOTAL_DIM)
    if norm(v_ev_raw[1:PHASE_DIM]) > 1e-10
        v_event .= rotate_phase(Float64.(v_ev_raw), π / 2.0)
    end
    
    # 3. متجه النتيجة
    v_res = zeros(Float64, TOTAL_DIM)
    v_res_raw = pv_fn(idea.result)
    if norm(v_res_raw[1:PHASE_DIM]) > 1e-10
        v_res .= v_res_raw ./ norm(v_res_raw[1:PHASE_DIM])
    end
    
    # دمج المكونات لبناء الموجة الدلالية المتعامدة للفكرة
    v_idea .= v_objs .+ v_event .+ v_res
    nrm = norm(v_idea[1:PHASE_DIM])
    if nrm > 1e-10
        v_idea[1:PHASE_DIM] ./= nrm
    end
    
    return v_idea
end

"""
    match_sentence_to_idea(sentence_words::Vector{String}, idea::Idea, pv_fn) -> Float64

قياس مدى مطابقة جملة معينة للفكرة المطلوبة. 
يقوم ببناء موجة الجملة المركبة ومقاطعتها رنينياً مع موجة الفكرة.
"""
function match_sentence_to_idea(sentence_words::Vector{String}, idea::Idea, pv_fn)
    if isempty(sentence_words)
        return 0.0
    end
    
    # 1. بناء المتجه الطوري للفكرة
    v_idea = compute_idea_vector(idea, pv_fn)
    
    # 2. بناء المتجه الموجي للجملة (جمع متجهات كلماتها)
    v_sent = zeros(Float64, TOTAL_DIM)
    for w in sentence_words
        v_w = pv_fn(w)
        if norm(v_w[1:PHASE_DIM]) > 1e-10
            v_sent .+= v_w ./ norm(v_w[1:PHASE_DIM])
        end
    end
    
    nrm_s = norm(v_sent[1:PHASE_DIM])
    nrm_i = norm(v_idea[1:PHASE_DIM])
    
    if nrm_s < 1e-10 || nrm_i < 1e-10
        return 0.0
    end
    
    # حساب التشابه الطوري (الضرب الداخلي) بين موجة الجملة وموجة الفكرة
    sim = dot(v_sent[1:PHASE_DIM], v_idea[1:PHASE_DIM]) / (nrm_s * nrm_i)
    return max(0.0, sim)
end

"""
    score_candidate_for_idea(candidate_word::String, context_words::Vector{String}, 
                             idea::Idea, pv_fn) -> Float64

حساب توافق الكلمة المرشحة مع الفكرة المستهدفة بالاعتماد على موقعها السياقي.
"""
function score_candidate_for_idea(candidate_word::String, context_words::Vector{String}, 
                                  idea::Idea, pv_fn)
    # تجميع سياق الجملة الحالي شاملاً الكلمة المرشحة
    sentence = copy(context_words)
    push!(sentence, candidate_word)
    
    return match_sentence_to_idea(sentence, idea, pv_fn)
end

end # module IdeaEngine
