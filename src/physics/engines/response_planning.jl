"""
مخطط الاستجابة والمسار الطوري — Response & Trajectory Planning.

يتحكم في توجيه التوليد اللغوي عبر مسار طوري مستمر في فضاء الطور 64D:
1. TrajectoryPlanner: يخطط مساراً هندسياً يمر بمعالم طورية متتالية مع جدول تشدد تدريجي (Tightening Schedule).
2. ResponsePlanner: يحدد الهدف العام للرد وصيغته الكلية بناءً على القصد المكتشف.
3. ResponseArchitect: يقسم الرد إلى مراحل معمارية بنيوية (مقدمة، تفصيل، خاتمة، إلخ) ويوجه التوليد نحو المفاهيم المناسبة لكل مرحلة.
"""
module ResponsePlanning

using LinearAlgebra, Random, Statistics

export ResponsePlanner, ResponseArchitect, TrajectoryPlanner, TrajectoryMilestone,
       plan!, plan_response!, plan_architect!, response_fidelity, architect_score, trajectory_score

# ═══════════════════════════════════════════════════════
# المعالم والمسار الطوري
# ═══════════════════════════════════════════════════════

struct TrajectoryMilestone
    name::String
    target_pv::Vector{Float64}
    tightness::Float64
    phase_position::Float64
    description::String
end

mutable struct TrajectoryPlanner
    milestones::Vector{TrajectoryMilestone}
    opening_tightness::Float64
    closing_tightness::Float64
end

TrajectoryPlanner() = TrajectoryPlanner(TrajectoryMilestone[], 0.2, 1.0)

# خرائط معالم النوايا
const INTENT_MILESTONES = Dict{String, Vector{Tuple{String, Float64, String}}}(
    "GREETING" => [
        ("opening", 0.4, "تحية واردة"),
        ("response", 0.7, "رد التحية"),
        ("warmth", 0.3, "دفء إضافي")
    ],
    "QUESTION" => [
        ("opening", 0.3, "مقدمة مباشرة"),
        ("elaboration", 0.5, "توضيح وتفصيل"),
        ("detail", 0.6, "شرح دقيق"),
        ("closing", 0.4, "خلاصة إجابة")
    ],
    "COMMAND" => [
        ("acknowledgment", 0.6, "إقرار بالطلب"),
        ("execution", 0.7, "بدء التنفيذ"),
        ("result", 0.5, "تأكيد النتيجة")
    ],
    "STATEMENT" => [
        ("acknowledgment", 0.4, "إقرار بالسياق"),
        ("expansion", 0.5, "توسيع الفكرة"),
        ("insight", 0.3, "طرح بصيرة")
    ],
    "OPINION" => [
        ("engagement", 0.4, "تفاعل مع الرأي"),
        ("nuance", 0.5, "تحليل الفروقات"),
        ("synthesis", 0.6, "تركيب نهائي")
    ],
    "DEFAULT" => [
        ("opening", 0.3, "بداية جملة"),
        ("development", 0.5, "تطور الفكرة"),
        ("closing", 0.4, "ختام متزن")
    ]
)

"""
    _detect_intent(prompt_tokens) -> String

كاشف نوايا داخلي مبسط لتوجيه التخطيط الطوري.
"""
function _detect_intent(prompt_tokens::Vector{String})::String
    isempty(prompt_tokens) && return "DEFAULT"
    
    greetings = Set(["سلام", "مرحبا", "اهلا", "أهلا", "صباح", "مساء", "hello", "hi", "hey"])
    questions = Set(["هل", "ما", "كيف", "لماذا", "متى", "أين", "من", "كم", "what", "how", "why", "where", "when", "who"])
    commands = Set(["افعل", "اكتب", "اذهب", "قل", "اعمل", "do", "go", "make", "write", "say"])
    requests = Set(["من فضلك", "لو سمحت", "أرجو", "please", "could", "would"])
    farewells = Set(["وداعا", "مع السلامة", "bye", "goodbye", "see you"])
    thanks = Set(["شكرا", "مشكور", "thanks", "thank"])

    for w in prompt_tokens
        wl = lowercase(w)
        if wl in greetings; return "GREETING"; end
        if wl in questions; return "QUESTION"; end
        if wl in commands; return "COMMAND"; end
        if wl in requests; return "REQUEST"; end
        if wl in farewells; return "FAREWELL"; end
        if wl in thanks; return "THANK"; end
    end
    return "STATEMENT"
end

"""
    plan!(tp::TrajectoryPlanner, prompt_tokens, pv_fn)

بناء مسار هندسي متكامل يمر بمعالم طورية متتالية لمدخل المستخدم.
"""
function plan!(tp::TrajectoryPlanner, prompt_tokens::Vector{String}, pv_fn)
    empty!(tp.milestones)
    isempty(prompt_tokens) && return

    # حساب المتجه الطوري المتوسط للـ prompt
    pvs = Vector{Float64}()
    for w in prompt_tokens
        try
            pv = Float64.(pv_fn(w))
            if norm(pv) > 1e-10
                push!(pvs, pv)
            end
        catch e
            @debug "Trajectory pv_fn failed for '$w': $e"
        end
    end

    start_pv = if !isempty(pvs)
        # المتوسط الحسابي لمتجهات الكلمات
        mean(pvs)
    else
        zeros(Float64, 10000) # البعد الكلي المعتاد
    end
    
    nrm = norm(start_pv)
    if nrm > 1e-10
        start_pv ./= nrm
    end

    # كشف القصد اللغوي لتحديد شكل المسار
    intent = _detect_intent(prompt_tokens)
    milestone_defs = get(INTENT_MILESTONES, intent, INTENT_MILESTONES["DEFAULT"])
    
    # تحديد عدد المعالم حسب طول المدخل
    n_words = length(prompt_tokens)
    n_milestones = min(length(milestone_defs), max(2, div(n_words, 3)))
    milestone_defs = milestone_defs[1:n_milestones]

    # استخدام مولد أرقام عشوائي محلي لضمان تماسك الضوضاء الطورية
    rng = MersenneTwister(42)

    for (i, (name, tightness, desc)) in enumerate(milestone_defs)
        phase_pos = i / length(milestone_defs)
        target_pv = zeros(Float64, 10000)
        t = tightness

        if i == 1
            target_pv = copy(start_pv)
            t = tp.opening_tightness
        elseif i == length(milestone_defs)
            # الهدف النهائي يدمج طاقة المدخل مع تقلب طوري عشوائي صغير يمثل الاستقرار
            target_pv = start_pv .* (0.3 + 0.2 * i) .+ randn(rng, 10000) .* 0.1
            t = tp.closing_tightness
        else
            # معالم وسطية تدرج بين طاقة البداية والانتقال الطوري
            blend = 0.4 + 0.15 * i
            target_pv = start_pv .* (1.0 - blend) .+ randn(rng, 10000) .* (0.15 * blend)
            t = tightness
        end

        pv_norm = norm(target_pv)
        if pv_norm > 1e-10
            target_pv ./= pv_norm
        end

        push!(tp.milestones, TrajectoryMilestone(name, target_pv, t, phase_pos, desc))
    end
end

"""
    trajectory_score(tp::TrajectoryPlanner, candidate_pv, position) -> Float64

حساب درجة مواءمة الكلمة المرشحة مع المسار الطوري المستمر المستنبط.
يتم استيفاء المتجه الطوري التدريجي بين المعالم حسب التقدم الفعلي للتوليد.
"""
function trajectory_score(tp::TrajectoryPlanner, candidate_pv::AbstractVector, position::Int; max_steps::Int=12)
    isempty(tp.milestones) && return 0.0
    
    progress = position / max(1, max_steps)
    n_milestones = length(tp.milestones)
    
    # تحديد المعلم الحالي والتالي
    milestone_idx = min(Int(floor(progress * n_milestones)) + 1, n_milestones)
    
    milestone = if milestone_idx < n_milestones
        next_m = tp.milestones[milestone_idx + 1]
        curr_m = tp.milestones[milestone_idx]
        local_progress = (progress * n_milestones) - (milestone_idx - 1)
        
        # استيفاء ناعم للمتجهات الطورية والتشدد
        blended_pv = curr_m.target_pv .* (1.0 - local_progress) .+ next_m.target_pv .* local_progress
        blended_tightness = curr_m.tightness * (1.0 - local_progress) + next_m.tightness * local_progress
        
        nrm = norm(blended_pv)
        if nrm > 1e-10
            blended_pv ./= nrm
        end
        
        TrajectoryMilestone(curr_m.name * "->" * next_m.name, blended_pv, blended_tightness, progress, "")
    else
        tp.milestones[milestone_idx]
    end
    
    n_cand = norm(candidate_pv)
    n_target = norm(milestone.target_pv)
    if n_cand < 1e-10 || n_target < 1e-10
        return 0.0
    end
    
    # تشابه جيب التمام الطوري × معامل التشدد الحالي
    align = dot(candidate_pv, milestone.target_pv) / (n_cand * n_target)
    return max(0.0, align) * milestone.tightness
end

# ═══════════════════════════════════════════════════════
# تخطيط الرد العام
# ═══════════════════════════════════════════════════════

mutable struct ResponsePlanner
    intent::String
    strategy::String
    target_words::Vector{String}
    target_pv::Vector{Float64}
    confidence::Float64
    tightening::Vector{Float64}
end

ResponsePlanner() = ResponsePlanner("STATEMENT", "direct", String[], zeros(Float64, 10000), 0.5, Float64[])

"""
    plan_response!(rp::ResponsePlanner, intent, pv_fn; max_words=8)

تخطيط المتجه الكلي المستهدف للرد وبناء جدول التشدد الطوري.
"""
function plan_response!(rp::ResponsePlanner, intent::String, pv_fn=nothing; max_words::Int=8)
    rp.intent = intent
    
    # اختيار استراتيجية الرد والكلمات المرجعية بناءً على القصد
    if intent == "GREETING"
        rp.target_words = ["وعليكم", "السلام", "أهلا", "مرحبا", "ورحمة", "الله"]
        rp.strategy = "greet_back"
    elseif intent == "QUESTION"
        rp.target_words = ["لأن", "بسبب", "الجواب", "نعم", "إن", "قد", "السبب"]
        rp.strategy = "answer"
    elseif intent == "COMMAND"
        rp.target_words = ["حاضر", "تم", "سأفعل", "إليك", "الآن"]
        rp.strategy = "acknowledge"
    elseif intent == "THANK"
        rp.target_words = ["العفو", "أهلاً", "شكر", "واجب"]
        rp.strategy = "politeness"
    elseif intent == "FAREWELL"
        rp.target_words = ["وداعا", "سلام", "أمان", "اللقاء"]
        rp.strategy = "farewell"
    else
        rp.target_words = ["إن", "قد", "هذا", "هناك", "كذلك", "أيضاً"]
        rp.strategy = "statement"
    end

    # بناء المتجه الطوري المستهدف
    rp.target_pv = zeros(Float64, 10000)
    if pv_fn !== nothing
        n_valid = 0
        for w in rp.target_words
            try
                pv = Float64.(pv_fn(w))
                if norm(pv) > 1e-10
                    rp.target_pv .+= pv
                    n_valid += 1
                end
            catch e
                @debug "Response planner pv_fn failed for '$w': $e"
            end
        end
        if n_valid > 0
            rp.target_pv ./= n_valid
        end
        nrm = norm(rp.target_pv)
        if nrm > 1e-10
            rp.target_pv ./= nrm
        end
    end

    # جدول التشدد الخطي من 0.2 (حرية عالية) إلى 1.0 (تقارب مطلق)
    rp.tightening = collect(range(0.2, 1.0; length=max_words))
    rp.confidence = 0.8
end

function response_fidelity(rp::ResponsePlanner, word::String)
    return word in rp.target_words ? 1.0 : 0.0
end

# ═══════════════════════════════════════════════════════
# المعماري الهيكلي للرد
# ═══════════════════════════════════════════════════════

mutable struct ResponseStage
    name::String
    target_words::Vector{String}
    target_pv::Vector{Float64}
    tightness::Float64
    progress::Float64
end

mutable struct ResponseArchitect
    stages::Vector{ResponseStage}
end

ResponseArchitect() = ResponseArchitect(ResponseStage[])

"""
    plan_architect!(ra::ResponseArchitect, intent, prompt_tokens, pv_fn)

بناء المراحل البنيوية للرد (المقدمة، المتن، الخاتمة) بناءً على القصد المكتشف ومتجهات المفاهيم.
"""
function plan_architect!(ra::ResponseArchitect, intent::String, prompt_tokens::Vector{String}, pv_fn)
    empty!(ra.stages)
    
    structure = if intent == "GREETING"
        ["greeting_return", "blessing"]
    elseif intent == "QUESTION"
        ["acknowledgment", "definition", "elaboration", "closing"]
    elseif intent == "THANK"
        ["acknowledgment", "politeness"]
    elseif intent == "FAREWELL"
        ["farewell_return", "wish"]
    elseif intent == "COMMAND"
        ["acknowledgment", "execution"]
    else
        ["acknowledgment", "addition"]
    end

    n_stages = length(structure)

    for (stage_idx, stage_name) in enumerate(structure)
        progress = stage_idx / max(n_stages, 1)

        target_words = if stage_name == "greeting_return"
            ["وعليكم", "السلام", "مرحبا", "أهلا", "welcome", "hello"]
        elseif stage_name == "blessing"
            ["ورحمة", "الله", "بركاته", "بك", "peace"]
        elseif stage_name == "acknowledgment"
            ["نعم", "بالطبع", "حسنا", "أفهم", "yes", "of", "course", "indeed"]
        elseif stage_name == "definition"
            ["هو", "تعني", "عبارة", "معنى", "means", "is", "تعريف"]
        elseif stage_name == "elaboration"
            ["كما", "و", "حيث", "لأن", "أيضا", "because", "also"]
        elseif stage_name == "closing"
            ["بشكل", "عام", "إذن", "ولذلك", "therefore", "thus", "overall"]
        elseif stage_name == "politeness"
            ["العفو", "ولا", "شكر", "welcome", "thank", "please"]
        elseif stage_name == "farewell_return"
            ["مع", "السلامة", "أمان", "الله", "bye", "peace", "safe"]
        elseif stage_name == "wish"
            ["أتمنى", "إن", "شاء", "hopefully", "wish"]
        elseif stage_name == "execution"
            ["تم", "إليك", "here", "done", "جاهز"]
        elseif stage_name == "addition"
            ["أيضا", "كذلك", "also", "moreover", "إضافة"]
        else
            String[]
        end

        target_pv = zeros(Float64, 10000)
        n_valid = 0
        for tw in target_words
            try
                pv = Float64.(pv_fn(tw))
                if norm(pv) > 1e-10
                    target_pv .+= pv
                    n_valid += 1
                end
            catch e
                @debug "Architect pv_fn failed for '$tw': $e"
            end
        end
        if n_valid > 0
            target_pv ./= n_valid
        end
        nrm = norm(target_pv)
        if nrm > 1e-10
            target_pv ./= nrm
        end

        # التشدد يتزايد مع تقدم مراحل الرد لضمان التماسك البنيوي
        tightness = 0.2 + 0.6 * progress

        push!(ra.stages, ResponseStage(stage_name, target_words, target_pv, tightness, progress))
    end
end

"""
    architect_score(ra::ResponseArchitect, step, total_steps, word) -> Float64

حساب توافق الكلمة المرشحة مع المرحلة البنيوية الحالية للرد.
"""
function architect_score(ra::ResponseArchitect, step::Int, total_steps::Int, word::String)
    isempty(ra.stages) && return 0.0

    progress = step / max(1, total_steps)
    n_stages = length(ra.stages)
    stage_idx = min(Int(floor(progress * n_stages)) + 1, n_stages)

    current_stage = ra.stages[stage_idx]

    score = 0.0
    if word in current_stage.target_words
        score += 1.0 * current_stage.tightness
    end

    return score
end

end # module ResponsePlanning
