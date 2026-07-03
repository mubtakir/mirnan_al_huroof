# tune_gravity.jl - الضبط التلقائي لعوامل الجاذبية الطورية والموضعية
using Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "src", "MirnanNew.jl")); using .MirnanNew, SparseArrays, JSON, LinearAlgebra, Statistics

# إدراج مسبب التحميل الموحد
include(joinpath(@__DIR__, "train.jl"))

# 1. تعريف الجمل الصحيحة والجمل المبعثرة (Word Salad)
const FLUENT_SENTENCES = [
    "العلم نور ساطع يضيء دروب الجهل",
    "الفضيله هي جوهر الحكمه",
    "ينفع العلم عندما يقترن بالعمل الصالح",
    "الصبر مفتاح الفرج وطريق الارتقاء",
    "والتجربه مع الفضيله تولد الحكمه وتكشف اسرار المعرفه"
]

const SALAD_SENTENCES = [
    "الجهل دروب يضيء ساطع العلم نور",
    "الحكمه جوهر هي الفضيله",
    "يقترن بالعمل العلم الصالح عندما ينفع",
    "الارتقاء وطريق الفرج مفتاح الصبر",
    "المعرفه اسرار وتكشف الحكمه تولد الفضيله مع والتجربه"
]

# 2. دالة حساب تقييم الجملة
function score_sentence(gen, sentence::String)
    # تطبيع النص ليتوافق تماماً مع المعجم المدرب
    normalized_s = _normalize_arabic_text(sentence)
    tokens = split(normalized_s)
    total_score = 0.0
    all_pv = Vector{Float64}[]
    prompt_pv = Vector{Float64}[]
    
    # استخراج الدوال الداخلية للمحرك
    _score = MirnanNew.Physics.Generator._score
    _pv = MirnanNew.Physics.Generator._pv
    
    for (step, word) in enumerate(tokens)
        word_str = String(word)
        prev_word = step > 1 ? String(tokens[step-1]) : nothing
        context_words = step > 1 ? String[String(w) for w in tokens[1:step-1]] : String[]
        context_ids = [get(gen.vocab, w, 0) for w in context_words]
        
        # نقيم الكلمة باستخدام used فارغ لتفادي عقاب الكلمات المتكررة الطبيعية
        s = _score(gen, word_str, Set{String}(), all_pv, prompt_pv;
                   gen_pos=step, total_pos=length(tokens), prev_word=prev_word,
                   context_ids=context_ids, context_words=context_words, fast_only=false)
                   
        if isfinite(s)
            total_score += s
        else
            total_score += -5.0 # عقوبة قصوى للانتقال المستحيل
        end
        
        push!(all_pv, _pv(gen, word_str))
    end
    return total_score / length(tokens)
end

# 3. دالة تحديث ملف الإعدادات yaml مع الحفاظ على التنسيق والتعليقات
function update_config_yaml(sem_val, pos_val)
    path = joinpath(@__DIR__, "config.yaml")
    if isfile(path)
        content = read(path, String)
        content = replace(content, r"semantic_factor:\s*[0-9.]+" => "semantic_factor: $sem_val")
        content = replace(content, r"positional_factor:\s*[0-9.]+" => "positional_factor: $pos_val")
        write(path, content)
        println("✓ [config.yaml] تم تحديث العوامل بنجاح في ملف الإعدادات.")
    else
        println("⚠ لم يتم العثور على config.yaml لتحديثه.")
    end
end

function main()
    println("==========================================================")
    println("          مِرنان V8 — الضبط التلقائي لعوامل الجاذبية        ")
    println("==========================================================")
    
    model_dir = joinpath(@__DIR__, "model")
    data = load_model(model_dir)
    if data === nothing
        println("❌ خطأ: لم يعثر على نموذج مدرب في model/. يرجى تشغيل التدريب أولاً.")
        return
    end
    
    gen = MirnanGenerator(data["vocab"], data["K_sem"];
                          K_syn=data["K_syn"], K_dialogue=data["K_dial"], K_causal=get(data, "K_causal", nothing))
                          
    println("✓ تم تحميل النموذج بنجاح.")
    println()
    
    # النطاق المستهدف للضبط (Grid Search)
    factors = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    best_sem = 1.0
    best_pos = 1.0
    best_gap = -Inf
    
    # تخزين النتائج لعرض جدول كامل
    results_grid = []
    
    for sem in factors
        for pos in factors
            # تحديث العوامل مؤقتاً في إعدادات المولد
            gen.config["gravity"] = Dict{String,Any}(
                "semantic_factor" => sem,
                "positional_factor" => pos
            )
            
            fluent_scores = [score_sentence(gen, s) for s in FLUENT_SENTENCES]
            salad_scores = [score_sentence(gen, s) for s in SALAD_SENTENCES]
            
            mean_fluent = mean(fluent_scores)
            mean_salad = mean(salad_scores)
            gap = mean_fluent - mean_salad
            
            push!(results_grid, (sem=sem, pos=pos, fluent=mean_fluent, salad=mean_salad, gap=gap))
            
            if gap > best_gap
                best_gap = gap
                best_sem = sem
                best_pos = pos
            end
        end
    end
    
    # عرض النتائج كجدول فصيح ومفرّق
    println("----------------------------------------------------------")
    println("Semantic Factor    | Positional Factor  | Fluent Score | Salad Score  | Gap (Diff)")
    println("----------------------------------------------------------")
    for r in results_grid
        # طباعة نتائج منسقة
        @views println(
            rpad(string(r.sem), 18), " | ",
            rpad(string(r.pos), 18), " | ",
            rpad(string(round(r.fluent, digits=3)), 12), " | ",
            rpad(string(round(r.salad, digits=3)), 12), " | ",
            round(r.gap, digits=3)
        )
    end
    println("----------------------------------------------------------")
    println()
    
    println("==========================================================")
    println("النتائج المثلى المكتشفة:")
    println("  ← أفضل معامل جاذبية طورية (دلالية): $best_sem")
    println("  ← أفضل معامل جاذبية موضعية (نحوية): $best_pos")
    println("  ← الفارق القياسي المحقق (Gap): $(round(best_gap, digits=3))")
    println("==========================================================")
    println()
    
    # تحديث ملف الإعدادات بالقيم المثالية المكتشفة
    update_config_yaml(best_sem, best_pos)
end

main()
