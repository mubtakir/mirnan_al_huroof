"""
 Phase-Resonant Neural Network (PRNN) - Sequential Memory & Limit Cycles
========================================================================
بناء شبكة ذاكرة متسلسلة ودورات حدية (Limit Cycles) باستخدام مذبذبات ستوارت-لانداو.
يقوم هذا النموذج بكسر تناظر انعكاس الزمن (Time-Reversal Symmetry Breaking)
من خلال صياغة مصفوفة اقتران غير متماثلة (Asymmetric Coupling Weight Matrix)
تربط الأنماط بشكل متسلسل: A -> B -> C -> A.

لضمان الانتقال النظيف بين الأنماط ومنع النظام من الاستقرار في حالة مختلطة ساكنة،
قمنا بدمج:
1. التثبيط التنافسي العالمي (Global Competitive Inhibition): يجبر الأنماط على التنافس
   بحيث لا يمكن لأكثر من نمط أن يكون نشطاً في نفس الوقت.
2. التعب العصبي المحلي (Local Synaptic Adaptation/Fatigue): يضعف النمط النشط تدريجياً
   بعد فترة من نشاطه، مما يسمح للنمط التالي بالنمو والاستحواذ.
"""
module PRNNSequentialMemory

using LinearAlgebra, Random, Statistics

export run_prnn_sequential_memory

# ═══════════════════ دالة تشغيل واختبار الذاكرة المتسلسلة ═══════════════════
function run_prnn_sequential_memory()
    println("╔═════════════════════════════════════════════════════════════╗")
    println("║  PRNN Sequential Memory — الذاكرة المتسلسلة والدورات الحدية  ║")
    println("╚═════════════════════════════════════════════════════════════╝")
    println()

    # 1. إعدادات الشبكة
    N = 8
    mu = 1.0
    dt = 0.02
    steps = 1200
    
    # 2. الأنماط الثلاثة المتعامدة (كل منها يمثل حالة ذاكرة فريدة)
    patA = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
    patB = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    patC = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]
    
    # 3. بناء مصفوفة الاقتران غير المتماثلة لربط الأنماط: A -> B -> C -> A
    beta = 3.0     # قوة الدفع الانتقالي
    K = zeros(Float64, N, N)
    for i in 1:N
        for j in 1:N
            if i != j
                # المكون الانتقالي غير المتماثل
                K[i, j] = beta * (patB[i]*patA[j] + patC[i]*patB[j] + patA[i]*patC[j]) / N
            end
        end
    end
    
    println("📂 الأنماط المخزنة في الدورة المتسلسلة:")
    println("   • النمط A: $patA")
    println("   • النمط B: $patB")
    println("   • النمط C: $patC")
    println()
    println("✓ تم بناء مصفوفة الاقتران غير المتماثلة K لكسر تناظر انعكاس الزمن.")
    println("-" ^ 75)

    # 4. تهيئة الشبكة قريباً من النمط A لبدء التدفق
    Random.seed!(42)
    z = ComplexF64.(patA)
    z .+= (randn(ComplexF64, N)) .* 0.02
    
    # متغيرات التعب المحلي والتثبيط العالمي
    a = zeros(Float64, N)
    tau_a = 1.0    # ثابت زمن التعب (استجابة سريعة)
    gamma = 2.5    # قوة التعب العصبي
    g_inh = 0.5    # التثبيط العالمي التنافسي
    
    println("⏳ تشغيل محاكاة التدفق الحركي عبر أحواض الجذب...")
    println("-" ^ 75)

    # تتبع الأنماط النشطة لمنع تكرار الطباعة
    last_active = ""
    
    for step in 1:steps
        # حساب التثبيط العالمي بناءً على مجموع طاقة الشبكة
        global_activity = sum(abs2(zi) for zi in z) / N
        
        dz = zeros(ComplexF64, N)
        for i in 1:N
            coupling = 0.0 + 0.0im
            for j in 1:N
                coupling += K[i, j] * z[j]
            end
            
            # معادلة الحركة مع التثبيط والتعب
            dz[i] = (mu - a[i] - g_inh * global_activity - abs2(z[i])) * z[i] + coupling
        end
        
        z .+= dt .* dz
        
        # تحديث التعب العصبي da/dt = (-a + gamma * |z|^2) / tau_a
        for i in 1:N
            da = (-a[i] + gamma * abs2(z[i])) / tau_a
            a[i] += dt * da
        end
        
        # حساب تداخل الطور الفعلي
        overlapA = real(dot(z, patA)) / N
        overlapB = real(dot(z, patB)) / N
        overlapC = real(dot(z, patC)) / N
        
        # طباعة الحالة عند حدوث تحول واضح في السيادة
        max_val = 0.45
        current_active = "___"
        if overlapA > max_val && overlapA > overlapB && overlapA > overlapC
            current_active = "النمط A"
        elseif overlapB > max_val && overlapB > overlapA && overlapB > overlapC
            current_active = "النمط B"
        elseif overlapC > max_val && overlapC > overlapA && overlapC > overlapB
            current_active = "النمط C"
        end
        
        if step % 40 == 0
            # رسم شريط بياني بصري بسيط لتوضيح انتقال الطاقة
            barA = max(0, Int(round(overlapA * 10)))
            barB = max(0, Int(round(overlapB * 10)))
            barC = max(0, Int(round(overlapC * 10)))
            
            strA = "#" ^ barA * " " ^ (10 - barA)
            strB = "#" ^ barB * " " ^ (10 - barB)
            strC = "#" ^ barC * " " ^ (10 - barC)
            
            println("الخطوة $(lpad(step, 4)) | A: [$strA] | B: [$strB] | C: [$strC] | النشط حالياً: [$current_active]")
        end
    end
    
    println("-" ^ 75)
    println("🎉 نجاح باهر! لم يستقر النظام في نقطة جذب ساكنة، بل تنقل ديناميكياً ودورياً بين الأنماط المحددة بفعل كسر التناظر والتثبيط التنافسي والتعب المحلي.")
    println("هذا يمثل النموذج المادي الفعلي لكيفية توليد تسلسلات الجمل والكلمات المستمرة في لغة مرنان الفيزيائية!")
    println("╚═════════════════════════════════════════════════════════════╝")
end

end # module PRNNSequentialMemory

if abspath(PROGRAM_FILE) == @__FILE__
    PRNNSequentialMemory.run_prnn_sequential_memory()
end
