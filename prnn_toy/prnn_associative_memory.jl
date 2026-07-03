"""
 Phase-Resonant Neural Network (PRNN) - Associative Memory & Pattern Completion
===============================================================================
نموذج متقدم للذاكرة الترابطية وإكمال الأنماط التالفة باستخدام مذبذبات ستوارت-لانداو
(Stuart-Landau Oscillators) في الفضاء المركب.

يقوم هذا النموذج بتخزين نمطين طوريين مختلفين في مصفوفة الاقتران المركب K:
  1. النمط الأول: نصف متزامن عند 0 ونصف عند pi.
  2. النمط الثاني: نمط متناوب (0، pi، 0، pi...).

عند تقديم نمط تالف (عناصر مفقودة أو سعات صفرية أو أطوار مقلوبة)، تتطور معادلات
الشبكة تلقائياً لسحب النظام بأكمله نحو أقرب "جاذب طوري" (Phase Attractor Basin)
واسترجاع النمط الأصلي بالكامل بسعات كاملة ومستقرة.
"""
module PRNNAssociativeMemory

using LinearAlgebra, Random, Statistics

export run_prnn_associative_memory

# ═══════════════════ محاكي ستوارت-لانداو (Stuart-Landau Simulator) ═══════════════════
function simulate_stuart_landau!(z::Vector{ComplexF64}, K::Matrix{Float64}, omega::Vector{Float64},
                                 mu::Float64, clamped_nodes::Vector{Int}, steps::Int, dt::Float64, noise_level::Float64)
    N = length(z)
    dz = zeros(ComplexF64, N)
    
    for step in 1:steps
        for i in 1:N
            if i in clamped_nodes
                dz[i] = 0.0 + 0.0im
                continue
            end
            
            # اقتران انتشاري (Diffusive Coupling)
            coupling = 0.0 + 0.0im
            for j in 1:N
                coupling += K[i, j] * (z[j] - z[i])
            end
            
            # ديناميكية ستوارت-لانداو
            dz[i] = (mu - abs2(z[i]) + im * omega[i]) * z[i] + coupling
        end
        
        # ضوضاء لانجفان الحرارية
        noise = noise_level * sqrt(dt) .* randn(ComplexF64, N)
        for node in clamped_nodes
            noise[node] = 0.0 + 0.0im
        end
        z .+= dt .* dz .+ noise
    end
    return z
end

# ═══════════════════ دالة تشغيل واختبار الذاكرة الترابطية ═══════════════════
function run_prnn_associative_memory()
    println("╔═════════════════════════════════════════════════════════════╗")
    println("║   PRNN Associative Memory — الذاكرة الترابطية وإكمال الأنماط  ║")
    println("╚═════════════════════════════════════════════════════════════╝")
    println()

    # 1. إعدادات الشبكة
    N = 8 # 8 مذبذبات
    mu = 1.0
    dt = 0.03
    sim_steps = 600
    
    Random.seed!(42)
    
    # ترددات ذاتية صغيرة جداً لكسر التناظر الطوبولوجي
    omega = (rand(Float64, N) .- 0.5) .* 0.02
    
    # 2. الأنماط المراد تخزينها (أطوار ثنائية القيمة 0 و pi)
    # النمط 1: النصف الأول متزامن عند الطور 0، والنصف الثاني عند الطور pi
    pat1 = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
    
    # النمط 2: متناوب طورياً (0، pi، 0، pi...)
    pat2 = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    
    println("📂 الأنماط المخزنة في حوض جذب الشبكة:")
    println("   • النمط الأول: $pat1")
    println("   • النمط الثاني: $pat2")
    println()

    # 3. بناء مصفوفة الأوزان الهيبية (Hebbian Prescription)
    # K_ij = (1/N) * sum_p (xi_i * xi_j)
    K = zeros(Float64, N, N)
    for i in 1:N
        for j in 1:N
            if i != j
                K[i, j] = (pat1[i]*pat1[j] + pat2[i]*pat2[j]) / N
            end
        end
    end
    println("✓ تم بناء مصفوفة الأوزان الترابطية K بنجاح.")
    println("-" ^ 65)

    # ═══════════════════ الاختبار الأول: النمط 1 المشوش ═══════════════════
    println("\n🔍 الاختبار 1: استرجاع [النمط الأول] من إدخال تالف ومفقود البيانات:")
    
    # نقوم بتشويه النمط 1:
    # 1. قلب طور العنصر الأول (وضعناه -1 بدلاً من 1)
    # 2. تصفير وتقليل سعة العقد 3، 4، 5 (سعات منخفضة جداً تحاكي فقدان البيانات)
    z_corrupted1 = ComplexF64[
        -1.0, # مقلوب طورياً (كان يجب أن يكون 1.0)
        1.0,
        0.1,  # مفقود (السعة تقترب من الصفر)
        -0.2, # مفقود ومشوش
        0.0,  # مفقود تماماً
        -1.0,
        -1.0,
        -1.0
    ]
    
    println("   ← المدخل التالف (الأطوار):   ", round.(angle.(z_corrupted1) ./ pi, digits=2), " pi")
    println("   ← المدخل التالف (السعات):   ", round.(abs.(z_corrupted1), digits=2))
    
    # تشغيل محاكاة الاسترخاء الحر
    z_recall1 = copy(z_corrupted1)
    simulate_stuart_landau!(z_recall1, K, omega, mu, Int[], sim_steps, dt, 0.0)
    
    # استخلاص الاتجاهات بعد استقرار التزامن
    pred_signs1 = [cos(angle(z)) > 0.0 ? 1.0 : -1.0 for z in z_recall1]
    
    println("   → الأطوار المسترجعة:        ", round.(angle.(z_recall1) ./ pi, digits=2), " pi")
    println("   → السعات المستقرة:          ", round.(abs.(z_recall1), digits=2))
    println("   → النمط المستخلص:           ", pred_signs1)
    
    success1 = (pred_signs1 == pat1) || (pred_signs1 == -pat1)
    status_symbol1 = success1 ? "✓ ناجح" : "❌ فشل"
    println("   [الحالة]: $status_symbol1")
    
    # ═══════════════════ الاختبار الثاني: النمط 2 المشوش ═══════════════════
    println("\n🔍 الاختبار 2: استرجاع [النمط الثاني] من إدخال تالف ومفقود البيانات:")
    
    # تشويه النمط 2: تصفير العقد 2 و 5 و 7 وتشويه العقدة 4
    z_corrupted2 = ComplexF64[
        1.0,
        0.0,  # مفقود
        1.0,
        -0.5, # مشوش ومفقود جزئياً
        0.2,  # مفقود
        -1.0,
        0.0,  # مفقود تماماً
        -1.0
    ]
    
    println("   ← المدخل التالف (الأطوار):   ", round.(angle.(z_corrupted2) ./ pi, digits=2), " pi")
    println("   ← المدخل التالف (السعات):   ", round.(abs.(z_corrupted2), digits=2))
    
    # محاكاة الاسترخاء
    z_recall2 = copy(z_corrupted2)
    simulate_stuart_landau!(z_recall2, K, omega, mu, Int[], sim_steps, dt, 0.0)
    
    pred_signs2 = [cos(angle(z)) > 0.0 ? 1.0 : -1.0 for z in z_recall2]
    
    println("   → الأطوار المسترجعة:        ", round.(angle.(z_recall2) ./ pi, digits=2), " pi")
    println("   → السعات المستقرة:          ", round.(abs.(z_recall2), digits=2))
    println("   → النمط المستخلص:           ", pred_signs2)
    
    success2 = (pred_signs2 == pat2) || (pred_signs2 == -pat2)
    status_symbol2 = success2 ? "✓ ناجح" : "❌ فشل"
    println("   [الحالة]: $status_symbol2")
    
    println("-" ^ 65)
    if success1 && success2
        println("🎉 نجاح باهر! نجحت مذبذبات ستوارت-لانداو في التصرف كذاكرة ترابطية مستقرة؛ استطاع النظام سحب الأنماط التالفة وإعادة بناء المعلومات والأطوار المفقودة بنسبة 100%!")
    else
        println("⚠️ لم تنجح عملية الاسترجاع بالكامل لجميع الأنماط.")
    end
    println("╚═════════════════════════════════════════════════════════════╝")
end

end # module PRNNAssociativeMemory

if abspath(PROGRAM_FILE) == @__FILE__
    PRNNAssociativeMemory.run_prnn_associative_memory()
end
