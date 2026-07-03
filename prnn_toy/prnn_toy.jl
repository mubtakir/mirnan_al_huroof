"""
 Phase-Resonant Neural Network (PRNN) - Toy Model
=================================================
إثبات مفهوم للشبكة الرنينية الطورية لحل معضلة XOR المنطقية.
تعتمد الشبكة بالكامل على تزامن المذبذبات الطورية (Kuramoto Oscillators)
والتعلم المحلي الهيبي التبايني (Contrastive Hebbian Learning) بدون استخدام Backpropagation.

الهيكل (7 مذبذبات):
- 1، 2: المدخلات (مغلقة طورياً عند 0 أو pi)
- 3: مذبذب التحيز المرجعي / Bias Oscillator (مغلق طورياً دائماً عند 0.0 لتوفير مرجع طوري وكسر التناظر الطوري العام)
- 4، 5، 6: المذبذبات الخفية (تتعلم الميزات وتنشئ تمثيلاً غير خطي)
- 7: المخرج (مغلق في المرحلة الموجبة، وحر في السالبة وفي مرحلة الاختبار)
"""
module PRNNToy

using LinearAlgebra, Random, Statistics

export run_prnn_xor

# ═══════════════════ محاكي تطور الأطوار (Kuramoto Simulator) ═══════════════════
function simulate_kuramoto!(phi::Vector{Float64}, K::Matrix{Float64}, omega::Vector{Float64},
                           clamped_nodes::Vector{Int}, steps::Int, dt::Float64; noise_level::Float64=0.01)
    N = length(phi)
    dphi = zeros(Float64, N)
    
    for step in 1:steps
        for i in 1:N
            if i in clamped_nodes
                dphi[i] = 0.0 # المذبذبات المغلقة لا تتغير أطوارها
                continue
            end
            
            # معادلة كوراموتو للرنين والاقتران
            coupling_sum = 0.0
            for j in 1:N
                coupling_sum += K[i, j] * sin(phi[j] - phi[i])
            end
            dphi[i] = omega[i] + coupling_sum
        end
        # تحديث الأطوار مع إضافة ضوضاء لانجفان الحرارية (Langevin Thermal Noise)
        noise = noise_level * sqrt(dt) .* randn(Float64, N)
        for node in clamped_nodes
            noise[node] = 0.0
        end
        phi .+= dt .* dphi .+ noise
    end
    # حصر الأطوار بين -pi و pi
    phi .= mod2pi.(phi .+ pi) .- pi
    return phi
end

# ═══════════════════ دالة التدريب والتشغيل ═══════════════════
function run_prnn_xor(; epochs::Int=600, lr::Float64=0.1, dt::Float64=0.04, sim_steps::Int=500, init_scale::Float64=1.0)
    println("╔══════════════════════════════════════════════╗")
    println("║    PRNN Toy Model — حل XOR بالتزامن الطوري    ║")
    println("╚══════════════════════════════════════════════╝")
    println()

    # 1. إعدادات الشبكة
    N = 7
    
    # تثبيت البذرة العشوائية لضمان تكرار النتائج
    Random.seed!(42)
    
    # كسر التناظر الهيكلي: منح كل مذبذب ترددًا ذاتيًا عشوائيًا طفيفًا
    omega = (rand(Float64, N) .- 0.5) .* 0.05
    omega[3] = 0.0 # مذبذب التحيز ليس له تردد ذاتي نشط
    
    # مصفوفة الاقتران K (أوزان متناظرة وقابلة للتعديل)
    # نربط كل العصبونات ببعضها مع أوزان عشوائية صغيرة
    K = (rand(N, N) .- 0.5) .* init_scale
    # جعلها متناظرة وبدون اتصال ذاتي
    for i in 1:N
        K[i, i] = 0.0
        for j in (i+1):N
            K[j, i] = K[i, j]
        end
    end

    # فرض هيكلية الطبقات: إلغاء الاتصال المباشر بين المدخلات والمخرجات
    K[1, 7] = K[7, 1] = 0.0
    K[2, 7] = K[7, 2] = 0.0
    # إلغاء الاتصال المباشر بين المدخلين
    K[1, 2] = K[2, 1] = 0.0

    # بيانات XOR
    dataset = [
        # (x1, x2) -> target
        (0.0, 0.0) => 0.0,
        (1.0, 0.0) => 1.0,
        (0.0, 1.0) => 1.0,
        (1.0, 1.0) => 0.0
    ]

    clamped_pos = [1, 2, 3, 7]
    clamped_neg = [1, 2, 3]

    println("⏳ جاري تدريب الشبكة الرنينية الطورية باستخدام مذبذب مرجعي (Bias)...")
    
    for epoch in 1:epochs
        total_error = 0.0
        
        for (input_pair, target_val) in dataset
            x1, x2 = input_pair
            
            # تهيئة الأطوار (Zeros أو تهيئة قريبة من الصفر لتكون متسقة)
            phi_init = (rand(Float64, N) .- 0.5) .* 0.2
            
            # --- المرحلة الموجبة (Positive Phase - Clamped Target) ---
            phi_pos = copy(phi_init)
            phi_pos[1] = x1 * pi
            phi_pos[2] = x2 * pi
            phi_pos[3] = 0.0 # تثبيت المرجعية عند الصفر
            phi_pos[7] = target_val * pi
            
            # محاكاة التزامن
            simulate_kuramoto!(phi_pos, K, omega, clamped_pos, sim_steps, dt; noise_level=0.01)
            
            # --- المرحلة السالبة (Negative Phase - Free Target) ---
            phi_neg = copy(phi_init)
            phi_neg[1] = x1 * pi
            phi_neg[2] = x2 * pi
            phi_neg[3] = 0.0 # تثبيت المرجعية عند الصفر
            phi_neg[7] = phi_pos[7] # بدء التنبؤ من نفس الطور الموجب لضمان تماسك الحركة
            
            simulate_kuramoto!(phi_neg, K, omega, clamped_neg, sim_steps, dt; noise_level=0.01)
            
            # حساب الخطأ الدائري بين التنبؤ والهدف
            pred_phase = phi_neg[7]
            target_phase = target_val * pi
            diff = atan(sin(pred_phase - target_phase), cos(pred_phase - target_phase))
            total_error += abs(diff)
            
            # --- تحديث الأوزان محلياً (Contrastive Hebbian Weight Update) ---
            for i in 1:N
                for j in (i+1):N
                    if (i == 1 && j == 7) || (i == 7 && j == 1) ||
                       (i == 2 && j == 7) || (i == 7 && j == 2) ||
                       (i == 1 && j == 2) || (i == 2 && j == 1)
                        continue
                    end
                    
                    pos_coherence = cos(phi_pos[i] - phi_pos[j])
                    neg_coherence = cos(phi_neg[i] - phi_neg[j])
                    
                    # هيبي تبايني
                    dk = lr * (pos_coherence - neg_coherence)
                    
                    K[i, j] += dk
                    K[j, i] = K[i, j]
                end
            end
        end
        
        # تنظيم خفيف للأوزان لمنع التضخم اللانهائي (Weight Decay)
        K .*= 0.9995
        
        # طباعة التقدم كل 50 دورة
        if epoch % 50 == 0 || epoch == 1
            mean_err = total_error / 4.0
            println("   ↳ الدورة: $(lpad(epoch, 3)) | متوسط الخطأ الطوري: $(round(mean_err; digits=4)) راديان")
        end
    end
 
    println("\n✓ اكتمل التدريب. جاري اختبار التنبؤات والتحقق من التزامن...")
    println("-" ^ 65)
    
    # 3. اختبار التنبؤات النهائية
    success = true
    for (input_pair, target_val) in dataset
        x1, x2 = input_pair
        
        # اختبار طوري حر مع كسر تناظر البدء
        phi_test = (rand(Float64, N) .- 0.5) .* 0.2
        phi_test[1] = x1 * pi
        phi_test[2] = x2 * pi
        phi_test[3] = 0.0 # تثبيت المرجعية عند الصفر
        
        # في فحص الاختبار النهائي نقوم بتقليل الضوضاء للوصول إلى استقرار تام
        simulate_kuramoto!(phi_test, K, omega, clamped_neg, sim_steps, dt; noise_level=0.0)
        
        output_phase = phi_test[7]
        # التنبؤ بناءً على جيب التمام للطور (المحاذاة القريبة من 0 تعني 0، والمحاذاة القريبة من pi تعني 1)
        pred_val = cos(output_phase) > 0.0 ? 0.0 : 1.0
        
        status_symbol = pred_val == target_val ? "✓" : "❌"
        if pred_val != target_val; success = false; end
        
        println("   $status_symbol الإدخال: ($(Int(x1)), $(Int(x2))) | المستهدف: $(Int(target_val)) | طور المخرج: $(round(output_phase, digits=3)) راديان -> تنبؤ: $(Int(pred_val))")
    end
    
    println("-" ^ 65)
    if success
        println("🎉 نجاح باهر! استطاعت الشبكة الطورية PRNN حل معضلة XOR بالكامل (4/4) عبر التزامن المحلي والـ Contrastive Hebbian Learning دون استخدام أي مشتقات أو Backpropagation!")
    else
        println("⚠️ لم تتقارب الشبكة بشكل كامل لحل جميع الحالات، قد تتطلب ضبطاً طفيفاً لمعاملات الخطوة الزمنية dt أو معدل التعلم lr.")
    end
    println("╚══════════════════════════════════════════════╝")
end

end # module PRNNToy

# تشغيل فوري إذا تم استدعاء الملف مباشرة
if abspath(PROGRAM_FILE) == @__FILE__
    PRNNToy.run_prnn_xor()
end
