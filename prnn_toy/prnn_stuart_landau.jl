"""
 Phase-Resonant Neural Network (PRNN) - Stuart-Landau Model
===========================================================
نموذج مطور ثنائي الأبعاد (سعة وطور) للشبكة الرنينية الطورية لحل XOR.
تعتمد الشبكة على مذبذبات ستوارت-لانداو (Stuart-Landau Oscillators) في الفضاء المركب:
  dz_i/dt = (mu - |z_i|^2 + j*omega_i)*z_i + sum( K_ij * (z_j - z_i) )

الميزات المتقدمة المدمجة:
1. ديناميكيات السعة (Amplitude Dynamics): تمثل السعة |z| مدى "الثقة" أو "الانتباه" (Dynamic Gating).
2. كسر التناظر عبر المرجعية (Bias): مذبذب مرجعي (Node 3) مثبت عند 1.0 (طور 0.0).
3. جدولة الضوضاء (Simulated Annealing): تبريد تدريجي لضوضاء لانجفان للوصول لاستقرار تام.
4. التعلم الهيبي التبايني (CHL) الموزون بالسعة: تحديث الأوزان محلياً عبر المكون الحقيقي للضرب المرافق.
"""
module PRNNStuartLandau

using LinearAlgebra, Random, Statistics

export run_prnn_stuart_landau

# ═══════════════════ محاكي ستوارت-لانداو (Stuart-Landau Simulator) ═══════════════════
function simulate_stuart_landau!(z::Vector{ComplexF64}, K::Matrix{Float64}, omega::Vector{Float64},
                                 mu::Float64, clamped_nodes::Vector{Int}, steps::Int, dt::Float64, noise_level::Float64)
    N = length(z)
    dz = zeros(ComplexF64, N)
    
    for step in 1:steps
        for i in 1:N
            if i in clamped_nodes
                dz[i] = 0.0 + 0.0im # الخلايا المغلقة لا تتغير حالتها
                continue
            end
            
            # اقتران انتشاري في الفضاء المركب (Diffusive Coupling)
            coupling = 0.0 + 0.0im
            for j in 1:N
                coupling += K[i, j] * (z[j] - z[i])
            end
            
            # معادلة ستوارت-لانداو
            dz[i] = (mu - abs2(z[i]) + im * omega[i]) * z[i] + coupling
        end
        
        # ضوضاء لانجفان الحرارية في الفضاء المركب
        noise = noise_level * sqrt(dt) .* (randn(ComplexF64, N))
        for node in clamped_nodes
            noise[node] = 0.0 + 0.0im
        end
        z .+= dt .* dz .+ noise
    end
    return z
end

# ═══════════════════ دالة التدريب والتشغيل ═══════════════════
function run_prnn_stuart_landau(; epochs::Int=300, lr::Float64=0.1, dt::Float64=0.03, sim_steps::Int=500, init_scale::Float64=1.0, mu::Float64=1.0)
    println("╔══════════════════════════════════════════════════════════╗")
    println("║  PRNN Stuart-Landau Model — حل XOR بالرنين والسعة المركبة  ║")
    println("╚══════════════════════════════════════════════════════════╝")
    println()

    # 1. إعدادات الشبكة
    N = 7 # 1,2: مداخل | 3: مرجع طوري (Bias) | 4,5,6: خفية | 7: مخرج
    
    Random.seed!(42)
    
    # ترددات ذاتية صغيرة جداً لكسر التناظر الهيكلي
    omega = (rand(Float64, N) .- 0.5) .* 0.05
    omega[3] = 0.0 # المرجعية ليس لها تردد ذاتي نشط
    
    # تهيئة مصفوفة الاقتران التناظرية K
    K = (rand(N, N) .- 0.5) .* init_scale
    for i in 1:N
        K[i, i] = 0.0
        for j in (i+1):N
            K[j, i] = K[i, j]
        end
    end
    
    # فرض طوبولوجيا الطبقات (عزل المدخلات عن المخرجات وعن بعضها)
    K[1, 7] = K[7, 1] = 0.0
    K[2, 7] = K[7, 2] = 0.0
    K[1, 2] = K[2, 1] = 0.0

    # بيانات XOR
    dataset = [
        (0.0, 0.0) => 0.0,
        (1.0, 0.0) => 1.0,
        (0.0, 1.0) => 1.0,
        (1.0, 1.0) => 0.0
    ]

    clamped_pos = [1, 2, 3, 7]
    clamped_neg = [1, 2, 3]

    initial_noise = 0.015

    println("⏳ جاري تدريب مذبذبات ستوارت-لانداو مع جدولة الضوضاء والتبريد...")
    
    for epoch in 1:epochs
        # جدولة الضوضاء (Simulated Annealing): تبريد تدريجي
        noise_level = initial_noise * (1.0 - (epoch / epochs)^2)
        
        total_error = 0.0
        
        for (input_pair, target_val) in dataset
            x1, x2 = input_pair
            
            # تهيئة الحالة البدئية (أطوار عشوائية خفيفة وسعات قريبة من 1.0)
            phi_rand = (rand(Float64, N) .- 0.5) .* 0.2
            z_init = exp.(im .* phi_rand)
            
            # --- المرحلة الموجبة (Positive Phase - Clamped Target) ---
            z_pos = copy(z_init)
            z_pos[1] = exp(im * x1 * pi)
            z_pos[2] = exp(im * x2 * pi)
            z_pos[3] = 1.0 + 0.0im # المرجع مثبت عند الطور 0
            z_pos[7] = exp(im * target_val * pi) # المخرج مثبت عند الهدف
            
            simulate_stuart_landau!(z_pos, K, omega, mu, clamped_pos, sim_steps, dt, noise_level)
            
            # --- المرحلة السالبة (Negative Phase - Free Target) ---
            # تبدأ من نفس الحالة البدئية لتجنب تجميد المخرج
            z_neg = copy(z_init)
            z_neg[1] = exp(im * x1 * pi)
            z_neg[2] = exp(im * x2 * pi)
            z_neg[3] = 1.0 + 0.0im
            
            simulate_stuart_landau!(z_neg, K, omega, mu, clamped_neg, sim_steps, dt, noise_level)
            
            # قياس خطأ المخرج الطوري
            pred_phase = angle(z_neg[7])
            target_phase = target_val * pi
            diff = atan(sin(pred_phase - target_phase), cos(pred_phase - target_phase))
            total_error += abs(diff)
            
            # --- تحديث الأوزان محلياً (Contrastive Hebbian Learning) ---
            for i in 1:N
                for j in (i+1):N
                    if (i == 1 && j == 7) || (i == 7 && j == 1) ||
                       (i == 2 && j == 7) || (i == 7 && j == 2) ||
                       (i == 1 && j == 2) || (i == 2 && j == 1)
                        continue
                    end
                    
                    # الاقتران الموزون بالسعة: Re(z_i * conj(z_j)) = r_i * r_j * cos(phi_i - phi_j)
                    # هذا يقيد التعلم تلقائياً بناءً على سعة التنشيط (الانتباه)
                    pos_coherence = real(z_pos[i] * conj(z_pos[j]))
                    neg_coherence = real(z_neg[i] * conj(z_neg[j]))
                    
                    dk = lr * (pos_coherence - neg_coherence)
                    K[i, j] += dk
                    K[j, i] = K[i, j]
                end
            end
        end
        
        # اضمحلال الأوزان العام وتثبيت القيم لمنع التضخم اللانهائي
        K .*= 0.9995
        
        if epoch % 50 == 0 || epoch == 1
            mean_err = total_error / 4.0
            println("   ↳ الدورة: $(lpad(epoch, 3)) | الخطأ الطوري: $(round(mean_err; digits=4)) راديان | مستوى الضوضاء: $(round(noise_level; digits=4))")
        end
    end

    println("\n✓ اكتمل التدريب. جاري اختبار التنبؤات والتحقق من السعة والطور...")
    println("-" ^ 85)
    
    # 3. اختبار التنبؤات النهائية
    success = true
    for (input_pair, target_val) in dataset
        x1, x2 = input_pair
        
        # اختبار طوري حر كامل بدون ضوضاء
        z_test = [exp(im * (rand() - 0.5) * 0.2) for _ in 1:N]
        z_test[1] = exp(im * x1 * pi)
        z_test[2] = exp(im * x2 * pi)
        z_test[3] = 1.0 + 0.0im
        
        simulate_stuart_landau!(z_test, K, omega, mu, clamped_neg, sim_steps, dt, 0.0)
        
        out_phase = angle(z_test[7])
        out_amp = abs(z_test[7])
        pred_val = cos(out_phase) > 0.0 ? 0.0 : 1.0
        
        status_symbol = pred_val == target_val ? "✓" : "❌"
        if pred_val != target_val; success = false; end
        
        # طباعة تفصيلية للطور والسعة (تمثيل الثقة في اتخاذ القرار)
        println("   $status_symbol الإدخال: ($(Int(x1)), $(Int(x2))) | المستهدف: $(Int(target_val)) | طور المخرج: $(round(out_phase, digits=3)) راديان | السعة: $(round(out_amp, digits=3)) -> تنبؤ: $(Int(pred_val))")
    end
    
    println("-" ^ 85)
    if success
        println("🎉 نجاح باهر! تقاربت مذبذبات ستوارت-لانداو بنسبة 100% وحلت بوابة XOR باستخدام ديناميكيات السعة والطور المشتركة!")
    else
        println("⚠️ لم تتقارب الشبكة بالكامل، قد تتطلب المعاملات ضبطاً أدق.")
    end
    println("╚══════════════════════════════════════════════════════════╝")
end

end # module PRNNStuartLandau

if abspath(PROGRAM_FILE) == @__FILE__
    PRNNStuartLandau.run_prnn_stuart_landau()
end
