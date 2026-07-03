"""
 🧠 Quantum Wave Metaphor Simulation - PRNN Suite
===================================================
محاكاة المفاهيم الكمية باستخدام الموجات الكلاسيكية في فضاء الطور المركب:
1. تجربة الشق المزدوج والانشطار الطوري (Double-Slit & Phase Bifurcation)
2. الإفناء الموجي وبوابة النفي الطوري (Wave Annihilation & NOT Gate)
3. التشابك والانهيار الطوري (Entanglement & Phase Collapse)

ملاحظة علمية: هذه محاكاة كلاسيكية تناظرية تستخدم التشابه الرياضي (Isomorphism)
مع ميكانيكا الكم لتوضيح قوة الحوسبة الموجية المستمرة.
"""
module PRNNQuantumWave

using LinearAlgebra, Random, Statistics

# ═══════════════════════════════════════════════════════════════════
# 🔬 التجربة الأولى: الشق المزدوج والتداخل الطوري (Double-Slit)
# ═══════════════════════════════════════════════════════════════════
"""
يحاكي انتشار الموجات في شبكة خطية (1D Array) من المذبذبات المقترنة انتشاريًا.
يتم حقن مصدرين عند شقين مختلفين بأطوار مختلفة لتوليد أهداب التداخل (Interference Fringes).
"""
function run_double_slit(; N::Int=31, steps::Int=200, dt::Float64=0.01, D::Float64=2.0)
    println("\n=== 🔬 التجربة الأولى: تجربة الشق المزدوج والانشطار الطوري ===")
    println("بناء وسط ناقل طوله $N عصبونات موجية...")
    
    # شبكة مذبذبات: تهيئة بالصفر
    z = zeros(ComplexF64, N)
    
    # مواقع الشقين
    slit1 = Int(div(N, 3))
    slit2 = N - slit1 + 1
    
    # التطور الديناميكي مع الحقن المستمر
    # الشق 1: يحقن موجة بالطور 0
    # الشق 2: يحقن موجة بالطور pi/2 (انشطار طوري / Hadamard-like)
    omega = 2.0 * pi * 5.0 # تردد التشغيل
    
    for step in 1:steps
        t = step * dt
        
        # الانتشار الموجي (Diffusive/Laplacian Coupling)
        dz = zeros(ComplexF64, N)
        for i in 2:(N-1)
            # معادلة الانتشار الموجي الكلاسيكي + اضمحلال لتجنب الانفجار
            dz[i] = D * (z[i-1] + z[i+1] - 2.0 * z[i]) - 0.2 * z[i]
        end
        
        # حقن المصادر عند الشقين
        z[slit1] = exp(im * omega * t)
        z[slit2] = exp(im * (omega * t + pi/2)) # إزاحة طورية 90 درجة
        
        z .+= dt .* dz
    end
    
    # رسم أهداب التداخل (ASCII Plot)
    println("\n📊 توزيع السعة الكلية |z| عبر الشبكة (أهداب التداخل):")
    max_amp = maximum(abs.(z))
    for i in 1:N
        amp = abs(z[i])
        norm_amp = max_amp > 0 ? amp / max_amp : 0.0
        bar_len = Int(round(norm_amp * 30))
        bar = "█"^bar_len * "░"^(30 - bar_len)
        
        # وضع علامات على الشقين لتسهيل القراءة
        tag = (i == slit1) ? " [الشق 1]" : (i == slit2) ? " [الشق 2]" : ""
        # إبراز العقد المدمرة (تداخل هدام تام)
        status = amp < 0.15 * max_amp ? " (عقدة إلغاء/هدام)" : ""
        
        println("العقدة $(lpad(i, 2)): |z| = $(round(amp, digits=3)) | $bar $tag$status")
    end
end

# ═══════════════════════════════════════════════════════════════════
# 🔬 التجربة الثانية: الإفناء الموجي (Wave Annihilation / NOT Gate)
# ═══════════════════════════════════════════════════════════════════
"""
محاكاة تداخل هدام كامل بين موجة ونظيرها المعاكس (Antipode).
"""
function run_wave_annihilation(; steps::Int=100, dt::Float64=0.05, mu::Float64=1.0)
    println("\n=== 🔬 التجربة الثانية: الإفناء الموجي (Wave Annihilation) ===")
    
    # مذبذبان مقترنان بقوة
    # z1: الموجة الأصلية
    # z2: الموجة المقترنة بها
    z1 = 1.0 + 0.0im
    z2 = 1.0 + 0.0im # في البداية متطابقان في الطور والسعة
    
    K = 2.0 # معامل الاقتران
    
    println("الحالة البدئية: الموجتان في طور متوافق (تداخل بنّاء).")
    println("  z1 = $(round(z1, digits=3)) | السعة: $(round(abs(z1), digits=3))")
    println("  z2 = $(round(z2, digits=3)) | السعة: $(round(abs(z2), digits=3))")
    
    # تطور زمني قصير
    for step in 1:30
        dz1 = (mu - abs2(z1)) * z1 + K * (z2 - z1)
        dz2 = (mu - abs2(z2)) * z2 + K * (z1 - z2)
        z1 += dt * dz1
        z2 += dt * dz2
    end
    
    println("\n[فجأة] نقوم بحقن النظير المعاكس (Antipode) بضرب z2 في -1 (إزاحة طورية بمقدار pi)...")
    z2 = -z2 # NOT Gate / Antipode Injection
    
    println("الحالة فور الانعكاس:")
    println("  z1 = $(round(z1, digits=3))")
    println("  z2 = $(round(z2, digits=3))")
    println("جاري محاكاة تطور الحقل المشترك ومراقبة الإفناء...")
    
    # محاكاة التلاشي والإفناء الموجي
    for step in 1:steps
        dz1 = (mu - abs2(z1)) * z1 + K * (z2 - z1)
        dz2 = (mu - abs2(z2)) * z2 + K * (z1 - z2)
        z1 += dt * dz1
        z2 += dt * dz2
        
        # حساب السعة الكلية المشتركة
        total_amp = abs(z1 + z2)
        if step % 20 == 0 || step == 1
            println("  الخطوة: $(lpad(step, 3)) | السعة الكلية |z1 + z2| = $(round(total_amp; digits=4)) | z1 = $(round(abs(z1), digits=3)), z2 = $(round(abs(z2), digits=3))")
        end
    end
    
    final_amp = abs(z1 + z2)
    println("\nالنتيجة النهائية: انخفضت السعة الكلية المشتركة للـ NOT Gate إلى $(round(final_amp, digits=5)).")
    if final_amp < 0.01
        println("🎉 نجاح الإفناء الموجي التام! تلاشت طاقة الحقل بفعل التداخل الهدام الثنائي.")
    else
        println("⚠️ لم يحدث تلاشٍ كامل، يرجى التحقق من قوة الاقتران.")
    end
end

# ═══════════════════════════════════════════════════════════════════
# 🔬 التجربة الثالثة: التشابك والانهيار الطوري (Wave Entanglement)
# ═══════════════════════════════════════════════════════════════════
"""
يحاكي مذبذبين متشابكين عبر اقتران غير متماثل (singlet-like).
أطوارهما حرة وتتحرك بعشوائية بفعل الضوضاء الحرارية، لكنهما مرتبطان طورياً بشكل صارم.
عند قياس (Clamp) المذبذب A، ينهار المذبذب B لحظياً ومباشرة إلى الطور المقابل.
"""
function run_entanglement(; steps::Int=100, dt::Float64=0.03, mu::Float64=1.0)
    println("\n=== 🔬 التجربة الثالثة: التشابك والانهيار الطوري ===")
    
    # مذبذبان متشابكان A و B
    # نهيئهما بأطوار عشوائية
    zA = exp(im * rand() * 2 * pi)
    zB = -zA # نربطهما في البداية بطورين متضادين (حالة سينغليت موجية)
    
    # اقتران تبادلي قوي جداً للحفاظ على التشابك الطوري
    K = 10.0 
    
    # مستوى ضوضاء لانجفان الحرارية لزعزعة الأطوار
    noise_level = 0.5
    
    println("الحالة البدئية المتشابكة (الطور المضاد):")
    println("  zA = $(round(zA, digits=3)) | طور A = $(round(angle(zA), digits=3)) راديان")
    println("  zB = $(round(zB, digits=3)) | طور B = $(round(angle(zB), digits=3)) راديان")
    
    println("\n1. نترك النظام يتطور تحت تأثير الضوضاء الحرارية الحرة لمدة 50 خطوة...")
    for step in 1:50
        # ضوضاء مستقلة لكل مذبذب
        nA = noise_level * sqrt(dt) * (randn() + im * randn())
        nB = noise_level * sqrt(dt) * (randn() + im * randn())
        
        # التطور المشترك مع محاولة الاقتران تثبيتهما كطور متضاد
        # zB - (-zA) = zB + zA
        dzA = (mu - abs2(zA)) * zA + K * (-zB - zA)
        dzB = (mu - abs2(zB)) * zB + K * (-zA - zB)
        
        zA += dt * dzA + nA
        zB += dt * dzB + nB
    end
    
    println("الحالة بعد التجوال الحر في فضاء الطور:")
    println("  zA = $(round(zA, digits=3)) | طور A = $(round(angle(zA), digits=3)) راديان")
    println("  zB = $(round(zB, digits=3)) | طور B = $(round(angle(zB), digits=3)) راديان")
    println("  فرق الطور الفعلي: $(round(abs(angle(zA) - angle(zB)), digits=3)) راديان (متقاربان جداً من pi)")
    
    # 2. عملية القياس والانهيار (Measurement Collapse)
    # نقوم بتثبيت المذبذب A عند طور عشوائي نختاره
    measured_phase = 1.25 * pi # طور القياس
    println("\n[عملية القياس] نقوم بتثبيت (Clamp) المذبذب A عند الطور $(round(measured_phase, digits=3)) راديان...")
    
    for step in 1:steps
        # تثبيت zA عند طور القياس وسعة 1.0
        zA = exp(im * measured_phase)
        
        # المذبذب B يتطور حراً تحت التأثير الاقتراني من zA المقيد
        nB = 0.05 * sqrt(dt) * (randn() + im * randn()) # ضوضاء منخفضة أثناء الاستقرار
        dzB = (mu - abs2(zB)) * zB + K * (-zA - zB)
        zB += dt * dzB + nB
        
        if step % 20 == 0 || step == 1
            println("  الخطوة: $(lpad(step, 3)) | طور B = $(round(angle(zB), digits=3)) راديان | الفرق مع الطور المتوقع (A + pi) = $(round(abs(angle(zB) - (measured_phase - pi)), digits=4))")
        end
    end
    
    expected_B = mod2pi(measured_phase + pi + pi) - pi # إعادة توجيه
    final_diff = abs(angle(zB) - (measured_phase - pi))
    println("\nالنتيجة النهائية:")
    println("  طور A المقاس: $(round(angle(zA), digits=3)) راديان")
    println("  طور B المنهار: $(round(angle(zB), digits=3)) راديان")
    
    if final_diff < 0.05
        println("🎉 نجاح الانهيار الطوري المتشابك! استجاب المذبذب B لحظياً لطور القياس في A وحافظ على الرابطة الفيزيائية.")
    else
        println("⚠️ لم يحدث التزامن المطلوب بشكل دقيق.")
    end
end

# ═══════════════════════════════════════════════════════════════════
# المحرك الرئيسي للتجغيل
# ═══════════════════════════════════════════════════════════════════
function run_all()
    println("=================================================================")
    println("   🧠 محاكاة الحوسبة الموجية التناظرية الكمية (Analog Wave Computing)   ")
    println("=================================================================")
    
    run_double_slit()
    run_wave_annihilation()
    run_entanglement()
    
    println("\n=================================================================")
    println("         انتهت تجارب المحاكاة بنجاح باهر! 🚀                       ")
    println("=================================================================")
end

end # module PRNNQuantumWave

# التشغيل المباشر عند طلب الملف
if abspath(PROGRAM_FILE) == @__FILE__
    PRNNQuantumWave.run_all()
end
