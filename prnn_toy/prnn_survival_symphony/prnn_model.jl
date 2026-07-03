"""
🧠 Phase-Resonant Neural Network (PRNN) Model - Survival Symphony Edition
=======================================================================
هذا الملف يحتوي على التعريف الرياضي والديناميكي للشبكة الرنينية الطورية المكونة من
10 مذبذبات ستوارت-لانداو (Stuart-Landau) في الفضاء المركب z = r * exp(j*phi).

يتم تدريب الشبكة عبر قانون التعلم التبايني الهيبي الموزون بالسعات المركبة.
"""
module PRNNModel

using LinearAlgebra, Random, Statistics

export PRNN, simulate_stuart_landau!, update_contrastive!, update_reward_hebbian!, save_state, restore_state!, is_connected

mutable struct PRNN
    N::Int                  # عدد المذبذبات (10)
    K::Matrix{Float64}      # مصفوفة أوزان الاقتران التناظرية
    omega::Vector{Float64}  # الترددات الطبيعية الذاتية للمذبذبات
    z::Vector{ComplexF64}   # الحالة المركبة الحالية لكل مذبذب
    a::Vector{Float64}      # متغير التعب العصبي (Fatigue/Adaptation) لكل مذبذب
    mu::Float64             # حد النمو الطبيعي للسعة (Stuart-Landau mu)
    g_inh::Float64          # التثبيط التنافسي العالمي
    gamma::Float64          # قوة التعب العصبي
    tau_a::Float64          # ثابت زمن التعب العصبي
    lr::Float64             # معدل التعلم (Learning Rate)
    learning_mode::Symbol   # نمط التعلم: :contrastive أو :hebbian
end

"""
تحقق من وجود اتصال هيكلي بين عقدتين (i, j) بناءً على طوبولوجيا اللعبة:
- 1، 2، 3: مستشعرات الإدخال (خطر، فرق تردد 1، فرق تردد 2)
- 4: مذبذب التحيز المرجعي (Bias)
- 5: التردد الحالي للاعب (Player Freq input)
- 6، 7، 8: الطبقة الخفية (Hidden Layer)
- 9: مخرج التحكم بالتردد (Freq Output)
- 10: مخرج التحكم بالدرع (Shield Output)
"""
function is_connected(i::Int, j::Int)
    i == j && return false
    
    # تصنيف المجموعات
    is_sensor(k) = (k >= 1 && k <= 3)
    is_bias(k) = (k == 4)
    is_player_in(k) = (k == 5)
    is_hidden(k) = (k >= 6 && k <= 8)
    is_output(k) = (k == 9 || k == 10)
    
    # 1. منع الاتصال المباشر بين المداخل والمخارج
    if (is_sensor(i) || is_player_in(i)) && is_output(j); return false; end
    if (is_sensor(j) || is_player_in(j)) && is_output(i); return false; end
    
    # 2. منع الاتصال المباشر بين المداخل نفسها (مستشعرات ومدخل اللاعب)
    if (is_sensor(i) || is_player_in(i)) && (is_sensor(j) || is_player_in(j)); return false; end
    
    # 3. عصبونة التحيز تتصل بالطبقة الخفية والمخارج فقط
    if is_bias(i) && !(is_hidden(j) || is_output(j)); return false; end
    if is_bias(j) && !(is_hidden(i) || is_output(i)); return false; end
    
    # بقية الاتصالات (خفية-خفية، خفية-مدخل، خفية-مخرج، خفية-تحيز، تحيز-مخرج) مسموحة
    return true
end

"""
إنشاء وتهيئة كائن شبكة PRNN بالمعاملات الافتراضية.
"""
function PRNN(; lr::Float64=0.08, mu::Float64=1.0, g_inh::Float64=0.4, gamma::Float64=2.0, tau_a::Float64=1.5)
    N = 10
    
    # تهيئة الترددات الذاتية العشوائية الصغيرة لكسر التناظر الهيكلي
    omega = (rand(Float64, N) .- 0.5) .* 0.05
    omega[4] = 0.0 # عصبونة التحيز ثابتة التردد
    
    # تهيئة مصفوفة الأوزان عشوائياً وبشكل متناظر
    K = (rand(Float64, N, N) .- 0.5) .* 0.5
    for i in 1:N
        K[i, i] = 0.0
        for j in (i+1):N
            if is_connected(i, j)
                K[j, i] = K[i, j]
            else
                K[i, j] = K[j, i] = 0.0
            end
        end
    end
    
    # تهيئة الحالات
    z = [exp(im * (rand() * 2pi)) for _ in 1:N]
    z[4] = 1.0 + 0.0im # التحيز يبدأ وينتهي عند طور 0.0 (سعة 1.0)
    
    a = zeros(Float64, N)
    
    return PRNN(N, K, omega, z, a, mu, g_inh, gamma, tau_a, lr, :contrastive)
end

"""
محاكاة خطوة زمنية (أو عدة خطوات) لتطور معادلات Stuart-Landau التفاعلية للشبكة.
"""
function simulate_stuart_landau!(nn::PRNN, clamped_nodes::Vector{Int}, clamped_phases::Dict{Int, Float64}, 
                                 steps::Int, dt::Float64; noise_level::Float64=0.01)
    N = nn.N
    dz = zeros(ComplexF64, N)
    
    for step in 1:steps
        # حساب التثبيط التنافسي بناء على النشاط الكلي للمذبذبات الحرة
        free_count = count(i -> !(i in clamped_nodes), 1:N)
        global_activity = free_count > 0 ? sum(abs2, nn.z) / N : 0.0
        
        # فرض قيم الأطوار المقفلة
        for i in clamped_nodes
            if haskey(clamped_phases, i)
                amp = 1.0
                nn.z[i] = amp * exp(im * clamped_phases[i])
            elseif i == 4
                nn.z[4] = 1.0 + 0.0im # تثبيت التحيز بشكل آمن
            end
        end
        
        # حساب المشتقات
        for i in 1:N
            if i in clamped_nodes
                dz[i] = 0.0 + 0.0im
                continue
            end
            
            # اقتران انتشاري
            coupling = 0.0 + 0.0im
            for j in 1:N
                if nn.K[i, j] != 0.0
                    coupling += nn.K[i, j] * (nn.z[j] - nn.z[i])
                end
            end
            
            # معادلة ستوارت-لانداو مع التعب المحلي والتثبيط العالمي
            dz[i] = (nn.mu - nn.a[i] - nn.g_inh * global_activity - abs2(nn.z[i]) + im * nn.omega[i]) * nn.z[i] + coupling
        end
        
        # ضوضاء لانجفان الحرارية للمذبذبات الحرة فقط
        free_indices = [i for i in 1:N if !(i in clamped_nodes)]
        if !isempty(free_indices)
            noise = zeros(ComplexF64, N)
            free_noise = noise_level * sqrt(dt) .* (randn(ComplexF64, length(free_indices)))
            for (k, idx) in enumerate(free_indices)
                noise[idx] = free_noise[k]
            end
            nn.z .+= dt .* dz .+ noise
        else
            nn.z .+= dt .* dz
        end
        
        # حماية من قيم NaN و Inf المباغتة
        for i in 1:N
            if isnan(nn.z[i]) || isinf(nn.z[i])
                nn.z[i] = exp(im * rand() * 2*pi)
            end
        end
        
        # تحديث التعب العصبي da/dt = (-a + gamma * |z|^2) / tau_a
        for i in 1:N
            if !(i in clamped_nodes)
                da = (-nn.a[i] + nn.gamma * abs2(nn.z[i])) / nn.tau_a
                nn.a[i] += dt * da
            else
                nn.a[i] = 0.0 # المداخل لا يصيبها التعب العصبي
            end
        end
    end
    return nn.z
end

"""
تحديث مصفوفة الأوزان باستخدام قاعدة التعلم التبايني الهيبي الموزون بالسعات المركبة.
"""
function update_contrastive!(nn::PRNN, z_pos::Vector{ComplexF64}, z_neg::Vector{ComplexF64})
    N = nn.N
    for i in 1:N
        for j in (i+1):N
            if is_connected(i, j)
                # ...
                pos_coherence = real(z_pos[i] * conj(z_pos[j]))
                neg_coherence = real(z_neg[i] * conj(z_neg[j]))
                
                dk = nn.lr * (pos_coherence - neg_coherence)
                # تقييد الأوزان لمنع الانفجار الرياضي (Weight Clamping)
                nn.K[i, j] = clamp(nn.K[i, j] + dk, -2.0, 2.0)
                nn.K[j, i] = nn.K[i, j]
            end
        end
    end
    
    # تنظيم الأوزان لمنع الانتفاخ اللانهائي (Weight Decay)
    nn.K .*= 0.9995
end

"""
تحديث مصفوفة الأوزان باستخدام قاعدة هيبي المعززة بالمكافأة.
ΔK_ij = η · R · Re(z_i · conj(z_j))
"""
function update_reward_hebbian!(nn::PRNN, reward::Float64)
    N = nn.N
    for i in 1:N
        for j in (i+1):N
            if is_connected(i, j)
                coh = real(nn.z[i] * conj(nn.z[j]))
                dk = nn.lr * reward * coh
                nn.K[i, j] = clamp(nn.K[i, j] + dk, -2.0, 2.0)
                nn.K[j, i] = nn.K[i, j]
            end
        end
    end
    nn.K .*= 0.9995
end

"""
حفظ نسخة من الحالة الديناميكية الحالية للشبكة.
"""
function save_state(nn::PRNN)
    return (copy(nn.z), copy(nn.a))
end

"""
استعادة الحالة الديناميكية المحفوظة للشبكة.
"""
function restore_state!(nn::PRNN, state::Tuple{Vector{ComplexF64}, Vector{Float64}})
    nn.z .= state[1]
    nn.a .= state[2]
end

end # module PRNNModel
