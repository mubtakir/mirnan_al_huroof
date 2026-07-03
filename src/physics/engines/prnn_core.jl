"""
النواة الفيزيائية للشبكة العصبية الرنينية الطورية — PRNN Core.
يحتوي على معادلات ديناميكيات ستوارت-لانداو والربط الهولوغرافي الطوري.
"""
module PRNNCore

using LinearAlgebra, Random, Statistics
using ..Constants: TOTAL_DIM

export PRNNState, LowRankPRNN, simulate_stuart_landau!, bind_phase, unbind_phase,
       complete_pattern!, prnn_noise_sample

# ═══════════════════ البنية الهيكلية لنموذج ستوارت-لانداو منخفض الرتبة ═══════════════════

mutable struct PRNNState
    z::Vector{ComplexF64}      # أطوار وسعات المذبذبات (N)
    a::Vector{Float64}         # التعب العصبي المحلي (N)
    omega::Vector{Float64}     # الترددات الذاتية لكسر التناظر (N)
    rng::MersenneTwister       # مولد الأرقام العشوائية للضوضاء
end

function PRNNState(N::Int)
    rng = MersenneTwister(42)
    # ترددات عشوائية طفيفة لمنع التزامن العشوائي المطلق
    omega = (rand(rng, Float64, N) .- 0.5) .* 0.002
    return PRNNState(zeros(ComplexF64, N), zeros(Float64, N), omega, rng)
end

function PRNNState(N::Int, seed::Int)
    rng = MersenneTwister(seed)
    omega = (rand(rng, Float64, N) .- 0.5) .* 0.002
    return PRNNState(zeros(ComplexF64, N), zeros(Float64, N), omega, rng)
end

struct LowRankPRNN
    transitions::Vector{Tuple{Vector{ComplexF64}, Vector{ComplexF64}, Float64}} # روابط سببية موزونة (v_curr, v_next, weight)
    beta::Float64 # معامل قوة الاقتران السببي
end

# ═══════════════════ محاكي الديناميكيات غير الخطية ═══════════════════

"""
    simulate_stuart_landau!(state::PRNNState, model::LowRankPRNN; 
                            mu=1.0, g_inh=0.5, gamma=2.0, tau_a=1.5, steps=40, dt=0.02,
                            noise_amp=0.0, anneal_rate=0.0)

محاكاة تطور المذبذبات تحت تأثير معادلة ستوارت-لانداو مع:
- تثبيط تنافسي عالمي
- تعب عصبي محلي (Synaptic Fatigue)
- اقتران سببي هولوغرافي
- ضوضاء لانجفان (Langevin noise) للاستكشاف والهروب من الجاذبات المحلية
- تبريد محاكى (Simulated Annealing): T(t) = noise_amp * exp(-anneal_rate * t)

noise_amp = 0.0  ← لا ضوضاء (السلوك الأصلي)
noise_amp > 0.0  ← استكشاف إبداعي
anneal_rate > 0.0  ← تهدئة تدريجية نحو الجاذب
"""
function simulate_stuart_landau!(state::PRNNState, model::LowRankPRNN; 
                                 mu::Float64=1.0, g_inh::Float64=0.5, 
                                 gamma::Float64=2.0, tau_a::Float64=1.5, 
                                 steps::Int=40, dt::Float64=0.02,
                                 noise_amp::Float64=0.0, anneal_rate::Float64=0.0)
    z = state.z
    a = state.a
    omega = state.omega
    rng = state.rng
    N = length(z)
    beta = model.beta
    transitions = model.transitions
    
    dz = zeros(ComplexF64, N)
    
    # تحضير عيّنات ويينر (Wiener) مسبقاً لتسريع الحساب
    dW_real = zeros(Float64, N)
    dW_imag = zeros(Float64, N)
    
    for step in 1:steps
        # التثبيط التنافسي العالمي
        global_activity = sum(abs2(zi) for zi in z) / N
        norm_z = sqrt(global_activity)
        
        # حساب الاقتران السببي هولوغرافياً بأسلوب منخفض الرتبة O(T * N)
        coupling = zeros(ComplexF64, N)
        active_count = 0
        for (v_curr, v_next, weight) in transitions
            # dot(v_curr, z) يحسب conj(v_curr) * z
            overlap = dot(v_curr, z)
            similarity = abs(overlap) / (N * max(0.01, norm_z))
            if similarity > 0.15  # عتبة التشابه الدلالي الطوري لتفادي الضوضاء
                coupling .+= (weight * beta / N * overlap) .* v_next
                active_count += 1
            end
        end
        
        # معايرة الاقتران لمنع تراكم الطاقة وتضخم السعة (Synaptic Scaling)
        if active_count > 1
            coupling ./= active_count
        end
        
        # ضوضاء لانجفان مع تبريد محاكى
        # T(t) = noise_amp * exp(-anneal_rate * step)
        # √(2T·dt)·dW حيث dW ~ N(0,1) معقدة
        noise_scale = 0.0
        if noise_amp > 0.0
            T_t = noise_amp * exp(-anneal_rate * step)
            noise_scale = sqrt(2.0 * T_t * dt)
        end
        
        # معادلة ستوارت-لانداو الفيزيائية مع حد الضوضاء
        if noise_scale > 0.0
            randn!(rng, dW_real)
            randn!(rng, dW_imag)
        end
        
        for i in 1:N
            dz_i = (mu - a[i] - g_inh * global_activity - abs2(z[i]) + im * omega[i]) * z[i] + coupling[i]
            if noise_scale > 0.0
                dz_i += noise_scale * ComplexF64(dW_real[i], dW_imag[i])
            end
            dz[i] = dz_i
        end
        z .+= dt .* dz
        
        # تحديث التعب العصبي المحلي da/dt = (-a + gamma * |z|^2) / tau_a
        for i in 1:N
            da = (-a[i] + gamma * abs2(z[i])) / tau_a
            a[i] += dt * da
        end
    end
end

"""
    prnn_noise_sample(state::PRNNState, model::LowRankPRNN; 
                      noise_amp=0.3, anneal_rate=0.05, steps=100, kwargs...)

توليد عيّنة من حقل PRNN بالاستكشاف الحر مع ضوضاء لانجفان.
تُعيد متجه z بعد المحاكاة.
مفيدة لاختبار التنوع والتوليد الإبداعي.
"""
function prnn_noise_sample(state::PRNNState, model::LowRankPRNN;
                           noise_amp::Float64=0.3, anneal_rate::Float64=0.05,
                           steps::Int=100, kwargs...)
    simulate_stuart_landau!(state, model; 
        noise_amp=noise_amp, anneal_rate=anneal_rate, 
        steps=steps, kwargs...)
    return copy(state.z)
end

# ═══════════════════ إكمال الأنماط عبر الجاذبات الطورية ═══════════════════

"""
    complete_pattern!(state::PRNNState, model::LowRankPRNN; 
                      steps=200, tol=1e-6, max_stable=20, kwargs...)

إكمال النمط (Pattern Completion) عبر ترك النظام يتطور نحو أقرب جاذب طوري.
- يستقبل حالة مبدئية `state.z` (قد تكون تالفة أو ناقصة)
- يُجري محاكاة ستوارت-لانداو حتى الاستقرار
- يُعيد `true` إذا استقر، و`false` إذا لم يصل للاستقرار

المعاملات:
- `steps`: الحد الأقصى لخطوات المحاكاة
- `tol`: عتبة الاستقرار (تغيّر الحالة < tol)
- `max_stable`: عدد الخطوات المتتالية المستقرة المطلوبة
- `kwargs`: تُمرّر إلى simulate_stuart_landau!
"""
function complete_pattern!(state::PRNNState, model::LowRankPRNN;
                           steps::Int=200, tol::Float64=1e-6, max_stable::Int=20,
                           mu::Float64=1.0, g_inh::Float64=0.5,
                           gamma::Float64=2.0, tau_a::Float64=1.5,
                           dt::Float64=0.02, noise_amp::Float64=0.0, anneal_rate::Float64=0.0)
    z = state.z
    N = length(z)
    stable_count = 0
    
    # مرحلة الاستكشاف الأولي (noise عالٍ نسبياً) ← الهروب من الجاذبات الخاطئة
    if noise_amp > 0.0
        simulate_stuart_landau!(state, model;
            mu=mu, g_inh=g_inh, gamma=gamma, tau_a=tau_a,
            steps=div(steps, 4), dt=dt,
            noise_amp=noise_amp, anneal_rate=anneal_rate)
    end
    
    # مرحلة الاستقرار (noise منخفض أو معدوم)
    for step in 1:steps
        z_prev = copy(z)
        
        simulate_stuart_landau!(state, model;
            mu=mu, g_inh=g_inh, gamma=gamma, tau_a=tau_a,
            steps=1, dt=dt,
            noise_amp=max(noise_amp * exp(-anneal_rate * step), 0.0), anneal_rate=0.0)
        
        # قياس التغيّر
        max_change = maximum(abs.(z .- z_prev)) / (maximum(abs.(z_prev)) + 1e-10)
        
        if max_change < tol
            stable_count += 1
            stable_count >= max_stable && return true
        else
            stable_count = 0
        end
    end
    
    return stable_count >= max_stable
end

# ═══════════════════ العمليات الهولوغرافية الطورية (VSA) ═══════════════════

"""
    bind_phase(v_a, v_b)

الربط الهولوغرافي الطوري (Element-wise multiplication)
"""
bind_phase(v_a, v_b) = v_a .* v_b

"""
    unbind_phase(v_bound, v_context)

فك الربط الهولوغرافي الطوري (Unbinding via complex conjugate)
"""
unbind_phase(v_bound, v_context) = v_bound .* conj(v_context)

end # module PRNNCore
