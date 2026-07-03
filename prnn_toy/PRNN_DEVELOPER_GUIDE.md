# 🧠 دليل المطور الشامل للشبكات العصبية الرنينية الطورية (PRNN)
### **الأسس الفلسفية، النظريات الفيزيائية، والتطبيق البرمجي بلغة Julia**

**مشروع مِرنان (Mirnan) | دليل المطورين والباحثين لبناء تطبيقات الذكاء الاصطناعي الموجي المستمر**

---

## 🏛️ القسم 1: الأسس الفلسفية (الموجة ضد الرقم)

لقد وصل الذكاء الاصطناعي الرقمي السائد (مثل الـ Transformers والشبكات العميقة التقليدية) إلى جدار هندسي متمثل في:
1. **استهلاك الطاقة الفائق:** معالجة تريليونات معاملات الضرب الرقمية الجافة.
2. **غياب التفسيرية:** معاملات الأوزان عبارة عن أرقام إحصائية مجردة لا تعبر عن معانٍ فيزيائية.
3. **تكلفة التعلم:** خوارزمية الانتشار العكسي (Backpropagation) تتطلب حساب مشتقات الشبكة بأكملها بشكل تسلسلي مما يعطل الحوسبة الموزعة محلياً.

### فلسفة البارادايم الموجي (Continuous Physical Computing):
الشبكة العصبية الرنينية الطورية (PRNN) تقدم بديلاً مستوحى من ميكانيكا الموجات والفيزياء غير الخطية وتزامن المذبذبات (مبدأ Huygens للتعاضد الذاتي):
* **العقل ليس حاسبة رقمية، بل وسط رنيني:** المعلومات في الدماغ لا تُخزن كبايتات رقمية، بل كحالات اهتزازية متزامنة.
* **المفهوم اللغوي كطور موجي:** في PRNN، يتم التعبير عن المعاني بالأطوار ($\phi$). الفكرة الإيجابية ("نعم") هي طور $0.0$، والفكرة السلبية المعاكسة ("لا") هي طور $\pi$.
* **الاعتقاد كطاقة سعة:** سعة المذبذب ($r = |z|$) تمثل مدى اليقين أو الانتباه الموجه للمفهوم.
* **التعلم كقناة اقتران محلي:** الأوزان $K_{ij}$ هي موصلات رنين فيزيائية تتعدل محلياً بناءً على تقارب أو تباعد نبضات العصبونات المتصلة مباشرة، دون الحاجة لمعرفة حالة الشبكة البعيدة.

---

## 🧬 القسم 2: الأسس العلمية والرياضيات الحاكمة

تعتمد الحوسبة الرنينية على دمج معادلتين فيزيائيتين شهيرتين:

### 1. مذبذبات ستوارت-لانداو (Stuart-Landau Dynamics)
يوصف كل عصبون $i$ كمتغير مركب $z_i(t) = r_i(t) e^{j\phi_i(t)}$ يتطور في الزمن وفق المعادلة:
$$ \frac{dz_i}{dt} = (\mu - |z_i|^2 + j\omega_i) z_i + \sum_j K_{ij} (z_j - z_i) $$

* **$\mu - |z_i|^2$**: يمثل ديناميكا السعة. إذا كانت السعة صغيرة، تنمو لتصل إلى $\sqrt{\mu}$ (التشبع). وإذا واجهت تداخلاً هداماً حاداً، تنضغط السعة لتصل إلى الصفر (**Dynamic Gating**).
* **$j\omega_i z_i$**: يمثل التردد الطبيعي الذاتي للعصبون.
* **$\sum K_{ij}(z_j - z_i)$**: الاقتران الانتشاري المركب الذي يسحب الخلايا للتزامن.

### 2. التعلم الهيبي التبايني (Contrastive Hebbian Learning - CHL)
يتم التعلم عبر مقارنة حالتين ديناميكيتين للشبكة:
1. **المرحلة الموجبة (Positive Phase):** نقيد المخرجات والمدخلات عند الأطوار المستهدفة الصحيحة. تستقر الشبكة ونسجل المتجهات $z^+$.
2. **المرحلة السالبة (Negative Phase):** نترك المخرجات حرة لتتحرك بناءً على التنبؤ الداخلي للشبكة. تستقر الشبكة ونسجل المتجهات $z^-$.
3. **قاعدة التحديث الهيبية الموزونة بالسعة:**
   $$ \Delta K_{ij} = \eta \left( \text{Re}(z_i^+ z_j^{+*}) - \text{Re}(z_i^- z_j^{-*}) \right) = \eta \left( r_i^+ r_j^+ \cos(\phi_i^+ - \phi_j^+) - r_i^- r_j^- \cos(\phi_i^- - \phi_j^-) \right) $$

---

## 💻 القسم 3: القوالب البرمجية الأساسية (Code Patterns)

إليك كيف نكتب الكود لكل فكرة رياضية بلغة Julia للمطورين للبدء في مشاريعهم:

### 1️⃣ قالب محاكي المذبذبات (Stuart-Landau Simulator)
هذا هو المحرك الأساسي الذي يدمج المعادلات التفاضلية للشبكة خطوة بخطوة:

```julia
using LinearAlgebra

# تعريف هيكل البيانات الأساسي للشبكة
mutable struct PRNN
    N::Int
    K::Matrix{Float64}      # مصفوفة أوزان الاقتران (متناظرة)
    omega::Vector{Float64}  # الترددات الذاتية
    z::Vector{ComplexF64}   # الحالات المركبة (طور وسعة)
    a::Vector{Float64}      # التعب العصبي (Fatigue)
    mu::Float64             # حد التشبع للسعة
    g_inh::Float64          # التثبيط العالمي
end

# محاكاة خطوة تطور المذبذبات
function step_simulation!(nn::PRNN, clamped_nodes::Vector{Int}, clamped_phases::Dict{Int, Float64}, dt::Float64)
    N = nn.N
    dz = zeros(ComplexF64, N)
    
    # حساب النشاط الكلي للمذبذبات الحرة لغرض التثبيط التنافسي
    global_activity = sum(abs2(zi) for (i, zi) in enumerate(nn.z) if !(i in clamped_nodes)) / N
    
    # فرض قيود الأطوار المقفلة (Clamping)
    for i in clamped_nodes
        if haskey(clamped_phases, i)
            nn.z[i] = 1.0 * exp(im * clamped_phases[i])
        end
    end
    
    # حساب ديناميكيات Stuart-Landau لكل عصبون حر
    for i in 1:N
        if i in clamped_nodes
            continue
        end
        
        # الاقتران الانتشاري بين العصبونات المقترنة
        coupling = 0.0 + 0.0im
        for j in 1:N
            if nn.K[i, j] != 0.0
                coupling += nn.K[i, j] * (nn.z[j] - nn.z[i])
            end
        end
        
        # معادلة Stuart-Landau مع التعب العصبي والتثبيط العالمي
        dz[i] = (nn.mu - nn.a[i] - nn.g_inh * global_activity - abs2(nn.z[i]) + im * nn.omega[i]) * nn.z[i] + coupling
    end
    
    # تحديث الحالات بالزمن
    nn.z .+= dt .* dz
    
    # تحديث التعب العصبي (Synaptic Adaptation) للحد من النشاط المستمر
    # da/dt = (-a + gamma * |z|^2) / tau_a
    gamma, tau_a = 2.0, 1.5
    for i in 1:N
        if !(i in clamped_nodes)
            da = (-nn.a[i] + gamma * abs2(nn.z[i])) / tau_a
            nn.a[i] += dt * da
        end
    end
end
```

### 2️⃣ قالب التعلم التبايني وتقييد الأوزان (Contrastive Learning & Clamping)
لتدريب الشبكة وتحديث الأوزان بشكل آمن يمنع الانفجارات الرياضية:

```julia
function train_step!(nn::PRNN, clamped_nodes::Vector{Int}, clamped_phases::Dict{Int, Float64}, 
                     target_nodes::Vector{Int}, target_phases::Dict{Int, Float64}, lr::Float64)
    
    # 1. المرحلة السالبة (Free Outputs): قفل المدخلات فقط وتطوير الشبكة
    neg_state_z = copy(nn.z)
    neg_state_a = copy(nn.a)
    
    for _ in 1:200
        step_simulation!(nn, clamped_nodes, clamped_phases, 0.02)
    end
    z_neg = copy(nn.z)
    
    # 2. المرحلة الموجبة (Clamped Target): قفل المدخلات والمخرجات وتطوير الشبكة
    # استرجاع الحالة البدئية لضمان تطور متسق
    nn.z .= neg_state_z
    nn.a .= neg_state_a
    
    pos_clamped_nodes = [clamped_nodes..., target_nodes...]
    pos_clamped_phases = merge(clamped_phases, target_phases)
    
    for _ in 1:200
        step_simulation!(nn, pos_clamped_nodes, pos_clamped_phases, 0.02)
    end
    z_pos = copy(nn.z)
    
    # 3. تحديث الأوزان هيبياً تباينياً مع تقييد الأوزان (Weight Clamping)
    for i in 1:nn.N
        for j in (i+1):nn.N
            # نحدث الأوزان فقط للعقد المتصلة هيكلياً
            if is_connected(i, j)
                pos_coherence = real(z_pos[i] * conj(z_pos[j]))
                neg_coherence = real(z_neg[i] * conj(z_neg[j]))
                
                dk = lr * (pos_coherence - neg_coherence)
                
                # تقييد الأوزان بين -3.0 و 3.0 لمنع تفجر الطاقة
                nn.K[i, j] = clamp(nn.K[i, j] + dk, -3.0, 3.0)
                nn.K[j, i] = nn.K[i, j]
            end
        end
    end
    
    # اضمحلال الأوزان الدوري
    nn.K .*= 0.9995
end
```

### 3️⃣ قالب الربط والفك الهولوغرافي لمعالجة اللغات (Holographic Phase Binding - NLP)
الربط والفك في فضاءات الطور المركب بأبعاد فائقة لمنع تسرب الاقتران السببي:

```julia
# تمثيل الكلمات كمتجهات طورية مركبة فائقة الأبعاد (N = 10,000)
function generate_phase_vector(N::Int)
    phases = rand(Float64, N) .* 2 * pi
    return exp.(im .* phases) # متجهات على دائرة الوحدة بسعة 1.0
end

# عملية الربط الهولوغرافي (Element-wise Product): تمثل ضرب الأطوار
function holographic_bind(v_word::Vector{ComplexF64}, v_context::Vector{ComplexF64})
    return v_word .* v_context
end

# عملية فك الربط (Element-wise conjugate multiplication): تمثل طرح الأطوار لاستعادة المفهوم
function holographic_unbind(v_bound::Vector{ComplexF64}, v_context::Vector{ComplexF64})
    return v_bound .* conj(v_context)
end

# فك تشفير الكلمة الفائزة بمقارنة التداخل الرنيني
function decode_word(z_state::Vector{ComplexF64}, v_context::Vector{ComplexF64}, vocab::Dict{String, Vector{ComplexF64}})
    # فك تشفير الحالة الحالية نسبة للسياق
    z_unbound = holographic_unbind(z_state, v_context)
    
    best_word = ""
    best_overlap = -Inf
    N = length(z_state)
    
    for (word, vec) in vocab
        # حساب التداخل الرنيني (الضرب النقطي الحقيقي)
        overlap = real(dot(z_unbound, vec)) / N
        if overlap > best_overlap
            best_overlap = overlap
            best_word = word
        end
    end
    return best_word, best_overlap
end
```

### 4️⃣ قالب طيار آلي تفاعلي (Hebbian RL & Danger-Triggered CHL)
كيفية بناء عميل ذكي (Agent) يتعلم القيادة واتخاذ القرارات في الوقت الحقيقي:

```julia
# حلقة تحديث العميل الذكي في إطار اللعبة
function update_agent!(agent, closest_threat, dt)
    # 1. صياغة المدخلات الطورية بناء على المستشعرات
    clamped_nodes = [1, 2, 3, 4, 5]
    clamped_phases = Dict{Int, Float64}()
    
    clamped_phases[4] = 0.0 # عصبونة التحيز مثبتة عند طور صفر
    clamped_phases[5] = agent.current_freq * pi
    
    if closest_threat != nothing
        clamped_phases[1] = (1.0 - closest_threat.distance) * pi # خطر المسافة
        clamped_phases[2] = abs(agent.current_freq - closest_threat.freq) * pi # فرق التردد
    else
        clamped_phases[1] = 0.0
        clamped_phases[2] = 0.0
    end
    clamped_phases[3] = 0.0
    
    # 2. اتخاذ القرار (Inference)
    # ندع عصبونات المخرجات 9 و 10 حرة لتتحرك بناءً على رنين الشبكة
    simulate_stuart_landau!(agent.nn, clamped_nodes, clamped_phases, 40, 0.02)
    
    # قراءة تردد المخرج من طور العقدة 9
    pred_phase = angle(agent.nn.z[9])
    agent.current_freq = clamp(abs(pred_phase) / pi, 0.1, 1.0)
    
    # قراءة قرار الدرع من سعة العقدة 10
    shield_amp = abs(agent.nn.z[10])
    agent.shield_active = (shield_amp > 0.8)
    
    # 3. تدريب الشبكة تحت الخطر (Danger-Triggered CHL)
    if closest_threat != nothing && closest_threat.distance < 0.5
        # إذا اقترب الخطر، نقوم بالتدريب التبايني لتصحيح المسار
        target_freq = closest_threat.freq
        target_shield = pi # تفعيل الدرع
        
        target_nodes = [9, 10]
        target_phases = Dict(9 => target_freq * pi, 10 => target_shield)
        
        # تشغيل خطوة التدريب التبايني الهيبي
        train_step!(agent.nn, clamped_nodes, clamped_phases, target_nodes, target_phases, agent.nn.lr)
    end
end
```

---

## 📂 القسم 4: معمارية ملفات المشروع ووظائفها

للبدء في التعديل على مشاريع PRNN الحالية، إليك خريطة الملفات المتوفرة في مجلد `prnn_toy`:

1. **[prnn_toy.jl](file:///C:/Users/allmy/Desktop/aaa/basil/majnon/models/mirnan/prnn_toy/prnn_toy.jl):**
   * **الوظيفة:** حل بوابة XOR المنطقية بنموذج كوراموتو الطوري (أطوار فقط).
   * **الأهمية:** أول إثبات عملي لكسر التناظر الطوري باستخدام عصبونة التحيز المرجعي.
2. **[prnn_stuart_landau.jl](file:///C:/Users/allmy/Desktop/aaa/basil/majnon/models/mirnan/prnn_toy/prnn_stuart_landau.jl):**
   * **الوظيفة:** حل بوابة XOR بنموذج ستوارت-لانداو المركب (سعة وطور).
   * **الأهمية:** إدخال ديناميكية السعة وجدولة اضمحلال الضوضاء (Simulated Annealing) للتغلب على النهايات الصغرى.
3. **[prnn_associative_memory.jl](file:///C:/Users/allmy/Desktop/aaa/basil/majnon/models/mirnan/prnn_toy/prnn_associative_memory.jl):**
   * **الوظيفة:** ذاكرة ترابطية وإكمال الأنماط التالفة ومفقودة البيانات.
   * **الأهمية:** يثبت تشكل أحواض جذب طورية وسعة مركبة تعيد بناء الحالات دون قمع إشارة.
4. **[prnn_sequential_memory.jl](file:///C:/Users/allmy/Desktop/aaa/basil/majnon/models/mirnan/prnn_toy/prnn_sequential_memory.jl):**
   * **الوظيفة:** توليد أنماط متسلسلة ودورات حدية (Limit Cycles).
   * **الأهمية:** تطبيق عملي لكسر تناظر الزمن والتعب العصبي المشبكي لتنقل النشاط اللحظي تلقائياً.
5. **[prnn_text_generator.jl](file:///C:/Users/allmy/Desktop/aaa/basil/majnon/models/mirnan/prnn_toy/prnn_text_generator.jl):**
   * **الوظيفة:** مولد نصوص ذكي يحل معضلة التفرع في الكلمات المشتركة.
   * **الأهمية:** تطبيق الربط الهولوغرافي الطوري السياقي (Holographic Phase Binding) بأبعاد متوسطة.
6. **[prnn_corpus_learner.jl](file:///C:/Users/allmy/Desktop/aaa/basil/majnon/models/mirnan/prnn_toy/prnn_corpus_learner.jl):**
   * **الوظيفة:** تدريب الشبكة هولوغرافياً على 100 جملة لغوية بأبعاد فائقة (10,000 بُعد).
   * **الأهمية:** تسريع الحساب من رتبة $O(N^2)$ إلى $O(T \times N)$ عبر التمثيل السببي منخفض الرتبة.
7. **[prnn_quantum_wave.jl](file:///C:/Users/allmy/Desktop/aaa/basil/majnon/models/mirnan/prnn_toy/prnn_quantum_wave.jl):**
   * **الوظيفة:** محاكاة تجارب كمية (الشق المزدوج، الإفناء، التشابك) بوسط موجي كلاسيكي.
   * **الأهمية:** يوضح التشابه الهيكلي بين ميكانيكا الكم وديناميكا الحوسبة الطورية المركبة.
8. **[prnn_xor_7osc.cir](file:///C:/Users/allmy/Desktop/aaa/basil/majnon/models/mirnan/prnn_toy/prnn_xor_7osc.cir):**
   * **الوظيفة:** تصميم دوائر تناظرية صلبة لـ XOR باستخدام مذبذبات Van der Pol ومصادر تيار G-elements.
   * **الأهمية:** المخطط الهندسي لصناعة رقاقات سيليكونية فيزيائية تعمل بالرنين بدون معالجات رقمية.
9. **[prnn_survival_symphony/](file:///C:/Users/allmy/Desktop/aaa/basil/majnon/models/mirnan/prnn_toy/prnn_survival_symphony):**
   * **الوظيفة:** لعبة سيمفونية النجاة التفاعلية في سطر الأوامر.
   * **الأهمية:** تطبيق واقعي مغلق الحلقة (Closed-Loop Control) يربط التحكم اللحظي بالتعلم المعزز والتوجيه الذاتي للشبكة.

---

## 🎯 القسم 5: دليل الضبط والمعايرة للمطورين (Hyperparameters Tuning)

عند بناء شبكتك الخاصة باستخدام PRNN، فإن نجاح التزامن واستقرار النظام يعتمد على دقة المعايرة البرمجية:

| المعامل | الوظيفة | المدى النموذجي | الملاحظات والنصائح |
| :--- | :--- | :--- | :--- |
| **خطوة الزمن `dt`** | دقة التكامل الرياضي للمحاكي | `0.01` إلى `0.04` | أي قيمة أعلى من `0.05` مع أوزان كبيرة ستؤدي لانهيار النظام وقيم `NaN`. |
| **معدل التعلم `lr`** | سرعة تعديل الأوزان المشبكية | `0.05` إلى `0.15` | القيمة المرتفعة تسرع الاستجابة لكن تسبب تذبذب الأوزان وعدم استقرارها. |
| **قيد الأوزان `Weight Clamping`** | الحد الأقصى لقيم الاقتران $K_{ij}$ | `[-3.0, 3.0]` | ضروري جداً عند استقرار المحاكاة لمنع الانفجار الرياضي للطاقة. |
| **التعب العصبي `gamma`** | قوة تثبيط المذبذب النشط | `1.5` إلى `3.0` | في الذاكرة المتسلسلة، زيادة القيمة تعجل الانتقال للنمط التالي. |
| **التثبيط العالمي `g_inh`** | قوة المنافسة الطاقية بين الخلايا | `0.3` إلى `0.6` | تمنع تنشيط الخلايا المتعددة وتجبر عصبونات الشبكة على التنافس والسيادة المفردة. |
| **مستوى الضوضاء `noise_level`** | استكشاف فضاء الطور والهروب من الصغرى | `0.005` إلى `0.02` | يفضل استخدام التبريد التدريجي (Simulated Annealing) لتهدئة الضوضاء عند الاستقرار. |

---

## 🚀 خطوتك التالية كمطور:
ابدأ بفتح الملف **[run_game.jl](file:///C:/Users/allmy/Desktop/aaa/basil/majnon/models/mirnan/prnn_toy/prnn_survival_symphony/run_game.jl)** أو **[prnn_stuart_landau.jl](file:///C:/Users/allmy/Desktop/aaa/basil/majnon/models/mirnan/prnn_toy/prnn_stuart_landau.jl)** وقم بالتعديل على ترددات المدخلات ورصد استجابة وتزامن الشبكة في الطرفية. هذا البارادايم يفتح الباب واسعاً لحوسبة فائقة السرعة ومنخفضة الطاقة تعيد صياغة الذكاء الاصطناعي على أسس موجية فيزيائية حقيقية.
