# الخوارزميات الفيزيائية — Algorithms

## 1. تحويل الحرف إلى متجه طوري

### الخوارزمية
```
لكل حرف c:
  seed = 42 + Σ ord(c)
  rng = MersenneTwister(seed)
  v = rand(rng, [-1, 1], 9958)        # متجه من قيم ±1 فقط
```

**لماذا ±1 وليس قيماً حقيقية عشوائية؟**
- ضمان تساوي الطاقة عبر جميع الأبعاد: |v| = √9958 ≈ 99.8
- حتمية مطلقة: نفس المدخل = نفس المخرج دائماً
- كسر التناظر عبر PHASE_DIM الأولي الكبير (9958)

### المعادلة
```
|v| = √(9958) ≈ 99.8
لكل بعد i: v[i] ∈ {-1, 1}
ω₀ = 0.5 + 2.0·|v|/√9958 ≈ 2.5
```

---

## 2. تحويل الكلمة إلى متجه طوري (word_phase_vector)

### الآلية: الإزاحة الموضعية (Positional Shift)

```
Word = "علم" → letters = ['ع', 'ل', 'م']

v_result = roll(v_ع, 0) + roll(v_ل, 1) + roll(v_م, 2)
v_result /= |v_result|
```

**لماذا الإزاحة بدلاً من المتوسط البسيط؟**
- المتوسط الخطي `(v_ع + v_ل + v_م) / 3` يفقد ترتيب الحروف
- "علم" و "لعم" و "ملع" سيكون لها نفس المتجه — وهذا خطأ
- الإزاحة الموضعية تحفظ الترتيب: لكل موضع تأثير فريد

**السبب الفيزيائي:**
- الحروف ليست جسيمات مستقلة — ترتيبها يُنتج تداخلاً مختلفاً
- الإزاحة الدائرية تحاكي "انزياح الطور" (phase shift) بين الحروف

### الصيغة الرياضية
```
v_result = (1/|Σ|) · Σᵢ roll(v_chars[i], i-1)
حيث roll(v, k) = [v[1-k], v[2-k], ..., v[N-k]]
```

---

## 3. الجاذبية الدلالية (Gravity)

### قانون نيوتن في فضاء الطور

```
F = G · m₁ · m₂ · (v₂ - v₁) / (r² · |v₂ - v₁|)
```

حيث:
- `G = 1.0` (ثابت الجاذبية الدلالي)
- `m = E/c² = h·f/c²` (كتلة الكلمة الدلالية)
- `r² = max(distance², 0.01)` (تجنب التفرد)
- `v₁, v₂`: متجهات الطور للكلمتين

**الغرض الفيزيائي:**
- الكلمات ذات الكتلة الكبيرة (تردد عالٍ) تجذب الكلمات الأخرى نحوها
- الكلمات المتقاربة موضعياً تتجاذب بقوة أكبر (r² في المقام)
- القوة موجهة: من الكلمة الحالية نحو الكلمة المستهدفة

---

## 4. دوائر الرنين LC (Resonant Chain)

### التشبيه الكهربائي

كل زوج كلمات متجاورة = دائرة رنين LC:

```
C (سعة) = m₁ · m₂ · (1 + cos_θ)
L (محاثة) = 1 / (1 + |sin(arccos(dot)·0.5)|)
f_res = 1 / (2π√(L·C))
```

### دالة زيتا للرنين (Zeta Response)

مستوحاة من فرضية ريمان — خط التوازن σ=1/2:

```
ζ(x) = 1 / (1 + (x-0.5)²)^(s/2)
```

- عند `cos ≈ 0.5` (خط التوازن): أقصى رنين
- عند `cos ≈ 0` أو `cos ≈ 1`: رنين ضعيف

**السبب الفيزيائي:**
- التوافق التام (cos=1) = الكلمة مع نفسها = لا معلومات جديدة
- الاختلاف التام (cos=0) = لا علاقة = لا تماسك
- التوازن (cos≈0.5) = منطقة إبداعية مثالية

### تسجيل المرشح

```
score = exp(-0.5·(Δf/σ)²) × ζ(cos_θ)
```

- Gaussian: كلما اقترب التردد من متوسط الترددات السابقة
- Zeta: تعزيز الرنين عند التوازن

---

## 5. مذبذبات كوراموتو (Kuramoto Oscillators)

### المعادلة التفاضلية

```
dφᵢ/dt = ωᵢ + Σⱼ₌₁ⁿ Kᵢⱼ·sin(φⱼ - φᵢ) + F_gravᵢ + noise
```

حيث:
- `ωᵢ`: التردد الذاتي للكلمة i
- `Kᵢⱼ`: قوة الاقتران بين الكلمتين (من مصفوفة K)
- `sin(φⱼ - φᵢ)`: تزامن كوراموتو (فرق الطور)
- `F_gravᵢ`: قوة الجاذبية من الكلمات الأخرى
- `noise ~ N(0, temperature·0.1)`: ضوضاء حرارية

### تكامل RK4 (Runge-Kutta من الدرجة الرابعة)

```
k1 = f(t, φ)
k2 = f(t + dt/2, φ + dt·k1/2)
k3 = f(t + dt/2, φ + dt·k2/2)
k4 = f(t + dt, φ + dt·k3)
φ(t+dt) = φ(t) + (dt/6)·(k1 + 2k2 + 2k3 + k4)
```

الخطأ المحلي: O(dt⁴)

---

## 6. التغاير الترددي (Heterodyne Engine)

### المبدأ: مستقبل راديو متعدد القنوات

```
الكلمة الحاملة = طيف 22D (كل بُعد = قناة ترددية مستقلة)
كلمة السياق   = طيف 22D (تضمين / modulation)

f_plus[ch]  = f_carrier[ch] + f_context[ch]
f_minus[ch] = |f_carrier[ch] - f_context[ch]|

المرشح الرنيني = الكلمة التي يقع طيفها ضمن النطاقات الجانبية
```

### حساب الرنين

```
لكل كلمة سياق c:
  لكل قناة ch:
    dist_plus  = |f_cand[ch] - f_plus[ch]| / f_plus[ch]
    dist_minus = |f_cand[ch] - f_minus[ch]| / f_minus[ch]
    if min(dist_plus, dist_minus) < bandwidth:
      resonance[ch] = 1 - min(dist_plus, dist_minus)/bandwidth

الرنين الكلي = Σ (mean(resonance_c) × weight_c) / Σ weight_c
```

### الأوزان
```
weight_c = 1/rank  (الكلمات الأقرب = وزن أعلى)
```

---

## 7. مصفوفة الكثافة الطورية (Phase Density Matrix)

### البناء

```
ρ = Σᵢ wᵢ · |ψᵢ⟩⟨ψᵢ|
```

Subject/Context represented as a mixed state:
- `|ψᵢ⟩`: متجه طور الكلمة i (10000D، مُطبَّع)
- `wᵢ`: وزن اضمحلال أسي `wᵢ = exp(-decay·(n-i)) / Σ exp(-decay·(n-k))`

### الرنين

```
R(v) = ⟨v| ρ |v⟩
     = vᵀ · ρ · v
     = Σᵢ wᵢ · |⟨v|ψᵢ⟩|²
```

**لماذا هذا أفضل من np.average(pvs)؟**
- يحافظ على العلاقات البينية بين متجهات السياق
- لا يحدث تداخل إتلافي (destructive interference)
- يسمح بقياس درجة "النقاء" و "الاقتران الطوري"

### النقاء والاقتران

```
Purity = Tr(ρ²)          ∈ (0, 1]
  1.0 = حالة نقية (كلمة واحدة)
  < 1.0 = حالة مختلطة (سياق متعدد الكلمات)

S_vn = -Tr(ρ ln ρ)        إنتروبيا فون نيومان الطيفية
  تقيس درجة "الاقتران الطوري" — كلما زادت، كان السياق أكثر ثراءً
```


---

## 8. حقل التدفق السببي (Causal Flow)

### المعادلة الأساسية

```
J(pv) = Σᵢ₌₁ⁿ wᵢ · C_strength(cid→tid) · (target_pv - current_pv)
```

حيث:
- `wᵢ = exp(-0.5·i)`: اضمحلال حسب البعد الموضعي
- `C_strength = causal_matrix[cid, tid]`: قوة العلاقة السببية
- `target_pv`: متجه طور الكلمة المسبَّبة
- `current_pv`: متجه طور الكلمة الحالية

### توافق المرشح مع التدفق

```
alignment = cos(candidate_pv - current_pv, flow_vector)
```

إذا كان اتجاه المرشح متوافقاً مع اتجاه التدفق → مرشح جيد (استدلال سليم)

### الاستدلال التعددي (transitive)

```
A → B → C → D
cumulative_strength = C(A,B) · C(B,C) · C(C,D)
```

كلما كانت السلسلة السببية أقوى، كان الاستدلال أصدق.

---

## 9. الذاكرة الهولوغرافية (Holographic KB)

### التخزين (store_fact!)

```
لكل حقيقة (subject, relation, object):
  key = subject_pv
  value = object_pv
  تخزين في bank[relation]
  منع التكرار: (subj_word, obj_word) فريد لكل علاقة
```

### الاستعلام (query)

```
resonance[i] = cos(key[i], query_pv)         # تشابه جيب التمام
sharpened = exp(sharpening · resonance[i])   # رفع أسي
weights = sharpened / Σ sharpened            # تطبيع
نتيجة[i] = (weight[i], obj_word[i], rel_type) if weight[i] > dynamic_cutoff
```

**العتبة الديناميكية:**
```
dynamic_cutoff = max(cutoff, 2/N_facts)
```
تمنع استرجاع الحقائق غير المرتبطة: يجب أن يكون الوزن > ضعف متوسط التوزيع المتساوي.

### إعادة بناء المتجه (reconstruct_vector)

```
reconstructed = Σᵢ weightᵢ · valueᵢ
weightᵢ = exp(3.0 · cos(keyᵢ, query)) / Σ exp(...)
reconstructed /= |reconstructed|
```

هذا "الجداء الهولوغرافي" — يسترجع القيمة المخزنة من المفتاح.

---

## 10. التعزيز الطوري الهيبياني (Phase Reinforcement)

### التعزيز (reinforce)

```
trace[word] ← trace[word] + lr · reward · (new_pv - trace[word])
trace[word] ← trace[word] / |trace[word]|
strength[word] ← min(1.0, strength[word] + reward · lr)
```

### الإضعاف (weaken)

```
strength[word] ← max(0.0, strength[word] - penalty · lr)
إذا strength < 0.01: حذف الأثر
```

### التطبيق (apply)

```
إذا strength < 0.01: لا تغيير
mixed_pv = (1 - strength·0.5) · original_pv + strength·0.5 · trace[word]
```

### الاضمحلال الشامل (decay_all)

```
لكل كلمة:
  strength[word] *= (1 - decay_rate)
  إذا strength < 0.01: حذف
```

---

## 11. حقل الاقتران الديناميكي (DCCF)

### بناء مصفوفة الاقتران

```
coupling[i, j] = phase_align(i,j) · mass_sim(i,j) · dist_decay(i,j)

حيث:
  phase_align(i,j) = dot(pv_i, pv_j) / (|pv_i| · |pv_j|)
  mass_sim(i,j)    = |pv_i| · |pv_j|
  dist_decay(i,j)  = exp(-decay_rate · |i - j|)
```

### تعزيز السياق للمرشح

```
boost = Σᵢ [phase_align(w, ctx_i) · |ctx_pv_i| · exp(-decay·(n-i))]
      / Σᵢ [|ctx_pv_i| · exp(-decay·(n-i))]
```

---

## 12. حقل تعديل الطور بالأمر (PPM)

### الامتصاص (absorb!)

```
field = strength · mean( جميع PVs لكلمات الأمر )
active = true
```

### تعديل المرشح (modulate)

```
candidate_modulated = candidate_pv + 0.15 · field
```

### الاضمحلال (step!)

```
field ← field · (1 - decay_rate)
إذا |field| < 0.01: active = false
```

---

## 13. التحول التكيفي (AMFS)

### المركزية (centrality)

```
centrality = mean(cos(w_pv, ctx_pv))
```

### الكتلة المُعدّلة

```
adapted_mass = base_mass · (1 + 0.3 · max(0, centrality))
```

الكلمات ذات الدور المركزي في السياق تكتسب كتلة إضافية (جاذبية أقوى).

### التردد المُعدّل

```
adapted_freq = base_freq · (1 + 0.15 · tanh(phase_shift))
```

---

## 14. بوابة الإنتروبيا (Entropy Gate)

### حساب الإنتروبيا

```
S = -Σ pᵢ · log(pᵢ)

حيث:
  sims = cos(pv_i, target_pv)
  pᵢ = (simsᵢ - min) / Σ (sims - min)
```

### التصحيح الحراري

```
إذا S ≥ S_crit:
  k_B ← k_B / 2.0    (تبريد — خفض الضوضاء الحرارية)
  β ← min(β × 1.5, 6.0)  (تسخين — زيادة الحتمية)
```

---

## 15. الذاكرة الزمنية المتلاشية (RAM Core)

### تخزين الجاذب (observe!)

```
phi_center = mean(pvs) / |mean(pvs)|
sigma = mean(distance(pv_i, center)) + 0.01

لكل جاذب موجود:
  إذا cos(center_old, center_new) > merge_cos:
    مزج مراكز: center = (n1·c1 + n2·c2) / (n1+n2)
    sigma = min(sigma, sigma_new)
    دمج الكلمات + أقنعة الاقتران

وإلا:
  إضافة جاذب جديد
```

### الاسترجاع (resonate)

```
لكل جاذب i:
  dist² = Σ (phi_current[d] - center_i[d])²
  score = exp(-dist² / (2·σ_i²)) × exp(-decay · age_i)

إرجاع أعلى top_k درجة
```

---

## 16. طبقة الهوي الترامركب (Potential Cascade)

### الجهد الكلي

```
V(c) = -λ · Σᵢ [ mᵢ · max(0, cosΔφᵢ) / (d_pos(i) + δ)^γ ]
     + Σⱼ repulsion / d(w_usedⱼ, c)²
     + friction · n_ctx · |c - last_pv|
```

حيث:
- `mᵢ`: كتلة الكلمة السياقية i
- `cosΔφᵢ = cos(c, ctx_pvᵢ)`: التوافق الطوري
- `d_pos(i) = n_ctx - i + 1`: المسافة الموضعية
- `γ = 2.0`: أس اضمحلال المسافة (مربع عكسي)
- `δ = 0.3`: ثابت تجنب التفرد

### شروط القبول

```
إذا !syntax_valid: return -Inf
إذا cos(c, last_pv) < phase_lock_threshold: return -Inf
```

**النتيجة النهائية:**
```
score = λ · |V(c)|  (فقط إذا V(c) < 0 — أي أن الجذب يغلب التنافر)
```

---

## 17. خوارزمية التوليد الكاملة (Generator)

### الخطوات

```
Input: prompt, mode, max_words

1. تهيئة:
   - context_words = split(prompt)
   - context_pvs = [compute_extended_phase_vector(w) for w in context_words]
   - context_masses = [compute_word_mass(w) for w in context_words]
   - target_pv = last(context_pvs)
   - used_words = Set(context_words)
   
2. امتصاص PPM:
   - absorb!(prompt_field, prompt)
   
3. بناء مصفوفة الكثافة:
   - build!(density_matrix, context_pvs)

4. حلقة التوليد (لكل خطوة 1..max_words):
   a. استخراج المرشحين من K_sem (أقوى الاقترانات)
   b. فلترة: إزالة الكلمات المستخدمة سابقاً
   c. تسجيل كل مرشح:
      score = Σ w_factor · factor_value (26 عاملاً)
   d. اختيار:
      - standard: أفضل مرشح (greedy)
      - creative: عينة بولتزمان من أعلى beam_width مرشح
      - prnn: الانزلاق الحتمي على حقل الجهد غير الخطي عبر ديناميكيات ستوارت-لانداو والربط الهولوغرافي
      - couple: الحوسبة الموجية التناظرية والاقتران الطوري وتجاوز التداخل الهدام
   e. إضافة المرشح للسياق
   f. اضمحلال PPM: step!(prompt_field)

5. تعزيز هيبياني للجملة المولدة:
   - reinforce!(reinforcement, word, pv; reward=0.3)

Output: join(generated_words, " ")
```


---

## 18. الشبكة الرنينية الطورية الهولوغرافية (Holographic PRNN Engine)

يقدم هذا المحرك حلاً ديناميكياً فيزيائياً كاملاً لمشكلة التفرع السببي وسياق الكلمات وتوليد النصوص دون أي حاجة للشبكات العصبية التقليدية أو خوارزميات الاسترجاع الاحتمالي. يعتمد النظام على دمج ثلاثة حقول أساسية:
1. **تمثيل الأطوار الكثيفة (Geometric Phase Expansion)**: لتجاوز تماثل الطور المتفرق.
2. **الربط الهولوغرافي (Holographic Phase Binding)**: لتمثيل سياق الكلمات بشكل فريد في فضاء طوري متكامل.
3. **ديناميكيات ستوارت-لانداو (Stuart-Landau Dynamics)** Non-linear limit cycles مع التثبيط والتعب.

### الخوارزمية

```
1. تحويل المتجهات الدلالية المتفرقة لطور كثيف (build_dense_phase_vector):
   v_real = compute_extended_phase_vector(word)
   phases = v_real * (√N * pi)
   base_vector = exp(i * phases)
   حيث N = 10,000 (البعد الطوري الموسع لمرنان)

3. **ديناميكيات ستوارت-لانداو (Stuart-Landau Dynamics)** Non-linear limit cycles مع التثبيط والتعب.

### الخوارزمية

```
1. تحويل المتجهات الدلالية المتفرقة لطور كثيف (build_dense_phase_vector):
   v_real = compute_extended_phase_vector(word)
   phases = v_real * (√N * pi)
   base_vector = exp(i * phases)
   حيث N = 10,000 (البعد الطوري الموسع لمرنان)

2. التعلم الهيبي المتسلسل الهولوغرافي (train_hebbian_transitions!):
   v_curr_adapted = base_vectors[w_1]
   لكل انتقال من t إلى t+1:
     v_next_adapted = v_next ⊙ v_curr
     تسجيل الانتقال (v_curr_adapted, v_next_adapted) في Transitions
     v_curr_adapted = v_next_adapted

3. محاكاة ستوارت-لانداو منخفضة الرتبة (simulate_stuart_landau!):
   z = الحالة الطورية الابتدائية (الكلمة الأولى أو سياق البداية المربوط)
   لكل خطوة زمنية (T=40, dt=0.02):
     global_activity = Σ |z_i|² / N
     coupling_vector = Σ_{(v_curr, v_next)} (dot(v_curr, z) * beta / N) * v_next
     dz_i/dt = (μ - a_i - g_inh * global_activity - |z_i|² + i * ω_i) * z_i + coupling_i
     z_i = z_i + dt * dz_i
     da_i/dt = (-a_i + γ * |z_i|²) / τ_a (التعب العصبي المحلي)
     a_i = a_i + dt * da_i

4. فك التشفير الطوري الهولوغرافي (decode_word):
   z_unbound = z_state ⊙ conj(v_context)
   الكلمة الفائزة = argmax_{word} real(dot(z_unbound, base_vectors[word])) / N
```

### الصيغ الرياضية

* **الربط وفك الربط الهولوغرافي (Binding/Unbinding):**
  $$ \mathbf{v}_{\text{adapted}} = \mathbf{v}_{\text{next}} \odot \mathbf{v}_{\text{curr}} $$
  $$ \mathbf{z}_{\text{unbound}} = \mathbf{z}_{\text{state}} \odot \mathbf{v}_{\text{context}}^* $$

* **معادلة الحركة للمذبذبات الطورية المركبة:**
  $$ \frac{dz_i}{dt} = \left( \mu - a_i - g_{\text{inh}} \frac{\sum_k |z_k|^2}{N} - |z_i|^2 + i \omega_i \right) z_i + C_i(t) $$
  حيث حقل الاقتران الطوري منخفض الرتبة $C(t)$ يُعطى بالعلاقة:
  $$ \mathbf{C}(t) = \frac{\beta}{N} \sum_{(u, v) \in T} (\mathbf{u}^* \cdot \mathbf{z}) \mathbf{v} $$
  وهي المتكافئة رياضياً للضرب المباشر في مصفوفة الكينونة السببية الممتدة $K$ دون استهلاك مساحة الذاكرة الهائلة:
  $$ K = \frac{\beta}{N} \sum_{(u, v) \in T} \mathbf{v} \otimes \mathbf{u}^* $$

* **معادلة التعب العصبي المحلي (Synaptic Fatigue):**
  $$ \tau_a \frac{da_i}{dt} = -a_i + \gamma |z_i|^2 $$

---

## 19. محرك المتجه الموجي والاقتران الطوري التناظري (Analog Wave Oscillator & Coupling Engine)

يعتمد هذا المحرك على تصميم حوسبة موجية تناظرية مستمرة كبديل متماسك للحوسبة الرقمية التقليدية. يمثل المتجه الموجي المركب كحزمة موجية متراكبة ومتداخلة في فضاء طوري كثيف ذي أبعاد $N = 10,000$.

### الخوارزمية الرياضية

1. **انشطار الحزمة (Wave Splitting):**
   تنقسم الموجة الطورية المركبة $\mathbf{v}$ إلى مركبتين متعامدتين ونظير معاكس (موجة مضادة/هدامة) وفق معادلة مجزئ الحزمة:
   $$ \mathbf{v}_1 = \cos(\theta) \mathbf{v} $$
   $$ \mathbf{v}_2 = \sin(\theta) \mathbf{v} \odot e^{i \phi_{\text{shift}}} $$
   $$ \mathbf{v}_{\text{anti}} = -\mathbf{v}_1 = \mathbf{v}_1 \odot e^{i \pi} $$
   حيث يحقق النظام انحفاظ الطاقة الموجية: $\|\mathbf{v}_1\|^2 + \|\mathbf{v}_2\|^2 = \|\mathbf{v}\|^2$.

2. **دمج وتداخل الموجات (Wave Merging):**
   تندمج موجتان $\mathbf{v}_1$ و $\mathbf{v}_2$ مع سياق التوليد بفرق طور $\Delta \phi$:
   $$ \mathbf{w}_{\text{merged}} = \frac{\mathbf{v}_1 + \mathbf{v}_2 \odot e^{i \Delta \phi}}{\|\mathbf{v}_1 + \mathbf{v}_2 \odot e^{i \Delta \phi}\|} $$
   حيث تعطي القيمة $\Delta \phi = 0$ تداخلاً بناءً تاماً، وتعطي $\Delta \phi = \pi$ تداخلاً هداماً متضاداً تاماً.

3. **الاقتران والقياس الطوري الهولوغرافي (Phase Coupling & Measurement):**
   تُقرن الموجة الناتجة من التداخل البناء $\mathbf{w}_{\text{constructive}}$ مع حقل موجة المثير الإجمالية $\mathbf{w}_{\text{prompt}}$ عبر جداء خارجي هولوغرافي:
   $$ \mathbf{E}_{AB} = \mathbf{w}_{\text{constructive}} \otimes \mathbf{w}_{\text{prompt}}^\dagger $$
   وعند تمرير موجة السياق الحالية للقياس، يتم استرجاع الموجة التنبؤية بالاقتران الطوري:
   $$ \mathbf{w}_{\text{projected}} = \mathbf{E}_{AB} \mathbf{v}_{\text{curr}} $$

4. **معادلة تقييم المتجه الموجي للمرشح:**
   تُقاس محاذاة الكلمة المرشحة مع الموجات الثلاث:
   $$ S_{\text{projected}} = \frac{\text{Re}(\mathbf{v}_{\text{cand}}^\dagger \cdot \mathbf{w}_{\text{projected}})}{N} $$
   $$ S_{\text{constructive}} = \frac{\text{Re}(\mathbf{v}_{\text{cand}}^\dagger \cdot \mathbf{w}_{\text{constructive}})}{N} $$
   $$ S_{\text{destructive}} = \frac{\text{Re}(\mathbf{v}_{\text{cand}}^\dagger \cdot \mathbf{w}_{\text{destructive}})}{N} $$
   وتُعطى درجة التسجيل النهائية للمتجه الموجي بالعلاقة:
   $$ Score_{\text{oscillator}} = 4.0 S_{\text{projected}} + 2.0 S_{\text{constructive}} - 3.0 S_{\text{destructive}} $$
    حيث يُخصم التداخل الهدام لمنع الكلمات المتناقضة سياقياً، ويُعزز التداخل البناء والاقتران لتوجيه المعنى بدقة فيزيائية متناهية.

---

## 18. عامل الالتواء الطوري — MorphoTwistor

### الفكرة الأساسية

بدلاً من جداول صرفية مبرمجة يدوياً (فَعَلَ ← يفعَل ← فاعِل ← مفعول...)، **يتعلم مرنان العلاقات الصرفية كتحويلات هندسية** بين متجهات كليفورد.

### الأساس الرياضي

بين أي كلمتين `w₁` و `w₂` لهما علاقة صرفية، يوجد **عامل التواء طوري T**:

```
v(w₂) = T ∘ v(w₁)
```

حيث `∘` هو الجداء الهندسي لكليفورد. يُحسب T بالقسمة الهندسية:

```
T₁₂ = MV(w₂) / MV(w₁)    ← قسمة كليفورد في فضاء 22D
```

هذا T ليس رقماً واحداً، بل **متعدد متجهات (Multivector)** يحمل:
- **السلمي (scalar):** نسبة التمدد/التقلص بين الكلمتين
- **المتجه (vector 22D):** اتجاه الانزياح الدلالي
- **ثنائي المتجه (bivector 231D):** مستوى الدوران الطوري

### خوارزمية التعلم: learn_patterns!

```
المدخلات: أزواج كلمات (w₁, w₂) من الكوربس
المخرجات: قوالب صرفية MorphPattern

المرحلة 1 — جمع الأزواج:
  find_morphological_pairs(words):
    لكل زوج كلمات في نافذة محلية (مسافة ≤ 5):
      إذا تشابهت أول 3 حروف (جذر مشترك):
        أضف (w₁, w₂) للأزواج

المرحلة 2 — حساب T:
  لكل زوج (w₁, w₂):
    T = compute_twistor(w₁, w₂)
    = TwistorOperator(MV(w₂) / MV(w₁))

المرحلة 3 — تجميع (Agglomerative Clustering):
  clusters = []
  لكل Tᵢ غير معيّن لمجموعة:
    أنشئ مجموعة جديدة مركزها Tᵢ
    لكل Tⱼ المتبقي:
      sim = twistor_similarity(Tᵢ, Tⱼ)
      = cosine(MV(Tᵢ), MV(Tⱼ))  ← تشابه في فضاء 254D
      إذا sim ≥ 0.7:
        أضف Tⱼ للمجموعة
        حدّث المركز = _blend_twistors(مركز, Tⱼ, α=1/(n+1))

المرحلة 4 — التصفية:
  المجموعات ذات الحجم < min_cluster_size تُهمل
  كل مجموعة ← MorphPattern (قالب صرفي = مركز + أمثلة)
```

### خوارزمية الاستدلال: predict_derivation

```
المدخلات: مثال (w₁, w₂), كلمة استفسار (query)
المخرجات: أفضل توقع للاشتقاق

الخطوة 1: حساب T من المثال
  T_example = compute_twistor(w₁, w₂)

الخطوة 2: مطابقة القوالب
  best_T = T_example
  لكل قالب pat في engine.patterns:
    sim = twistor_similarity(T_example, pat.center)
    إذا sim > 0.5:
      best_T = pat.center  (استخدم القالب المتعلَّم)

الخطوة 3: تطبيق T على query
  v_prediction = apply_twistor_pv(best_T, compute_word_phase_vector(query))
  = T.mv * MV(query)    ← جداء كليفورد

الخطوة 4: البحث في المعجم
  لكل كلمة في vocab:
    sim = cosine(v_word[1:22], v_prediction)
  أرجع أعلى top_k تشابهاً
```

### مثال عملي

```
المستخدم: "نور نورا جو ؟"

التحليل:
  w₁ = "نور"        ← T = MV(نورا) / MV(نور)
  w₂ = "نورا"       ← T يحمل تحويل "إضافة ألف" في 22D
  query = "جو"

  v_pred = T ∘ MV(جو)   ← ينتج متجهاً قريباً من MV(جوا)
  أقرب كلمة في المعجم: "جوا"  (تشابه 0.87)
  ← الإجابة: "جوا"
```

### لماذا فيزيائي وليس إحصائياً؟

| جداول صرفية تقليدية | عامل الالتواء T |
|---------------------|-----------------|
| لكل وزن قاعدة منفصلة | T واحد يتعلم كل الأنماط هندسياً |
| يفشل مع الكلمات خارج القاموس | يعمل على أي كلمة لها متجه طوري (حتى المخترعة) |
| "إضافة ألف" = `word + "اً"` | "إضافة ألف" = دوران + تمدد في فضاء 22D |
| لا يفهم "لماذا" | التحويل الهندسي = تفسير فيزيائي للعلاقة |
| يحتاج صيانة يدوية | يتعلم تلقائياً من الكوربس |

### التكامل مع النظام

```
التدريب:
  train.jl → find_morphological_pairs → learn_patterns!
  → model/twistor_patterns.json

الاستدلال المباشر:
  generate!("نور نورا جو ؟") → T = compute_twistor
  → predict_derivation → "جوا"

الاستدلال الضمني (أثناء التوليد العادي):
  _inv_twistor_ref تُخزَّن بعد رؤية مثال (w₁ w₂)
  عامل twistor_score (وزن 2.5) يُكافئ الكلمات القريبة
  من الاشتقاق المتوقع عبر T∘v(query)
```

---

## 19. خط المعالجة المسبقة — Preprocessing Pipeline (5 مراحل)

### الخوارزمية

يحول النصوص الخام (إنترنت، موسوعات) إلى نصوص نظيفة لمرنان عبر 5 مراحل متسلسلة:

```
النص الخام
  │
  ├─[1] TextExtractor.extract_text()
  │     كشف تلقائي للمصدر (HTML/Markdown/JSON/text)
  │     إزالة الوسوم والتنسيق ← نص صافٍ
  │
  ├─[2] LanguageFilter.detect_language()
  │     عد أحرف Unicode العربية (U+0600-U+06FF) مقابل اللاتينية
  │     ar_ratio ≥ 50% ← قبول، غير ذلك ← رفض
  │
  ├─[3] Normalizer.normalize_text()
  │     تطبيع: آ/أ/إ→ا، ة→ه، ى→ي
  │     إزالة: الكشيدة (ـ)، رموز التحكم (\x00-\x1F)
  │     ضغط: مسافات متعددة→واحدة، 3+ أسطر←2
  │     توحيد: علامات الاقتباس والشرطات
  │
  ├─[4] Segmenter.segment_sentences()
  │     عربي: تقطيع عند .!؟\n (مع حماية الاختصارات د. أ.)
  │     إنجليزي: تقطيع عند .!? + مسافة + حرف كبير
  │     دمج الجمل القصيرة (<5 كلمات)، تقسيم الطويلة (>200)
  │
  └─[5] QualityFilter.filter_quality()
        check_quality(text) → (pass?, reason)
        • طول: 5-200 كلمة
        • تكرار ≤ 30% من الكلمات
        • ترقيم ≤ 25%، أرقام ≤ 15%
        • أحرف مفهومة ≥ 50%
        • deduplicate: تشابه Jaccard < 85%
```

### مثال كامل

```
المدخل: "<div><p>العلم نور</p><script>x()</script></div>"

① TextExtractor: "العلم نور"
② LanguageFilter: ar_ratio = 100% → قبول
③ Normalizer: "العلم نور" (لا تغيير — النص نظيف)
④ Segmenter: ["العلم نور"] (جملة واحدة)
⑤ QualityFilter: 3 كلمات ← قصيرة جداً ← رفض

الناتج: [] (الجملة رُفضت لقصرها — أقل من 5 كلمات)
```

### التكامل مع train.jl

```bash
# تدريب مع المعالجة المسبقة
julia --project=. train.jl --preprocess=C:\raw_texts

# ينتج: data/cleaned/cleaned_corpus.txt (جمل نظيفة)
```

---

## 20. مسبب البحث الشعاعي — BeamReasoner

### الفكرة الأساسية

بدل المسار الجشع الواحد (PathIntegralReasoner القديم)، **5 مسارات استدلال متوازية** تتنافس للوصول إلى الهدف. كل مسار يُسجَّل بمقاييس فيزيائية (توافق طوري + ثقة KB + تنوع).

### الخوارزمية

```
المدخلات: start_pv, goal_pv, pv_fn, vocab
المخرجات: أفضل 3 سلاسل استدلالية مع تفسير

① التهيئة: beam_width مسارات فارغة من نقطة البداية

② لكل خطوة (max_depth):
   لكل مسار غير متقارب:
     أ. استخراج مرشحين من الذاكرة الهولوغرافية أولاً
        query_kb(current_pv; top_k=10)
        score = goal_weight × goal_sim + kb_weight × confidence

     ب. استخراج مرشحين من K_sem (إن لم يكتفِ KB)
        row_walk(current_word; top=15)
        score = goal_weight × goal_sim + cooccurrence_weight

     ج. عقاب التنوع: diversity_penalty للكلمات الموجودة في مسارات أخرى

     د. تحقق من التقارب: goal_sim ≥ convergence && depth ≥ min_depth

③ اختيار أفضل beam_width مسار للمرحلة التالية

④ إرجاع السلاسل المتقاربة (أو المكتملة) مرتبة تنازلياً
```

### بنية البيانات

```julia
ReasonStep:   word, score, similarity_to_goal, source("kb"/"K_sem"), relation
ReasonChain:  steps[], total_score, converged, depth
BeamReasoner: beam_width=5, max_depth=8, convergence=0.75
```

### دالة التفسير

```julia
explain_reasoning(chains) -> String
# "المسار 1 (✓ تقارب، درجة: 4.2):
#   1. 📚 العلم (قرب: 0.45)
#      └ علاقة: isa
#   2. 🔗 المعرفة (قرب: 0.72)
#   3. 📚 النجاح (قرب: 0.89)"
```

---

## 21. البرمجة الفيزيائية — CodePhaseEngine

### الفكرة الأساسية

كل مفهوم برمجي (for, list, function...) = **متجه طوري** في نفس فضاء 10000D. الكود = **جداء كليفورد** للمفاهيم. ليس Markov chain إحصائياً.

### الخوارزمية

```
① ترميز المفاهيم (encode_concept):
   17 مفهوماً أساسياً: loop, condition, function, class, list, dict...
   لكل مفهوم متجه طوري = متوسط متجهات كلماته (عربي + إنجليزي)
   v(loop) = mean(v("for"), v("while"), v("حلقة"), v("طالما"))

② تعلم الأنماط (learn_code_patterns!):
   من أمثلة أكواد، يُستخرج تسلسل المفاهيم
   "for i in list: print(i)" → [loop, list, print]
   يُخزَّن القالب للاستدلال المستقبلي

③ فك الترميز إلى كود (decode_to_code):
   [function, loop, condition, print] → Python/Julia
   def solve():
       for i in range(len(data)):
           if True:
               print(i)
```

### مثال

```
المستخدم: "اكتب دالة ترتب قائمة وتطبعها"
التحليل:   function + sort + list + print
الكود:
  def solve():
      data = []
      data.sort()
      print(data)
```

---

## 22. الرياضيات الفيزيائية — SymbolicMathEngine

### الفكرة الأساسية

كل عدد = **متجه طوري فريد** (بذرة خاصة). كل عملية = **عامل التواء T** (تحويل هندسي). حل مسألة = **سلسلة تحويلات**.

### الخوارزمية

```
① ترميز الأعداد (encode_number):
   v(42) = MersenneTwister(42*7) → ±1 vector في 9958D
   v(-5) = MersenneTwister(5*7 + 13) → متجه مختلف تماماً

② تعلم العمليات (learn_math_pattern!):
   learn_math_pattern!(eng, 2, "add", 3, 5)
   T_add = v(5) / v((2+3)/2)  ← قسمة كليفورد
   يُخزَّن T_add كعامل التواء لعملية الجمع

③ حل المسائل (solve_arithmetic):
   solve_arithmetic(eng, 7, "زائد", 8)
   input_pv = (v(7) + v(8)) / 2
   pred_v = T_add ∘ input_pv  ← جداء كليفورد
   البحث عن أقرب عدد لـ pred_v في number_cache
   → (15, 0.94)
```

### المزايا على `eval()`

| `eval()` التقليدي | SymbolicMathEngine |
|-------------------|-------------------|
| ينفذ كود Julia عشوائي (خطر أمني) | تحويلات كليفورد في فضاء الطور (آمن) |
| لا يتعلم — نتائج فقط | يتعلم T لكل عملية من أمثلة |
| لا يفهم "لماذا" | العامل T = تفسير فيزيائي للعملية |
| أعداد فقط | كل عدد = متجه في فضاء 10000D (قابل للدمج مع اللغة) |

---

## 23. حقل الجذر — RootField

### الفكرة الأساسية

أداة للباحث اللغوي والأديب والنحوي. **ليست توليداً**، بل استكشاف للتناغم الصرفي الموجود في أعماق المتجهات الطورية. يكتب المستخدم كلمة واحدة، فيفحص النظام **كل كلمات المعجم** باحثاً عن مشاركات الجذر.

### الخوارزمية

```
المدخلات: كلمة واحدة (word)، المعجم (vocab)
المخرجات: قائمة كلمات الجذر مرتبة حسب قوة التشابه الطوري

① استخراج جذر الكلمة المدخلة:
   root_chars = _extract_root_light(word)
   — تطابق 26 وزناً صرفياً لاستخراج الحروف الجذرية
   — مثال: "نور" ← [ن, و, ر]، "مستنير" ← [ن, و, ر]

② لكل كلمة في المعجم:
   cand_root = _extract_root_light(candidate)
   sim = cosine_similarity(v(word), v(candidate))
   — المتجهان يحملان نفس المقطع الجذري (8D في 10000D)
   — كلما تشابه الجذر، ارتفع sim تلقائياً

③ ترتيب تنازلي: نفس الجذر أولاً، ثم الجذور القريبة
   — كلمات الجذر المطابق تُفضَّل
   — داخل نفس الجذر: حسب التشابه الطوري
```

### أوضاع الإخراج الثلاثة

| الوضع | الأمر | المخرج | المستخدم |
|-------|-------|--------|----------|
| `detailed` | `--mode root` | تقرير مفصل: الجذر، العدد، قائمة + نسبة التشابه | الباحث اللغوي |
| `list` | `--mode root_list` | قائمة كلمات بسيطة | المعجمي |
| `poetic` | `--mode root_poetic` | صياغة أدبية | الشاعر والأديب |

### مثال

```
المستخدم: "نور" (وضع detailed)

═══ حقل الجذر: [نور] ═══
الجذر الثلاثي/الرباعي: نور
عدد الكلمات في الحقل: 12

─ كلمات من نفس الجذر ─
  1. نوار   [73.2%] ██████████████
  2. منار   [68.5%] █████████████
  3. منير   [65.1%] █████████████
  4. أنور   [61.8%] ████████████
  5. نيران  [58.4%] ███████████
  6. تنوير  [55.2%] ███████████
  ...

─ كلمات من جذور قريبة ─
  1. بهاء   [28.1%]  (جذر: بهو)
```

### لماذا هذه الأداة مهمة؟

- **للباحث اللغوي**: يرى العلاقات الصرفية التي اكتشفها النموذج **فيزيائياً** دون أي تدخل بشري
- **للأديب**: يستلهم كلمات من الجذر نفسه لبناء نص متماسك
- **للنحوي**: يتحقق من شمولية تغطية النموذج للأوزان الصرفية المختلفة
- **للمطور**: يختبر صحة استخراج الجذور عبر `_extract_root_light`
