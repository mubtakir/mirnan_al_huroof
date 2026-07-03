# معمارية نموذج مِرنان V9 (Julia) — Architecture

يوفر هذا الدليل التفصيلي نظرة عميقة على معمارية نموذج مِرنان V9، وهيكل الفضاء الطوري الكثيف، والتحسينات المضافة حديثاً.

---

## 1. الرسم البياني للوحدات والتبعيات (Module Graph)

بدلاً من الهيكل المتجانس القديم، تم تقسيم محركات مِرنان الفيزيائية والمعرفة الإدراكية إلى **10 مجموعات منظمّة** لمنع التبعيات الدائرية وتسريع البناء البرمجي:

```text
Mirnan (src/MirnanNew.jl)
├── Physics (src/physics/Physics.jl)
│   ├── Groups (src/physics/groups/)
│   │   ├── CoreGroup (config, constants, letter_db, word_physics, clifford_math, wave_field)
│   │   ├── BasicPhysicsGroup (gravity, oscillator, resonant_chain, heterodyne_engine, chaos)
│   │   ├── QuantumGroup (entropy_gate, density_matrix, phase_reinforcement)
│   │   ├── MemoryGroup (causal_flow, holographic_kb, potential_cascade, RAMCore)
│   │   ├── MorphologyGroup (word_fusion, semantic_arithmetic, lexical_oracle, weight_resonance)
│   │   ├── ArabicGroup (al_lisan, al_code, al_tadbir, al_hisab, al_hisban_al_dalali)
│   │   ├── WaveGroup (wave_oscillators, continuous_coupling)
│   │   ├── InfraGroup (infra_engines, base_classes)
│   │   ├── PRNNGroup (prnn_core, prnn_learner, prnn_generator)
│   │   └── GenerationGroup (generator, response_planning, orchestrator)
│   │
│   ├── Al-Aql (src/physics/al_aql/al_aql.jl)  # فضاء الاستدلال المعرفي والسببية
│   │
│   └── Strategies (src/physics/engines/strategies/)  # استراتيجيات التوليد الـ 15
│       ├── base, shared_scoring, finisher, resonant_strategy, dialogue_strategy,
│       ├── yesno_relations, evidence_relations, definition_strategy, code_strategy,
│       ├── math_strategy, relation_strategy, lexical_strategy, difference_and_gate,
│       └── nisba_relations, tadbir_strategy
│
├── Grammar (src/grammar/Grammar.jl)          # النحو العربي (مرتكزات 6D)
└── Semantics (src/semantics/Semantics.jl)    # الدلالة العربية
```

---

## 2. الفضاء الطوري الكثيف والثوابت الفيزيائية

يتم تمثيل الكلمات والكيانات في فضاء طوري كثيف بأبعاد إجمالية تساوي $10,000$ بُعد:
* **الأبعاد الأساسية للحرف (`PHASE_DIM`)**: $9958$ بُعداً (وهو عدد أولي كبير لضمان التوزيع العشوائي الحتمي ومنع التصادم الموجي).
* **الأبعاد الدلالية للنحو (`SYNTAX_DIMS`)**: $6$ أبعاد.
* **الأبعاد الدلالية للفلسفة (`SEMANTIC_DIMS`)**: $16$ بُعداً.
* **الأبعاد القصديّة وسياق الحوار (`PRAGMATIC_DIMS`)**: $6$ أبعاد.
* **أبعاد الجذر المورفولوجي (`ROOT_DIMS`)**: $8$ أبعاد.
* **أبعاد DDE الإضافية (`EXTRA_DIMS`)**: $6$ أبعاد.
* **الإجمالي (`TOTAL_DIM`)**: $10,000$ بُعداً.

### الثوابت الفيزيائية الكونية في النموذج
* ثابت بلانك الدلالي: $h = 1.0$
* سرعة الضوء الدلالي: $c = 1.0$
* ثابت الجاذبية الدلالي: $G = 1.0$
* ثابت بولتزمان (الحرارة الفوضوية): $k_B = 0.1$

---

## 3. تحسينات الأداء ومواءمة النوع (Float32 & Mmap)

### تثبيت المسار الساخن على Float32
بسبب الأبعاد الضخمة للمتجهات ($10,000$ بُعد)، فإن معالجة الكلمات وتخزينها المؤقت يستهلك مساحة ذاكرة كبيرة جداً.
* تم تحويل قنوات التخزين الساخنة `pv_cache` و `syntax_cache` إلى `Vector{Float32}` بدلاً من `Float64` لتوفير 50% من المساحة في الذاكرة العشوائية العشوائية وتسريع الضرب النقطي والعمليات الجبرية بنسبة 40%.
* يتم التحويل إلى `Float64` فقط في الحسابات الفيزيائية الدقيقة (حساب كتل الجاذبية وكوزين التشابه النهائي) لتجنب تراكم أخطاء التقريب.

### التخريط الفوري بالذاكرة (Mmap Cache Alignment)
عند تشغيل النظام، يتم ربط ملفات `model/pv_cache.bin` و `model/mass_cache.bin` الحاوية على متجهات وكتل جميع كلمات المعجم (532,848 كلمة) مباشرة بالذاكرة الافتراضية عبر `Mmap`.
* **الرأس الثنائي لملف الكتل (`mass_cache.bin`)**: تم تصميمه برأس `Int64` لضمان محاذاة البيانات على حدود 8 بايت، وهو ما يمنع أخطاء قراءة مؤشرات الذاكرة الغير محاذية (`ArgumentError: pointer is not properly aligned to 8 bytes`).
* يتم الاستعلام O(1) مباشرة من القرص دون تحميل الملف بالكامل في الذاكرة، مما يقلل RAM البدء إلى ما يقارب الصفر.

---

## 4. المحركات المحدثة في الإصدار V9

### تخميد مذبذبات كوراموتو (Kuramoto Damping Physics)
نموذج كوراموتو المحدث لمحاكاة التزامن الطوري عبر INTEGRATOR 4 (RK4):
$$\frac{d\phi_i}{dt} = \omega_i + \sum_j K_{ij} \sin(\phi_j - \phi_i) - \lambda_{\text{damping}} \frac{d\phi_i}{dt} + F_{\text{grav}} + \text{noise}$$
* **التخميد الديناميكي**: تمت إضافة معامل التخميد $\lambda_{\text{damping}} = 0.1$ لامتصاص الترددات الشاذة وتحقيق الاستقرار الموجي السريع.
* **قص المطال (Clipping)**: تصفية وتقييد طاقة المشتقة لمنع JIT من التشتت اللانهائي عند التكامل الزمني.

### متحكم المخيخ التكيفي المغلق (Cerebellum PID Controller)
يقوم المخيخ بمراقبة إنتروبيا النص المولّد في بوابة الإنتروبيا ويقارنها بالقيمة المستهدفة لتحديث الأوزان الـ 26 ديناميكياً:

```mermaid
graph LR
    EntropyGate[بوابة الإنتروبيا] -- إشارة الجودة --> PID[متحكم PID في المخيخ]
    PID -- تصحيح الأوزان --> SharedScoring[محرك حساب الدرجات الـ 26]
    SharedScoring -- اختيار الكلمة التالية --> EntropyGate
```

* **الخطأ المستهدف**:
  $$\text{Error} = \text{setpoint} - \text{current\_entropy}$$
* **تعديل الوزن**:
  $$\text{correction} = K_p \cdot e(t) + K_i \int e(t) dt + K_d \frac{de(t)}{dt}$$
  $$\text{weight}_{\text{new}} = \text{weight}_{\text{old}} \cdot (1 + 0.1 \cdot \text{correction})$$
* **حماية التكامل (Anti-Windup Clamping)**: تقييد حد التكامل المتراكم بين $[-10.0, 10.0]$ لمنع انحياز الأوزان المفرط في الجمل الطويلة.

---

## 5. معادلة التسجيل الفيزيائي الإجمالي للكلمة المرشحة

يتم احتساب درجة كل كلمة مرشحة $w$ عبر تراكب العوامل الفيزيائية والمعرفية الـ 26 كالتالي:
$$\text{Score}(w) = w_{\text{align}} \cdot \text{Similarity}(w, \text{context}) + w_{\text{gravity}} \cdot \frac{G \cdot m_w \cdot m_{\text{last}}}{r^2 + \delta} + w_{\text{resonant}} \cdot f_{\text{LC}}(m, pv) + w_{\text{PID}} \cdot \text{PID\_correction}$$
$$- w_{\text{diversity}} \cdot \text{RepetitionPenalty} - w_{\text{repulsion}} \cdot \text{Repulsion}(w, \text{used\_words})$$
+ بقية عوامل الذاكرة التراكمية والهولوغرافية واللسان المعايرة ديناميكياً.
