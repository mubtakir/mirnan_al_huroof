# دليل الإعدادات — config.yaml

جميع المعاملات التالية تتحكم في سلوك المحرك الفيزيائي أثناء التوليد والتدريب.
يمكن تعديلها دون تغيير الكود.

---

## هيكل الملف

```yaml
# ═══ فضاء الاندماج الهندسي ═══
semantic_fusion:
  method: "projection"       # طريقة التوليف (linear, remnant, projection)
  vowel_modulation: true     # تعديل الحركات بنصف قيمة حرف المد
  regularize: true           # انتظام هندسي لتفادي أصفار الرتب

# ═══ خلط السياق ═══
alpha: 0.5                   # عامل خلط السياق في المتجه الطوري (0-1)

# ═══ حقل الاقتران الديناميكي (DCCF) ═══
dccf:
  decay_rate: 0.5            # سرعة اضمحلال تأثير الكلمات البعيدة
  mass_threshold: 0.3        # عتبة التوافق الطوري الأدنى

# ═══ طبقة الهوي التراكمي (Potential Cascade) ═══
cascade:
  enabled: true              # تفعيل الهوي التراكمي
  lambda_cascade: 3.0        # شدة طبقة الهوي
  gamma: 2.0                 # أس اضمحلال المسافة (N²)
  delta: 0.3                 # ثابت تجنب التفرد
  phase_lock_threshold: 0.65 # عتبة القفل الطوري
  repulsion_strength: 2.5    # شدة التنافر
  friction_decay: 0.3        # اضمحلال الاحتكاك

# ═══ التعزيز الطوري ═══
phase_reinforcement:
  learning_rate: 0.05        # سرعة التعلم الطوري
  decay: 0.01                # اضمحلال التعزيز

# ═══ حقل الأمر (PPM) ═══
prompt_field:
  decay_rate: 0.1            # سرعة اضمحلال الحقل
  strength: 1.0              # شدة التأثير

# ═══ عوامل التحكم بقوة الجاذبية ═══
gravity:
  semantic_factor: 0.3       # معامل قوة الجاذبية الطورية الدلالية
  positional_factor: 2.5     # معامل قوة الجاذبية الموضعية النحوية

# ═══ بوابة الإنتروبيا ═══
entropy:
  S_crit: 1.0                # العتبة الحرجة للإنتروبيا

# ═══ معاملات التوليد ═══
generation:
  beam_width: 5              # عرض الشعاع (عدد مرشحين محتفظ بهم)
  top_k: 500                 # عدد المرشحين من K
  beta: 2.0                  # ثابت بولتزمان للتنشيط الطوري

# ═══ معايرة الرنين ═══
calibration:
  lr: 0.02                   # معدل تعلم المعايرة
  min_samples: 50            # الحد الأدنى لعينات التدريب
  max_iterations: 10         # أقصى عدد جولات
  convergence: 0.001         # عتبة التقارب

# ═══ تطور المتجهات الطورية ═══
phase_evolution:
  lr_cooc: 0.03              # معدل تعلم المزامنة الهيبيانية
  lr_contrast: 0.015         # معدل تعلم التباين
  lr_spectral: 0.02          # معدل تعلم الإزاحة الطيفية
  min_cooc: 2                # الحد الأدنى للتشارك
  max_shift: 0.25            # أقصى إزاحة للمتجه

# ═══ مشهد القصد ═══
intent_landscape:
  adaptation_rate: 0.01      # معدل تكييف الآبار

# ═══ أثر التفاعل ═══
interaction_trace:
  decay_rate: 0.95           # معدل اضمحلال الحقل
  max_attractors: 50         # أقصى عدد مناطق الجذب
  learning_rate: 0.05        # معدل تعلم الحقل

# ═══ الذاكرة الهرمية ═══
hierarchical_memory:
  word_capacity: 100         # سعة مستوى الكلمة
  phrase_capacity: 40        # سعة مستوى العبارة
  paragraph_capacity: 15     # سعة مستوى الفقرة
  conversation_capacity: 5   # سعة مستوى المحادثة

# ═══ أوزان التسجيل ═══
scoring_weights:
  align: 5.0                 # التوافق الطوري (cosine similarity)
  gravity: 5.0               # الجاذبية الدلالية
  resonant_chain: 5.0        # السلسلة الرنينية LC + زيتا
  heterodyne: 4.0            # التغاير الترددي
  beamform: 3.5              # تشكيل الشعاع الطوري
  dccf: 4.0                  # الاقتران الديناميكي (بديل self-attention)
  causal_flow_align: 3.0     # محاذاة التدفق السببي
  density_resonance: 3.0     # رنين مصفوفة الكثافة
  prompt_align: 3.0           # التوافق مع الأمر
  constraint_align: 4.0      # قيد Dirichlet
  kb_knowledge: 2.5          # ذاكرة هولوغرافية

  syntax: 5.0                # النحو
  morpho: 1.5                # الصرف
  diversity: 2.5             # التنوع (منع التكرار حرفياً)
  repulsion: 2.0             # التنافر الطوري (طرد الكلمات المتشابهة دلالياً)
  novelty: 1.2               # الجدة ومكافأة الكلمات الجديدة
  spectral: 2.0              # الرنين الطيفي
  thermo: 1.5                # البوابة الحرارية

  dialogue: 3.0              # الحوار
  ppm: 2.0                   # حقل الأمر
  amfs: 1.5                  # التحول التكيفي
  carrier: 3.0               # الموجة الحاملة
  causal: 2.5                # السببية
  contextual_spectra: 5.0    # الأطياف السياقية

  # إنجليزية (مُعطّلة حالياً)
  eng_morpho: 1.0
  eng_grammar: 1.0

# ═══ محرك التغاير ═══
heterodyne:
  bandwidth: 0.15            # عرض نطاق التطابق الترددي
  context_window: 8          # نافذة السياق للتحليل الترددي

# ═══ التدريب ═══
training:
  window_sem: 10             # نافذة K_sem
  window_syn: 4              # نافذة K_syn
  window_dial: 2             # نافذة K_dialogue

# ═══ الوضع الإبداعي ═══
creative:
  tau: 1.2                   # حرارة Boltzmann (أكبر = تنوع أعلى)
  beam_width: 8              # شعاع أوسع للاستكشاف
  tunnel_prob: 0.05          # احتمال النفق الكمومي
  perturbation: 0.15          # شدة الاضطراب الطوري

# ═══ مصفوفة الكثافة ═══
density_matrix:
  decay_rate: 0.8            # اضمحلال الأهمية الموضعية

# ═══ حقل التدفق السببي ═══
causal_flow:
  strength: 1.0              # شدة تيار التدفق (γ)
  transitive_depth: 3        # عمق الاستدلال التعددي

# ═══ تبريد محاكى ═══
annealing:
  tau_max: 2.5               # حرارة البدء (استكشاف عالٍ)
  tau_min: 0.15              # حرارة النهاية (تبلور حتمي)
  tau_decay: 5.0             # ثابت الاضمحلال

# ═══ حقل قيود Dirichlet ═══
prompt_constraint:
  k_spring: 3.0              # ثابت النابض
  damping: 0.15              # تخميد النابض

# ═══ ذاكرة هولوغرافية ═══
holographic_kb:
  enabled: true              # تفعيل القاعدة المعرفية
  max_facts: 500             # أقصى عدد حقائق
  curated_path: "data/facts_expanded.json"

# ═══ استدلال ═══
goal_gravity_weight: 0.3     # وزن جاذبية الهدف (λ)
kb_influence: 0.4            # وزن تأثير KB (0-1)

reasoning:
  beam_width: 10             # عرض شعاع الاستدلال
  max_depth: 8               # أقصى عمق استدلالي
  convergence_threshold: 0.75 # عتبة تشابه التقارب
```

---

## دليل الضبط

### إذا كان الإخراج متكرراً جداً
```yaml
generation:
  beam_width: 10     # زِد عرض الشعاع
  beta: 1.5          # اخفض β (حرارة أعلى)
creative:
  tau: 1.5           # زِد حرارة Boltzmann
  perturbation: 0.25 # زِد الاضطراب الطوري
```

### إذا كان الإخراج عشوائياً جداً
```yaml
generation:
  beta: 3.0          # ارفع β (حتمية أعلى)
  beam_width: 3      # قلل عرض الشعاع
cascade:
  lambda_cascade: 5.0 # زِد شدة الهوي التراكمي
  phase_lock_threshold: 0.75 # ارفع عتبة القفل
```

### إذا كان بطيئاً
```yaml
generation:
  top_k: 100         # قلل عدد المرشحين
  beam_width: 3      # قلل عرض الشعاع
```

### إذا كنت تريد دقة حقائق أكثر
```yaml
kb_influence: 0.7              # ارفع وزن KB
holographic_kb:
  enabled: true
scoring_weights:
  kb_knowledge: 5.0            # ارفع وزن المعرفة
```

### إذا كنت تريد إبداعاً أكثر
```yaml
kb_influence: 0.1              # اخفض وزن KB
generation:
  beta: 1.0                    # اخفض β
scoring_weights:
  diversity: 3.0               # ارفع وزن التنوع
  novelty: 2.0                 # ارفع وزن الجدة
  repulsion: 1.5               # ارفع وزن التنافر
creative:
  tau: 2.0                     # حرارة عالية
  tunnel_prob: 0.10            # نفق كمومي أعلى
```

---

## أوزان التسجيل — القيم المثلى حسب السياق

### للتوليد الواقعي (Factual)
```yaml
scoring_weights:
  kb_knowledge: 5.0
  align: 5.0
  constraint_align: 5.0
  prompt_align: 4.0
```

### للتوليد الإبداعي (Creative)
```yaml
scoring_weights:
  diversity: 3.0
  novelty: 2.0
  repulsion: 1.5
  thermo: 3.0
```

### للتوليد الشعري (Poetic)
```yaml
scoring_weights:
  resonant_chain: 7.0
  spectral: 4.0
  carrier: 5.0
  contextual_spectra: 7.0
```

### للحوار (Dialogue)
```yaml
scoring_weights:
  dialogue: 5.0
  ppm: 4.0
  prompt_align: 5.0
```

### للاستدلال المنطقي (Reasoning)
```yaml
scoring_weights:
  causal_flow_align: 7.0
  causal: 5.0
  constraint_align: 5.0
  kb_knowledge: 4.0
```

---

## ═══ الإعدادات الطورية المتقدمة (خارج الصندوق) ═══

### 1. التداخل الطوري الهدّام (`destructive_phase`)
يستخدم هذا الخيار لحظر تسرب كلمات النظام والبرمجة وسجلات العمليات إلى الأنماط الحوارية الطبيعية عن طريق عكس متجهها الطوري ($v' = -v$) لإنتاج تداخل هدّام يمنع تفعيلها.

```yaml
destructive_phase:
  enabled: true                 # تفعيل التداخل الطوري الهدام
  system_words:                 # الكلمات الحساسة للنظام المراد حظرها حوارياً
    - "الملف"
    - "بنجاح"
    - "النافذة"
    - "اغلاق"
    - "توليد"
    - "فيزيائي"
    - "مرنان"
```
*ملاحظة: محرك التوليد يقوم بمطابقة الكلمات بطريقة غير حساسة للتشكيل أو الحركات النحوية لضمان حظر الكلمة بكافة تصريفاتها.*

### 2. الجاذبية الطورية المعكوسة (`gravity.semantic_factor`)
يمكن تعديل معامل قوة الجاذبية الدلالية ليكون سالباً لتحويل حقل الجذب إلى حقل تنافري، مما يمنع التكرار اللفظي ويحفز التوليد المتنوع والإبداعي.

```yaml
gravity:
  semantic_factor: -0.2        # قيمة سالبة تفعل التنافر الدلالي (تمنع آبار الجذب)
  positional_factor: 2.5       # جاذبية موضعية نحوية ثابتة
```

