# خطة تطوير HoloPRNN_Vision — المرحلة التالية
## Development Plan: Style Profiles, Demo Refactor, and SemanticScene→Vision Bridge

> **المبدأ التوجيهي:** لا نغير جوهر المعادلات الفيزيائية (Stuart-Landau, Laplacian, CHL).  
> نضيف طبقات فصل وإعادة استخدام وتكامل مع SemanticScene في مرنان.

---

## المحاور الثلاثة

### المحور أ: فصل استخراج Profile الأسلوب
### المحور ب: تصغير demo_classification وإزالة تكرار الكود
### المحور ج: تصميم جسر SemanticScene → حقل بصري

---

## المحور أ: فصل استخراج Profile الأسلوب

**الهدف**: جعل `run_style_transfer` قابلة للفصل إلى خطوتين مستقلتين:
1. استخراج بصمة الأسلوب من صورة مرجعية ← `StyleProfile`
2. تطبيق البصمة على أي صورة محتوى ← `apply_style`

### أ.1 — هيكل `StyleProfile`

```julia
struct StyleProfile
    H::Int
    W::Int
    K_h::Matrix{ComplexF64}   # علاقات الطور الأفقية
    K_v::Matrix{ComplexF64}   # علاقات الطور العمودية
    params::OscillatorParams   # البارامترات المثلى المستخرجة
end
```

### أ.2 — `extract_style_profile`

```julia
"""
    extract_style_profile(img::Matrix{<:Colorant}, params=OscillatorParams()) -> StyleProfile

تحليل صورة الأسلوب واستخراج مصفوفة الاقتران المحلي (العلاقات الطورية بين الجيران).
لا تحتاج أي محاكاة — مجرد تحويل وتحليل طوري مباشر.

الخطوات:
1. image_to_wavefield(img) → z
2. حساب K_h[y,x] = exp(i * (θ[y,x+1] - θ[y,x])) أفقي
3. حساب K_v[y,x] = exp(i * (θ[y+1,x] - θ[y,x])) عمودي
4. تخزين النتائج في StyleProfile
"""
```

### أ.3 — `apply_style`

```julia
"""
    apply_style(z_content::Matrix{ComplexF64}, profile::StyleProfile, params=OscillatorParams();
                D_spatial=0.1, steps=200) -> Matrix{ComplexF64}

تطبيق بصمة الأسلوب على حقل المحتوى.
خطوة المحاكاة نفسها في run_style_transfer الحالية، لكن باستخدام StyleProfile المُستخرج مسبقاً.
"""
```

### أ.4 — حفظ وتحميل

```julia
save_style_profile(profile::StyleProfile, path::String)
load_style_profile(path::String) -> StyleProfile
```

### أ.5 — تعديل `run_style_transfer`

تصبح wrapping بسيطاً:

```julia
function run_style_transfer(img_content, img_style, params; ...)
    profile = extract_style_profile(img_style, params)
    z_content = image_to_wavefield(img_content)
    z_result = apply_style(z_content, profile, params; ...)
    return wavefield_to_image(z_result)
end
```

### أ.6 — التبعية مع المحور ج

المحور ج سيستخدم `apply_style` لتطبيق أسلوب بصري على المشاهد الدلالية — مثلاً:
- `Scene(actor="أسد", action="زأر")` → تلوين أحمر/برتقالي (أسلوب الخطر)
- `Scene(actor="نهر", action="جريان")` → تموجات زرقاء (أسلوب الماء)

---

## المحور ب: تصغير demo_classification وإزالة تكرار الكود

### المشكلة

`demo_classification.jl` الأسطر 56-81 تعيد تنفيذ الـ forward pass كاملاً لقياس الخسارة، وهذا نسخة طبق الأصل من `train_classifier!`. لو تغير `train_classifier!` في المستقبل، سينفصل demo عنها.

### ب.1 — إضافة `compute_classification_loss`

في `HoloPRNN_Vision.jl`:

```julia
"""
    compute_classification_loss(clf::PhasNetClassifier, X, Y) -> Float64

تحسب متوسط مربع الخطأ (MSE) بين مخرجات التصنيف والهدف.
تستخدم negative phase (free run) ثم تحسب |z_out|² - target.
تصلح للمراقبة أثناء التدريب.
"""
```

### ب.2 — تعديل `train_classifier!`

قبل الخروج، تحسب الخسارة عبر `compute_classification_loss` وتطبعها أو تخزنها في حقل جديد `clf.loss_history`.

### ب.3 — تعديل `demo_classification.jl`

- إزالة الأسطر 56-81 (الـ forward pass المكرر)
- استدعاء `compute_classification_loss` بدلاً منه
- مراقبة الخسارة عبر `clf.loss_history` أو عبر `compute_classification_loss`

### ب.4 — إضافة `predict_batch`

```julia
predict_batch(clf::PhasNetClassifier, X::Vector{Vector{Float64}}) -> Vector{Int}
```

لتوحيد واجهة التنبؤ بدلاً من استدعاء `predict_classifier!` في حلقة.

---

## المحور ج: جسر SemanticScene → حقل بصري

### ج.1 — الرؤية العامة

```
┌─────────────────────────────────────────────────────────────────┐
│                       Mirnan (مرنان)                            │
│  ┌─────────────────┐     ┌──────────────────────────┐          │
│  │ SemanticScene   │────→│ SceneToVisionBridge     │          │
│  │ (مشهد دلالي)    │     │ (جسر المشهد → بصر)      │          │
│  └─────────────────┘     └───────────┬──────────────┘          │
│                                      │                         │
└──────────────────────────────────────┼─────────────────────────┘
                                       ▼
              ┌─────────────────────────────────────────┐
              │           HoloPRNN_Vision               │
              │  (حقل موجي بصري — عين مرنان)            │
              │  image_to_wavefield ← wavefield_to_image│
              │  simulate_wave_field! ← CHL ← Style    │
              └─────────────────────────────────────────┘
```

### ج.2 — هيكل `SceneToVisionBridge`

```julia
struct SceneToVisionBridge
    # مرجع للمشهد الدلالي (أو نسخة)
    scene::Union{Nothing, Any}   # SemanticScene أو dict
    # معاملات المحاكاة
    params::OscillatorParams
    # حقل البصريات الأساسي
    z_background::Union{Nothing, Matrix{ComplexF64}}
    # أسلوب افتراضي للتقديم
    default_style::Union{Nothing, StyleProfile}
    # دالة تحويل الكيان → لون/طور (تثبت لاحقاً)
    entity_color_map::Dict{String, Float64}
end
```

### ج.3 — `render_scene!`

```julia
"""
    render_scene!(bridge::SceneToVisionBridge, scene_kwargs...;
                  resolution=(64,64), steps=200) -> Matrix{RGB{N0f8}}

الخطوات:
1. إنشاء حقل z من Scene:
   - لكل كيان في المشهد، تحديد موضع (y,x) في الحقل
   - تعيين اللون عبر entity_color_map (طور)
   - تعيين السطوع عبر أهمية الكيان أو ثقته
2. إضافة خلفية (أسلوب افتراضي أو موجة صفرية)
3. تشغيل simulate_wave_field! لاستقرار المشهد
4. تحويل الناتج إلى صورة
5. إعادة الصورة
"""
```

### ج.4 — مبدأ الترميز البصري للمشهد

| عنصر المشهد | التمثيل البصري |
|-------------|----------------|
| `actor` | موضع مركزي، طور ثابت يمثل هوية الفاعل |
| `action` | تردد ذاتي (ω) عالٍ للفعل، نمط حركي في الـ Laplacian |
| `patient` | موضع قريب من الفاعل، طور متأثر بالاقتران |
| `effect_candidates` | بقع نشر حول الفاعل/المفعول بألوان مختلفة |
| `instrument` | خط اتجاهي بين الفاعل والمفعول |
| `place` | خلفية — حقل منتشر بنمط لوني محدد |
| `time_marker` | تأثير إضاءة (سعة الموجة): نهار ← سعة عالية، ليل ← سعة منخفضة |
| `affect_tone` | saturation في HSV: حيادي ← 0.5، إيجابي ← 1.0، سلبي ← 0.2 |

### ج.5 — `SemanticScene` من مرنان — نظرة سريعة

من `test_semantic_imagination.jl`، هيكل المشهد الحالي:

```julia
scene.actor           # "خالد"
scene.action          # "ضرب"
scene.patient         # "الكرة"
scene.instrument      # "bat"
scene.place           # "yard"
scene.time_marker     # "sunset"
scene.state_before    # "stable"
scene.state_after     # "moved and changed position"
scene.affect_tone     # "neutral"
scene.effect_candidates  # ["حركة", "ابتعاد"]
```

### ج.6 — خريطة التطور

| الخطوة | الوصف | المخرجات |
|--------|-------|----------|
| **ج.1** | تعريف `SceneToVisionBridge` + `render_scene!` الأساسي | ينتج صورة بدائية من المشهد |
| **ج.2** | إضافة `entity_color_map` ديناميكي (كل كيان يكتسب لوناً تلقائياً من phase vector) | لكل فاعل لون فريد |
| **ج.3** | ربط `effect_candidates` بمصفوفة اقتران لا-محلي (non-local coupling) | التأثيرات تنتشر في الحقل |
| **ج.4** | تطبيق `apply_style` مع `default_style` كمرشح جمالي نهائي | المشهد يُقدَّم بأسلوب موحد |
| **ج.5** | تدوير المشهد (زوايا كاميرا مختلفة) عبر تدوير الحقل المركب | منظورات متعددة |

### ج.7 — تكامل مع Mirnan

```julia
# مثال تخيلي للاستخدام النهائي:
scene = extract_semantic_scene(calculus, "Khalid hit the ball with bat in yard before sunset.")
bridge = SceneToVisionBridge(params=OscillatorParams())
image = render_scene!(bridge, scene; resolution=(64,64))
save("scene_output.png", image)
```

---

## الجدول الزمني المقترح

| الأسبوع | المحور | المهام |
|---------|--------|--------|
| 1 | **أ** | `StyleProfile` + `extract_style_profile` + `apply_style` + حفظ/تحميل |
| 2 | **أ** | تعديل `run_style_transfer` لاستخدام `extract_style_profile` + اختبارات أساسية |
| 3 | **ب** | `compute_classification_loss` + تعديل `train_classifier!` + تنظيف demo |
| 4 | **ج** | تعريف `SceneToVisionBridge` + `render_scene!` أساسي |
| 5 | **ج** | خريطة `entity_color_map` + تأثيرات المشهد (effect_candidates) |
| 6 | **ج** | ربط `apply_style` كمرشح نهائي + تكامل مع Mirnan |

---

## مبادئ عدم التغيير (ما لا نلمسه)

- ✅ `OscillatorParams` — يبقى كما هو
- ✅ `Stuart-Landau dynamics` في `simulate_wave_field!` — لا تغيير
- ✅ `image_to_wavefield` و `wavefield_to_image` — يبقيان دون تغيير
- ✅ `laplacian2d` — يبقى كما هو
- ✅ `PhasNetClassifier` و CHL — لا تغيير في المعادلات
- ✅ `crystallization` — لا تغيير

---

*الأولوية: المحور أ > المحور ب > المحور ج*
*المبدأ: كل ما نبنيه اليوم يكون جاهزاً لعين مرنان غداً*
