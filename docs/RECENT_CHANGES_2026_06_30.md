# تحديثات مرنان 2026-06-30

هذه الوثيقة تلخص محور الإصلاحات والتطويرات الأخيرة في مرنان: تثبيت مسارات الأسئلة، إزالة الأجوبة المعرفية الجاهزة، بناء طبقة `RelationFrame`، إضافة مفاتيح الكمية، وبداية حقل الحس الدلالي.

## الحالة العامة

- الاختبار الشامل قبل حقل الحس الدلالي: `Passed: 43 / 43`.
- `runtests.jl` القديم: `430 / 430`.
- بعد إضافة `test_semantic_imagination.jl`: اختبار الحقل الجديد منفردا مر `17 / 17`.
- لا توجد أية إضافات معرفية جاهزة عامة. الردود الجاهزة المسموحة محصورة في الذاكرة الاجتماعية القريبة.

## إصلاح مسارات الأسئلة

تم تثبيت مسارات الأسئلة الأساسية في طبقة التوليد والاستراتيجيات:

- `هل`: حماية اتجاه العلاقة ومنع قلب السبب والنتيجة.
- `لماذا`: اعتماد أدلة العلاقة والسبب بدلا من القوالب.
- `كيف`: تمرير السؤال عبر مخطط النية/الآلية.
- `ما معنى`: تحسين اختيار التعريف المطابق.
- `ما الفرق`: صياغة الفرق من تعريفين مضبوطين مع جملة تركيز.
- `السلام`: فصل معنى السلام المعرفي عن تحية السلام الاجتماعية.

الاختبارات الحارسة:

- `test_question_paths_from_hiwar.jl`
- `test_question_type_matrix.jl`
- `test_intent_response_planner.jl`
- `test_social_reply_memory_boundary.jl`
- `test_strict_no_templates.jl`

## حد الأجوبة الجاهزة

القاعدة المثبتة:

- الردود الاجتماعية القريبة مسموحة كذاكرة جاهزة: التحية، الاسم، الشكر، كيف الحال.
- الأجوبة المعرفية لا تؤخذ من قوالب جاهزة.
- المعرفة تأتي من الذاكرة، العلاقة، الحسبان الدلالي، الأدلة، أو مسارات الاستنباط.

هذا الحد موثق ومختبر في:

- `QUESTION_AND_SOCIAL_MEMORY_GUARDS_2026_06_29.md`
- `READY_ANSWER_AUDIT_2026_06_29.md`

## RelationFrame

أضيفت طبقة علاقات مستقلة داخل `al_istinbat.jl` لتمثيل العلاقات التي تكشفها مفاتيح اللغة.

البنية:

```julia
mutable struct RelationFrame
    left_terms::Vector{String}
    marker::String
    right_terms::Vector{String}
    relation_type::String
    polarity::Int
    direction::Int
    confidence::Float64
    source_sentence::String
end
```

الأنواع المدعومة في هذه المرحلة:

- `purpose`
- `conditional`
- `temporal`
- `spatial`
- `state`

الدوال الأساسية:

- `relation_type_for_marker(marker)`
- `extract_relation_frames(text)`
- `learn_relation_frames_from_text!(mem, text)`
- `select_relation_frame_attention(mem, prompt; active_paras=...)`
- `relation_frame_diagnostic(mem, prompt)`

المبدأ المعماري:

- الاستخراج مستقل.
- التخزين يضيف سجلات ولا يكسر سجلات `causal` القديمة.
- الاختيار داخلي فقط.
- التشخيص يعرض ما اختير دون توليد.

## Purpose answer

أضيف مسار غائي محدود جدا:

- `purpose_answer(mem, prompt)`
- يعمل فقط مع أسئلة الغاية مثل `لماذا`، `لأي غاية`، `من أجل ماذا`.
- يتجاهل أسئلة `هل`.
- يتطلب تطابقا وثقة كافيين.
- لا يعتمد على قوالب معرفة جاهزة.

ثم أضيفت مقارنة خارجية:

- `PurposeComparisonRecord`
- `compare_purpose_strategies(mem, generate_func, prompt)`

هذه المقارنة لا تغير التوليد، بل تقارن جواب `purpose_answer` بجواب `generate!`.

## RelationFrameStrategy

أضيفت استراتيجية مستقلة:

- `RelationFrameStrategy <: GenerationStrategy`
- الملف: `strategies/relation_frame_strategy.jl`
- تدعم حاليا `purpose` فقط.
- لا تدخل في `generate!` افتراضيا.

بوابة التشغيل:

```powershell
$env:MIRNAN_ENABLE_RELATION_FRAME_STRATEGY="1"
```

عند غياب البوابة أو ضبطها إلى `0` يبقى السلوك العام كما كان.

الموضع الحالي:

- الاستراتيجية موجودة في الكود.
- مضافة إلى قائمة الاستراتيجيات فقط خلف البوابة.
- لا تجيب عن `هل`.
- الشامل مر بعد إضافتها.

## QuantityFrame

أضيفت المرحلة الأولى لمفاتيح الكمية والعدد، استخراج فقط دون تخزين أو توليد.

البنية:

```julia
mutable struct QuantityFrame
    marker::String
    quantity_type::String
    target::String
    value::String
    polarity::Int
    confidence::Float64
end
```

الأنواع:

- `quantity_count`
- `quantity_measure`
- `quantity_comparison`
- `quantifier_scope`
- `vague_quantity`

الدوال:

- `relation_type_for_quantity_marker(marker)`
- `extract_quantity_frames(text)`

الحدود:

- لا تخزين في ذاكرة الاستنباط بعد.
- لا ربط بـ `generate!`.
- لا خلط مع `al_hisab`: ليست كل كمية مسألة حسابية.

## حقل الحس الدلالي

بدأت المرحلة الأولى من حقل الحس الدلالي كطبقة رصد فقط.

الملف:

- `src/physics/engines/semantic_imagination.jl`

البنية:

```julia
struct SemanticScene
    sentence::String
    actor::String
    action::String
    patient::String
    effect_candidates::Vector{String}
    confidence::Float64
    guidance_relation::String
    source::String
end
```

الدوال:

- `extract_semantic_scene(mem, sentence)`
- `scene_effect_terms(scene)`

مثال:

```julia
scene = extract_semantic_scene(mem, "ضرب خالد الكرة")
```

ينتج مشهدا دلاليا:

- `actor = "خالد"`
- `action = "ضرب"`
- `patient = "الكرة"`
- `effect_candidates = ["حركة", "ابتعاد", "تغير موضع"]`

مصادر الآثار:

- أولا: `al_hisban_al_dalali` عند وجود ذاكرة دلالية مناسبة.
- ثانيا: سوابق دلالية فيزيائية محافظة لأفعال واضحة مثل الضرب والدفع والكسر والإضاءة.

حدود المرحلة:

- لا ذاكرة جديدة.
- لا توليد.
- لا تغيير في `generate!`.
- لا ربط بالاستراتيجيات.

الاختبار:

- `test_semantic_imagination.jl`
- النتيجة المنفردة: `17 / 17`.

## إصلاح عرض العربية في الطرفية

أصبح منفذا الطرفية في مجنون ونسخة مرنان/باسل يمران مخرجات النص العربي عبر:

- `models/mirnan/scripts/fix-arabic.jl`

المواضع:

- `src/terminal_executor.jl`
- `models/mirnan/basil_agent/src/tools/terminal_executor.jl`

السلوك:

- لا يغير المخرجات التي لا تحتوي حروفا عربية.
- يعمل تلقائيا على المخرجات العربية.
- يستعمل `--force` افتراضيا لإضافة علامات RTL عند الحاجة.
- عند فشل السكربت لأي سبب، يرجع النص الأصلي دون كسر الأمر.

متغيرات التحكم:

```powershell
$env:MAJNON_FIX_ARABIC_TERMINAL="0"   # تعطيل المعالجة في منفذ مجنون
$env:MIRNAN_FIX_ARABIC_TERMINAL="0"   # تعطيل المعالجة في منفذ مرنان/باسل
$env:MAJNON_FIX_ARABIC_FORCE="0"      # عدم إجبار RTL
$env:MIRNAN_FIX_ARABIC_FORCE="0"      # عدم إجبار RTL
```

## الملفات الجديدة البارزة

- `src/physics/engines/semantic_imagination.jl`
- `src/physics/engines/strategies/relation_frame_strategy.jl`
- `test/test_relation_frame.jl`
- `test/test_semantic_imagination.jl`
- `test/test_question_type_matrix.jl`
- `test/test_question_paths_from_hiwar.jl`
- `test/test_social_reply_memory_boundary.jl`

## قواعد التطوير اللاحقة

1. كل طبقة جديدة تبدأ باستخراج مستقل.
2. ثم تخزين.
3. ثم اختيار داخلي.
4. ثم تشخيص.
5. ثم جواب مستقل.
6. ثم استراتيجية خلف بوابة.
7. ثم فقط بعد الشامل تدخل في السلوك العام.

هذه القاعدة هي التي استخدمت في:

- `RelationFrame`
- `QuantityFrame`
- بداية `SemanticScene`

## المرحلة التالية المقترحة

بعد نجاح الشامل بعد `test_semantic_imagination.jl`:

1. توسيع `SemanticScene` بحذر لاستخراج:
   - الفعل المركزي.
   - المتأثر.
   - الأثر المتوقع.
   - درجة المشهد.
2. إضافة مقارنة تشخيصية بين المشهد الدلالي و`al_hisban_al_dalali`.
3. عدم ربط الحقل بالتوليد قبل عدة مراحل اختبار.
