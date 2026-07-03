# ذاكرات النمط والحسبان الدلالي في مرنان

تاريخ التحديث: 2026-06-18

هذه الوثيقة تلخص الطبقات التي أضيفت فوق مصفوفات الاقتران الفيزيائية في مرنان. الفكرة ليست استبدال `K_sem` أو `K_syn` أو `K_causal`، بل إضافة ذاكرات منظمة تساعد المولد على اختيار هيئة الجواب قبل اختيار الكلمات.

## الفكرة العامة

مرنان كان يعرف قرب الكلمات وجاذبيتها ورنينها، لكنه قد ينزلق أحيانا إلى مخرجات تشبه قائمة كلمات. لذلك أضيفت طبقات تحفظ أنماطا أعلى:

| الطبقة | وظيفتها | ملف النموذج |
| --- | --- | --- |
| `al_lisan` | ذاكرة الجملة: القوالب، الخانات، ترتيب القول، الحوار، وأدوات الربط | `model/al_lisan.json` |
| `al_code` | ذاكرة الكود: قوالب الدوال، الحلقات، الشروط، الاستدعاءات، والكتل البرمجية | `model/al_code.json` |
| `al_tadbir` | ذاكرة التدبير: خطوات العمل، التشخيص، التنفيذ، التحقق، والتقرير | `model/al_tadbir.json` |
| `al_hisab` | ذاكرة الحساب المتحقق: تعبيرات رياضية ونتائج قابلة للفحص | `model/al_hisab.json` |
| `al_ta3rif` | ذاكرة التعريفات والعلاقات العامة: ماهية، معنى، صفة، ارتباط، جزء/نوع | `model/al_ta3rif.json` |
| `al_hisban_al_dalali` | الحسبان الدلالي بجبر كليفورد: قياس حركة المعنى بين السؤال والجواب | `model/al_hisban_al_dalali.json` |

يبقى `al_aql` طبقة العقل المنطقي والسببي، بينما هذه الطبقات تصف هيئة الكلام والعمل والحساب وحركة المعنى.

## موضعها في التدريب

أثناء `train.jl` يتعلم مرنان المصفوفات الأساسية أولا، ثم يبني الذاكرات النمطية من النصوص نفسها:

```text
corpus
  -> vocab
  -> K_sem / K_syn / K_dialogue / K_causal
  -> al_lisan
  -> al_code
  -> al_tadbir
  -> al_hisab
  -> al_ta3rif
  -> al_hisban_al_dalali
  -> model/*.json
```

هذه الملفات مستقلة، لذلك لا تغير صيغة مصفوفات `K` ولا تكسر النماذج القديمة. إذا لم يوجد ملف ذاكرة معين، يعمل المولد بالمسار القديم مع فقدان تلك الطبقة فقط.

## تجهيز كود مجنون كخبرات تدريبية

لإضافة كود مشروع مجنون نفسه إلى سجل خبرات مرنان، استعمل السكربت:

```powershell
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=. tools\seed_mirnan_code_experiences.jl
```

يفحص السكربت كود المشروع ووحداته وملحقاته، ويستبعد النسخ المجمعة والمكتبات الخارجية الضخمة مثل `dist/` و`build/` و`llama.cpp/` ونسخة Blender داخل `studio/5.1`. ثم يكتب:

```text
.agent/mirnan/training_corpus.txt
.agent/mirnan/experiences.jsonl
models/mirnan/data/toy_corpus/majnon_code_corpus/
```

كل ملف يضاف كبنية خبرة فيها وصف عربي وقالب كود fenced، وهذا مناسب لـ `al_code` و`al_tadbir` و`al_hisban_al_dalali`. ويمكن تجربة المعاينة دون كتابة:

```powershell
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=. tools\seed_mirnan_code_experiences.jl --dry-run
```

## موضعها في التوليد

عند إنشاء `MirnanGenerator` يحاول تحميل ملفات الذاكرة من مجلد النموذج. ثم يسير التوليد تقريبا هكذا:

```text
prompt
  -> كشف النمط: لغة / كود / خطة / حساب / سؤال دلالي
  -> استشارة al_ta3rif إذا كان السؤال يطلب تعريفا أو معنى أو علاقة عامة
  -> استشارة al_hisban_al_dalali لحركة المعنى
  -> اختيار قالب مناسب من al_lisan أو al_code أو al_tadbir أو al_hisab
  -> ملء الخانات من K_sem / K_syn / K_causal مع توجيه دلالي
  -> مراجعة ذاتية
  -> إصلاح إن ظهرت مشكلة بنيوية أو دلالية
```

بهذا لا يختار مرنان الكلمات وحدها؛ بل يختار أيضا شكل الجملة أو الخطة أو العبارة البرمجية أو الحل الرياضي.

## `al_lisan`: ذاكرة اللسان

`al_lisan` يحفظ القالب المجرد للجملة وعدد تكراره وأمثلة قصيرة والكلمات التي ملأت كل خانة. مثال مبسط:

```json
{
  "shape": "NOUN VERB OBJECT",
  "count": 1,
  "examples": ["العلم ينير العقول"],
  "slots": {
    "subject": ["العلم"],
    "verb": ["ينير"],
    "object": ["العقول"]
  }
}
```

لا تشترط هذه الطبقة وجود فعل في كل جملة؛ فهي تقبل الجمل الاسمية، المضاف والمضاف إليه، شبه الجملة، السؤال والجواب، أدوات الشرط، القول، والاستثناء. الكلمات العربية الخاصة مثل `لطالما` و`ريثما` و`بينما` و`حبذا` موضوعة في ملف قابل للتوسعة:

```text
config/al_lisan_markers.json
```

## `al_code`: ذاكرة الكود

`al_code` تحفظ هيئة المسألة البرمجية قبل تفاصيلها. أمثلة الأنماط:

- تعريف دالة.
- شرط.
- حلقة.
- إسناد.
- استدعاء.
- كتلة `try/catch`.
- مخطط خطوات داخل تعليق أو نص تقني.

هذه الطبقة تمنع جواب الكود من أن يصبح كلمات تقنية متجاورة، وتدفعه إلى بنية قابلة للقراءة.

## `al_tadbir`: ذاكرة التخطيط

`al_tadbir` تحفظ تسلسل العمل في المسائل العملية:

```text
observe -> diagnose -> execute -> verify -> report
```

تستخدم عند طلب خطة، إصلاح، تحليل خطأ، أو إجراء متدرج. وهي مناسبة جدا لطلبات مثل:

```text
كيف أصلح هذا الخطأ؟
ضع خطة تدريب.
ما خطوات اختبار هذه الوحدة؟
```

## `al_hisab`: ذاكرة الحساب

`al_hisab` تحفظ مسائل حسابية ورمزية صغيرة مع نتيجة أو صيغة متحققة. الهدف ليس أن تكون بديلا عن نظام رياضيات رمزي كامل، بل أن تمنع المولد من اختلاق نتيجة عندما يملك شكلا حسابيا واضحا.

تستخدم هذه الطبقة أولا في مطالب الرياضيات الواضحة، ثم يعود المولد إلى المسار العام إذا لم يجد نمطا مناسبا.

## `al_ta3rif`: ذاكرة التعريفات والعلاقات العامة

`al_ta3rif` تحفظ المعرفة التعريفية البسيطة التي يحتاجها المستخدم في أسئلة مثل `ما هو X؟` و`ما معنى X؟` و`اشرح X`. وهي ليست بديلا عن `al_aql`: الفرق أن `al_aql` عقل سببي/منطقي داخلي، أما `al_ta3rif` فهي طبقة عرض ومعرفة عامة تحفظ ما يصلح أن يقال مباشرة للمستخدم.

تتعلم الطبقة من صيغ مثل:

```text
العلم هو معرفة منظمة تكشف أسباب الأشياء.
العدل هو إعطاء كل ذي حق حقه.
العرش يرتبط ب الملك.
الشجاعة صفة الثبات عند الخوف.
Knowledge is organized understanding.
```

وتحفظ لكل موضوع:

- تعريفات مختصرة.
- علاقات عامة مثل جزء من، نوع من، يرتبط ب، يدل على.
- صفات عامة.
- أمثلة قصيرة من corpus.

عند التوليد يستشيرها مرنان قبل عرض إجابة `al_aql`، حتى لا تظهر للمستخدم سجلات داخلية مثل `كيان مسجل` أو `__classes`. وإذا لم تجد الطبقة معرفة مفيدة، يواصل المولد مساره الفيزيائي المعتاد.

## `al_hisban_al_dalali`: الحسبان الدلالي

هذه الطبقة هي أقرب الطبقات إلى فكرة اللغة الداخلية في مرنان. فهي تحسب تمثيل السؤال والجواب بجبر كليفورد وفضاء الحروف، ثم تحفظ الحركة بينهما:

```text
source_signature -> semantic_transform -> target_signature
```

الحركة قد تكون مثلا:

| الحركة | المعنى العملي |
| --- | --- |
| `definition` | السؤال يطلب ماهية أو تعريفا |
| `method` | السؤال يطلب كيف أو آلية |
| `cause` | السؤال يطلب لماذا أو سبب |
| `comparison` | السؤال يطلب فرقا أو مقارنة |
| `result` | السؤال يطلب أثرا أو نتيجة |

ثم تنتج الطبقة `semantic_guidance` فيه:

- نوع الحركة.
- ثقة تقريبية.
- كلمات هدف مرشحة.
- خطة جواب دلالية.
- فرق scalar وbivector بين السؤال والجواب.

هذه الإشارة لا تلغي فيزياء مرنان؛ بل ترجح الكلمات والقوالب التي تسير في الاتجاه الدلالي المطلوب.

## التكامل مع المراجعة الذاتية

المراجعة الذاتية تقرأ الآن إشارة `semantic_guidance`. إذا كان السؤال يطلب آلية مثلا، لكن الجواب خرج كقائمة أسماء أو كتعريف ساكن، تسجل المراجعة مشكلة من نوع:

```text
semantic_movement_mismatch
```

ثم تعطي هدف إصلاح:

```text
semantic_guidance
```

وبعدها يستخدم المولد هذه الإشارة في محاولة إصلاح الجواب.

## التقرير الموحد

يمكن فحص حالة الذاكرات عبر:

```julia
gen = MirnanGenerator(vocab; model_dir="model")
summary = pattern_memory_summary(gen)
```

أو ضمن التقرير الفيزيائي:

```julia
report = get_physics_report(gen)
report["pattern_memories"]
```

يعرض التقرير هل الذاكرات محملة وعدد الأنماط أو السجلات الأساسية فيها.

## الملفات الرئيسية

| الملف | الدور |
| --- | --- |
| `src/physics/engines/al_lisan.jl` | تدريب وحفظ وتحميل ذاكرة اللسان |
| `src/physics/engines/al_code.jl` | تدريب وحفظ وتحميل ذاكرة الكود |
| `src/physics/engines/al_tadbir.jl` | تدريب وحفظ وتحميل ذاكرة التدبير |
| `src/physics/engines/al_hisab.jl` | تدريب وحفظ وتحميل ذاكرة الحساب |
| `src/physics/engines/al_ta3rif.jl` | تدريب وحفظ وتحميل ذاكرة التعريفات والعلاقات العامة |
| `src/physics/engines/al_hisban_al_dalali.jl` | تدريب وحفظ وتحميل الحسبان الدلالي |
| `src/physics/engines/generator.jl` | تحميل الذاكرات واستعمالها أثناء التوليد |
| `src/physics/engines/self_review.jl` | مراجعة الحركة الدلالية وإصلاحها |
| `train.jl` | بناء الذاكرات وحفظها أثناء التدريب |

## الاختبارات

الاختبارات الصغيرة المباشرة:

```powershell
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\test\test_al_lisan.jl
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\test\test_al_code.jl
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\test\test_al_tadbir.jl
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\test\test_al_hisab.jl
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\test\test_al_ta3rif.jl
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\test\test_al_hisban_al_dalali.jl
```

اختبارات التكامل والمراجعة:

```powershell
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\test\test_pattern_memory_integration.jl
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\test\test_self_review.jl
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\test\run_all_tests.jl
```

آخر حالة موثقة قبل هذا التحديث: جميع اختبارات مرنان الشاملة نجحت.

## مبدأ التصميم

```text
al_aql              = العقل: منطق، سبب، علاقة، حكم.
al_lisan            = اللسان: قالب، ترتيب، أسلوب، حوار.
al_code             = الكود: بنية برمجية قابلة للإكمال.
al_tadbir           = التدبير: خطوات عمل ومراجعة.
al_hisab            = الحساب: نتيجة أو صيغة متحققة.
al_ta3rif           = التعريف: ماهية وصفة وعلاقة عامة قابلة للعرض.
al_hisban_al_dalali = الحسبان: حركة المعنى في فضاء كليفورد.
```

كل طبقة تساعد أختها. فإذا كانت المصفوفات الفيزيائية تقول "هذه الكلمات قريبة"، فإن ذاكرة اللسان تقول "كيف ترتبها في جملة"، والحسبان الدلالي يقول "في أي اتجاه يجب أن يتحرك الجواب".
