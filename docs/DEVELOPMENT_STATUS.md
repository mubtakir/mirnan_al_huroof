# حالة تطوير مرنان الجديد

تاريخ آخر تحديث: 2026-06-16

صاحب أفكار مرنان: باسل يحيى عبدالله (basil Yahya Abdullah)

هذه الوثيقة تلخص الحالة الحالية لمشروع `mirnan_new` بعد بدء نقل أهم أفكار
النموذج القديم `mirnan_julia` إلى البنية الجديدة. هدفها أن تكون خريطة عمل
سريعة: ما الذي أصبح موجودا، ما الذي تغير، وكيف نختبره قبل البناء فوقه.

## ملخص تنفيذي

مرنان الجديد هو نموذج لغوي فيزيائي لا يعتمد على شبكة عصبية تقليدية. نواة
النموذج الحالية أصبحت تعمل على فضاء طور كبير مأخوذ من النموذج القديم:

- `PHASE_DIM = 9958`
- `TOTAL_DIM = 10000`
- `ENHANCED_DIM = 27` ما زال محفوظا كمسار تشخيص/قياس للحروف المحسنة

المسار الرسمي الآن:

```text
حرف -> متجه طور 9958D
كلمة -> compute_word_phase_vector(word) 9958D
كلمة موسعة -> compute_extended_phase_vector(word) 10000D
توليد -> MirnanGenerator + scoring فيزيائي/لغوي
توجيه -> MirnanCerebellum يضبط نمط التشغيل وأوزانه
تراكب معنى -> SenseSuperposition عند الكلمات الملتبسة
مراجعة -> SelfReviewEngine يفحص الناتج ويقترح العلاج
تعلم علاجي -> ذاكرة تشخيص وذاكرة علاج لما نجح في الإصلاح
```

بهذا لم يعد مرنان مجرد مجموعة محركات متجاورة. صار لديه مسار تشغيل مغلق نسبيا:
يرى السؤال، يختار سياسة، يولد، يراجع الناتج، يحاول إصلاحه عند الحاجة، ثم يتعلم
أي علاج أفاد في حالة مشابهة.

## نقل الأبعاد من النموذج القديم

تمت إعادة أبعاد مرنان القديمة إلى:

```julia
PHASE_DIM = 9958
ROOT_DIMS = 8
EXTRA_DIMS = 6
SYNTAX_DIMS = 6
SEMANTIC_DIMS = 16
PRAGMATIC_DIMS = 6
TOTAL_DIM = 10000
```

المعنى العملي لهذه الأبعاد:

| الجزء | الأبعاد | الغرض |
| --- | ---: | --- |
| فضاء الطور الأساسي | 9958 | بصمة الكلمة الموجية من متجهات الحروف |
| الجذر | 8 | أثر الجذر العربي الخفيف |
| DDE/إضافي | 6 | أبعاد ديناميكية مساعدة |
| النحو | 6 | إشارات نحوية مختصرة |
| الدلالة | 16 | إشارات دلالية/فلسفية |
| القصد/السياق | 6 | أبعاد براغماتية |
| المجموع | 10000 | متجه الكلمة الكامل |

الدوال المركزية:

```julia
compute_word_phase_vector("سلام")      # 9958D
compute_extended_phase_vector("سلام")  # 10000D
compute_word_enhanced_vector("سلام")   # 27D
```

## ملفات النواة التي يجب معرفتها

| الملف | الدور |
| --- | --- |
| `src/physics/core/constants.jl` | الثوابت والأبعاد المركزية |
| `src/physics/core/letter_db_original.jl` | قاعدة الحروف القديمة ذات فضاء 9958D |
| `src/physics/core/letter_db.jl` | مسار المعادلات الحرفية 27D |
| `src/physics/core/word_physics.jl` | تحويل الكلمة إلى متجه طور/كتلة/تردد |
| `src/physics/core/clifford_math.jl` | جبر كليفورد للكلمات والحروف |
| `src/physics/engines/generator.jl` | المولد المركزي ومسار scoring |
| `src/physics/engines/mirnan_cerebellum.jl` | متحكم خفيف يوجه أنماط التوليد ويتعلم من النتائج |
| `src/physics/engines/sense_superposition.jl` | تراكب معاني الكلمة وانهيار المعنى بالسياق |
| `src/physics/engines/self_review.jl` | مراجعة ذاتية للناتج وذاكرة تشخيص/علاج |
| `src/physics/engines/bayan_logic_kernel.jl` | تدقيق منطقي محلي صغير مستوحى من بيان |
| `src/physics/Physics.jl` | تجميع كل المحركات وإعادة التصدير |

## تحسين مسار التوليد

كان المولد قد يرجع نصا فارغا عندما لا توجد مصفوفة `K_sem`، أو عندما تكون
المفردات صغيرة جدا. تم تحسين fallback بحيث يستخدم scoring فيزيائيا وسياقيا
بدلا من التوقف.

الإشارات الحالية داخل `_score`:

- `align`: توافق المرشح مع آخر متجه سياقي.
- `prompt_align`: توافق المرشح مع prompt.
- `gravity`: جاذبية الكلمة مع كلمات السياق.
- `syntax`: توافق نحوي مختصر مع الكلمة السابقة.
- `density_resonance`: رنين المرشح في مصفوفة الكثافة.
- `dccf`: تعزيز السياق الديناميكي.
- `ppm`: أثر prompt field.
- `root_affinity`: تقارب الجذر العربي مع السياق.
- `surface_affinity`: تقاطع الحروف/السطح اللفظي مع السياق.

أوزان الإشارات موجودة في `DEFAULT_WEIGHTS` داخل:

```text
src/physics/engines/generator.jl
```

### طبقة التوجيه والمراجعة

أضيف حول التوليد مسار جديد:

1. `MirnanCerebellum` يقرأ prompt ويختار سياسة تشغيل مثل `standard` أو
   `resonant` أو مسار رياضي/برمجي، ويطبق مضاعفات مؤقتة على أوزان التسجيل.
2. `SenseSuperposition` يفعل عند الأسئلة عن معنى كلمة ملتبسة مثل `عين`، حيث
   تبقى الكلمة في تراكب معان حتى يرجح السياق أحدها.
3. `SelfReviewEngine` يراجع الناتج بدرجات للغة والمنطق والاتساق والتكرار
   وتوافق prompt.
4. ذاكرة التشخيص تتوقع نوع الخلل المتكرر في أسئلة مشابهة.
5. ذاكرة العلاج تتعلم أي إصلاح رفع الدرجة فعلا: زيادة التنوع، تقوية الالتزام
   بالسؤال، الرجوع إلى `standard_mode`، أو استعمال `fallback`.

الشرح التفصيلي في:

```text
docs/MIRNAN_SELF_REVIEW_AND_META_CONTROL.md
```

مثال تحقق سريع:

```julia
vocab = Dict(w => i for (i, w) in enumerate([
    "العلم", "نور", "تعلم", "معلم", "العالم", "ضياء", "كتاب", "قلم",
]))

gen = MirnanNew.Physics.MirnanGenerator(vocab)
MirnanNew.Physics.generate!(gen, "العلم نور"; mode="standard", max_words=3)
```

النتيجة المتوقعة يجب أن تكون غير فارغة وقريبة من السياق، مثل:

```text
تعلم معلم العالم
```

## سياسة الاختبارات الحالية

يوجد مساران للاختبار:

### 1. الاختبار الرسمي

```powershell
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=. test\runtests.jl
```

النتيجة الحالية:

```text
83 / 83 passed
```

### 2. الاختبارات التشخيصية الواسعة

```powershell
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=. test\run_all_tests.jl
```

النتيجة الحالية:

```text
21 / 21 passed
All tests passed.
```

ملاحظة مهمة: `test/run_all_tests.jl` يشغل كل اختبار في عملية Julia مستقلة.
هذا مقصود لأن بعض الاختبارات القديمة تعرف وحدات محلية مثل `CliffordLocal`
و`LetterDBLocal`، وتشغيلها كلها داخل `Main` واحد يسبب تصادم أسماء كاذبا.

## الاختبارات الجديدة أو المعدلة

| الملف | الغرض |
| --- | --- |
| `test/run_all_tests.jl` | runner معزول لكل اختبار |
| `test/test_generator_fallback.jl` | يثبت أن fallback يولد نصا سياقيا عند غياب `K_sem` |
| `test/test_contextual_learning.jl` | يثبت تعلم التراكيب ونية السؤال والمجاز والأثر والتصحيح من feedback |
| `test/test_mirnan_cerebellum.jl` | يثبت توجيه المخيخ الداخلي وتعلمه من نتيجة التوليد |
| `test/test_sense_superposition.jl` | يثبت تراكب المعنى وانهياره بالسياق |
| `test/test_self_review.jl` | يثبت المراجعة الذاتية، منطق بيان المحلي، ذاكرة التشخيص، وذاكرة العلاج |
| اختبارات enhanced القديمة | أصبحت تحمل `MirnanNew.Physics.WordPhysics` بدلا من include مباشر |
| `test/test_phase2.jl` | أصبح يستخدم `MirnanNew.Physics` |
| `test/test_semantic_gravity.jl` | تم إصلاح خلط أنواع `Multivector22` المحلية |

## قواعد تطوير مهمة

1. لا تجعل ملفات `src/physics/core/*.jl` تعمل كملفات مستقلة إذا كانت مصممة
   كجزء من `Physics.jl`. الاختبارات يجب أن تحمل الحزمة من الباب الصحيح.
2. حافظ على المسارين:
   - 10000D هو المسار الرسمي للنموذج.
   - 27D هو مسار قياس/تشخيص للحروف والمعادلات.
3. عند إضافة محرك جديد، اربطه من `src/physics/Physics.jl` ثم اختبره من
   `MirnanNew.Physics`.
4. عند تعديل التوليد، أضف اختبارا يثبت السلوك، لا مجرد أن الدالة تعيد `String`.
5. إذا ظهر فشل في `run_all_tests.jl` لكن الاختبار ينجح منفردا، افحص تصادم
   الوحدات المحلية قبل اتهام النواة.

## حالة المشروع الآن

ما أصبح مستقرا:

- أبعاد 9958D/10000D.
- واجهة `compute_word_phase_vector`.
- واجهة `compute_extended_phase_vector`.
- مسار 27D المحسن.
- محركات Phase 2.
- اختبارات كليفورد التشخيصية.
- fallback التوليد عند غياب `K_sem`.
- طبقة `ContextualLearning` للتراكيب المركبة ونوع السؤال والمجاز والأثر وfeedback.
- `MirnanCerebellum` كمتحكم داخلي خفيف لا يعتمد على نموذج لغوي خارجي.
- `SenseSuperposition` لفكرة الكلمة كموجة معان متعددة ينهار معناها بالسياق.
- `SelfReviewEngine` لمراجعة الناتج قبل اعتماده.
- `BayanLogicKernel` كتدقيق منطقي محلي لا يعتمد على مجلد `BayanLanguage` الكامل.
- ذاكرة تشخيص وذاكرة علاج تحفظان ما تكرر من خلل وما نجح من إصلاح.
- runner معزول للاختبارات الواسعة.

ما يحتاج تطويرا لاحقا:

- تحسين جودة الجمل لا مجرد اختيار كلمات قريبة.
- معايرة أوزان `DEFAULT_WEIGHTS` على corpus حقيقي.
- إدخال إشارات صرفية أعمق من `_extract_root_light`.
- ربط أفضل بين `K_sem` و`root_affinity` عند وجود مصفوفة تدريب.
- توحيد ترميز بعض ملفات التوثيق القديمة التي تظهر مشوهة في الطرفية.
- إضافة اختبارات جودة للتوليد تقيس الاتساق والتنوع وعدم التكرار.
- توسيع معجم تراكب المعاني من أمثلة عربية محدودة إلى قاموس واسع.
- جعل تقرير المراجعة يشرح مساهمة المحركات في كل قرار توليد بصورة أعمق.
- نقل ذاكرة العلاج لاحقا من JSON عملي إلى مخزن معرفي قابل للنمو الكبير.

## أوامر مفيدة

فحص الأبعاد:

```powershell
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=. -e "include(joinpath(pwd(), \"src\", \"MirnanNew.jl\")); using .MirnanNew; println(MirnanNew.Physics.Constants.PHASE_DIM); println(MirnanNew.Physics.Constants.TOTAL_DIM); println(length(MirnanNew.Physics.compute_word_phase_vector(\"سلام\"))); println(length(MirnanNew.Physics.compute_extended_phase_vector(\"سلام\")))"
```

تشغيل الاختبارات الرسمية:

```powershell
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=. test\runtests.jl
```

تشغيل الاختبارات الواسعة:

```powershell
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=. test\run_all_tests.jl
```
