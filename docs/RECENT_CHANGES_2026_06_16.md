# سجل تغييرات 2026-06-16

صاحب أفكار مرنان: باسل يحيى عبدالله (basil Yahya Abdullah)

هذا سجل مختصر للتغييرات التي ركزت على جعل مرنان يراجع نفسه ويوجه محركاته بدل
أن يكتفي بالتوليد المباشر.

## المخيخ الداخلي

أضيف `src/physics/engines/mirnan_cerebellum.jl`.

الغرض:

- ملاحظة نوع السؤال قبل التوليد.
- اختيار سياسة تشغيل مثل `standard` أو `resonant`.
- تفعيل قياس المعنى عند الكلمات الملتبسة.
- تطبيق مضاعفات مؤقتة على أوزان التسجيل.
- التعلم من جودة النتيجة عبر مكافأة بسيطة.

## تراكب معنى الكلمة

أضيف `src/physics/engines/sense_superposition.jl`.

الغرض:

- تمثيل الكلمة متعددة المعنى كحالة احتمالات.
- قياس السياق لترجيح معنى معين.
- إرجاع شرح عربي يوضح هل انهار المعنى أم بقي مختلطا.

أمثلة حالية: `عين`، `قلب`، `لسان`، `جذر`، `شحن`.

## المراجعة الذاتية

أضيف `src/physics/engines/self_review.jl`.

الغرض:

- فحص اللغة والمنطق والاتساق والتكرار والالتزام بالسؤال.
- إنتاج `GenerationReview` يحتوي درجة وقائمة مشكلات وهدف إصلاح.
- تعلم أنماط الخلل المتكرر في `diagnostic_memory`.
- تعلم العلاج الذي حسن النتيجة فعلا في `treatment_memory`.

## نواة بيان المنطقية المحلية

أضيف `src/physics/engines/bayan_logic_kernel.jl`.

هذه ليست عودة إلى مسار `BayanLanguage` الكامل. هي نواة صغيرة داخل مرنان
تستعير فكرة التدقيق المنطقي البسيط لاكتشاف تناقضات مباشرة أثناء المراجعة.

## تكامل المولد

تغير `MirnanGenerator` بحيث يملك:

```julia
cerebellum::MirnanCerebellum
self_review::SelfReviewEngine
```

وصار مسار `generate!`:

```text
ملاحظة السؤال -> سياسة تشغيل -> توليد -> مراجعة -> إصلاح عند الحاجة -> تعلم العلاج
```

كما أضيف حفظ/تحميل حالة المخيخ والمراجعة الذاتية ضمن `runtime_learning.json`.

## الاختبارات

أضيفت أو حدثت:

- `test/test_mirnan_cerebellum.jl`
- `test/test_sense_superposition.jl`
- `test/test_self_review.jl`
- `test/run_all_tests.jl`

آخر تشغيل يدوي موثق قبل تحديث الوثائق:

```text
Passed: 21 / 21
All tests passed.
```

## التوثيق

أضيفت الوثيقة المركزية:

```text
docs/MIRNAN_SELF_REVIEW_AND_META_CONTROL.md
```

وتم تحديث:

- `docs/README.md`
- `docs/INDEX.md`
- `docs/ARCHITECTURE.md`
- `docs/DEVELOPMENT_STATUS.md`
- `docs/MIRNAN_TRAINING_MESSAGE_FLOW.md`
- `docs/MIRNAN_VIDEO_SCRIPT.md`
