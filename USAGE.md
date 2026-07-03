# دليل الاستخدام لنموذج مِرنان V9 (Julia)

يوضح هذا الدليل كيفية استخدام نموذج مِرنان V9 عملياً للتحليل والتوليد، مع أمثلة تطبيقية لكافة المستويات.

---

## 1. البدء السريع

تحميل المكتبة وتفعيل المشروع في بيئة Julia:
```julia
using Pkg
Pkg.activate(".")

using MirnanNew
```

---

## 2. التحليل الإحصائي (MirnanPipeline)

تُستخدم واجهة خط الأنابيب لتشغيل كافة الطبقات اللغوية والفيزيائية دفعة واحدة للتحليل:

```julia
# 1. تهيئة خط الأنابيب
pipeline = MirnanPipeline()

# 2. تشغيل التحليل على نص عربي
result = analyze_text(pipeline, "العلم نور والجهل ظلام دامس")

# 3. الوصول إلى المؤشرات والتحليلات
println("الكلمات المستخرجة: ", result.words)
println("متوسط الكثافة الدلالية: ", result.semantic_summary.avg_semantic_density)
println("الطاقة الكلية للنص: ", result.physics_summary.total_energy)

# 4. طباعة التقرير الشامل المنسق
report = get_analysis_report(result)
println(report)
```

---

## 3. التوليد والتعلم التكيفي (MirnanGenerator)

يمثل `MirnanGenerator` الواجهة الديناميكية لإنتاج النصوص وحل المسائل وحفظ المهارات.

### تهيئة المولد مع التحسينات الجديدة
يتم ربط التخزين المؤقت المستمر على القرص (`pv_cache.bin` و `mass_cache.bin`) تلقائياً لضمان بدء تشغيل فوري:
```julia
# تحميل مفردات ومصفوفة دلالية افتراضية
vocab = Dict("العلم" => 1, "ينير" => 2, "العقول" => 3)
K_sem = spzeros(3, 3)

# إنشاء المولد مع تفعيل المخيخ PID
gen = MirnanGenerator(vocab, K_sem; model_dir="model")
```

### تشغيل التوليد بالأنماط المختلفة
```julia
# التوليد القياسي (Standard)
res1 = generate!(gen, "ما هو العلم؟"; mode="standard", max_words=10)

# التوليد الإبداعي (Creative)
res2 = generate!(gen, "اكتب بيتاً شعرياً"; mode="creative", max_words=15)

# التوليد الحواري (Dialogue)
res3 = generate!(gen, "أهلاً بك"; mode="dialogue", max_words=8)
```

### إدارة التغذية الراجعة والتعلم المستمر
```julia
# تزويد النموذج بتغذية راجعة لتعزيز المسارات الناجحة
learn_from_feedback!(gen, "ما هو العلم؟", "العلم نور ينير العقول"; rating=1.0)

# حفظ التعلم وقواعد السببية المكتسبة إلى ملفات القرص
save_path = save_runtime_learning!(gen)
println("تم حفظ التعلم في: ", save_path)
```

---

## 4. فضاء العقل وقواعد ADL (Al-Aql Logic)

كتابة قواعد السببية والتصنيف بشكل مقروء في فضاء العقل:

```julia
# تجميع قواعد ADL
compile_adl!(gen.aql_space, """
مفتاح مادة < شيء { }
حديد : مادة { }
حرارة : فعل { الهدف: تمدد, مقياس: 5.0 }
قاعدة : حرارة > 0.5 -> تمدد
فكرة : حديد يتمدد بالحرارة
""")

# الاستعلام والتحقق من السببية
frames = infer_event!(gen.aql_space, "حديد", "حرارة", "تمدد")
for frame in frames
    println("النتيجة المستنتجة: ", frame.result)
end
```

---

## 5. ضبط وتعديل متحكم المخيخ PID

يتيح لك مرنان V9 التحكم في معاملات PID لتصحيح الأوزان الـ 26 ديناميكياً لتوليد نصوص متناسقة:

```julia
# تفعيل متحكم PID في المخيخ
gen.cerebellum.pid_enabled = true

# تعديل معاملات التناسب والنسب المستهدفة
gen.cerebellum.pid.Kp = 0.8
gen.cerebellum.pid.Ki = 0.1
gen.cerebellum.pid.Kd = 0.05
gen.cerebellum.pid.setpoint = 0.45  # قيمة الإنتروبيا المستهدفة

# إعادة تعيين تاريخ أخطاء الـ PID عند بدء جلسة جديدة
reset_pid!(gen.cerebellum.pid)
```

---

## 6. واجهة الأوامر التفاعلية (CLI)

يمكنك تشغيل واجهة الأوامر التفاعلية من الطرفية واستخدام الخيارات لتحديد نمط التوليد:
```bash
# التشغيل التفاعلي
julia --project=. cli.jl -i

# توليد نص مباشر بحد أقصى للكلمات وبوضع حواري
julia --project=. cli.jl --mode dialogue --max-words 15 "كيف حالك اليوم؟"
```
