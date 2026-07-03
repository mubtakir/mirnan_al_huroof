# صيغ بيانات التدريب في مرنان

هذه الوثيقة تحدد أين توضع بيانات التدريب الجديدة وكيف تكتب، خصوصا الكود البرمجي والمسائل الرياضية. الهدف أن يتعلم مرنان من المادة الخام مع بقاء الطبقات المتخصصة في موضعها الصحيح: النص للفيزياء اللغوية، الكود لـ `al_code` و`K_code`، والمسائل لـ `al_hisab`.

## أين أضع الملفات؟

المكان الافتراضي لكل ملفات التدريب هو:

```text
models/mirnan/data
```

يمكن وضع الملفات مباشرة داخله، أو داخل مجلدات فرعية. التدريب الحالي يقرأ المجلدات الفرعية أيضا. تقسيم مقترح:

```text
models/mirnan/data/corpus        نصوص عامة وفقرات
models/mirnan/data/exercises     أسئلة، أجوبة، تدريبات
models/mirnan/data/poetry        شعر ونصوص إيقاعية
models/mirnan/data/novels        سرد طويل
models/mirnan/data/toy_corpus    عينات صغيرة وموجهة
```

وللكود والمسائل يمكن إنشاء مجلدات واضحة مثل:

```text
models/mirnan/data/code_corpus
models/mirnan/data/math_corpus
```

ليست الأسماء إلزامية، لكنها تجعل المراجعة أسهل.

## النصوص والفقرات

التدريب الافتراضي يعمل على مستوى الفقرة. لذلك أفضل صيغة هي فقرة قصيرة أو متوسطة، ثم سطر فارغ، ثم فقرة أخرى.

قواعد النص:

1. الجملة التامة تنتهي بنقطة.
2. الجملة غير التامة أو الممتدة تنتهي بفاصلة.
3. الجملة المكملة أو السببية تنتهي بفاصلة منقوطة.
4. جمل القول تنتهي بنقطتين قبل الكلام المنقول.
5. علامات الترقيم تكون ملاصقة للكلمة السابقة.
6. افصل بين الفقرات بسطر فارغ.
7. الحوار يكتب بسطرين يفصل بينهما tab، أو بصيغة `سؤال:` و`جواب:`.

مثال:

```text
العلم نور؛ لأنه يفتح طريق الفهم.
يتعلم الإنسان بالملاحظة والتجربة والتكرار.

سؤال: كيف يثبت الفهم؟
جواب: يثبت الفهم بالمراجعة والعمل.
```

## الكود البرمجي

مرنان يتعلم الكود بطريقتين:

1. ملفات كود مباشرة داخل `models/mirnan/data` أو مجلداتها الفرعية.
2. كتل كود داخل ملفات نصية محاطة بسياج Markdown.

الصيغة المفضلة عند كتابة كود داخل ملف تدريبي:

````text
المطلوب: اكتب دالة تجمع عددين في Python.

```python
def add(a, b):
    return a + b
```
````

استخدم اسم اللغة بعد السياج كلما أمكن:

````text
```julia
function add(a, b)
    return a + b
end
```

```javascript
function add(a, b) {
  return a + b;
}
```
````

إرشادات مهمة للكود:

- اجعل المثال صغيرا ومكتمل المعنى.
- لا تكثر من ملفات ضخمة بلا وصف، إلا إذا كان الهدف تدريب `K_code` على بنية مشروع كامل.
- اكتب قبل الكود جملة قصيرة تصف الغرض، فهذا يساعد `al_code` على ربط الطلب بالشكل البرمجي.
- لا تخلط كودا غير مكتمل مع كود صحيح في المثال نفسه إلا إذا كان المثال عن التصحيح.

## إدخال أكواد مشروع مجنون

لإضافة أكواد المشروع نفسه إلى سجل تجارب مرنان، استخدم السكربت:

```powershell
cd C:\Users\allmy\Desktop\aaa\basil\majnon
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=. tools\seed_mirnan_code_experiences.jl
```

للمعاينة دون كتابة:

```powershell
cd C:\Users\allmy\Desktop\aaa\basil\majnon
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=. tools\seed_mirnan_code_experiences.jl --dry-run
```

هذا السكربت يجهز الكود في سجل الخبرات، ويصنع corpus لكل ملف، ويحفظ manifest للمراجعة.

## المسائل الرياضية

المسائل الرياضية موجهة إلى `al_hisab`. أفضل صيغة أن تكون المسألة، ثم الحل، ثم التحقق.

مثال بسيط:

```text
مسألة: 2 + 3
حل: 5
تحقق: 2 + 3 = 5
```

مثال أوضح:

```text
مسألة: إذا كان x = 4، فما قيمة 2x + 1؟
خطوات:
1. نعوض x بالقيمة 4.
2. نحسب 2 * 4 + 1.
3. الناتج 9.
جواب: 9
تحقق: 2 * 4 + 1 = 9
```

إرشادات للمسائل:

- اكتب الأرقام العربية أو الغربية، فكلاهما مدعوم.
- اجعل الجواب النهائي ظاهرا بكلمة `جواب:` أو `حل:`.
- أضف `تحقق:` عندما يكون ذلك ممكنا.
- لا تضع مسائل كثيرة جدا في سطر واحد؛ الأفضل مسألة قصيرة لكل فقرة.

## تشغيل التدريب

التدريب الافتراضي يقرأ بيانات `models/mirnan/data` على مستوى الفقرة:

```powershell
cd C:\Users\allmy\Desktop\aaa\basil\majnon
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=. models\mirnan\train.jl
```

يمكن التصريح بالمستوى يدويا عند الحاجة:

```powershell
$env:MIRNAN_SEGMENT_LEVEL="paragraph"
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=. models\mirnan\train.jl
```

## فحص سريع بعد التدريب

بعد التدريب، افحص جودة التوليد:

```powershell
cd C:\Users\allmy\Desktop\aaa\basil\majnon
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan scratch\check_mirnan_quality_probe.jl
```

وللاختبارات الشاملة:

```powershell
cd C:\Users\allmy\Desktop\aaa\basil\majnon
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\test\run_all_tests.jl
```

## المبدأ الحاكم

البيانات لا ينبغي أن تجعل طبقة واحدة تبتلع بقية الطبقات. في مرنان:

- `al_lisan` يرشد شكل الجملة ولا يستبدل الفيزياء.
- `al_hisban_al_dalali` يغذي الحساب الدلالي ولا يسيطر على التوليد.
- `al_code` يتعلم شكل الكود وسياق الطلب البرمجي.
- `al_hisab` يحفظ المسائل المحققة وخطواتها.
- الفيزياء اللغوية تظل هي المجال المركزي الذي تتعاون حوله هذه الذاكرات.

## تصدير نسخة مستقلة من مرنان

إذا أردت حفظ مرنان كنموذج متكامل مستقل عن بقية مشروع مجنون، استخدم أداة التصدير:

```powershell
cd C:\Users\allmy\Desktop\aaa\basil\majnon
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan tools\export_mirnan_standalone.jl
```

للمعاينة دون كتابة:

```powershell
cd C:\Users\allmy\Desktop\aaa\basil\majnon
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan tools\export_mirnan_standalone.jl --dry-run
```

النسخة الناتجة تحفظ الشكل النسبي نفسه:

```text
models/mirnan
.agent/mirnan
.agent_workspace/.agent/mirnan
tools/seed_mirnan_code_experiences.jl
scratch/check_mirnan_quality_probe.jl
```

بهذا يستطيع `train.jl` داخل النسخة قراءة بيانات مرنان وملفات الخبرة الخارجية دون الحاجة إلى مشروع مجنون كاملا.
