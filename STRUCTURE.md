# الهيكلية النهائية لمشروع مِرنان V9

## المجلدات والملفات

```text
mirnan/
├── config.yaml                # إعدادات المشروع المركزية
├── Project.toml               # تبعيات المشروع لـ Julia
├── Manifest.toml              # شجرة التبعيات الدقيقة
├── CLI.jl                     # واجهة الأوامر التفاعلية (Command Line Interface)
├── train.jl                   # سكربت تدريب النموذج وبناء المصفوفات K
├── api_server.jl              # خادم API المحلي
├── docs/                      # التوثيق التفصيلي للمشروع
├── model/                     # مجلد حفظ النموذج والمصفوفات المدربة
│   ├── vocab.json             # معجم الكلمات
│   ├── K_sem.npz              # مصفوفة الاقتران الدلالي
│   ├── K_syn.npz              # مصفوفة الاقتران النحوي
│   ├── pv_cache.bin           # التخزين المؤقت المستمر للمتجهات (Mmap)
│   └── mass_cache.bin         # التخزين المؤقت المستمر للكتل (Mmap)
├── config/                    # مجلد إعدادات الأنماط
│   ├── al_lisan_markers.json  # علامات اللسان
│   ├── base.yaml              # إعدادات التوليد الأساسية
│   ├── creative.yaml          # إعدادات التوليد الإبداعي
│   ├── dialogue.yaml          # إعدادات التوليد الحواري
│   ├── grammar.yaml           # إعدادات النحو
│   └── physics.yaml           # إعدادات الفيزياء
├── src/                       # الكود المصدري للمشروع
│   ├── MirnanNew.jl           # الوحدة البرمجية الرئيسية (Main Entrypoint)
│   ├── preprocessing/         # طبقة المعالجة المسبقة (5 مراحل)
│   │   ├── Preprocessing.jl   # خط الأنابيب والفلترة
│   │   ├── text_extractor.jl  # استخراج النصوص
│   │   ├── language_filter.jl # تصفية اللغة
│   │   ├── normalizer.jl      # تطبيع الأحرف والتشكيل
│   │   ├── segmenter.jl       # تقطيع الجمل
│   │   └── quality_filter.jl  # تصفية جودة الكلمات
│   ├── semantics/             # طبقة الدلالة (stubs)
│   │   └── Semantics.jl
│   ├── grammar/               # طبقة النحو
│   │   └── Grammar.jl         # محددات النحو 6D
│   ├── sio/                   # الذكاء التوليفي والتنسيق
│   │   └── SIO.jl
│   ├── api/                   # خادم API
│   │   └── server.jl
│   ├── utils/                 # الأدوات المساعدة
│   │   └── DataLoader.jl
│   ├── integration/           # الربط المتكامل
│   │   └── Integration.jl
│   └── physics/               # المحركات الفيزيائية (القلب النابض للنموذج)
│       ├── Physics.jl         # نقطة التجميع الرئيسية وتصدير الدوال
│       ├── constants.jl       # الثوابت الفيزيائية (THESE)
│       ├── al_aql/            # فضاء العقل وقواعد الحكم والسببية الاستدلالية
│       │   └── al_aql.jl
│       ├── core/              # النواة الأساسية
│       │   ├── constants.jl   # ثوابت النواة
│       │   ├── letter_db.jl   # قاعدة متجهات الحروف (SEED)
│       │   ├── word_physics.jl# حساب الكتلة والتردد والطاقة للكلمات
│       │   └── clifford_math.jl # جبر كليفورد الهندسي (Multivector22)
│       ├── groups/            # مجموعات المحركات الـ 10 (تنظيم التبعيات)
│       │   ├── core_group.jl
│       │   ├── basic_physics_group.jl
│       │   ├── quantum_group.jl
│       │   ├── memory_group.jl
│       │   ├── morphology_group.jl
│       │   ├── arabic_group.jl
│       │   ├── wave_group.jl
│       │   ├── infra_group.jl
│       │   ├── prnn_group.jl
│       │   └── generation_group.jl
│       └── engines/           # المحركات الفيزيائية والمعرفية
│           ├── gravity.jl     # الجاذبية نيوتن في فضاء الطور
│           ├── oscillator.jl  # مذبذبات كوراموتو المحدثة
│           ├── resonant_chain.jl # رنين LC وسلسلة زيتا
│           ├── mirnan_cerebellum.jl # متحكم المخيخ مع PID التكيفي
│           ├── self_review.jl # المراجعة الذاتية للناتج
│           ├── generator.jl   # موجه التوليد الاستراتيجي
│           ├── strategies/    # استراتيجيات التوليد الـ 15 المفصّلة
│           │   ├── base.jl    # الهيكل التجريدي للاستراتيجية
│           │   ├── shared_scoring.jl # حساب درجات التوليد الموحدة (26 وزناً)
│           │   ├── finisher.jl # التحقق والإنهاء اللغوي
│           │   ├── resonant_strategy.jl # التوليد بالرنين الكلي
│           │   ├── dialogue_strategy.jl # التوليد الحواري
│           │   ├── yesno_relations.jl # استراتيجية إجابات نعم/لا
│           │   ├── evidence_relations.jl # استراتيجية الاستدلال بالبينات
│           │   ├── definition_strategy.jl # استراتيجية التعريفات والشرح
│           │   ├── code_strategy.jl # استراتيجية توليد الكود البرمجي
│           │   ├── math_strategy.jl # استراتيجية العمليات الرياضية
│           │   ├── relation_strategy.jl # استراتيجية العلاقات العامة
│           │   ├── lexical_strategy.jl # استراتيجية معجم الحروف
│           │   ├── difference_and_gate.jl # استراتيجية الفروق والإنكار
│           │   ├── nisba_relations.jl # استراتيجية علاقات النسبة
│           │   └── tadbir_strategy.jl # استراتيجية التخطيط والتدبير
│           └── ...            # باقي المحركات الفيزيائية (85 محركاً فرعياً)
```

---

## الوحدات البرمجية والربط

### 1. MirnanNew.jl - الوحدة الرئيسية

تمثل نقطة المدخل العام للمشروع وتربط جميع المكونات تحت واجهة موحدة:
```julia
module MirnanNew
    # تضمين جميع الوحدات الفرعية
    include("preprocessing/Preprocessing.jl")
    include("semantics/Semantics.jl")
    include("grammar/Grammar.jl")
    include("physics/Physics.jl")
    include("sio/SIO.jl")
    include("api/server.jl")
    include("utils/DataLoader.jl")
    include("integration/Integration.jl")
    
    # إعادة التصدير للرموز والدوال الأساسية
    using .Preprocessing
    using .Semantics
    using .Grammar
    using .Physics
    using .SIO
    using .APIServer
    using .DataLoader
    using .Integration
end
```

### 2. Physics.jl - بنية المحركات والمجموعات الـ 10

بدلاً من تضمين جميع المحركات بشكل مسطح متداخل، تم تنظيم الفيزياء في 10 مجموعات تضمن التحميل المرتب حسب التبعيات البرمجية:
1. **Core Group** (`groups/core_group.jl`): تحميل الثوابت، وقاعدة الحروف، وجبر كليفورد، وفيزياء الكلمة الأساسية.
2. **Basic Physics Group** (`groups/basic_physics_group.jl`): الجاذبية، المذبذبات، سلاسل LC، والتداخل الترددي.
3. **Quantum Group** (`groups/quantum_group.jl`): بوابات الإنتروبيا، مصفوفة الكثافة الكمومية، والتعزيز الطوري الهيبي.
4. **Memory Group** (`groups/memory_group.jl`): الذاكرة الهولوغرافية، التدفق السببي، وذاكرة الجذب RAM.
5. **Morphology Group** (`groups/morphology_group.jl`): الصرف الكليفوردي، الجواذب اللغوية، وعلاقات الأوزان.
6. **Arabic Group** (`groups/arabic_group.jl`): ذاكرات الأنماط العربية وتفعيل الحسبان الدلالي.
7. **Wave Group** (`groups/wave_group.jl`): تداخل الموجات المستمر والاقتران الطوري المركب.
8. **Infra Group** (`groups/infra_group.jl`): الطبقات الأساسية المساعدة للبنى التحتية الفيزيائية.
9. **PRNN Group** (`groups/prnn_group.jl`): الشبكة الرنينية الطورية وديناميكيات ستوارت-لانداو.
10. **Generation Group** (`groups/generation_group.jl`): محرك التوليد المركزي والمخططين والاستراتيجيات المرافقة.

---

## ذاكرات النمط والحسبان الدلالي

يتعلم `train.jl` خمس طبقات ذاكرة مستقلة تؤثر على التوليد الفيزيائي بشكل توجيهي دون تغيير قيم المصفوفات الأساسية:
- **اللسان (`al_lisan`)**: لحفظ قوالب ومؤشرات الجمل العربية الفصيحة.
- **الكود (`al_code`)**: لحفظ قوالب وتركيب الكود البرمجي.
- **التدبير (`al_tadbir`)**: لتوجيه التخطيط وحل المشكلات التتابعية.
- **الحساب (`al_hisab`)**: لحفظ قواعد العمليات الحسابية والرياضية.
- **الحسبان الدلالي (`al_hisban_al_dalali`)**: لإدارة المعاني الفيزيائية عبر جبر كليفورد الهندسي.

---

## ترتيب التحميل والتشغيل

يتم تحميل الوحدات البرمجية بالترتيب التسلسلي التالي لضمان توفر التبعيات:
1. `Preprocessing` (معالجة النص وتطبيعه).
2. `Semantics` (أدوات الدلالة).
3. `Grammar` (أدوات النحو).
4. `Physics` (المحركات المجموعية + فضاء العقل + استراتيجيات التوليد).
5. `SIO` (الذكاء التوليفي والتنظيم).
6. `APIServer` (خادم API للتواصل الخارجي).
7. `DataLoader` (تحميل الموديلات والبيانات).
8. `Integration` (خط الأنابيب الكامل للتحليل والرد).
