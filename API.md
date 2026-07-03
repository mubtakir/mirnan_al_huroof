# مرجع واجهة البرمجة (API Reference) لنموذج مِرنان V9

يوضح هذا الدليل الواجهات البرمجية للدوال والهياكل في مِرنان V9، مقسمة إلى: واجهة خط الربط المتكامل (Integration Pipeline) وواجهة المحركات الفيزيائية والتوليد (Generator & Physics Engine).

---

## 1. واجهة الربط المتكامل (Integration Pipeline)

تُستخدم للتحليل الإحصائي السريع لخصائص النصوص النحوية والدلالية والفيزيائية.

### المكونات والهياكل

#### `MirnanPipeline`
```julia
struct MirnanPipeline
    preprocessor::TextPreprocessor
    semantic_analyzer::SemanticAnalyzer
    grammar_analyzer::GrammarAnalyzer
end
```
* **الوصف**: ينشئ خط معالجة كامل يربط طبقات المعالجة والنحو والدلالة.
* **البناء**: `pipeline = MirnanPipeline()`

---

### الدوال الرئيسية للربط

#### `analyze_text(pipeline, text)`
* **الوصف**: يقوم بتحليل نص كامل وتطبيعه، ويحسب جميع الملخصات الدلالية والفيزيائية والنحوية.
* **المعاملات**:
  - `pipeline::MirnanPipeline`
  - `text::AbstractString`
* **الإرجاع**: `PipelineResult`

#### `analyze_word(pipeline, word, context)`
* **الوصف**: يحلل كلمة واحدة ضمن سياق جملتها لإنتاج قيم التشابه والدور النحوي.
* **الإرجاع**: `WordAnalysis`

#### `analyze_sentence_full(pipeline, sentence, context_words)`
* **الوصف**: يحلل تماسك الجملة دلالياً وصحتها نحوياً.
* **الإرجاع**: `SentenceAnalysis`

---

### هياكل نتائج التحليل

#### `PipelineResult`
```julia
struct PipelineResult
    original_text::String           # النص الخام المدخل
    normalized_text::String         # النص بعد المعالجة والتنظيف
    words::Vector{String}           # الكلمات المستخرجة
    sentences::Vector{String}       # الجمل المستخرجة
    word_analyses::Vector{WordAnalysis}     # تحليل تفصيلي لكل كلمة
    sentence_analyses::Vector{SentenceAnalysis} # تحليل تفصيلي لكل جملة
    semantic_summary::SemanticSummary       # ملخص المؤشرات الدلالية
    grammar_summary::GrammarSummary         # ملخص المؤشرات النحوية
    physics_summary::PhysicsSummary         # ملخص الطاقة والكتلة ورنين الطور
    response::String                # الرد الموجي المتولد
end
```

#### `PhysicsSummary`
```julia
struct PhysicsSummary
    total_energy::Float64           # الطاقة الإجمالية للنص (E=hf)
    avg_mass::Float64               # متوسط كتلة الكلمات (m=E/c²)
    phase_coherence::Float64        # تماسك متجهات الطور
    semantic_gravity::Float64       # شدة التجاذب الدلالي بين الكلمات
    syntactic_oscillation::Float64  # رنين التذبذب النحوي
end
```

---

## 2. واجهة محرك التوليد والفيزياء (Generator & Physics Engine)

المحرك الرئيسي لتوليد النصوص، التعلم المستمر، والتكامل مع قاعدة المعرفة الإدراكية (Al-Aql).

### المكونات والهياكل

#### `MirnanGenerator`
* **الوصف**: الهيكل الحركي التوليدي المركزي الذي يدير 10 مجموعات من المحركات و 15 استراتيجية توليد ومتحكم المخيخ PID.
* **البناء**:
  ```julia
  # بناء افتراضي بمفردات ومصفوفة اقتران
  gen = MirnanGenerator(vocab::Dict{String,Int}, K_sem::AbstractMatrix; model_dir="model", ...)
  ```

---

### الدوال الرئيسية للتوليد والتعلم

#### `generate!(gen, prompt; mode="standard", max_words=30)`
* **الوصف**: يولد استجابة بناءً على الأمر المدخل والنمط المحدد.
* **المعاملات**:
  - `gen::MirnanGenerator`: محرك التوليد.
  - `prompt::String`: موجه التوليد الأساسي.
  - `mode::String`: نمط التوليد (`"standard"`, `"creative"`, `"dialogue"`, `"prnn"`, `"entangle"`).
  - `max_words::Int`: الحد الأقصى للكلمات المتولدة.
* **الإرجاع**: `String` (النص المتولد).

#### `compile_adl!(space::SimulationSpace, adl_code::String)`
* **الوصف**: يترجم لغة ADL (Aql Description Language) لبناء الكيانات وقواعد السببية في فضاء العقل.
* **المعاملات**:
  - `space::SimulationSpace`: فضاء محاكاة العقل (متاح عبر `gen.aql_space`).
  - `adl_code::String`: كود ADL.

#### `learn_from_feedback!(gen, prompt, response; rating=1.0, note="")`
* **الوصف**: يعزز المسارات الطورية للردود الصحيحة ويعدل أوزان الذاكرة الهيبيانية بناءً على تقييم المستخدم.

#### `save_runtime_learning!(gen) -> String`
* **الوصف**: يحفظ جميع القواعد والكيانات والذاكرات التي تعلمها المولد أثناء التشغيل إلى القرص.
* **الإرجاع**: مسار الملف المحفوظ.

#### `load_runtime_learning!(gen)`
* **الوصف**: يستعيد جميع القواعد المتعلمة والمحفوظة مسبقاً من القرص.

---

## 3. واجهات خادم الويب (APIServer)

يوفر الخادم نقاط نهاية HTTP للتكامل الخارجي:
* `POST /analyze`: يستقبل نصاً خاماً ويعيد نتيجة `PipelineResult` في هيئة JSON.
* `POST /generate`: يستقبل أمراً (`prompt`) وخيارات التوليد ويعيد النص المتولد والتحليل الفيزيائي المرافق له.
