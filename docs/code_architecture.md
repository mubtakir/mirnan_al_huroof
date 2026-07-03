# الدليل الهيكلي والتفصيلي لأكواد نموذج مرنان (Mirnan V8)

يُعد نموذج **مرنان (Mirnan)** نموذجاً لغوياً ديناميكياً فريداً مبنياً بالكامل على **الفيزياء اللغوية (Linguistic Physics)** ومحاكاة حركة الأمواج وتوافق الأطوار والربط الهولوغرافي، بدلاً من بنى الشبكات العصبية التقليدية (كالمحولات أو Transformers).

تشرح هذه الوثيقة بالتفصيل وظيفة ومهمة كل ملف كود ومجلد فرعي في تطبيق مرنان المكتوب بلغة جوليا (Julia) والموجود في المجلد الرئيسي `src/`.

---

## 1. الملف الرئيسي لنقطة الدخول: `src/Mirnan.jl`

* **الملف الرئيسي:** [Mirnan.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/Mirnan.jl)
* **المهمة والوظيفة:**
  هو المنسق ونقطة التجميع (Orchestrator & Entry Point) للمكتبة بأكملها. يقوم بـ:
  1. تضمين جميع المجلدات الفرعية والمكتبات التابعة عبر عبارات `include`.
  2. تنظيم العلاقات الموديولية وتمرير المتغيرات العامة.
  3. إعادة تصدير (re-exporting) الدوال والمحركات الهامة ليتسنى للمستخدمين والـ API استدعاؤها مباشرة باستخدام `using Mirnan`.
  4. تصدير البنيات الفيزيائية الجوهرية مثل `MirnanGenerator`, `BeamReasoner`, `CodePhaseEngine`, و `SymbolicMathEngine`.

---

## 2. مجلد المعالجة المسبقة والتطهير: `src/preprocessing/`

يحتوي هذا المجلد على محركات تنظيف الكوربس وتهيئته للنصوص العربية والإنجليزية عبر خط معالجة (Pipeline) خماسي المراحل:

* **[Preprocessing.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/preprocessing/Preprocessing.jl):**
  يُدير ويعرّف `PipelineConfig` (إعدادات التنظيف والتطبيع والتقطيع والفلترة) و `PipelineStats` (التقرير الإحصائي للمعالجة). يحتوي على الدالة الرئيسية `run_pipeline` التي تمرر النصوص الخام عبر المراحل الخمس لإنتاج جمل فصيحة ونظيفة تماماً.

* **[text_extractor.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/preprocessing/text_extractor.jl):**
  يقوم باستخراج النصوص النقية وإزالة الوسوم البرمجية من الملفات بصيغ (HTML, Markdown, JSON, Text).

* **[language_filter.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/preprocessing/language_filter.jl):**
  يكتشف لغة النص (عربي، إنجليزي) ويحسب كثافة الحروف ليقوم بإلغاء واستبعاد المستندات المشوهة أو المكتوبة بلغة غير المستهدفة.

* **[normalizer.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/preprocessing/normalizer.jl):**
  يقوم بتوحيد أشكال الحروف المتشابهة (آ/أ/إ ← ا، ة ← ه، ى ← ي)، وإزالة الكشيدة والمد، **مع الحفاظ التام على علامات التشكيل وحركات الإعراب** (الفتحة، الضمة، الكسرة، التنوين، الشدة) لأن التشكيل في مرنان يُعد عنصراً فيزيائياً يحدد تردد تذبذب الكلمة وصيغتها الصرفية.

* **[segmenter.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/preprocessing/segmenter.jl):**
  يُقطع النصوص الكبيرة إلى مستويات تقسيم محددة: جمل (`sentence`) بناءً على علامات الترقيم العربية، أو فقرات (`paragraph`) مفصولة بسطور فارغة، أو أسطر (`line`) للبيانات المهيكلة.

* **[quality_filter.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/preprocessing/quality_filter.jl):**
  يستبعد الجمل التالفة أو القصيرة جداً، ويحتوي على خوارزمية إزالة المكررات (Deduplication) للمحافظة على تنوع الكوربس التدريبي وتجنب الانحياز التكراري.

---

## 3. مجلد الفيزياء اللغوية الأساسية والمتقدمة: `src/physics/`

هذا المجلد هو "قلب مرنان النابض" ويضم 88 ملفاً فيزيائياً تقوم بتمثيل الكلمات ككتل ونواسات رنانة تدور في فضاء طوري متعدد الأبعاد.

### أ. الفيزياء الأساسية والمذبذبات
* **[constants.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/constants.jl):**
  يحدد الثوابت الفيزيائية الكونية للنظام، مثل أبعاد فضاء الطور الأساسي `PHASE_DIM = 256` وإجمالي الأبعاد الدلالية الطورية والصرفية `TOTAL_DIM = 512` وثابت الجاذبية اللغوية `GRAVITY_G = 6.674e-11`.
* **[letter_db.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/letter_db.jl):**
  يخزن الخصائص الطورية والترددية والكتلية لكل حرف عربي وإنجليزي منفرد. فكل حرف له متجه طور فريد يمثل بصمته الموجية الأساسية.
* **[word_physics.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/word_physics.jl):**
  يحتوي على المعادلات الرياضية لحساب طاقة وكتلة الكلمة (Word Mass) اعتماداً على مجموع كتل حروفها وحركاتها الإعرابية، وحساب متجهات الأطوار الممتدة (Extended Phase Vectors).
* **[clifford_math.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/clifford_math.jl):**
  يوفر العمليات الرياضية لجبر كليفورد (Clifford Algebra) وهندسة التدوير الطوري متعددة الأبعاد للتعامل مع متجهات الحالات المركبة والتحولات الطورية.
* **[gravity.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/gravity.jl):**
  يحاكي قانون نيوتن للجاذبية في فضاء دلالات اللغة:
  $$F = G \frac{m_1 m_2}{r^2}$$
  حيث الكتل هي الأوزان المعرفية والمسافة $r$ هي الفرق الطوري بين الكلمات. يجذب المتجهات الدلالية لتقريب الكلمات المتقاربة بالمعنى أثناء التوليد.
* **[oscillator.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/oscillator.jl):**
  يستخدم **نموذج مذبذبات كوراموتو (Kuramoto Oscillators)** لمحاكاة التزامن الديناميكي غير الخطي لمجموعة من متجهات الطور للكلمات داخل السياق، ويتم حل تطورها الطوري زمنياً باستخدام محاكي RK4 (Runge-Kutta 4th Order).
* **[resonant_chain.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/resonant_chain.jl):**
  يمثل الجملة كـ **سلسلة رنينية من دوائر RLC الكهربائية المتتالية**. السعة الدلالية $C$ تحسب من كتل الكلمات المجاورة، والمحاثة $L$ من توافق الأطوار. التردد الرنيني يُعزز وفقاً لـ **دالة زيتا (Zeta Response)** الرياضية لمحاكاة استقرار المعنى عند خط التوازن الحرج للريمان ($\sigma=1/2$).

### ب. بدائل آليات الانتباه وحقول الطاقة
* **[dccf.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/dccf.jl):**
  حقل الاقتران السياقي الديناميكي (Dynamic Contextual Coupling Field). يُعد البديل الفيزيائي لـ self-attention، حيث يربط الكلمات الحالية بالسياق التاريخي البعيد مع تطبيق عامل اضمحلال أسي للمسافة ($e^{-\lambda \cdot dist}$).
* **[potential_cascade.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/potential_cascade.jl):**
  طبقة الهوي التراكمي (Potential Cascade). كل كلمة يتم توليدها تولّد بئر جهد جاذبية يجذب الكلمة التالية. تحتوي الصيغة على قوى جذب للمسارات المتفقة، وقوى تنافر (Repulsion) لمنع تكرار الكلمات، واحتكاك دلالي (Friction) لتهدئة قفزات التوليد وضمان التدفق الطبيعي.
* **[ppm.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/ppm.jl):**
  نموذج التنبؤ بنقاط الطور القريبة (Phase Point Prediction) لتصحيح وتوجيه التوليد اللغوي نحو الاتساق الإحصائي.
* **[amfs.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/amfs.jl):**
  حقل القوى الدلالية الترابطية النشطة (Associative Memory Force Field) الذي يغذي عملية البحث عن الكلمات التالية من خلال بوابات طاقة ترابطية.
* **[causal_flow.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/causal_flow.jl):**
  محرك التدفق السببي في فضاء الطور. يضمن أن تتبع الأزواج التوليدية قانوناً سببياً يمنع حدوث "سلاسل الكلمات العشوائية" (Word Salad).
* **[density_matrix.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/density_matrix.jl):**
  حساب مصفوفة الكثافة الكمية للحالات الدلالية، ويقيس التداخل والتشابك بين الأطوار المعرفية المختلفة.
* **[holographic_kb.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/holographic_kb.jl):**
  قاعدة البيانات الهولوغرافية التوليفية. تخزن العلاقات الدلالية والحقائق وتسترجعها عبر بصمات التردد الهولوغرافي دون الحاجة للبحث الخطي التقليدي.

### ج. محركات الصرف والنحو العربي ومصفوفات الجذب
* **[root_field.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/root_field.jl):**
  حقل الجذور العربية. يتعرف على الجذور الثلاثية والرباعية للكلمات ويحسب قوة الجذب بين الكلمات التي تنتمي لنفس العائلة الجذرية (مثل: ك-ت-ب تجمع كتاب، كاتب، مكتوب).
* **[weight_resonance.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/weight_resonance.jl):**
  رنين الأوزان الصرفية (Morphological Weight Resonance). يقيس رنين صيغة الكلمة الصرفية (على ميزان فَعَلَ، فاعِل، مَفْعول) مع سياق الإعراب المحيط.
* **[morpho_phasic.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/morpho_phasic.jl) & [morpho_twistor.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/morpho_twistor.jl):**
  تطبق ميكانيكا **الالتواء الصرفي (Morphological Twistors)** استناداً إلى رياضيات التواءات كليفورد. تمثل هذه الملفات الاشتقاق اللغوي والتحول الصرفي كعمليات دوران وتماسك في الفضاء الطوري المركّب.
* **[syntax_field_full.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/syntax_field_full.jl) & [syntax_monitor.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/syntax_monitor.jl):**
  تتولى نمذجة العلاقات النحوية التوافقية ومتابعة صحة الترابط الإعرابي أثناء التوليد اللحظي.
* **[particles.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/particles.jl):**
  يحتوي على قائمة الحروف الهجائية النحوية والأدوات المعنوية (الروابط النحوية كـ: في، من، إلى، أن، إلخ) ويطبق قواعد كبت أو تخفيف أطوارها (Particle Penalty) لمنع هيمنتها الإحصائية على مصفوفات الدلالة.

### د. الشبكة الرنينية الطورية الهولوغرافية (PRNN)
* **[prnn_core.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/prnn_core.jl):**
  النواة الحركية للـ Phasic Recurrent Neural Network. تحاكي مذبذبات Stuart-Landau الموصوفة بمعادلات تفاضلية غير خطية، وتطبق كبتاً تنافسياً عالمياً (competitive inhibition) لمنع انفجار الطاقة، وتوفر آليات التعب العصبي (neural fatigue)، والربط وفك الربط الهولوغرافي الطوري (VSA Complex Binding/Unbinding).
* **[prnn_learner.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/prnn_learner.jl):**
  يتولى تدريب متجهات الـ PRNN على الانتقالات السببية وحزم الأطوار الدلالية من الكوربس النظيف.
* **[prnn_generator.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/prnn_generator.jl):**
  المنفذ الخاص بتوليد الانتقالات المتتالية استناداً لحالة المذبذبات في شبكة الـ PRNN.

### هـ. الأمواج والطيف اللغوي
* **[spectral_wave_engine.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/spectral_wave_engine.jl):**
  محرك الأمواج الطيفية. يستخدم تحويلات فوريه السريعة (FFT) لحساب تداخل الترددات السياقية وإعادة إسقاطها على معجم الكلمات.
* **[spectral_context.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/spectral_context.jl) & [word_spectrum.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/word_spectrum.jl):**
  يحللان التموج الترددي للكلمات داخل الجملة لقياس مستوى التناغم الصوتي والمعنوي اللغوي.

### و. التوليد والتكامل المركزي
* **[generator.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/generator.jl):**
  المحرك المركزي للتوليد الصوتي والتركيبي لمرنان V8. يضم الهيكل المتكامل لـ `MirnanGenerator` ويحمل مصفوفات Co-occurrence العملاقة `K_sem` و `K_pos` والمعجم `vocab`. يقوم بحساب علامات التقييم الشاملة لكل كلمة مرشحة عبر جمع أوزان المعاملات الفيزيائية الـ 40+ (مثل الجاذبية الدلالية، والتحرك الرنيني، واستجابة زيتا، وقيود النحو، وعوامل التنافر الطوري).
* **[orchestrator.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/orchestrator.jl) & [multi_pass_generator.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/multi_pass_generator.jl):**
  تنسيق التوليد ذي الممرات المتعددة (Multi-Pass Generation). يُولد مسودات أولية للجمل ثم يُعيد تمريرها على محلل الأمواج لتنقيح وضبط أطوارها النحوية والصرفية للحصول على نواتج غاية في الفصاحة والاستقرار المعنوي.
* **[gpu_accelerator.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/gpu_accelerator.jl):**
  محرك تسريع الحسابات باستخدام كرت الشاشة (CUDA) لتسريع عمليات ضرب المصفوفات الكثيفة وحساب متجهات الأطوار لمليارات الكلمات المرشحة.
* **[language_feedback.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/language_feedback.jl):**
  آلية التغذية الراجعة المغلقة التي تسمح للنموذج بتقييم المخرجات ذاتياً وإعادة استخدامها في شحن وتقوية الممرات الضعيفة في الشبكة.
* **[semantic_comprehension.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/physics/semantic_comprehension.jl):**
  دائرة إدراك المعاني، تربط التمثيل الطوري للمستوى الحرفي بمستوى الكلمة وصولاً لفهم البنية الموضوعية للنصوص الطويلة.

---

## 4. مجلد القواعد والمفردات الهيكلية: `src/grammar/`

* **[Grammar.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/grammar/Grammar.jl):**
  يحدد متجهات الإرساء البنيوي (Structural Anchors) مثل مواضع الأسماء، الأفعال، الفاعل، المفعول، النعت، والحروف النحوية. يحاكي هذا الموديول القوانين الصارمة للغة العربية مثل موازين الحالات النحوية وعلاقة الفاعل بالفعل وعوامل الرفع والنصب والجر، ممثلة كزوايا إرساء في الفضاء الطوري الدائري.

---

## 5. مجلد الفضاء الدلالي الكروي: `src/semantics/`

* **[Semantics.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/semantics/Semantics.jl):**
  يتولى الحوسبة الدلالية الكروية (Spherical Semantic Algebra). يمثل العلاقات الدلالية كإحداثيات كروية متعددة الأبعاد، ويوفر دوال ضرب وفصل المتجهات ومقاييس البعد المعرفي، كما يحتوي على مصفوفات الربط الدلالي المشترك التي يتم تدريبها عبر قراءة سياقات الكوربس.

---

## 6. مجلد منسق الذكاء التوليفي المتقدم: `src/sio/`

* **[SIO.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/sio/SIO.jl):**
  يتولى دور منسق الذكاء التوليفي (Synthetic Intelligence Orchestrator). يُمثل محرك العقل المفكر لمرنان الذي يستقبل المهمة أو الهدف من المستخدم (مثل: "كتابة كود" أو "إنشاء تقرير") ويتبع مساراً ذكياً لحل هذه المهام يتألف من:
  1. **GoalParser:** تحليل وتفكيك الهدف لمراحل تخطيط وتصميم وبناء ومراجعة.
  2. **PhasePlanner:** وضع خطة زمنية ومتابعة تقدم تنفيذ المراحل.
  3. **PhaseExecutor:** توليد المخرجات البرمجية أو اللغوية لكل مرحلة باستخدام مولد مرنان المركزي.
  4. **SelfMonitor:** المراقبة الذاتية للمخرجات وقياس تماسكها وصحتها النحوية والدلالية، وطلب إعادة المحاولة (Retry) في حال عدم اجتياز التقييم.
  5. **Integrator:** جمع ودمج وتنسيق نواتج المراحل المختلفة لتقديم منتج متكامل للمستخدم.

---

## 7. مجلد خادم الواجهة والاتصال: `src/api/`

* **[server.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/src/api/server.jl):**
  يُدير واجهة خادم التطبيق المبني على مكتبة HTTP بجوليا. يتيح تشغيل خادم محلي يستقبل طلبات التوليد اللغوي أو معالجة الأكواد البرمجية أو الاستدلال المنطقي من تطبيقات المستخدم الخارجية أو واجهة المستخدم الرسومية (UI) عبر بروتوكولات REST API و WebSockets للتعامل الفوري واللحظي.
