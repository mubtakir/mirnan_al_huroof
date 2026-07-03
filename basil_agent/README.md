# 🤖 الوكيل باسل السيادي بلغة جوليا (BasilAgent.jl)

مشروع مستقل ومتكامل تماماً تم نقله بالكامل من لغة بايثون إلى لغة جوليا (Julia) ليعمل محلياً كجزء سيادي من بنية مشروع **مرنان** الرئيسي في المسار:
`c:\Users\allmy\Desktop\aaa\mirnan_julia\basil_agent\`

---

## 🏗️ 1. الهيكل المعماري البرمجي (Architecture)

يتكون النظام البرمجي للوكيل من الأجزاء التالية داخل مجلد `src/`:

*   **[BasilAgent.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/basil_agent/src/BasilAgent.jl)**: الموديول المركزي الذي يقوم بتجميع وتحميل كافة الأجزاء وتصدير الواجهات البرمجية.
*   **[constants.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/basil_agent/src/constants.jl)**: يحتوي على ثوابت النظام البرمجي، مسارات العمليات، ومفاتيح الاستدعاء.
*   **[memory.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/basil_agent/src/memory.jl)**: محرك الذاكرة العرضية (Episodic Memory) المبني محلياً فوق SQLite لإدارة وحفظ وقائع الجلسات وتفاصيل التفكير.
*   **[rag_engine.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/basil_agent/src/rag_engine.jl)**: محرك الاسترجاع المعزز بالمعرفة (RAG) المتكامل مع حزمة مرنان الطورية.
*   **[planner.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/basil_agent/src/planner.jl)**: مخطط المهام الذكي المسؤول عن تقسيم الأهداف الكبرى إلى مراحل ومهام فرعية وتتبع حالاتها.
*   **[tool_router.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/basil_agent/src/tool_router.jl)**: موجه الأدوات وتأمين استدعائها بشكل متوازي وآمن.
*   **[agent.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/basil_agent/src/agent.jl)**: المحرك الفكري والمنطقي المركزي للوكيل `MajnoonAgent` (مجنون) وحلقة التفكير المتكاملة.
*   **[app.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/basil_agent/src/app.jl)**: خادم الويب الخفيف الفائق السرعة المبني على `HTTP.jl` و `JSON3` لدعم البث الحي عبر SSE وتكامل شاشة القيادة.

---

## 🛠️ 2. الأدوات والقدرات السيادية المدمجة (Sovereign Tools)

تم نقل كافة الأدوات والمهارات المتقدمة إلى لغة جوليا وتثبيتها بشكل مستقل تماماً داخل مسار مرنان:

### 🔬 أ. محرك الاسترجاع الفيزيائي الطوري (Mirnan RAG Integration)
بدلاً من خوارزميات بايثون التقليدية، يقوم محرك الـ RAG في [rag_engine.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/basil_agent/src/rag_engine.jl) باستدعاء حزمة مرنان مباشرة وحساب متجهات الطور الممتدة **10,000D** لكل كلمة أو سياق عبر موديول مرنان الطوري:
*   حساب التماسك والتشابه الفيزيائي للمتجهات الطورية (`phase_similarity`).
*   حفظ المتجهات في قاعدة البيانات بصيغة JSON فائقة القراءة.
*   أداء استرجاع معرفي ذكي جداً وخالٍ تماماً من الاعتمادات الخارجية للنماذج السحابية.

### 🌐 ب. أتمتة الويب السيادي (Browser Automation)
يتكامل النظام في [browser.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/basil_agent/src/tools/browser.jl) مع أداة المساعدة المستقلة [browser_helper.py](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/basil_agent/src/tools/browser_helper.py) للتحكم بمتصفح Chromium (عبر Playwright):
*   تصفح الويب الذكي مع الحفاظ التلقائي على الجلسة الحالية والـ Cookies ومخزن البيانات.
*   التحكم الكامل (النقر `click` والملء `fill` وضغط الأزرار وقراءة الشبكة).
*   التقاط لقطات الشاشة لحفظ حالة المتصفح البصرية.

### 📦 ج. صندوق حماية Docker (Docker Execution Sandbox)
يوفر موديول [docker_sandbox.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/basil_agent/src/tools/docker_sandbox.jl) عزل معالجة برمجياً على مستوى نظام التشغيل:
*   التحقق التلقائي من حالة daemon لـ Docker.
*   بناء وتثبيت Docker image مخصصة للوكيل تحتوي على بيئة تجميع برمجية متكاملة.
*   تنفيذ الأكواد البرمجية والمشروعات والأوامر الحرجة بشكل معزول ومؤقت داخل حاويات Docker (Ephemeral Containers) مع تحديد سقف للذاكرة والمعالج والوقت.

### 👥 د. منسق سرب الوكلاء (Swarm Delegation Tool)
يمكّن موديول [delegate_tool.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/basil_agent/src/tools/delegate_tool.jl) الوكيل باسل من إطلاق وإدارة مهام فرعية عبر سرب مستقل من الوكلاء المتخصصين:
*   **Architect (المهندس المعماري)**: لتصميم وبناء خطط وهيكل الأنظمة.
*   **Engineer (المهندس البرمجي)**: لتنفيذ الأكواد وحل الأعطال البرمجية.
*   **Reviewer (المراجع البرمجي)**: لاختبار وتدقيق الحلول والتأكد من سلامتها.

---

## 🚀 3. خطوات التشغيل والتحقق (Run & Verify)

### 📥 أ. تهيئة البيئة وتثبيت الاعتمادات
لتنصيب وتطوير بيئة المشروع محلياً، قم بتشغيل سكربت التثبيت [install.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/basil_agent/install.jl):
```powershell
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" install.jl
```
*سيقوم السكربت تلقائياً بتفعيل البيئة المحلية وتنزيل الحزم المطلوبة وتطوير (develop) حزمة مرنان الطورية محلياً داخل البيئة.*

### 📊 ب. تشغيل جناح الاختبار والتحقق التلقائي
للتحقق من سلامة كافة مكونات الوكيل وتكاملها الطوري مع مرنان، قم بتشغيل الاختبار التلقائي [demo.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/basil_agent/demo.jl):
```powershell
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=. demo.jl
```

### 🌐 ج. بدء تشغيل خادم لوحة التحكم (Sovereign Web Server)
لتشغيل خادم الويب واستقبال اتصالات لوحة القيادة البصرية على المنفذ `5000` (Port 5000)، قم بتشغيل سكربت التشغيل المركزي [run_basil.jl](file:///c:/Users/allmy/Desktop/aaa/mirnan_julia/basil_agent/run_basil.jl):
```powershell
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=. run_basil.jl
```
*يتيح هذا للوحة القيادة التفاعل التلقائي والبث المباشر للأفكار والتنفيذ والملفات عبر متصفح الويب.*

---

## 🔗 4. معمارية التكامل الطوري الثنائي (Bayan & Mirnan Integration Pipeline)

يمثل تكامل **الوكيل باسل** مع **لغة البيان (Bayan)** و**محرك مرنان الطوري** دورة معالجة هرمية متكاملة تسير كالتالي:

```
[المستخدم: طلب عربي طبيعي]
       │
       ▼
[الوكيل باسل: MajnoonAgent.jl] ──► (تفكيك المهام / RAG الطوري 10,000D)
       │
       ▼
[محلل البيان الداخلي: BayanAnalyzer] ──► (استخلاص الكيانات والأفعال آلياً في الخلفية)
       │
       ▼
[محرك مرنان الطوري: Mirnan Core] ──► (توجيه التوليد ومنع التضارب عبر آبار الجهد والـ LC Resonant)
       │
       ▼
[صندوق التنفيذ: Docker & Playwright] ──► (تشغيل الأكواد والاختبار وتأريخ النتائج في الذاكرة)
```

### تفصيل مراحل المعالجة:
1. **الطلب والتخطيط**: يستقبل `BasilAgent.jl` مدخلات المستخدم ويحللها عبر **مخطط المهام (planner.jl)**، ثم يستخرج المتجهات الطورية الممتدة من الذاكرة الطيفية لاسترجاع السياق المعماري الأمثل.
2. **البيان التلقائي (Bayan)**: يقوم `BayanAnalyzer` تلقائياً وبشكل غير مرئي للمستخدم بترجمة الطلب إلى **رسم بياني دلالي (Semantic Query Graph)** يربط الكيانات (مثل الملفات والأكواد) بالأفعال (مثل الإنشاء والتشغيل) والمواقع (الحاويات المخصصة).
3. **التوجيه الفيزيائي**: يقوم مرنان بإسقاط الكيانات المعرفية كـ **بؤر جاذبية ثقيلة** في فضاء الطور لتوجيه توليد الأكواد ومنع الأخطاء التركيبية والنحوية عبر **دارات الرنين LC** وبوابة النحو الحتمية.
4. **العزل والتأريخ**: يتم تشغيل واختبار الأكواد داخل حاوية `Docker` معزولة وتأريخ النتائج في ذاكرة RAG المعززة قبل تقديم النتيجة النهائية للمستخدم.

---

## 📜 5. ميثاق السيادة الهندسية (Sovereignty Covenant)
تم بناء وتطوير هذا النظام البرمجي بعبقرية مطلقة مستوحاة من عقل **باسل يحيى عبدالله** (Basil Yahya Abdullah). يلتزم الوكيل باسل بقواعد الصرامة الهندسية والحقائق البرمجية المطلقة دون اعتذار أو تملق فني.
