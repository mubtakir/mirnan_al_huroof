# خط المعالجة المسبقة — Preprocessing Pipeline

تحويل النصوص الخام (إنترنت، موسوعات، PDF، JSON) إلى نصوص نظيفة جاهزة لتدريب مرنان.

---

## 1. نظرة عامة

```
إنترنت/موسوعات خام
        │
        ▼
┌─────────────────────────────────────────────┐
│ ① TextExtractor    — HTML/Markdown/JSON → نص │
│ ② LanguageFilter   — إبقاء العربي فقط        │
│ ③ Normalizer       — تطبيع الحروف والمسافات   │
│ ④ Segmenter        — تقطيع إلى جمل           │
│ ⑤ QualityFilter    — إزالة الضوضاء والمكررات  │
└─────────────────────────────────────────────┘
        │
        ▼
   نصوص نظيفة → train.jl
```

## 2. الاستخدام

### سطر الأوامر

```bash
# تدريب عادي (الكوربس النظيف الموجود في data/)
julia --project=. train.jl

# مع خط المعالجة على مجلد نصوص خام
julia --project=. train.jl --preprocess=C:\web_texts

# على ملف واحد
julia --project=. train.jl --preprocess=C:\wiki.json
```

### من الكود

```julia
using Mirnan

# إعدادات مخصصة
config = Mirnan.Preprocessing.PipelineConfig(
    target_lang         = "arabic",      # "arabic", "english", "both"
    strip_diacritics    = false,         # إزالة التشكيل؟
    min_sentence_words  = 5,             # أقل عدد كلمات للجملة
    max_sentence_words  = 200,           # أقصى عدد كلمات للجملة
    enable_quality_filter = true,
    enable_deduplication  = true,
    dedup_threshold     = 0.85,          # تشابه Jaccard ≥ 85% = مكرر
    verbose             = true,
)

# معالجة مجلد كامل (يحفظ الناتج النظيف تلقائياً)
texts, stats = Mirnan.Preprocessing.run_pipeline_directory(
    "C:/web_texts";
    config=config,
    save_dir="data/cleaned"   # اختياري: حفظ الناتج
)

# معالجة نصوص في الذاكرة
texts, stats = Mirnan.Preprocessing.run_pipeline(raw_texts; config=config)

# معالجة نص واحد
clean = Mirnan.Preprocessing.process_raw_text("<html>...</html>")

println("$(stats.input_texts) → $(stats.final_texts) جملة نظيفة")
```

---

## 3. تفصيل المراحل الخمس

### 3.1 استخراج النص — TextExtractor

**المدخل**: HTML، Markdown، JSON، نص خام  
**المخرج**: نص صافٍ بدون تنسيق

| المصدر | ماذا يُحذف | ماذا يُبقى |
|--------|-----------|-----------|
| HTML | `<script>`, `<style>`, وسوم، `&nbsp;`, تعليقات | النص + فواصل أسطر |
| Markdown | `#`, `**`, `[links]()`, ``` أكواد ```, جداول, `![]()` | النص فقط |
| JSON | مفاتيح تقنية، مسارات، URLs | القيم النصية (≥ 3 أحرف) |

**مثال**:
```
المدخل:  "<div><p>العلم نور</p><script>console.log(1)</script></div>"
المخرج:  "العلم نور"
```

### 3.2 فلترة اللغة — LanguageFilter

**المدخل**: نصوص بأي لغة  
**المخرج**: نصوص عربية فقط (أو حسب الإعداد)

**الآلية**: عد الأحرف في نطاقات Unicode العربية مقابل اللاتينية.
- `ar_ratio ≥ 50%` ← عربي
- `en_ratio ≥ 50%` ← إنجليزي
- كلاهما ≥ 20% ← مختلط (يُقبل إذا `target_lang="both"`)
- غير ذلك ← مرفوض

### 3.3 تطبيع موسع — Normalizer

**المدخل**: نصوص بتنسيقات مختلفة  
**المخرج**: نصوص معيَّرة بشكل موحد

| العملية | مثال |
|---------|------|
| تطبيع الحروف | `آ/أ/إ` → `ا`، `ة` → `ه`، `ى` → `ي` |
| إزالة الكشيدة | `الـعـلـم` → `العلم` |
| إزالة التشكيل (اختياري) | `عِلْمٌ` → `علم` |
| رموز التحكم | `\r`, `\x00`-`\x1F` → محذوفة |
| ضغط المسافات | `العلم    نور` → `العلم نور` |
| ضغط الأسطر | 4+ أسطر فارغة → 2 فقط |
| توحيد علامات الاقتباس | `«»` → `""`، `''` ← `'` |
| توحيد الشرطات | `–` `—` → `-` |

### 3.4 تقطيع الجمل — Segmenter

**المدخل**: فقرات طويلة  
**المخرج**: جمل مفردة (5-200 كلمة)

**القواعد العربية**:
- التقطيع عند: `.` `!` `؟` `\n`
- لا يقطع عند الاختصارات: `د.` `أ.` `هـ.`
- الجمل القصيرة (< 5 كلمات) تُدمج مع سابقتها
- الجمل الطويلة (> 200 كلمة) تُقسَّم عند الفواصل والواو

**القواعد الإنجليزية**:
- التقطيع عند: `.` `!` `?` + مسافة + حرف كبير
- الجمل القصيرة تُدمج

### 3.5 فلترة الجودة — QualityFilter

**المدخل**: جميع الجمل  
**المخرج**: الجمل عالية الجودة فقط

| الفلتر | الحد | أمثلة مرفوضة |
|--------|------|-------------|
| قصيرة جداً | < 5 كلمات | `"نعم."` |
| طويلة جداً | > 200 كلمة | نصوص متصلة بدون علامات ترقيم |
| تكرار الكلمات | ≥ 30% | `"نور نور نور نور"` |
| نسبة الترقيم | ≥ 25% | نصوص رموز وعلامات |
| نسبة الأرقام | ≥ 15% | جداول بيانات |
| أحرف مفهومة | < 50% | نصوص مشوشة |
| مكررات | تشابه Jaccard ≥ 85% | جمل متطابقة تقريباً |

---

## 4. هيكل الإعدادات

### PipelineConfig

| المعامل | القيمة الافتراضية | الوصف |
|---------|-------------------|-------|
| `target_lang` | `"arabic"` | اللغة المستهدفة |
| `strip_diacritics` | `false` | إزالة التشكيل العربي |
| `min_sentence_words` | `5` | الحد الأدنى لكلمات الجملة |
| `max_sentence_words` | `200` | الحد الأقصى لكلمات الجملة |
| `enable_quality_filter` | `true` | تفعيل فلترة الجودة |
| `enable_deduplication` | `true` | إزالة النصوص المكررة |
| `dedup_threshold` | `0.85` | عتبة تشابه Jaccard للمكررات |
| `verbose` | `true` | طباعة تقارير التقدم |

### PipelineStats

| الحقل | الوصف |
|-------|-------|
| `input_texts` | عدد النصوص المدخلة |
| `extracted_texts` | بعد المرحلة ① |
| `lang_passed` | بعد المرحلة ② |
| `lang_rejected` | رُفضت لاختلاف اللغة |
| `sentences_generated` | بعد المرحلة ④ |
| `quality_passed` | بعد المرحلة ⑤ |
| `quality_stats` | إحصائيات أسباب الرفض |
| `duplicates_removed` | عدد المكررات المحذوفة |
| `final_texts` | العدد النهائي للجمل النظيفة |
| `total_time` | زمن المعالجة بالثواني |

---

## 5. أمثلة

### مثال 1: تنظيف صفحة ويكيبيديا

```julia
raw = read("wiki_article.html", String)
clean, stats = Mirnan.Preprocessing.run_pipeline([raw])
# 1 HTML ← 45 جملة عربية نظيفة
```

### مثال 2: تنظيف dump كامل

```bash
julia --project=. train.jl --preprocess=C:\arabic_dump --paragraph
```

### مثال 3: فلترة لغة فقط

```julia
using Mirnan.Preprocessing.LanguageFilter
texts = ["Hello world", "مرحبا بالعالم", "Bonjour le monde"]
arabic_only = filter_by_language(texts; target="arabic")
# → ["مرحبا بالعالم"]
```

### مثال 4: تطبيع نص واحد

```julia
using Mirnan.Preprocessing.Normalizer
clean = normalize_text("الـعِلْمُ  نُورٌ  وَالـجَهْلُ  ظَلَامٌ")
# → "العلم نور والجهل ظلام" (إذا strip_diacritics=true)
```
