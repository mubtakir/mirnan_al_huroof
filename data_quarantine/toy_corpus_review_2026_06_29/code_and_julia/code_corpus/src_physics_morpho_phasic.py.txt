"""MorphoPhasicEngine — محرك الصرف الطوري.

يستخرج الجذور العربية (150+ جذر)، يطابق الأوزان الصرفية (15 وزناً)،
ويحدد حالة الإعراب (3 حالات). يبني متجهات طورية صرفية.
"""
import re
import numpy as np
from src.physics.word_physics import compute_extended_phase_vector, _normalize_letters
from src.physics.letter_db import LetterDB
from src.physics.constants import TOTAL_DIM
from src.semantics.arabic_semantics import ArabicRootExtractor
from src.semantics.english_semantics import EnglishRootExtractor, EnglishCharacterEmbedding

_arabic_extractor = ArabicRootExtractor()
_english_extractor = EnglishRootExtractor()
_eng_embedding = EnglishCharacterEmbedding(dim=TOTAL_DIM)


# -------- Arabic Roots (expanded: 500+) --------

_ROOTS = [
    # الأفعال الأساسية
    "كتب", "قرء", "علم", "جلس", "خرج", "دخل", "اخذ", "اكل", "شرب", "نظر",
    "سمع", "قول", "فعل", "عمل", "صنع", "بنى", "رسل", "نصر", "ضرب", "قتل",
    "جعل", "حبب", "صدق", "كذب", "ظلم", "عدل", "حكم", "فكر", "ذكر", "شكر",
    "صبر", "غفر", "رحم", "سلم", "كرم", "حسن", "جمل", "وصل", "فصل", "قسم",
    "جمع", "فرق", "قرب", "بعد", "طلب", "سال", "جوب", "وجد", "قدر", "عرف",
    "نكر", "ملك", "فتح", "غلق", "رفع", "خفض", "قدم", "لحق", "سبق", "نزل",
    "صعد", "وقف", "جري", "مشى", "حمل", "وضع", "قطع", "دعا", "اجر", "حفظ",
    "كشف", "ستر", "غسل", "مسح", "غيب", "شهد", "حضر", "غاب", "زار", "سكن",
    "عمر", "خرب", "زرع", "حصد", "غرس", "نبت", "باع", "شرى", "حسب", "وزن",
    "حقق", "قرر", "قضى",

    # الحركة والانتقال
    "سير", "ذهب", "عود", "رجع", "قفز", "طار", "سبح", "ركض", "زحف", "نقل",
    "حرك", "دار", "لفف", "ارسى", "هبط", "صعد", "انتقل", "عبر", "اجتاز", "تخطى",
    "اقبل", "ادبر", "توجه", "انطلق", "وصل", "بلغ", "ارتحل", "نزح", "هاجر", "رحل",

    # الإدراك والتفكير
    "درك", "فهم", "عقل", "دبر", "روى", "تامل", "لحظ", "رصد", "حدس", "استنتج",
    "استدل", "قاس", "وازن", "قيس", "فحص", "نقد", "حلل", "ركب", "استنبط", "استخلص",
    "تصور", "تخيل", "توهم", "ظنن", "شكك", "جزم", "ايقن", "ادرك", "استوعب", "استفهم",

    # الكلام والبيان
    "نطق", "لفظ", "صرح", "بين", "وضح", "شرح", "فسر", "عبر", "دل", "اشار",
    "اخبر", "نبا", "حدث", "روى", "ذكر", "وصف", "نعت", "سمى", "لقب", "كنى",
    "امر", "نهى", "دعا", "نادى", "صاح", "همس", "سرر", "اعلن", "بثث", "نشر",
    "قال", "اقر", "اعترف", "انكر", "رفض", "قبل", "رد", "اجاب", "سكت", "صمت",
    "حاور", "جادل", "ناقش", "خاصم", "لاحج", "احتج", "دافع", "فند", "نقض",

    # العلم والمعرفة
    "بحث", "درس", "تعلم", "فقه", "دقق", "تمحص", "تحقق", "اختبر", "جرب", "قيس",
    "نظر", "اشتغل", "اكتشف", "اخترع", "ابتكر", "ابدع", "انشا", "ولد", "انتج",
    "رصد", "قاس", "حسب", "احصى", "عد", "احصن", "سجل", "وثق", "ارخ", "دون",
    "طور", "حدث", "طبق", "نفذ", "استخدم", "وظف", "اعتمد", "استعمل",

    # القيم والأخلاق
    "امن", "كفر", "نفق", "اخلص", "غش", "خدع", "دلس", "خان", "وفى", "عهد",
    "التزم", "استقام", "اعوج", "فسق", "صلح", "فجر", "بر", "اساء", "اتقى", "جهل",
    "حلم", "غضب", "صفح", "عفا", "انتقم", "ظلم", "انصف", "قسط", "جار", "عدل",
    "كبر", "تواضع", "افتخر", "تذلل", "شهر", "اخفى", "قنع", "طمع", "زهد", "رغب",

    # الحياة والطبيعة
    "ولد", "مات", "عاش", "نما", "نضج", "ذبل", "يبس", "جفف", "روى", "سقى",
    "اشرق", "غرب", "طلع", "غاب", "ضاء", "اظلم", "حرق", "برد", "دفا", "اشتعل",
    "هطل", "هب", "عصف", "ثلج", "جمد", "ذاب", "تبخر", "تكثف", "فاض", "جزر",
    "امطر", "اعصف", "رعد", "برق", "زلزل", "فجر", "انفجر", "انبجس",

    # البناء والهندسة
    "شيد", "اقام", "رفع", "هدم", "نقض", "هدم", "حفر", "ملا", "سطح", "مهد",
    "قوى", "متن", "ضعف", "وطد", "دعم", "سند", "اسس", "اصل", "ربط", "فك",
    "لحم", "صهر", "قطع", "نشر", "ثقب", "نقب", "حفر", "دق", "سمر", "خاط",

    # التجارة والاقتصاد
    "اتجر", "ربح", "خسر", "استثمر", "انفق", "وفر", "ادخر", "اقرض", "استدان", "رهن",
    "اشترى", "باع", "تبادل", "صدر", "استورد", "انتج", "استهلك", "طلب", "عرض",
    "نافس", "احتكر", "سعر", "قيم", "خمن", "ضمن", "امن", "استام", "تسوق",

    # الجسد والصحة
    "شفى", "مرض", "داوى", "عالج", "ضمد", "جبر", "بتر", "بصر", "سمع", "ذاق",
    "شم", "لمس", "احس", "الم", "وجع", "اعيا", "نعس", "صحا", "نشط", "خمل",
    "هزل", "سمن", "قوى", "ضعف", "كبر", "صغر", "نضج", "شاخ", "شب", "ترعرع",

    # الاجتماع والسياسة
    "حكم", "ادار", "قاد", "وجه", "اشرف", "رقب", "حرس", "دافع", "هاجم", "صالح",
    "تفاوض", "عاهد", "ائتلف", "تحالف", "انشق", "تمرد", "ثار", "اطاع", "خضع",
    "انتخب", "عين", "اقال", "استقال", "ترشح", "نال", "حاز", "تولى", "نظم", "رتب",
    "جمع", "فرق", "حشد", "ناضل", "كافح", "جاهد", "صارع", "دافع", "حمى", "وقى",

    # الفن والجمال
    "رسم", "لون", "نحت", "صور", "زخرف", "جمل", "زين", "نظم", "شعر", "غنى",
    "عزف", "رقص", "مثل", "ابدع", "خلق", "صنع", "نسج", "حاك", "طرز", "زرد",
    "سرد", "قص", "حكى", "روى", "وصف", "نقش", "خطط", "رتب", "نسق", "ناسب",

    # العلوم والتقنية
    "برمج", "رمز", "شفر", "حوسب", "اوتمت", "دقق", "احكم", "ضبط", "ضبط", "معير",
    "قاس", "اختبر", "فحص", "رصد", "تتبع", "سجل", "ارسل", "استقبل", "نقل", "خزن",
    "استرجع", "حلل", "فسر", "صنف", "رتب", "نمذج", "محاكى", "توقع", "تنبا", "استشرف",
    "حوسب", "خوارزم", "شفر", "كود", "هندس", "انترب", "كمم", "رنن", "طفر", "استقر", "تزامن", "تذبذب", "تراكب", "تشتت",

    # الدين والروح
    "عبد", "صلى", "صام", "زكى", "حج", "دعا", "تضرع", "توكل", "توب", "استغفر",
    "شكر", "حمد", "سبح", "كبر", "تلا", "فقه", "تدبر", "تفكر", "تامل", "خشع",
    "بكى", "خاف", "رجا", "امل", "يئس", "قنط", "توسل", "ابتهل", "ناجى", "اذعن",
    "اسلم", "امن", "تقوى", "ورع", "زهد", "قنع", "عبد", "نسك", "تبتل", "اخبت",

    # الحرب والسلام
    "قاتل", "جاهد", "دافع", "هاجم", "حاصر", "فتح", "استسلم", "صالح", "هادن",
    "انتصر", "انهزم", "فر", "ثبت", "استبسل", "ضحى", "فدى", "انقذ", "نجا", "هلك",
    "اسر", "فدى", "فك", "حرر", "اطلق", "قيد", "اعتقل", "سجن", "طرد", "نفى",

    # الذكاء والعقل
    "ذكى", "فطن", "نبه", "استوعب", "تفطن", "ادرك", "لمح", "حدس", "تنبا", "توقع",
    "استنتج", "استنبط", "استخلص", "استدل", "قاس", "فحص", "دقق", "تمحص", "نظر",
    "تامل", "روى", "تدبر", "تفكر", "تخيل", "تصور", "تجريد", "عقل", "فكر", "قدر",

    # المشاعر والوجدان
    "فرح", "حزن", "غضب", "خاف", "امل", "يئس", "احب", "كره", "اشتاق", "حن",
    "وجد", "هيم", "عشق", "تاق", "شوق", "الف", "ود", "برم", "سئم", "ضجر",
    "دهش", "تعجب", "استغرب", "انبهر", "انذهل", "ارتبك", "اضطرب", "هدا", "سكن",
    "طمان", "اطمان", "توتر", "قلق", "اشفق", "عطف", "رثى", "شفق", "رق", "حنن",

    # الكون والفلسفة
    "وجد", "عدم", "كون", "فنى", "ابد", "ازل", "تعال", "تنزه", "اطلق", "قيد",
    "وحد", "كثر", "جزا", "كمل", "نقص", "عظم", "حقر", "شرف", "هان", "رفع",
    "خلد", "فنى", "حدث", "قدم", "اول", "اخر", "تقدم", "تاخر", "سبق", "لحق",

    # الطبيعة والكون
    "سما", "ارض", "جبل", "بحر", "نهر", "صحر", "غاب", "شجر", "زهر", "ثمر",
    "حجر", "تراب", "رمل", "طين", "ماء", "هواء", "نار", "ضوء", "ظلم", "شمس",
    "قمر", "نجم", "كوكب", "سحب", "مطر", "ثلج", "برد", "حر", "رياح", "عاصف",

    # الأسرة والمجتمع
    "ولد", "بنت", "اب", "ام", "اخ", "اخت", "زوج", "بيت", "اسر", "عشير",
    "نسب", "حسب", "قبيل", "عشيرة", "امة", "شعب", "جمهور", "حشد", "جيل", "تاريخ",
    "موروث", "تراث", "حضار", "تمدن", "تعليم", "تربية", "نشا", "ترعرع", "شب", "كبر",

    # جذور بلاغية وأدبية خاصة بالقوالب الجديدة
    "بلغ", "فصح", "بين", "برع", "تقن", "اتقن", "احسن", "اجاد", "ابدع", "ابتكر",
    "سجع", "نظم", "قفى", "عرض", "بدع", "طرب", "انشد", "اوزن", "قافى", "رجز",
    "مدح", "هجا", "رثى", "اطرى", "قرظ", "اثنى", "كيل", "نال", "تنقص", "ذم",
    "قدح", "اصاب", "اخطا", "صوب", "قوم", "عدل", "اصلح", "قحم", "جرا", "اقدم",
    "لاحج", "احتج", "دافع", "فند", "نقض", "رد", "ابطل", "اثبت", "اكد", "قرر",
    "شكك", "ارجح", "رجح", "ميل", "انحاز", "حيد", "انصف", "قسط", "جار", "ظلم",
    "تواضع", "تكبر", "افتخر", "تباهى", "تعجب", "دهش", "انذهل", "انبهر",
    "تهكم", "سخر", "استهزا", "عرض", "كنى", "لمز", "طعن", "انتقد",
    "حذر", "نبه", "انذر", "خوف", "رهب", "حث", "رغب", "شوق", "حرض",
    "تعزز", "توطد", "تاسس", "رسخ", "ثبت", "استقر", "تمكن", "ترسخ",
    "تقلص", "ضعف", "وهن", "زال", "انهار", "انكسر", "اندثر", "اندمر", "بلى",
    "حكم", "امثل", "ضرب", "مثل", "استشهد", "استدل", "احتج", "استند", "عزز",

    # الرياضيات والمنطق
    "جبر", "هندس", "حسب", "عدد", "رقم", "صفر", "واحد", "اثنان", "ثلث", "ربع",
    "خمس", "سدس", "عشر", "مئة", "الف", "ضعف", "مثلث", "مربع", "مكعب", "قطر",
    "محور", "زاوي", "وتر", "ضلع", "قوس", "دائرة", "نصف", "كسر", "بسط", "مقام",
    "نسبة", "تناسب", "متوسط", "مجموع", "جداء", "فرق", "باقي", "معدل", "تطابق", "تشابه",
    "احتمال", "توزيع", "منحنى", "دالة", "متغير", "ثابت", "حد", "مجال", "مدى", "اشتقاق",
    "تكامل", "مشتق", "معدل", "تغير", "تفاضل", "نهاي", "سلسلة", "متتالية", "منطق", "استقرا",

    # البرمجة والحوسبة
    "برمج", "شفر", "رمز", "خوارزم", "بيان", "حوسب", "معالج", "مخزن", "ذاكر", "قرص",
    "شاشة", "لوحة", "طابع", "ماسح", "شبكة", "خادم", "عميل", "متصفح", "موقع", "صفحة",
    "رابط", "تشعب", "بيانات", "ملف", "مجلد", "نظام", "تطبيق", "برنامج", "واجهة", "قاعدة",
    "استعلام", "جدول", "سجل", "حقل", "مفتاح", "فهرس", "نسخ", "لصق", "حذف", "ادراج",
    "تعديل", "استعلام", "ترتيب", "تصفية", "بحث", "استبدال", "تحويل", "دمج", "ربط", "فصل",

    # المنطق والفلسفة
    "علة", "سبب", "نتيجة", "مقدم", "تالي", "شرط", "جزاء", "استلزام", "استنتاج", "استدلال",
    "قياس", "تمثيل", "تجريد", "تعميم", "تخصيص", "تحليل", "تركيب", "تناقض", "تضاد", "تقابل",
    "افتراض", "مسلمة", "نظرية", "برهان", "دليل", "حجة", "بينة", "شاهد", "قرينة", "علامة",

    # العلوم والتجارب
    "تجرب", "مختبر", "عينة", "محلول", "تفاعل", "انحلال", "ترسيب", "تقطير", "ترشيح", "تبلور",
    "تسخين", "تبريد", "ضغط", "تمدد", "انكماش", "تأين", "تأكسد", "اختزال", "تحفيز", "تعادل",
    "خلية", "نسيج", "عضو", "جهاز", "كائن", "احياء", "وراث", "طفرة", "جين", "كروموسوم",
    "بكتير", "فيروس", "طفيلي", "فطر", "طحلب", "نبات", "حيوان", "انسان", "خمير", "عقار",
    "فضاء", "نجم", "كوكب", "مدار", "قمر", "شمس", "مجرة", "سديم", "ثقب", "انفجار",

    # الاقتصاد والاعمال
    "سوق", "مال", "مصرف", "سهم", "سند", "صك", "عقد", "اتفاق", "شراك", "مشروع",
    "ارباح", "خسائر", "ايرادات", "نفقات", "ديون", "اصول", "خصوم", "سيولة", "ارباح", "ضريبة",
    "تكلف", "سعر", "قيمة", "اجرة", "راتب", "اجر", "عمولة", "ربح", "فائدة", "هامش",
    "عرض", "طلب", "انتاج", "استهلاك", "تصدير", "استيراد", "جمرك", "تعرفة", "حصة", "سهم",

    # الفنون والثقافة
    "ثقاف", "تراث", "موروث", "ادب", "شعر", "نثر", "رواي", "قصة", "مسرح", "سينما",
    "موسيق", "لحن", "نغم", "ايقاع", "صوت", "غناء", "رقص", "تمثيل", "اخراج", "سيناريو",
    "لوحة", "تمثال", "نحت", "زخرف", "خطاط", "تصوير", "فوتوغراف", "جرافيك", "كاريكاتير", "هندام",

    # الإدارة والقيادة
    "خطط", "نظم", "ادار", "قاد", "وجه", "اشرف", "راقب", "قيم", "حفز", "درب",
    "فوض", "فات", "كلف", "ندب", "استأجر", "عاقب", "كافا", "شجع", "ثبط", "حاسب",
    "دون", "سجل", "بلغ", "احصى", "صنف", "رتب", "نسق", "وزع", "كلف", "اسند",

    # الصفات والمشاعر الإضافية
    "نبيل", "شريف", "كريم", "لطيف", "ظريف", "وسيم", "جميل", "انيق", "انيق", "رائع",
    "بهيج", "مرح", "نشيط", "كسول", "شجاع", "جبان", "حكيم", "جاهل", "عاقل", "مجنون",
    "صبور", "عجول", "غيور", "حقود", "ودود", "رؤوف", "عنيف", "لين", "قوي", "ضعيف",
    "دقيق", "غامض", "واضح", "جلي", "خفي", "مبهم", "مستقيم", "معوج", "فاسد", "صالح",
]

_db = None

def _get_db():
    global _db
    if _db is None:
        _db = LetterDB()
    return _db

def _compute_root_pv(root: str):
    if re.search(r'[a-zA-Z]', root):
        base = _eng_embedding.get_word_semantic_vector(root)
        if base.shape[0] < TOTAL_DIM:
            base = np.concatenate([base, np.zeros(TOTAL_DIM - base.shape[0])])
        else:
            base = base[:TOTAL_DIM]
        nrm = np.linalg.norm(base)
        if nrm > 1e-10:
            base = base / nrm
        return base

    db = _get_db()
    letters = [ch for ch in root if db.has(ch)]
    pvs = np.array([db.get_vector(ch) for ch in letters])
    if len(pvs) == 0:
        return np.zeros(TOTAL_DIM)
    avg = np.mean(pvs, axis=0)
    base = np.concatenate([avg, np.zeros(TOTAL_DIM - avg.shape[0])]) if avg.shape[0] < TOTAL_DIM else avg[:TOTAL_DIM]
    nrm = np.linalg.norm(base)
    if nrm > 1e-10:
        base = base / nrm
    return base


# -------- Weight Patterns --------

_WEIGHTS = [
    ("فَعَلَ", "", "", "", "basic perfective"),
    ("فَعَّلَ", "", "shadda_on_2nd", "", "intensive/causative"),
    ("فَاعَلَ", "", "alif_after_1st", "", "reciprocal"),
    ("أَفْعَلَ", "أ", "", "", "causative"),
    ("تَفَعَّلَ", "ت", "shadda_on_2nd", "", "reflexive"),
    ("تَفَاعَلَ", "ت", "alif_after_1st", "", "mutual"),
    ("اِنْفَعَلَ", "ان", "", "", "passive/reflexive"),
    ("اِفْتَعَلَ", "ا", "t_after_1st", "", "reflexive"),
    ("اِفْعَلَّ", "ا", "shadda_on_3rd", "", "stative"),
    ("مَفْعُول", "م", "waw_after_2nd", "", "passive participle"),
    ("فَعِيل", "", "ya_after_2nd", "", "adjective"),
    ("فَعَّال", "", "shadda_on_2nd", "ال", "intensive adjective"),
    ("مُفَعِّل", "م", "shadda_on_2nd", "", "active participle"),
    ("فُعُول", "", "waw_after_2nd", "", "plural pattern"),
    ("أَفْعَال", "ا", "", "ال", "broken plural"),
]

_POS_MAP = {
    "فَعَلَ": "verb",
    "فَعَّلَ": "verb",
    "فَاعَلَ": "verb",
    "أَفْعَلَ": "verb",
    "تَفَعَّلَ": "verb",
    "تَفَاعَلَ": "verb",
    "اِنْفَعَلَ": "verb",
    "اِفْتَعَلَ": "verb",
    "اِفْعَلَّ": "verb",
    "مَفْعُول": "noun",
    "فَعِيل": "adj",
    "فَعَّال": "adj",
    "مُفَعِّل": "noun",
    "فُعُول": "noun",
    "أَفْعَال": "noun",
}


# -------- Syntactic Cases (إعراب) --------

_CASES = {
    "مرفوع": {"anchor_weight": 0.4, "oscillation": 0.0, "label": "nominative"},
    "منصوب": {"anchor_weight": 0.2, "oscillation": 0.3, "label": "accusative"},
    "مجرور": {"anchor_weight": 0.1, "oscillation": 0.5, "label": "genitive"},
}


# -------- Stemming helpers --------

_DIACRITICS = str.maketrans({c: None for c in 'ًٌٍَُِّْـ'})
_PREFIXES = ['ال', 'بال', 'كال', 'فل', 'ول', 'بل', 'فال', 'وال', 'ب', 'ف', 'ك', 'ل', 'و', 'س']
_PRONOUNS = ['هم', 'هن', 'هما', 'ها', 'ه', 'كم', 'كن', 'كما', 'ك', 'نا', 'ني', 'ي']
_SUFFIXES = ['ها', 'هم', 'هن', 'كم', 'كن', 'نا', 'ني', 'كي', 'ك', 'ه', 'ي', 'ا', 'ون', 'ين', 'ات', 'ان', 'وا', 'تا', 'ت', 'ة']

def _stem(word: str) -> str:
    # Check if word is English
    if re.search(r'[a-zA-Z]', word):
        return word.lower().strip()
        
    w = word.translate(_DIACRITICS)
    w = _normalize_letters(w)
    if w.startswith('ال') and len(w) > 4:
        w = w[2:]
    for p in sorted(_PREFIXES, key=len, reverse=True):
        if w.startswith(p) and len(w) > len(p) + 2:
            if len(p) > 1 or (len(w) - len(p) >= 4):
                w = w[len(p):]
                break
    for pr in sorted(_PRONOUNS, key=len, reverse=True):
        if w.endswith(pr) and len(w) > len(pr) + 2:
            w = w[:-len(pr)]
            break
    for s in sorted(_SUFFIXES, key=len, reverse=True):
        if w.endswith(s) and len(w) - len(s) >= 3:
            w = w[:len(w)-len(s)]
            break
    return w

_EXTRACT_CACHE = {}

def _extract_root(word: str):
    if word in _EXTRACT_CACHE:
        return _EXTRACT_CACHE[word]
        
    if re.search(r'[a-zA-Z]', word):
        root, confidence = _english_extractor.extract(word)
    else:
        root, confidence = _arabic_extractor.extract(word)
    
    if confidence >= 0.8:
        _EXTRACT_CACHE[word] = root
        return root
        
    stem = _stem(word)
    best = None
    best_score = 0
    for r in _ROOTS:
        rn = _normalize_letters(r)
        if len(stem) < 3 or len(rn) < 3:
            continue
        si = 0
        matched = 0
        for rl in rn:
            idx = stem.find(rl, si)
            if idx >= 0:
                matched += 1
                si = idx + 1
        if matched >= 2 and rn[0] in stem[:3]:
            if matched == len(rn) and len(rn) <= 3:
                _EXTRACT_CACHE[word] = r
                return r
            if matched > best_score or (matched == best_score and len(rn) < len(best or '')):
                best = r
                best_score = matched
                
    if best and best_score >= 2:
        _EXTRACT_CACHE[word] = best
        return best
        
    _EXTRACT_CACHE[word] = root
    return root


def _infer_weight(stem, root):
    rn = _normalize_letters(root)
    pos = []
    si = 0
    for rl in rn:
        idx = stem.find(rl, si)
        if idx >= 0:
            pos.append(idx)
            si = idx + 1
        else:
            return None
    if len(pos) < 3:
        return None
    p0, p1, p2 = pos[0], pos[1], pos[2]
    L = len(stem)
    # Check for known patterns by stem structure
    if stem.startswith('م') and p0 == 1:
        if L >= 5 and stem.find('و', p1 + 1) >= 0:
            return "مَفْعُول"
        return "مُفَعِّل"
    if stem.startswith('ت') and p0 == 1:
        if L >= 5 and 'ا' in stem[p1+1:p2]:
            return "تَفَاعَلَ"
        if L >= 5 and p2 >= 3 and stem[p1] == stem[p1+1] if p1+1 < L else False:
            return "تَفَعَّلَ"
        return "تَفَعَّلَ"
    if stem.startswith('ا') and L >= 5:
        if p0 == 1 and 'ت' in stem[2:4]:
            return "اِفْتَعَلَ"
        if p0 == 1 and stem[1:3] == 'ن' + rn[0]:
            return "اِنْفَعَلَ"
    if stem.startswith('أ'):
        return "أَفْعَلَ"
    if L >= 4 and p0 == 0 and p1 >= 2 and 'ا' in stem[p0+1:p1]:
        return "فَاعَلَ"
    if L >= 5 and p2 >= 3 and p1 == 2 and p2 == 4 and 'ي' in stem[p1+1:p2]:
        return "فَعِيل"
    if L >= 4 and p0 == 0 and p1 == 1 and p2 == 3:
        return "فَعِيل"
    if L >= 3 and p0 == 0 and p1 == 1 and p2 == 2:
        return "فَعَلَ"
    if L >= 4 and p0 == 0 and p1 == 1 and p2 >= 3 and len(stem) >= 5 and stem[1] == stem[2]:
        return "فَعَّلَ"
    return "فَعَلَ"


# -------- Phase trajectory computation --------

def _weight_offset(weight_name):
    from src.physics.weight_resonance import _WEIGHT_EMBEDDINGS
    base = _WEIGHT_EMBEDDINGS.get(weight_name, np.zeros(22))
    return np.concatenate([base, np.zeros(TOTAL_DIM - 22)])

def _pad_to_total(v):
    if len(v) < TOTAL_DIM:
        return np.concatenate([v, np.zeros(TOTAL_DIM - len(v))])
    return v[:TOTAL_DIM]


# -------- Main Engine --------

class MorphoPhasicEngine:
    def __init__(self):
        self._root_pv_cache = {}
        self._case_cache = {}
        self._pos_cache = {}
        self._analyze_cache = {}

    def _get_root_pv(self, root):
        if root not in self._root_pv_cache:
            self._root_pv_cache[root] = _compute_root_pv(root)
        return self._root_pv_cache[root]

    def get_pos(self, word: str):
        if word in self._pos_cache:
            return self._pos_cache[word]
        a = self.analyze(word)
        if not a['has_morph'] or a['weight'] is None:
            self._pos_cache[word] = None
            return None
        pos = _POS_MAP.get(a['weight'], None)
        self._pos_cache[word] = pos
        return pos

    def analyze(self, word: str):
        if word in self._analyze_cache:
            return self._analyze_cache[word]
        w = _normalize_letters(word)
        stem = _stem(w)
        root = _extract_root(w)
        if root is None:
            result = {"root": None, "weight": None, "stem": stem, "has_morph": False}
        else:
            weight = _infer_weight(stem, root) or "فَعَلَ"
            result = {"root": root, "weight": weight, "stem": stem, "has_morph": True}
        self._analyze_cache[word] = result
        return result

    def compute_morph_phase(self, word: str, syntactic_case="مرفوع"):
        analysis = self.analyze(word)
        if not analysis["has_morph"] or analysis["root"] is None:
            return None

        root_pv = self._get_root_pv(analysis["root"])
        weight_off = _pad_to_total(_weight_offset(analysis["weight"]))

        case = _CASES.get(syntactic_case, _CASES["مرفوع"])
        case_anchor = np.array([case["anchor_weight"]] * TOTAL_DIM)
        oscillation = case["oscillation"]
        case_mod = case_anchor * (1.0 + oscillation * np.sin(np.arange(TOTAL_DIM) * 0.5))

        morph_pv = root_pv + weight_off * 0.3 + case_mod * 0.1
        nrm = np.linalg.norm(morph_pv)
        if nrm > 1e-10:
            morph_pv = morph_pv / nrm
        return morph_pv

    def score(self, word: str, prev_word: str = None, syntactic_case="مرفوع"):
        morph_pv = self.compute_morph_phase(word, syntactic_case)
        if morph_pv is None:
            return 0.0
        word_pv = compute_extended_phase_vector(word)
        align = float(np.mean(np.cos(morph_pv - word_pv)))
        score = max(0.0, (align - 0.7) * 3.0)
        return score

    def transition_score(self, word: str, prev_word: str = None):
        if prev_word is None:
            return 0.0
        wa = self.analyze(word)
        pa = self.analyze(prev_word)
        if not wa["has_morph"] or not pa["has_morph"]:
            return 0.0
        w_pv = compute_extended_phase_vector(word)
        p_morph = self.compute_morph_phase(prev_word)
        if p_morph is None:
            return 0.0
        align = float(np.mean(np.cos(w_pv - p_morph)))
        return max(0.0, (align - 0.6) * 2.0)
