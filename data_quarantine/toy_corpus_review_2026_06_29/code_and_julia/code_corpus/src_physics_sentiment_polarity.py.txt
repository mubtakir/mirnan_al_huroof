"""SentimentPolarity — تحليل المشاعر في فضاء الطور.

معجم قطبية للمفردات العربية + دالة احتساب polarity للجملة.
يُستخدم في _score() لضمان اتساق المشاعر بين المدخل والرد.

لا شبكات عصبية. لا باك بروب. مجرد معجم + استدلال.
"""
import re
import numpy as np

_POSITIVE = {
    # مشاعر إيجابية عامة
    'جميل', 'رائع', 'ممتاز', 'طيب', 'حسن', 'جيد', 'موفق', 'سعيد', 'فرح',
    'نشاط', 'حب', 'سلام', 'خير', 'نور', 'أمل', 'صحة', 'نجاح', 'جمال',
    'كرم', 'شجاع', 'ودود', 'رحيم', 'كريم', 'صبور', 'عفيف', 'شريف',
    'فاضل', 'عادل', 'صادق', 'أمين', 'وفي', 'نقي', 'طاهر',
    'بهجة', 'سرور', 'انشراح', 'رضا', 'يقين', 'ثقة', 'اطمئنان',
    # أفعال إيجابية
    'أحب', 'أفرح', 'أسعد', 'أنجح', 'أكرم', 'أعطي', 'أسامح',
    'يسر', 'يفرح', 'ينجح', 'يكرم', 'يعطي',
    # إنجليزية
    'good', 'great', 'excellent', 'wonderful', 'beautiful', 'nice',
    'happy', 'love', 'peace', 'success', 'hope', 'perfect',
    'amazing', 'fantastic', 'brilliant', 'awesome', 'superb',
    'thank', 'thanks', 'grateful', 'appreciate',
}

_NEGATIVE = {
    # مشاعر سلبية عامة
    'سيء', 'قبيح', 'خطأ', 'مشكلة', 'صعب', 'متعب', 'مزعج', 'مؤلم',
    'حزين', 'غضب', 'كره', 'حرب', 'شر', 'ظلم', 'فشل', 'مرض',
    'جهل', 'خوف', 'كذب', 'خيانة', 'غش', 'حسد', 'بغض', 'نفاق',
    'رياء', 'غرور', 'كبر', 'قسوة', 'جشع', 'ظلام', 'ضيق', 'هم',
    'غم', 'كرب', 'بلاء', 'محنة', 'ضلال',
    # أفعال سلبية
    'أكره', 'أغضب', 'أحزن', 'أفشل', 'أظلم', 'أكذب', 'أخون',
    'يسوء', 'يحزن', 'يغضب', 'يفشل', 'يخون', 'يكذب',
    # نفي + كلمات سلبية
    'لا يعمل', 'لا يصلح', 'لا يجدي', 'ليس جيدا',
    # إنجليزية
    'bad', 'terrible', 'awful', 'horrible', 'ugly', 'wrong',
    'problem', 'difficult', 'tired', 'annoying', 'painful',
    'sad', 'anger', 'hate', 'war', 'evil', 'failure', 'sick',
    'fear', 'lie', 'betray', 'cheat', 'fake', 'broken',
    'error', 'bug', 'crash', 'worse', 'worst',
}

# كلمات تعزيز — تزيد من حدة المشاعر
_INTENSIFIERS = {
    'جداً', 'جدًا', 'كثيراً', 'كثيرًا', 'جدا', 'للغاية',
    'بالغ', 'شديد', 'عظيم', 'هائل', 'ضخم',
    'very', 'so', 'extremely', 'incredibly', 'really',
    'highly', 'totally', 'completely',
}

# كلمات عكس — تقلب المشاعر
_NEGATORS = {
    'ليس', 'ليست', 'لستم', 'لست', 'ليسوا',
    'لا', 'لم', 'لن', 'ما', 'غير', 'دون',
    'not', "n't", 'never', 'no', 'without',
}


def compute_word_polarity(word):
    """Polarity of a single word: +1 (positive), -1 (negative), 0 (neutral)."""
    w = word.lower().strip('.,:;!?()[]{}"""\'`')
    if w in _POSITIVE:
        return 1.0
    if w in _NEGATIVE:
        return -1.0
    return 0.0


def compute_sentence_polarity(text):
    """Polarity of a sentence: -1 to +1.
    
    Accounts for intensifiers and negators.
    """
    words = text.split()
    if not words:
        return 0.0
    
    total = 0.0
    n = 0
    negate = False
    i = 0
    while i < len(words):
        w = words[i].lower().strip('.,:;!?()[]{}"\'-')
        # نفي
        if w in _NEGATORS:
            negate = not negate
            i += 1
            continue
        # تعزيز — ننظر للكلمة التالية
        if w in _INTENSIFIERS and i + 1 < len(words):
            next_w = words[i + 1].lower().strip('.,:;!?()[]{}"\'-')
            polarity = compute_word_polarity(next_w)
            if polarity != 0:
                total += 1.5 * polarity if not negate else -1.5 * polarity
                n += 1
                i += 2
                negate = False
                continue
        # كلمة عادية
        polarity = compute_word_polarity(w)
        if polarity != 0:
            total += polarity if not negate else -polarity
            n += 1
        negate = False
        i += 1
    
    if n == 0:
        return 0.0
    return max(-1.0, min(1.0, total / n))


def sentiment_fidelity(word, context_words):
    """How well does the word's polarity match the context polarity?
    
    Returns 0.0 to 1.0: 1.0 = perfect match, 0.0 = complete mismatch.
    """
    if not context_words:
        return 0.5
    
    ctx_polarity = compute_sentence_polarity(' '.join(context_words[-8:]))
    word_polarity = compute_word_polarity(word)
    
    if ctx_polarity == 0.0:
        return 0.5  # محايد — لا تأثير
    if word_polarity == 0.0:
        return 0.3  # كلمة محايدة في سياق عاطفي — غير مثالية
    
    # هل القطبية متوافقة؟
    match = (ctx_polarity * word_polarity) > 0
    if match:
        return min(1.0, abs(ctx_polarity) + 0.3)
    else:
        return max(0.0, 1.0 - abs(ctx_polarity))
