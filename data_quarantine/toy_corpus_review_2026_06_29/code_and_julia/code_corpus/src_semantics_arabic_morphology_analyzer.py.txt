#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محلل الصرف العربي الفطري
Arabic Morphological Analyzer
================================
برنامج مستقل تمامًا — لا يعتمد على أي مكتبات خارجية
يحدد نوع الكلمة العربية: اسم | صفة | حال | فعل
"""

import re
import sys


# ─── القوائم المغلقة ─────────────────────────────────────────────────────────

PRONOUNS = {
    'أنا', 'أنت', 'أنتِ', 'أنتما', 'أنتم', 'أنتن',
    'هو', 'هي', 'هما', 'هم', 'هن', 'نحن',
    'إياي', 'إياك', 'إياكِ', 'إياه', 'إياها',
    'إيانا', 'إياكم', 'إياهم', 'إياكن', 'إياهن',
}

DEMONSTRATIVES = {
    'هذا', 'هذه', 'هذان', 'هاتان', 'هؤلاء',
    'ذلك', 'تلك', 'ذانك', 'تانك', 'أولئك',
    'هنا', 'هناك', 'ثمة', 'ثَمَّ',
}

RELATIVES = {
    'الذي', 'التي', 'اللذان', 'اللتان',
    'الذين', 'اللواتي', 'اللاتي', 'اللائي',
    'ما', 'من', 'مهما', 'أيّ', 'أيّما',
}

CONJUNCTIONS = {'و', 'ف', 'ثم', 'أو', 'أم', 'لكن', 'بل', 'حتى', 'إذ', 'إذا', 'لما', 'كلما'}

PREPOSITIONS = {
    'في', 'من', 'إلى', 'على', 'عن', 'مع', 'عند', 'لدى',
    'حول', 'بين', 'أمام', 'خلف', 'فوق', 'تحت',
    'قبل', 'بعد', 'منذ', 'خلال', 'حتى', 'كي',
}

PARTICLES = {'لم', 'لن', 'لا', 'ما', 'قد', 'سوف', 'سـ', 'إن', 'أن', 'كي', 'هل', 'أ'}

FUNCTION_WORDS = CONJUNCTIONS | PREPOSITIONS | PARTICLES

# أسماء لا تبدأ بـ ي/ت/ن/أ لكن قد يُخطئ البرنامج في تصنيفها
KNOWN_NOUNS = {
    'يد', 'يوم', 'يمين', 'يسار', 'نهر', 'نور', 'نار',
    'تراب', 'تمر', 'تفاح', 'تفاحة', 'أمل', 'أخ', 'أخت',
    'أرض', 'أسد', 'أسرة', 'نعجة', 'نعمة',
}


# ─── دوال مساعدة ─────────────────────────────────────────────────────────────

def strip_diacritics(word: str) -> str:
    """إزالة علامات التشكيل (حركات) من الكلمة"""
    diacritics = re.compile(r'[\u064B-\u065F\u0670]')
    return diacritics.sub('', word)


def strip_al(word: str) -> tuple[str, bool]:
    """
    إزالة ال التعريف إن وُجدت.
    تُعيد (الكلمة_بدون_ال، هل_كان_فيها_ال)
    """
    s = strip_diacritics(word)
    # فقط نزيل "ال" الكاملة (حرفان) وليس الألف المنفردة
    if len(s) > 2 and s[:2] == 'ال':
        return s[2:], True
    if len(s) > 3 and s[:3] == 'آل':
        return s[3:], True
    return s, False



# ─── قاموس الاستثناءات الصرفية ─────────────────────────────────────────────
# كلمات أصيلة تبدأ بحروف تُوهم السوابق — لا تُجرَّد منها
LEXICAL_EXCEPTIONS = {
    # تبدأ بـ "و" — ليست عطفاً
    'وجد', 'وجه', 'وطن', 'وقت', 'ولد', 'وصل', 'ورد', 'وضع', 'ورق', 'وسط',
    'وحيد', 'وداع', 'ودود', 'وفاء', 'وفاة', 'وحشة', 'وعد', 'ولي', 'وراء',
    'وثيق', 'وليد', 'ومضة', 'وهم', 'وهج', 'وميض',
    # تبدأ بـ "ل" — ليست لام الجر
    'لبن', 'لون', 'لغة', 'لحظة', 'لعب', 'لجأ', 'لذة', 'لطف', 'لمس',
    'لواء', 'لهيب', 'لجنة', 'لئيم', 'لحم', 'لؤلؤ', 'لباس', 'لسان',
    'لحاء', 'لقاء', 'لحد', 'لتر', 'لصق', 'لغو',
    # تبدأ بـ "ب" — ليست باء الجر
    'بيت', 'بحر', 'بدر', 'برق', 'بئر', 'بدن', 'بطل', 'بكاء', 'بلاغ',
    'بنية', 'بهجة', 'بصر', 'بساط', 'بسمة', 'براء', 'باب', 'بدع',
    # تبدأ بـ "ف" — ليست فاء العطف
    'فجر', 'فخر', 'فكر', 'فقر', 'فهم', 'فلك', 'فطر', 'فيض', 'فضل',
    'فراق', 'فضاء', 'فداء', 'فتنة',
    # تبدأ بـ "ك" — ليست كاف التشبيه
    'كتاب', 'كلام', 'كرم', 'كمال', 'كنز', 'كثير', 'كيف', 'كذب', 'كسب',
}


def strip_prep_prefix(word: str, known_vocab: set | None = None) -> str:
    """
    إزالة حروف الجر/العطف المتصلة (ب، ل) — بصورة سياقية ذكية.

    المبدأ: نتحقق أولاً من أن الكلمة الكاملة ليست كلمةً مستقلةً معروفة.
    مثال:
      - 'لبن'  → لا نُزيل اللام (لأن "لبن" كلمة مستقلة تعني الحليب)
      - 'لمحمد' → نُزيل اللام (ويبقى "محمد")
      - 'وجد'  → لا نُزيل الواو (لأن "وجد" كلمة مستقلة)
      - 'بيت'  → لا نُزيل الباء (لأن "بيت" كلمة مستقلة)
    """
    s = strip_diacritics(word)

    # 1. هل الكلمة من الاستثناءات الصريحة؟ → لا تُجرَّد
    if s in LEXICAL_EXCEPTIONS:
        return s

    # 2. هل الكلمة موجودة في معجم النموذج النشط؟ → لا تُجرَّد
    if known_vocab and s in known_vocab:
        return s

    # 3. هل الجزء بعد إزالة السابقة منطقي (3 أحرف+)؟ → نُزيل
    if len(s) >= 4 and s[0] in 'بل':
        remainder = s[1:]
        # تأكد من أن الجزء المتبقي ليس كلمة أصيلة ذات معنى مفرد
        if len(remainder) >= 2 and remainder not in LEXICAL_EXCEPTIONS:
            if not (known_vocab and remainder not in known_vocab and len(remainder) < 3):
                return remainder
    return s



def has_tanwin_nasb(word: str) -> bool:
    """هل تنتهي الكلمة بتنوين النصب (ً قد يسبق الألف أو يأتي في النهاية)؟"""
    # تنوين النصب Unicode: \u064B
    # يكون إما في آخر الكلمة أو قبل الألف الأخيرة (راكضًا = ض + ً + ا)
    return '\u064B' in word or word.endswith('اً') or word.endswith('ً')


def tokenize(text: str) -> list[str]:
    """تقسيم النص إلى كلمات"""
    return [w for w in text.strip().split() if w]


# ─── المحلل الرئيسي ──────────────────────────────────────────────────────────

def analyze_word(word: str, context_words: list[str] | None = None) -> dict:
    """
    تحليل كلمة عربية واحدة وتحديد نوعها الصرفي.

    المُدخلات:
        word          - الكلمة المراد تحليلها
        context_words - قائمة كلمات الجملة كاملةً للاستفادة من السياق (اختياري)

    المُخرجات:
        dict يحتوي على:
            'word'       - الكلمة الأصلية
            'type'       - نوع الكلمة: فعل | اسم | صفة | حال
            'reasons'    - قائمة مبررات القرار
    """
    reasons = []

    # ── المعالجة المسبقة ────────────────────────────────────────────────────
    plain        = strip_diacritics(word)
    stripped_pre = strip_prep_prefix(plain)
    core, has_al = strip_al(stripped_pre)
    s            = core  # الجذر العملي للفحص

    if not s or len(s) < 2:
        return {'word': word, 'type': 'غير معروف', 'reasons': ['الكلمة قصيرة جدًا أو فارغة']}

    # ── الخطوة 1: القوائم المغلقة (ضمائر، إشارة، موصولة) ───────────────────
    for lst, label in [
        (PRONOUNS,       'ضمير منفصل'),
        (DEMONSTRATIVES, 'اسم إشارة'),
        (RELATIVES,      'اسم موصول'),
    ]:
        if plain in lst or s in lst:
            reasons.append(f'{label} من القوائم المغلقة → اسم')
            return {'word': word, 'type': 'اسم', 'reasons': reasons}

    # أدوات وحروف — لا تُحلَّل صرفيًا
    if plain in FUNCTION_WORDS:
        reasons.append('حرف أو أداة وظيفية → مبني، لا تُحلَّل صرفيًا')
        return {'word': word, 'type': 'أداة', 'reasons': reasons}

    # ── الخطوة 2: الفعل — الأولوية القصوى ──────────────────────────────────

    # 2أ: ضمائر الرفع المتصلة — نفحص الكلمة الأصلية (مع التشكيل) أيضًا
    verb_suffix_rules = [
        (r'[\u062A][\u064F]$|[\u062A][\u064E]$|[\u062A][\u0650]$',  'تاء الفاعل (ضمير رفع متحرك)'),
        (r'نا$',           'نا الفاعلين (ضمير رفع)'),
        (r'نَ$|نَّ$',      'نون النسوة (ضمير رفع)'),
        (r'ون$|وا$',       'واو الجماعة (ضمير رفع)'),
        (r'ين$',           'ياء المخاطبة + نون'),
        (r'نَّ$',          'نون التوكيد الثقيلة'),
        (r'نْ$',           'نون التوكيد الخفيفة'),
    ]
    # فحص تاء الفاعل في الكلمة الأصلية (مع التشكيل)
    if re.search(r'[تط][\u064F\u064E\u0650]$', word):
        reasons.append('اتصلت بها تاء الفاعل المتحركة → فعل')
        return {'word': word, 'type': 'فعل', 'reasons': reasons}

    for pattern, label in verb_suffix_rules:
        if re.search(pattern, s):
            # تمييز واو الجماعة في الفعل عن الجمع في الاسم
            if 'واو الجماعة' in label:
                if re.search(r'ون$', s) and has_al:
                    break  # كـ "المؤمنون" — اسم
            reasons.append(f'اتصلت بها {label} → فعل')
            return {'word': word, 'type': 'فعل', 'reasons': reasons}

    # 2ب: السياق — وجود ناصب أو جازم قبل الكلمة مباشرة
    if context_words:
        idx = None
        for i, w in enumerate(context_words):
            if strip_diacritics(w) == plain:
                idx = i
                break
        if idx is not None and idx > 0:
            prev = strip_diacritics(context_words[idx - 1])
            if prev in ('لم', 'لن', 'لا'):
                reasons.append(f'سبقتها الأداة "{prev}" التي تختص بالفعل المضارع → فعل')
                return {'word': word, 'type': 'فعل', 'reasons': reasons}
            if prev in ('سوف', 'سـ', 'قد'):
                reasons.append(f'سبقتها "{prev}" وهي خاصة بالفعل → فعل')
                return {'word': word, 'type': 'فعل', 'reasons': reasons}

    # 2ج: صيغة الأمر اِفْعَل (همزة وصل + 3 أحرف، بلا ألف في الوسط)
    if re.match(r'^[اإ][\u0600-\u06FF]{2,3}$', s) and not has_al:
        # تأكد أنه ليس على وزن فاعل (الذي تم فحصه فوق)
        if 'ا' not in s[1:]:  # لا ألف في وسط الكلمة
            reasons.append('تبدأ بهمزة وصل وعلى وزن "افعل" → فعل أمر')
            return {'word': word, 'type': 'فعل', 'reasons': reasons}

    # ── الخطوة 3: الصفة — الأوزان الحصرية (قبل فحص المضارع) ─────────────────

    # 3أ: وزن "أفعل" للون أو عيب (أحمر، أعرج، أعور) — 4 أحرف تبدأ بـ أ
    if re.match(r'^أ[\u0600-\u06FF]{3}$', s) and len(s) == 4:
        reasons.append('على وزن "أفعل" يدل على لون أو عيب → صفة مشبهة')
        return {'word': word, 'type': 'صفة', 'reasons': reasons}

    # 3ب: وزن "فعلان" (عطشان، جوعان، غضبان) — لا تبدأ بحرف مضارعة
    if re.match(r'^[\u0600-\u06FF]{2,4}ان$', s) and len(s) <= 6 and not has_al:
        # تأكد أن الحرف الأخير قبل "ان" ليس حرف علة يدل على مثنى فعلي
        root_without_an = s[:-2]
        if len(root_without_an) >= 2:
            reasons.append('على وزن "فعلان" ومؤنثه "فعلى" → صفة مشبهة')
            return {'word': word, 'type': 'صفة', 'reasons': reasons}

    # 2د: الفعل المضارع (يبدأ بحرف مضارعة ي/ت/ن/أ) — بعد فحص أفعل
    if (re.match(r'^[يتنأ][\u0600-\u06FF]{2,}$', s)
            and not has_al
            and 4 <= len(s) <= 8
            and plain not in KNOWN_NOUNS
            and s not in KNOWN_NOUNS):
        reasons.append('تبدأ بحرف مضارعة (ي/ت/ن/أ)، بلا "ال"، على طول مناسب → فعل مضارع')
        return {'word': word, 'type': 'فعل', 'reasons': reasons}

    # 3ج: وزن "فاعل" — اسم الفاعل (كاتب، جالس، قادم) — يجب فحصه قبل صيغة الأمر
    if re.match(r'^[\u0600-\u06FF]ا[\u0600-\u06FF]{1,2}$', s) and len(s) in (4, 5):
        reasons.append('على وزن "فاعل" → اسم الفاعل → صفة')
        return {'word': word, 'type': 'صفة', 'reasons': reasons}

    # ── الخطوة 4: الحال والصفة — المشتقات حسب السياق ───────────────────────

    # 4أ: اسم المفعول "مفعول" (مكتوب، مفتوح)
    is_maf3ul = bool(re.match(r'^م[\u0600-\u06FF]{1,2}و[\u0600-\u06FF]$', s) or
                     re.match(r'^م[\u0600-\u06FF]+ول$', s))
    if is_maf3ul:
        if not has_al and has_tanwin_nasb(word):
            reasons.append('اسم مفعول، نكرة، منصوب (تنوين النصب) → حال')
            return {'word': word, 'type': 'حال', 'reasons': reasons}
        reasons.append('اسم مفعول على وزن "مفعول" → صفة')
        return {'word': word, 'type': 'صفة', 'reasons': reasons}

    # 4ب: وزن "فعيل" (سريع، كريم، شريف)
    if re.match(r'^[\u0600-\u06FF]{1}[\u0600-\u06FF]ي[\u0600-\u06FF]{1}$', s) and len(s) == 4:
        if not has_al and has_tanwin_nasb(word):
            reasons.append('على وزن "فعيل"، نكرة، منصوب → حال')
            return {'word': word, 'type': 'حال', 'reasons': reasons}
        reasons.append('على وزن "فعيل" → صفة مشبهة')
        return {'word': word, 'type': 'صفة', 'reasons': reasons}

    # 4ج: أي مشتق نكرة منصوب (تنوين نصب) بلا ال → حال محتملة
    if not has_al and has_tanwin_nasb(word):
        reasons.append(
            'نكرة منصوبة (تنوين النصب) بلا "ال" التعريف، '
            'بعد جملة تامة → مرشح قوي للحال'
        )
        return {'word': word, 'type': 'حال', 'reasons': reasons}

    # ── الخطوة 5: الاسم ──────────────────────────────────────────────────────

    # 5أ: جمع تكسير ممنوع من الصرف (مفاعل، مفاعيل)
    if re.match(r'^م[\u0600-\u06FF]{4,}$', s) and len(s) >= 6:
        reasons.append('على وزن "مفاعل/مفاعيل" → جمع تكسير ممنوع من الصرف → اسم')
        return {'word': word, 'type': 'اسم', 'reasons': reasons}

    # 5ب: التاء المربوطة (شجرة، تفاحة) — ليست على وزن فاعلة
    if s.endswith('ة') and len(s) >= 3:
        if not re.match(r'^[\u0600-\u06FF]ا[\u0600-\u06FF]ة$', s):
            reasons.append('تنتهي بتاء مربوطة وليست على وزن "فاعلة" → اسم جامد')
            return {'word': word, 'type': 'اسم', 'reasons': reasons}

    # 5ج: مُعرَّفة بـ "ال" ولم تنطبق أوزان الصفة
    if has_al:
        reasons.append('مُعرَّفة بـ "ال" ولم تنطبق عليها أوزان الصفة أو الفعل → اسم')
        return {'word': word, 'type': 'اسم', 'reasons': reasons}

    # 5د: الحالة الافتراضية
    reasons.append('لم تنطبق عليها علامات الفعل أو الصفة أو الحال → الأصل اسم')
    return {'word': word, 'type': 'اسم', 'reasons': reasons}


# ─── تحليل الجملة ────────────────────────────────────────────────────────────

def analyze_sentence(text: str) -> list[dict]:
    """تحليل جملة كاملة وإعادة قائمة بنتائج كل كلمة"""
    words = tokenize(text)
    results = []
    for word in words:
        result = analyze_word(word, context_words=words)
        results.append(result)
    return results


# ─── طباعة النتائج ───────────────────────────────────────────────────────────

COLORS = {
    'فعل':  '\033[93m',   # أصفر
    'اسم':  '\033[94m',   # أزرق
    'صفة':  '\033[95m',   # بنفسجي
    'حال':  '\033[92m',   # أخضر
    'أداة': '\033[90m',   # رمادي
    'غير معروف': '\033[91m',  # أحمر
}
RESET  = '\033[0m'
BOLD   = '\033[1m'


def print_result(result: dict, index: int | None = None) -> None:
    word_type = result['type']
    color     = COLORS.get(word_type, '')
    prefix    = f'[{index}] ' if index is not None else ''

    print(f"\n{BOLD}{prefix}الكلمة:{RESET}  {result['word']}")
    print(f"{BOLD}النوع:{RESET}   {color}{BOLD}{word_type}{RESET}")
    print(f"{BOLD}التبريرات:{RESET}")
    for i, reason in enumerate(result['reasons'], 1):
        print(f"  {i}. {reason}")
    print('─' * 50)


# ─── الواجهة التفاعلية ───────────────────────────────────────────────────────

def interactive_mode() -> None:
    print('=' * 55)
    print('   محلل الصرف العربي الفطري — نسخة سطر الأوامر')
    print('=' * 55)
    print('  أدخل كلمة مفردة أو جملة كاملة للتحليل.')
    print('  اكتب "خروج" أو "exit" للإنهاء.\n')

    while True:
        try:
            text = input('>>> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nوداعًا!')
            break

        if not text:
            continue
        if text in ('خروج', 'exit', 'quit'):
            print('وداعًا!')
            break

        words = tokenize(text)
        if len(words) == 1:
            result = analyze_word(words[0])
            print_result(result)
        else:
            results = analyze_sentence(text)
            print(f'\n{BOLD}تحليل الجملة: "{text}"{RESET}')
            print('─' * 50)
            for i, res in enumerate(results, 1):
                print_result(res, index=i)


# ─── اختبارات جاهزة ──────────────────────────────────────────────────────────

TESTS = [
    ('يكتبون',                   'فعل'),
    ('كتبتُ',                    'فعل'),
    ('لم يذهب',                  'فعل'),    # جملة — يذهب فعل
    ('اجلس',                     'فعل'),
    ('أحمر',                     'صفة'),
    ('عطشان',                    'صفة'),
    ('كاتب',                     'صفة'),
    ('راكضًا',                   'حال'),
    ('مسرعًا',                   'حال'),
    ('الكتاب',                   'اسم'),
    ('شجرة',                     'اسم'),
    ('هذا',                      'اسم'),
    ('الذي',                     'اسم'),
    ('مساجد',                    'اسم'),
    ('هو',                       'اسم'),
]


def run_tests() -> None:
    print(f'\n{BOLD}{"="*55}')
    print('   تشغيل الاختبارات الآلية')
    print(f'{"="*55}{RESET}')
    passed = 0
    for text, expected in TESTS:
        words = tokenize(text)
        if len(words) == 1:
            result = analyze_word(words[0])
        else:
            # اختبر الكلمة الأخيرة في الجملة (المستهدفة)
            results = analyze_sentence(text)
            result  = results[-1]

        actual = result['type']
        ok     = actual == expected
        mark   = f'\033[92m✓{RESET}' if ok else f'\033[91m✗{RESET}'
        if ok:
            passed += 1
        status = f'{mark} {text!r:25} متوقع: {expected:8} حصلنا: {actual}'
        if not ok:
            status += f'  ← {result["reasons"]}'
        print(status)

    total = len(TESTS)
    print(f'\n{BOLD}النتيجة: {passed}/{total} اختبار ناجح{RESET}\n')


# ─── نقطة الدخول ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = sys.argv[1:]

    if '--test' in args or '-t' in args:
        run_tests()
    elif args:
        # تحليل ما مُرِّر مباشرةً في سطر الأوامر
        text = ' '.join(args)
        words = tokenize(text)
        if len(words) == 1:
            print_result(analyze_word(words[0]))
        else:
            results = analyze_sentence(text)
            print(f'\n{BOLD}تحليل الجملة: "{text}"{RESET}')
            print('─' * 50)
            for i, res in enumerate(results, 1):
                print_result(res, index=i)
    else:
        interactive_mode()
