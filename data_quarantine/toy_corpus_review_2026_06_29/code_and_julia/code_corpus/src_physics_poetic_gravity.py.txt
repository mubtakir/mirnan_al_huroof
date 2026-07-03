"""PoeticGravityEngine — محرك الجاذبية الشعرية الكامل.

البحور الـ 16: الطويل، المديد، البسيط، الوافر، الكامل، الهزج، الرجز، الرمل،
السريع، المنسرح، الخفيف، المضارع، المقتضب، المجتث، المتقارب، المتدارك.

V6: إكمال كل البحور + قافية + وزن حقيقي.
"""

import numpy as np
import re


def _ar_syllabify(word):
    """تقطيع الكلمة العربية إلى مقاطع (طويلة/قصيرة).
    
    م = 1 (ساكن/حرف مد), 0 = (متحرك قصير)
    مقطع طويل: C + V + C (1 0 1) أو C + V (1 0)
    مقطع قصير: C + V (1 0)
    """
    vowels = {'ا', 'و', 'ي', 'ى', 'آ', 'َ', 'ِ', 'ُ'}
    madd = {'ا', 'و', 'ي', 'آ'}
    sukun_chars = {'ْ', 'ّ'}
    
    pattern = []
    i = 0
    while i < len(word):
        c = word[i]
        if c in vowels:
            pattern.append(0)  # حركة
        elif c in (sukun_chars | {'ء', 'أ', 'إ', 'ؤ', 'ئ'}):
            pattern.append(1)  # سكون
        else:
            pattern.append(1)  # حرف ساكن
        i += 1
    return pattern


def _en_count_syllables(word):
    w = word.lower().strip()
    if not w:
        return 1
    count = 0
    prev_vowel = False
    vowels = set('aeiouy')
    for ch in w:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if w.endswith('e') and count > 1 and not w.endswith('le'):
        count -= 1
    if w.endswith('le') and len(w) > 2 and w[-3] not in vowels:
        count += 1
    return max(1, count)


def _en_stress_pattern(word):
    syl = _en_count_syllables(word)
    if syl <= 1:
        return [1]
    if syl == 2:
        if word.endswith(('ing', 'er', 'ly', 'y', 'ed', 'en', 'ful', 'less')):
            return [0, 1]
        if word.startswith(('a', 'be', 'de', 're', 'pre', 'pro', 'in', 'im')):
            return [0, 1]
        return [1, 0]
    if syl == 3:
        if word.endswith(('ity', 'tion', 'sion', 'ient')):
            return [0, 1, 0]
        if word.endswith(('ful', 'less', 'ness', 'ment', 'ate')):
            return [1, 0, 1]
        return [0, 1, 0]
    return [0, 1] + [0] * (syl - 3) + [1]


def _extract_en_rhythm(word):
    return _en_stress_pattern(word)


class PoeticGravityEngine:
    """محرك جاذبية شعري — كل البحور العربية الـ 16 + إنكليزية."""

    def __init__(self):
        # البحور الشعرية العربية الـ 16 (تفعيلات)
        # 1 = سبب ثقيل/وتد, 0 = سبب خفيف
        self.METERS = {
            # البحور العربية
            "tawil":       [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
            "madeed":      [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
            "baseet":      [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
            "wafir":       [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
            "kamil":       [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "hazaj":       [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
            "rajaz":       [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
            "ramal":       [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
            "saree":       [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
            "monsareh":    [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
            "khafeef":     [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
            "modare":      [0, 1, 0, 1, 0, 0, 1, 0, 1],
            "moqtadab":    [0, 1, 0, 0, 1, 0, 1, 0, 1],
            "mojtath":     [0, 1, 0, 0, 1, 0, 1, 0],
            "motakarib":   [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
            "motadarak":   [0, 1, 0, 1, 0, 1, 0, 1],
            # البحور الإنكليزية
            "iambic_pentameter":  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "trochaic_tetrameter":[1, 0, 1, 0, 1, 0, 1, 0],
            "anapestic_trimeter": [0, 0, 1, 0, 0, 1, 0, 0, 1],
            "dactylic_dimater":   [1, 0, 0, 1, 0, 0],
            "iambic_trimeter":    [0, 1, 0, 1, 0, 1],
            "iambic_tetrameter":  [0, 1, 0, 1, 0, 1, 0, 1],
        }
        
        # أسماء البحور العربية للتعرف التلقائي
        self.ARABIC_METER_NAMES = [
            "tawil", "madeed", "baseet", "wafir", "kamil",
            "hazaj", "rajaz", "ramal", "saree", "monsareh",
            "khafeef", "modare", "moqtadab", "mojtath",
            "motakarib", "motadarak",
        ]
        
        self.RHYME_GROUPS = {
            'ee': {'ee', 'ea', 'ie', 'y', 'ey', 'e', 'i'},
            'ay': {'ay', 'ai', 'a', 'ei', 'ey', 'eigh'},
            'oo': {'oo', 'ou', 'ue', 'ew', 'oe'},
            'ow': {'ow', 'ou'},
            'oy': {'oy', 'oi'},
            'ar': {'ar', 'aar'},
            'er': {'er', 'ir', 'ur', 'or', 'ear'},
            'ate': {'ate', 'eight'},
            'ight': {'ight', 'ite', 'yte'},
        }

        # الحروف العربية للمد
        self.AR_MADD = {'ا', 'و', 'ي', 'آ'}
        # حروف العلة
        self.AR_VOWELS = {'َ', 'ِ', 'ُ', 'ا', 'و', 'ي', 'ى', 'آ', 'أ', 'إ', 'ؤ', 'ئ'}
        # النون الساكنة للتنوين
        self.AR_TANWEEN = {'ً', 'ٍ', 'ٌ'}

    def _is_english(self, word):
        return bool(re.search(r'[a-zA-Z]', word))

    def _extract_rhythm(self, word):
        """استخراج النبض الإيقاعي لكلمة."""
        if self._is_english(word):
            return _extract_en_rhythm(word)
        return _ar_syllabify(word)

    def detect_meter(self, words, default='kamil'):
        """كشف البحر الشعري من كلمات النص."""
        if not words:
            return default
        
        all_rhythms = []
        for w in words:
            r = self._extract_rhythm(w)
            if r:
                all_rhythms.extend(r)
        
        if not all_rhythms:
            return default
        
        best_meter = default
        best_score = -1.0
        
        for mname in self.ARABIC_METER_NAMES:
            pattern = self.METERS[mname]
            plen = len(pattern)
            score = 0.0
            count = 0
            for i, pulse in enumerate(all_rhythms):
                if pulse == pattern[i % plen]:
                    score += 1.0
                count += 1
            if count > 0:
                score /= count
            if score > best_score:
                best_score = score
                best_meter = mname
        
        return best_meter

    def compute_rhythmic_resonance(self, word, current_position_index, meter_name):
        """حساب الرنين الإيقاعي لكلمة مع البحر.

        Returns:
            float: درجة الرنين (0..1)
            int: المؤشر التالي في البحر
        """
        if meter_name not in self.METERS:
            return 0.0, current_position_index
        
        meter_pattern = self.METERS[meter_name]
        word_rhythm = self._extract_rhythm(word)
        
        if not word_rhythm:
            return 0.0, current_position_index
        
        pattern_len = len(meter_pattern)
        match_count = 0
        
        for i, pulse in enumerate(word_rhythm):
            idx = (current_position_index + i) % pattern_len
            if pulse == meter_pattern[idx]:
                match_count += 1
        
        resonance = float(match_count) / max(len(word_rhythm), 1)
        next_idx = (current_position_index + len(word_rhythm)) % pattern_len
        
        return resonance, next_idx

    def compute_rhyme_gravity(self, word, target_rhyme_char):
        """حساب جاذبية القافية."""
        if not target_rhyme_char or not word:
            return 0.0
        
        if self._is_english(word):
            return self._compute_en_rhyme_gravity(word, target_rhyme_char)
        
        # تجريد الحروف الزائدة (التنوين، التاء المربوطة، إلخ)
        clean = word
        for v in ['ً', 'ٍ', 'ٌ', 'ا', 'و', 'ي', 'ه', 'ة']:
            if clean.endswith(v) and len(clean) > 2:
                clean = clean[:-1]
        
        # هل تنتهي بحرف القافية؟
        if clean.endswith(target_rhyme_char):
            return 1.0
        
        # تسكين: الحرف الأخير معروف
        if len(word) > 0 and word[-1] == target_rhyme_char:
            return 0.9
            
        return 0.0

    def _compute_en_rhyme_gravity(self, word, target):
        w_end = word[-2:].lower() if len(word) >= 2 else word.lower()
        t_end = target[-2:].lower() if len(target) >= 2 else target.lower()
        if w_end == t_end:
            return 1.0
        w_last3 = word[-3:].lower() if len(word) >= 3 else ''
        t_last3 = target[-3:].lower() if len(target) >= 3 else ''
        if w_last3 and w_last3 == t_last3:
            return 1.0
        for members in self.RHYME_GROUPS.values():
            w_in = any(w_end.endswith(m) for m in members)
            t_in = any(t_end.endswith(m) for m in members)
            if w_in and t_in:
                return 0.8
        return 0.0

    def get_meter_names(self, language='ar'):
        """أسماء البحور المتاحة."""
        if language == 'ar':
            return self.ARABIC_METER_NAMES[:]
        return [k for k in self.METERS if k not in self.ARABIC_METER_NAMES]

    def count_en_syllables(self, word):
        if not self._is_english(word):
            return 1
        return _en_count_syllables(word)

    def sentence_resonance(self, words, meter_name):
        """رنين جملة كاملة مع بحر — للإيقاع الكلي."""
        pos = 0
        total = 0.0
        count = 0
        for w in words:
            r, pos = self.compute_rhythmic_resonance(w, pos, meter_name)
            total += r
            count += 1
        return total / max(count, 1)
