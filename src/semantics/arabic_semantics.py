# -*- coding: utf-8 -*-
"""مرنان — تضمين دلالي فلسفي للحروف العربية (29 بُعداً).

المبادئ:
  1. لكل حرف معنى دلالي أساسي وضده (محور دلالي).
  2. الحروف المتشابهة صوتياً/شكلياً تتقارب، والمتضادة تتباعد.
  3. لا تطبيع للكلمة (Word Vector = مجموع موزون غير مطبَّع).
  4. أوزان أسية للمواقع (5|3|2) تكسر تناظر التباديل.
  5. التشابه: Cosine Similarity.
"""

import numpy as np

ARABIC_LETTERS = [
    "ء", "ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ",
    "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ",
    "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي",
]
NUM_ARABIC_LETTERS = len(ARABIC_LETTERS)

SELF_WEIGHT = 1.00        # قطر ذاتي قوي (تمييز الحرف عن غيره)
PHONETIC_BONUS = 0.30     # تقارب صوتي
SHAPE_BONUS = 0.25        # تقارب شكلي
CLUSTER_BONUS = 0.00      # معطّلة (تسبب تشويشاً دلالياً)
OPPOSITION_STRENGTH = 0.25  # تضاد دلالي
BIGRAM_WEIGHT = 0.45       # وزن المكافأة الثنائية (خفيف، لكسر التباديل فقط)
HARAKA_STRENGTH = 0.12    # وزن تأثير الحركة على متجه الحرف

# --- المجموعات الصوتية ---
PHONETIC_GROUPS = {
    "guttural":    {"ء", "ا", "ه", "ح", "ع", "غ", "خ"},
    "palatal":     {"ش", "ج", "ي"},
    "velar":       {"ك", "ق"},
    "dental":      {"ت", "د", "ط"},
    "interdental": {"ث", "ذ", "ظ"},
    "sibilant":    {"س", "ص", "ز"},
    "liquid":      {"ن", "ل", "ر"},
    "labial":      {"ب", "ف", "م", "و"},
}

# --- المجموعات الشكلية ---
SHAPE_GROUPS = {
    "ba_ta_tha": {"ب", "ت", "ث"},
    "jeem_ha_kha": {"ج", "ح", "خ"},
    "dal_thal": {"د", "ذ"},
    "ra_zay": {"ر", "ز"},
    "seen_sheen": {"س", "ش"},
    "sad_dad": {"ص", "ض"},
    "ta_dha": {"ط", "ظ"},
    "ayn_ghayn": {"ع", "غ"},
    "fa_qaf": {"ف", "ق"},
    "kaf_lam": {"ك", "ل"},
}

# --- الحقول الدلالية ---
SEMANTIC_CLUSTERS = {
    "authority":    {"م", "ك", "ق", "ح", "ل", "ع", "د", "ط", "س", "و"},
    "knowledge":    {"ع", "ل", "م", "ف", "ح", "ك", "ب", "ن", "د", "ر"},
    "light":        {"ن", "و", "ر", "ح", "ض", "ه", "ا", "ج", "ص"},
    "darkness":     {"ظ", "غ", "خ", "ي", "ش", "ث", "ذ", "ء"},
    "power_motion": {"ق", "و", "ر", "د", "ط", "ج", "ع", "ز", "ف", "ب"},
    "weakness":     {"ي", "خ", "ث", "ظ", "ذ", "ء", "ه"},
    "generosity":   {"ك", "ج", "و", "د", "ر", "ن", "ف", "ب", "م"},
    "creation":     {"ت", "ب", "ن", "خ", "ل", "ق", "م", "ص", "د"},
    "dispersion":   {"ث", "ش", "ظ", "ذ", "ز", "ف", "ر"},
}

OPPOSING_CLUSTERS = [
    ("light", "darkness"),
    ("power_motion", "weakness"),
    ("creation", "dispersion"),
    ("knowledge", "darkness"),
    ("authority", "weakness"),
    ("generosity", "dispersion"),
]

# --- الأوزان الأسية للمواقع (Position Exponential Weights) ---
# فاء الكلمة = 5×، عين الكلمة = 3×، لام الكلمة = 2×، الباقي = 1×
POSITION_WEIGHTS = [3.5, 2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3]

def _get_position_weight(pos, word_len):
    """الوزن الأسي للموقع: ينخفض تدريجياً بعد الموقع الثالث."""
    if pos < len(POSITION_WEIGHTS):
        return POSITION_WEIGHTS[pos]
    return POSITION_WEIGHTS[-1]

LETTER_MEANINGS = {
    "ء": "تعجب/خوف/صدمة/مفاجأة",
    "ا": "حنان/رفعة وضدهما",
    "ب": "دك/تشبع/نقل/حمل وضدها",
    "ت": "بناء/قذف/رص/ترتيب وضدها",
    "ث": "بعثرة/تشتت/تثبيط وضدها",
    "ج": "جمع/جذب/التحام وضدها",
    "ح": "حث/حياة/انتعاش وضدها",
    "خ": "خرق/خبيئة وضدها",
    "د": "عزم/ثبات/فتح وضدها",
    "ذ": "نفور/تلذذ وضدها",
    "ر": "حركة/تدفق/تكرار وضدها",
    "ز": "انزلاق/تزحلق/انزياح وضدها",
    "س": "سر/زحف/سور وضدها",
    "ش": "تشتت/تفرع وضدها",
    "ص": "مراقبة/إنصات وضدها",
    "ض": "كتم/ضغط وضدها",
    "ط": "طرق/استئذان وضدها",
    "ظ": "توهان/إيهام وضدها",
    "ع": "اقتلاع/عمق/سعة وضدها",
    "غ": "غياب/غموض/غور وضدها",
    "ف": "حفرة/نفخ/انفجار وضدها",
    "ق": "دقة/بعد وضدها",
    "ك": "عطاء/كفاية/كنز وضدها",
    "ل": "إحاطة/شمول/لامية وضدها",
    "م": "فهم/استيعاب/احتواء وضدها",
    "ن": "وضوح/تبيين/خلق/إنشاء وضدها",
    "ه": "جهد/ثمرة/نتيجة وضدها",
    "و": "تقدم/تهجم/هجوم/تدحرج وضدها",
    "ي": "إيلام/توجع/تقهقر وضدها",
}


def _build_opposition_map():
    char_clusters = {}
    for ch in ARABIC_LETTERS:
        char_clusters[ch] = set()
        for cname, cset in SEMANTIC_CLUSTERS.items():
            if ch in cset:
                char_clusters[ch].add(cname)
    opp_map = {}
    for ch_a in ARABIC_LETTERS:
        opp_map[ch_a] = set()
        clusters_a = char_clusters.get(ch_a, set())
        for ch_b in ARABIC_LETTERS:
            if ch_a == ch_b:
                continue
            clusters_b = char_clusters.get(ch_b, set())
            for c1, c2 in OPPOSING_CLUSTERS:
                if c1 in clusters_a and c2 in clusters_b:
                    opp_map[ch_a].add(ch_b)
                if c2 in clusters_a and c1 in clusters_b:
                    opp_map[ch_a].add(ch_b)
    return opp_map

OPPOSITION_MAP = _build_opposition_map()


class CharacterSemanticEmbedding:
    """تضمين دلالي للحروف العربية (29D متفرقة + أوزان أسية للمواقع).

    المبادئ:
    - متجه الحرف: 29D، القطر الذاتي قوي (1.0)، العلاقات العرضية ضعيفة (0.25-0.30)
      هذا يجعل المتجهات متفرقة ومتباعدة → تمييز أفضل بين مجموعات الحروف المختلفة.
    - لا تطبيع على مستوى الكلمة (يبقى المتجه خاماً بمجمعه الموزون).
    - أوزان أسية للمواقع (5, 3, 2, 1.5, ...) تقضي على تشابه التباديل.
    - التشابه: Cosine Similarity (يطبَّع داخلياً فقط لحظة القياس).
    """

    def __init__(self, dim: int | None = None):
        self.dim = dim if dim is not None else NUM_ARABIC_LETTERS
        self.arabic_letters = ARABIC_LETTERS
        self.letter_to_idx = {l: i for i, l in enumerate(self.arabic_letters)}
        self.semantic_categories = LETTER_MEANINGS
        self.num_categories = len(self.arabic_letters)
        self._cache = {}
        self._cluster_map = self._build_cluster_map()
        self._build_letter_vectors()
        self._build_haraka_vectors()

    def _build_cluster_map(self):
        cmap = {}
        for ch in self.arabic_letters:
            cmap[ch] = []
            for cname, cset in SEMANTIC_CLUSTERS.items():
                if ch in cset:
                    cmap[ch].append(cname)
        return cmap

    def _find_phonetic_group(self, ch):
        for gname, gset in PHONETIC_GROUPS.items():
            if ch in gset:
                return gname
        return None

    def _find_shape_group(self, ch):
        for gname, gset in SHAPE_GROUPS.items():
            if ch in gset:
                return gname
        return None

    def _build_letter_vectors(self):
        """بناء متجهات متفرقة: قطر قوي، وصلات عرضية ضعيفة، تضاد سالب."""
        n = self.dim
        self._letter_vectors = {}

        for ch_a in self.arabic_letters:
            if ch_a not in self.letter_to_idx:
                continue
            vec = np.zeros(n, dtype=np.float64)
            opp_set = OPPOSITION_MAP.get(ch_a, set())
            pg_a = self._find_phonetic_group(ch_a)
            sg_a = self._find_shape_group(ch_a)

            for ch_b in self.arabic_letters:
                if ch_b not in self.letter_to_idx:
                    continue
                idx_b = self.letter_to_idx[ch_b]
                if idx_b >= n:
                    continue

                if ch_a == ch_b:
                    vec[idx_b] = SELF_WEIGHT
                elif ch_b in opp_set:
                    vec[idx_b] = -OPPOSITION_STRENGTH
                else:
                    val = 0.0
                    pg_b = self._find_phonetic_group(ch_b)
                    if pg_a and pg_b and pg_a == pg_b:
                        val = max(val, PHONETIC_BONUS)
                    sg_b = self._find_shape_group(ch_b)
                    if sg_a and sg_b and sg_a == sg_b:
                        val = max(val, SHAPE_BONUS)
                    # مكافأة الحقول الدلالية المشتركة
                    clusters_a = set(self._cluster_map.get(ch_a, []))
                    clusters_b = set(self._cluster_map.get(ch_b, []))
                    shared = clusters_a & clusters_b
                    if shared:
                        val += CLUSTER_BONUS * len(shared)
                    vec[idx_b] = val

            self._letter_vectors[ch_a] = vec

    def get_character_vector(self, char):
        if char in self._cache:
            return self._cache[char]
        vec = self._letter_vectors.get(char, np.zeros(self.dim))
        self._cache[char] = vec
        return vec

    def _build_haraka_vectors(self):
        """بناء متجهات تأثير الحركات الثلاث الرئيسية على الحرف.

        - الفتحة (َ): انفتاح → تنشيط أبعاد الانفتاح (ا، ه، ح، و)
        - الضمة  (ُ): تجميع → تنشيط أبعاد الاحتواء (م، ق، ض، ب)
        - الكسرة (ِ): عمق  → تنشيط أبعاد العمق (ي، ع، غ، خ)
        - الشدة  (ّ): مضاعفة تأثير الحرف ×1.15
        - السكون (ْ): حيادي
        """
        n = self.dim
        # فتحة: أبعاد الانفتاح والارتفاع
        fatha_vec = np.zeros(n)
        for ch in ['ا', 'ه', 'ح', 'و', 'ر', 'ن']:
            if ch in self.letter_to_idx:
                fatha_vec[self.letter_to_idx[ch]] = 1.0
        fatha_vec /= np.linalg.norm(fatha_vec) if np.linalg.norm(fatha_vec) > 1e-10 else 1.0

        # ضمة: أبعاد الاحتواء والتجميع
        damma_vec = np.zeros(n)
        for ch in ['م', 'ق', 'ض', 'ب', 'ج', 'ل']:
            if ch in self.letter_to_idx:
                damma_vec[self.letter_to_idx[ch]] = 1.0
        damma_vec /= np.linalg.norm(damma_vec) if np.linalg.norm(damma_vec) > 1e-10 else 1.0

        # كسرة: أبعاد العمق والانخفاض
        kasra_vec = np.zeros(n)
        for ch in ['ي', 'ع', 'غ', 'خ', 'ذ', 'ظ']:
            if ch in self.letter_to_idx:
                kasra_vec[self.letter_to_idx[ch]] = 1.0
        kasra_vec /= np.linalg.norm(kasra_vec) if np.linalg.norm(kasra_vec) > 1e-10 else 1.0

        self._haraka_vectors = {
            0x064E: fatha_vec,   # فتحة
            0x064F: damma_vec,   # ضمة
            0x0650: kasra_vec,   # كسرة
            0x064B: fatha_vec,   # فتحتان
            0x064C: damma_vec,   # ضمتان
            0x064D: kasra_vec,   # كسرتان
        }
        self._shadda_code = 0x0651  # شدة

    def _parse_arabic_with_harakat(self, word: str) -> list[tuple[str, list[int]]]:
        """تفكيك الكلمة العربية إلى (حرف, [حركات]).

        مثلاً: 'مَلِكٌ' → [('م', [0x064E]), ('ل', [0x0650]), ('ك', [0x064C])]
        """
        result = []
        current_letter = None
        current_harakat = []

        for ch in word:
            cp = ord(ch)
            # تحقق إذا كان الحرف حرفاً عربياً أساسياً
            if 0x0621 <= cp <= 0x064A:  # نطاق الحروف العربية
                if current_letter is not None:
                    result.append((current_letter, current_harakat))
                current_letter = ch
                current_harakat = []
            elif 0x064B <= cp <= 0x0652:  # نطاق الحركات والتشكيل
                if current_letter is not None:
                    current_harakat.append(cp)
            elif current_letter is not None:
                current_harakat.append(cp)

        if current_letter is not None:
            result.append((current_letter, current_harakat))

        return result

    def get_word_semantic_vector(self, word: str) -> np.ndarray:
        """حساب المتجه الهجين للكلمة: أحادي + ثنائي + تأثير الحركات.

        - المكون الأحادي: مجموع موزون لمتجهات الحروف + تعديل الحركة.
        - المكون الثنائي: متوسطات الأزواج المتجاورة لكسر التباديل.
        - الحركات: الفتحة/الضمة/الكسرة/الشدة تؤثر على متجه الحرف.

        هذا يعني أن 'عَلِمَ' ≠ 'عُلِمَ' ≠ 'عِلْم' لأن حركاتها مختلفة!
        """
        parsed = self._parse_arabic_with_harakat(word)
        n = len(parsed)

        # --- كلمات قصيرة: مجموع بسيط ---
        if n <= 2:
            vectors = []
            for letter, harakat in parsed:
                vec = self.get_character_vector(letter)
                if np.any(vec):
                    vectors.append(vec)
            if not vectors:
                return np.zeros(self.dim)
            return np.sum(vectors, axis=0)

        # --- 1. المكون الأحادي مع تأثير الحركات ---
        unigram_vectors = []
        letters_for_bigram = []
        for i, (letter, harakat) in enumerate(parsed):
            vec = self.get_character_vector(letter).copy()
            if np.any(vec):
                # تأثير الحركات على المتجه
                for h in harakat:
                    if h == self._shadda_code:
                        vec = vec * 1.12  # شدة: تضاعف تأثير الحرف
                    elif h in self._haraka_vectors:
                        vec = vec + self._haraka_vectors[h] * HARAKA_STRENGTH

                letters_for_bigram.append(vec.copy())
                w = _get_position_weight(i, n)
                unigram_vectors.append(vec * w)

        # --- 2. المكون الثنائي (Bigram) ---
        bigram_vectors = []
        for i in range(len(letters_for_bigram) - 1):
            v1 = letters_for_bigram[i]
            v2 = letters_for_bigram[i + 1]
            if np.any(v1) and np.any(v2):
                bg = (v1 + v2) / 2.0
                bigram_vectors.append(bg * BIGRAM_WEIGHT)

        # --- 3. الدمج ---
        all_vectors = unigram_vectors + bigram_vectors
        if not all_vectors:
            return np.zeros(self.dim)

        return np.sum(all_vectors, axis=0)


# ============================================================
#  قاموس معاني الحروف (بحث مكرر) + تفكيك الكلمة إلى دلالاتها
# ============================================================

LETTER_DEFINITIONS = {
    "ء": "مفاجأة",
    "ا": "رفعة",
    "ب": "نقل",
    "ت": "بناء",
    "ث": "تشتت",
    "ج": "جمع",
    "ح": "حياة",
    "خ": "خبيئة",
    "د": "إثبات",
    "ذ": "نفور",
    "ر": "تدفق",
    "ز": "انزياح",
    "س": "سريان",
    "ش": "تفرع",
    "ص": "مراقبة",
    "ض": "ضغط",
    "ط": "طرق",
    "ظ": "إيهام",
    "ع": "مدافعة",
    "غ": "غياب",
    "ف": "انفجار",
    "ق": "دقة",
    "ك": "عطاء",
    "ل": "إلمام",
    "م": "فهم",
    "ن": "تبيين",
    "ه": "جهد",
    "و": "تقدم",
    "ي": "توجع",
}

_DEF_NORMALIZE = {
    '\u0622': '\u0627',  # آ → ا
    '\u0623': '\u0627',  # أ → ا
    '\u0625': '\u0627',  # إ → ا
    '\u0624': '\u0648',  # ؤ → و
    '\u0626': '\u064a',  # ئ → ي
    '\u0629': '\u0647',  # ة → ه
    '\u0649': '\u064a',  # ى → ي
}


def decompose_word_definition(word: str) -> list[tuple[str, str]]:
    """تفكيك الكلمة إلى حروفها مع معنى كل حرف.

    Returns:
        قائمة من (حرف, معناه) بالترتيب، مع تجاهل الحركات والتشكيل.
    """
    cleaned = []
    for ch in word:
        cp = ord(ch)
        if 0x064b <= cp <= 0x0652 or cp == 0x0640:
            continue
        ch = _DEF_NORMALIZE.get(ch, ch)
        if 0x0621 <= ord(ch) <= 0x064a:
            cleaned.append(ch)

    result = []
    for ch in cleaned:
        meaning = LETTER_DEFINITIONS.get(ch, "؟")
        result.append((ch, meaning))
    return result
