# -*- coding: utf-8 -*-
"""
Arabic Semantics - Adapted for Mirnan
=====================================

تضمين واعٍ بالمعاني الفلسفية للحروف والجذور العربية.
يدعم وضعين:
  - sparse (29D): نظام متفرق مع مجموعات صوتية/شكلية/دلالية + حركات + bigram
  - sinusoidal (16D): نظام جيبي كلاسيكي للتوافق مع التدريب السابق
"""

import json
import os
import math
import sqlite3
from typing import Dict, List, Optional, Tuple
import numpy as np

ARABIC_LETTERS = [
    "ء", "ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ",
    "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ",
    "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي",
]
NUM_ARABIC_LETTERS = len(ARABIC_LETTERS)

SPARSE_SELF_WEIGHT = 1.00
SPARSE_PHONETIC_BONUS = 0.30
SPARSE_SHAPE_BONUS = 0.25
SPARSE_CLUSTER_BONUS = 0.00
SPARSE_OPPOSITION_STRENGTH = 0.25
SPARSE_BIGRAM_WEIGHT = 0.45
SPARSE_HARAKA_STRENGTH = 0.12

SPARSE_POSITION_WEIGHTS = [3.5, 2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3]

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

SPARSE_OPPOSITION_MAP = _build_opposition_map()

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


class CharacterSemanticEmbedding:
    """تضمين دلالي للحروف العربية — يدعم وضعين:

    - sparse (29D): نظام متفرق مع مجموعات صوتية/شكلية/دلالية وحقول متضادة
                    + دعم الحركات (الفتحة/الضمة/الكسرة/الشدة)
                    + مكون ثنائي (bigram) لكسر تناظر التباديل
                    + أوزان أسية للمواقع

    - sinusoidal (أي بُعد آخر): نظام جيبي/جيب تمامي كلاسيكي
    """

    def __init__(self, dim: int = 29):
        self.dim = dim
        self.arabic_letters = ARABIC_LETTERS
        self.letter_to_idx = {l: i for i, l in enumerate(self.arabic_letters)}
        self.num_categories = 28
        self._cache = {}

        if dim == NUM_ARABIC_LETTERS:
            self._mode = "sparse"
            self._cluster_map = self._build_cluster_map()
            self._build_sparse_letter_vectors()
            self._build_haraka_vectors()
        else:
            self._mode = "sinusoidal"
            self.semantic_categories = {
                "أ": 0, "ب": 1, "ت": 2, "ث": 3, "ج": 4, "ح": 5, "خ": 6,
                "د": 7, "ذ": 8, "ر": 9, "ز": 10, "س": 11, "ش": 12,
                "ص": 13, "ض": 14, "ط": 15, "ظ": 16, "ع": 17, "غ": 18,
                "ف": 19, "ق": 20, "ك": 21, "ل": 22, "م": 23, "ن": 24,
                "ه": 25, "و": 26, "ي": 27,
            }
            self._haraka_vectors = None

    # --- sparse mode: cluster building ---
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

    def _build_sparse_letter_vectors(self):
        n = self.dim
        self._letter_vectors = {}

        for ch_a in self.arabic_letters:
            if ch_a not in self.letter_to_idx:
                continue
            vec = np.zeros(n, dtype=np.float64)
            opp_set = SPARSE_OPPOSITION_MAP.get(ch_a, set())
            pg_a = self._find_phonetic_group(ch_a)
            sg_a = self._find_shape_group(ch_a)

            for ch_b in self.arabic_letters:
                if ch_b not in self.letter_to_idx:
                    continue
                idx_b = self.letter_to_idx[ch_b]
                if idx_b >= n:
                    continue

                if ch_a == ch_b:
                    vec[idx_b] = SPARSE_SELF_WEIGHT
                elif ch_b in opp_set:
                    vec[idx_b] = -SPARSE_OPPOSITION_STRENGTH
                else:
                    val = 0.0
                    pg_b = self._find_phonetic_group(ch_b)
                    if pg_a and pg_b and pg_a == pg_b:
                        val = max(val, SPARSE_PHONETIC_BONUS)
                    sg_b = self._find_shape_group(ch_b)
                    if sg_a and sg_b and sg_a == sg_b:
                        val = max(val, SPARSE_SHAPE_BONUS)
                    clusters_a = set(self._cluster_map.get(ch_a, []))
                    clusters_b = set(self._cluster_map.get(ch_b, []))
                    shared = clusters_a & clusters_b
                    if shared:
                        val += SPARSE_CLUSTER_BONUS * len(shared)
                    vec[idx_b] = val

            self._letter_vectors[ch_a] = vec

    # --- sparse mode: haraka (diacritic) vectors ---
    def _build_haraka_vectors(self):
        n = self.dim
        fatha_vec = np.zeros(n)
        for ch in ['ا', 'ه', 'ح', 'و', 'ر', 'ن']:
            if ch in self.letter_to_idx:
                fatha_vec[self.letter_to_idx[ch]] = 1.0
        fatha_norm = np.linalg.norm(fatha_vec)
        if fatha_norm > 1e-10:
            fatha_vec /= fatha_norm

        damma_vec = np.zeros(n)
        for ch in ['م', 'ق', 'ض', 'ب', 'ج', 'ل']:
            if ch in self.letter_to_idx:
                damma_vec[self.letter_to_idx[ch]] = 1.0
        damma_norm = np.linalg.norm(damma_vec)
        if damma_norm > 1e-10:
            damma_vec /= damma_norm

        kasra_vec = np.zeros(n)
        for ch in ['ي', 'ع', 'غ', 'خ', 'ذ', 'ظ']:
            if ch in self.letter_to_idx:
                kasra_vec[self.letter_to_idx[ch]] = 1.0
        kasra_norm = np.linalg.norm(kasra_vec)
        if kasra_norm > 1e-10:
            kasra_vec /= kasra_norm

        self._haraka_vectors = {
            0x064E: fatha_vec,
            0x064F: damma_vec,
            0x0650: kasra_vec,
            0x064B: fatha_vec,
            0x064C: damma_vec,
            0x064D: kasra_vec,
        }
        self._shadda_code = 0x0651

    # --- sinusoidal mode ---
    def _generate_phase_vector(self, category_idx: int) -> np.ndarray:
        vec = np.zeros(self.dim)
        for i in range(self.dim):
            freq = (category_idx + 1) * math.pi / self.num_categories
            if i % 2 == 0:
                vec[i] = math.sin(freq * (i + 1))
            else:
                vec[i] = math.cos(freq * (i + 1))
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm
        return vec

    # --- public API ---
    def get_character_vector(self, char: str) -> np.ndarray:
        if char in self._cache:
            return self._cache[char]

        if self._mode == "sparse":
            vec = self._letter_vectors.get(char, np.zeros(self.dim))
        else:
            cat_idx = self.semantic_categories.get(char, -1)
            if cat_idx == -1:
                vec = np.zeros(self.dim)
            else:
                vec = self._generate_phase_vector(cat_idx)

        self._cache[char] = vec
        return vec

    def _parse_arabic_with_harakat(self, word: str) -> list:
        result = []
        current_letter = None
        current_harakat = []

        for ch in word:
            cp = ord(ch)
            if 0x0621 <= cp <= 0x064A:
                if current_letter is not None:
                    result.append((current_letter, current_harakat))
                current_letter = ch
                current_harakat = []
            elif 0x064B <= cp <= 0x0652:
                if current_letter is not None:
                    current_harakat.append(cp)
            elif current_letter is not None:
                current_harakat.append(cp)

        if current_letter is not None:
            result.append((current_letter, current_harakat))

        return result

    def _get_position_weight(self, pos, word_len):
        if pos < len(SPARSE_POSITION_WEIGHTS):
            return SPARSE_POSITION_WEIGHTS[pos]
        return SPARSE_POSITION_WEIGHTS[-1]

    def get_word_semantic_vector(self, word: str) -> np.ndarray:
        if self._mode == "sparse":
            return self._get_word_semantic_sparse(word)
        else:
            return self._get_word_semantic_sinusoidal(word)

    def _get_word_semantic_sinusoidal(self, word: str) -> np.ndarray:
        vectors = []
        for char in word:
            vec = self.get_character_vector(char)
            if np.any(vec):
                vectors.append(vec)
        if not vectors:
            return np.zeros(self.dim)
        semantic_vec = np.mean(vectors, axis=0)
        norm = np.linalg.norm(semantic_vec)
        if norm > 1e-10:
            semantic_vec = semantic_vec / norm
        return semantic_vec

    def _get_word_semantic_sparse(self, word: str) -> np.ndarray:
        parsed = self._parse_arabic_with_harakat(word)
        n = len(parsed)

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
                for h in harakat:
                    if h == self._shadda_code:
                        vec = vec * 1.12
                    elif h in self._haraka_vectors:
                        vec = vec + self._haraka_vectors[h] * SPARSE_HARAKA_STRENGTH

                letters_for_bigram.append(vec.copy())
                w = self._get_position_weight(i, n)
                unigram_vectors.append(vec * w)

        # --- 2. المكون الثنائي (Bigram) ---
        bigram_vectors = []
        for i in range(len(letters_for_bigram) - 1):
            v1 = letters_for_bigram[i]
            v2 = letters_for_bigram[i + 1]
            if np.any(v1) and np.any(v2):
                bg = (v1 + v2) / 2.0
                bigram_vectors.append(bg * SPARSE_BIGRAM_WEIGHT)

        # --- 3. الدمج ---
        all_vectors = unigram_vectors + bigram_vectors
        if not all_vectors:
            return np.zeros(self.dim)

        return np.sum(all_vectors, axis=0)


class ArabicRootExtractor:
    """
    مستخرج الجذور العربية الديناميكي والخوارزمي.
    يعتمد على المطابقة مع الأوزان الصرفية بدلاً من القاموس الثابت.
    """

    def __init__(self, semantic_db_path: Optional[str] = None):
        self.augmentation_letters = set("سألتمونيها")
        self.definite_article = "ال"

        self.attached_pronouns = [
            "هم", "هن", "هما", "ها", "ه", "ون", "ين", "ان", "ات",
            "كم", "كن", "كما", "ك", "تم", "تن", "تما", "وا",
            "نا", "ني", "ي", "ت"
        ]

        self.prefixes = ["و", "ف", "ب", "ل", "ك"]

        self.patterns = [
            ("استفعال", [3, 4, 6]),
            ("مستفعل", [3, 4, 5]),
            ("استفعل", [3, 4, 5]),
            ("يتفعلون", [3, 4, 5]),
            ("انفعال", [2, 3, 5]),
            ("افتعال", [1, 3, 5]),
            ("مفاعيل", [1, 3, 5]),
            ("مفاعلة", [1, 3, 4]),
            ("مفاعل", [1, 3, 4]),
            ("تفعيل", [1, 2, 4]),
            ("تفاعل", [1, 3, 4]),
            ("تفعّل", [1, 2, 4]),
            ("مفعول", [1, 2, 4]),
            ("فعائل", [0, 1, 4]),
            ("فواعل", [0, 3, 4]),
            ("أفعال", [1, 2, 4]),
            ("إفعال", [1, 2, 4]),
            ("أفعل", [1, 2, 3]),
            ("مفعل", [1, 2, 3]),
            ("فاعل", [0, 2, 3]),
            ("فعال", [0, 1, 3]),
            ("فعول", [0, 1, 3]),
            ("فعيل", [0, 1, 3]),
            ("فعلة", [0, 1, 2]),
            ("يفعل", [1, 2, 3]),
            ("تفعل", [1, 2, 3]),
            ("نفعل", [1, 2, 3]),
            ("أفعل", [1, 2, 3]),
        ]

        self.known_roots = {}
        self.word_to_root_cache = {}

        if semantic_db_path and os.path.exists(semantic_db_path):
            self._load_semantic_db(semantic_db_path)

        self.arramooz_conn = None
        self.arramooz_cursor = None
        try:
            db_path = os.path.join(os.path.dirname(__file__), '../../arabic_tools/arramooz/arramooz.db')
            db_path = os.path.abspath(db_path)
            if os.path.exists(db_path):
                self.arramooz_conn = sqlite3.connect(db_path, check_same_thread=False)
                self.arramooz_cursor = self.arramooz_conn.cursor()
        except Exception as e:
            print(f"Warning: Could not connect to Arramooz DB at {db_path}: {e}")

    def __del__(self):
        if hasattr(self, 'arramooz_conn') and self.arramooz_conn:
            self.arramooz_conn.close()

    def _load_semantic_db(self, path: str):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            roots = data.get("roots", {})
            for root_key, root_data in roots.items():
                root = root_data.get("root", root_key)
                self.known_roots[root_key] = root

                derivatives = root_data.get("derivatives", [])
                for deriv in derivatives:
                    self.word_to_root_cache[deriv] = root_key

        except Exception as e:
            import logging; logging.getLogger('arabic_semantics').debug(f'_build_root_cache: {e}')

    def extract(self, word: str) -> Tuple[str, float]:
        if word in self.word_to_root_cache:
            return self.word_to_root_cache[word], 1.0

        clean_word = self._clean_word(word)

        if clean_word in self.word_to_root_cache:
            return self.word_to_root_cache[clean_word], 0.9

        arramooz_root = self._lookup_arramooz(clean_word)
        if arramooz_root:
            self.word_to_root_cache[word] = arramooz_root
            self.word_to_root_cache[clean_word] = arramooz_root
            return arramooz_root, 1.0

        root, confidence = self._extract_algorithmic(clean_word)

        self.word_to_root_cache[word] = root

        return root, confidence

    def _lookup_arramooz(self, word: str) -> Optional[str]:
        if not self.arramooz_cursor:
            return None

        try:
            self.arramooz_cursor.execute("SELECT root FROM verbs WHERE unvocalized = ? LIMIT 1", (word,))
            row = self.arramooz_cursor.fetchone()
            if row and row[0]:
                return row[0]

            self.arramooz_cursor.execute("SELECT root FROM nouns WHERE unvocalized = ? LIMIT 1", (word,))
            row = self.arramooz_cursor.fetchone()
            if row and row[0]:
                return row[0]
        except Exception as e:
            import logging; logging.getLogger('arabic_semantics').debug(f'lookup_arramooz: {e}')
        return None

    def _clean_word(self, word: str) -> str:
        word = self._remove_diacritics(word)

        if word.startswith(self.definite_article) and len(word) > 4:
            word = word[2:]

        for _ in range(2):
            for prefix in self.prefixes:
                if word.startswith(prefix) and len(word) > 4:
                    word = word[1:]
                    break

        for pronoun in sorted(self.attached_pronouns, key=len, reverse=True):
            if word.endswith(pronoun) and len(word) > len(pronoun) + 2:
                word = word[:-len(pronoun)]
                break

        if word.endswith("ة") and len(word) > 3:
            word = word[:-1]

        return word

    def _remove_diacritics(self, text: str) -> str:
        diacritics = "ًٌٍَُِّْـ"
        return ''.join(c for c in text if c not in diacritics)

    def _extract_algorithmic(self, word: str) -> Tuple[str, float]:
        if len(word) <= 3:
            return "-".join(list(word)), 0.8

        for pattern, root_indices in self.patterns:
            if len(word) == len(pattern):
                root_chars = []
                match = True
                for i in range(len(word)):
                    if i in root_indices:
                        root_chars.append(word[i])
                    elif word[i] != pattern[i] and word[i] in self.augmentation_letters:
                        if pattern[i] == 'ا' and word[i] in 'أإآويا':
                            continue
                        elif pattern[i] == 'ء' and word[i] in 'ؤئء':
                            continue
                        else:
                            match = False
                            break
                    elif word[i] != pattern[i]:
                        match = False
                        break

                if match and len(root_chars) >= 3:
                    return "-".join(root_chars[:3]), 0.95

        root_letters = []
        for i, char in enumerate(word):
            if i == 0 or char not in self.augmentation_letters or len(root_letters) < 3:
                root_letters.append(char)

        if len(root_letters) >= 3:
            return "-".join(root_letters[:3]), 0.6
        elif len(root_letters) == 2:
            return "-".join(root_letters), 0.4
        else:
            return word[0] if word else "؟", 0.2

    def batch_extract(self, words: List[str]) -> List[Tuple[str, float]]:
        return [self.extract(word) for word in words]


# ============================================================
#  قاموس معاني الحروف (خفيف) + تفكيك الكلمة إلى دلالاتها
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
    '\u0622': '\u0627',
    '\u0623': '\u0627',
    '\u0625': '\u0627',
    '\u0624': '\u0648',
    '\u0626': '\u064a',
    '\u0629': '\u0647',
    '\u0649': '\u064a',
}


def decompose_word_definition(word: str) -> list:
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
        meaning = LETTER_DEFINITIONS.get(ch, "?")
        result.append((ch, meaning))
    return result
