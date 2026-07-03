# -*- coding: utf-8 -*-
"""
Holographic Knowledge Base — ذاكرة معرفية هولوغرافية (Resonant Bank)

المبدأ الفيزيائي: بنك مرشحات رنّان (Resonant Filter Bank).
- كل حقيقة = زوج (key, value) حيث المفتاح = phase_conjugate(subject).
- عند الاستعلام، يمر query عبر كل المرشحات في آنٍ واحد (تراكب كمومي).
- المرشح الأكثر رنيناً (أعلى dot product) يمرر قيمته المخزنة.
- الإجابة = تراكب موزون لجميع القيم المخزنة حسب درجة الرنين.

هذا يكافئ "ذاكرة المحتوى القابلة للعنونة" (CAM) في الإلكترونيات،
ويُطبّق مبدأ "التراكب الكمومي" — كل الحقائق تُستعلم دفعة واحدة.

الحماية من التداخل:
1. تجزئة حسب نوع العلاقة (IS_A, LOCATED_IN, ...)
2. رفع أسي للرنين (sharpening factor) لتمييز الإشارة عن الضوضاء
3. عتبة دنيا للرنين (cutoff threshold) تمنع استرجاع الحقائق غير المرتبطة
"""

import os
import re
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from src.physics.constants import TOTAL_DIM
except ImportError:
    TOTAL_DIM = 64

RELATION_TYPES = {
    "IS_A": 0,
    "HAS_PROPERTY": 1,
    "LOCATED_IN": 2,
    "CAPABLE_OF": 3,
    "PART_OF": 4,
    "SYNONYM": 5,
    "CURATED": 6,
    "CAPITAL_OF": 7,
    "MADE_OF": 8,
    "USED_FOR": 9,
    "BECOMES_WHEN": 10,
    "PRODUCES": 11,
}


class HolographicKB:
    """بنك مرشحات رنّان لتخزين واسترجاع المعرفة بالتراكب الكمومي."""

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.banks = {}         # rel_type → {'keys': [], 'values': [], 'words': []}
        self.fact_count = 0
        self._built = False

    def store_fact(self, subj_pv, obj_pv, rel_type, subj_word="", obj_word=""):
        """تخزين حقيقة: (subject, relation, object).

        يُستخدم word ID كفهرس لمنع التصادم بين الكلمات ذات المتجهات المتشابهة.
        """
        if rel_type not in RELATION_TYPES:
            return False

        if rel_type not in self.banks:
            self.banks[rel_type] = {'keys': [], 'values': [], 'subj_words': [], 'obj_words': []}

        bank = self.banks[rel_type]
        # Dedup: check (subject word, object word) pair
        for i, sw in enumerate(bank['subj_words']):
            if sw == subj_word and bank['obj_words'][i] == obj_word:
                return False  # exact duplicate
        bank['keys'].append(subj_pv.copy())
        bank['values'].append(obj_pv.copy())
        bank['subj_words'].append(subj_word)
        bank['obj_words'].append(obj_word)
        self.fact_count += 1
        self._built = True
        return True

    def query(self, query_pv, rel_type=None, top_k=10, sharpening=3.0, cutoff=0.0):
        """استعلام بنك المرشحات — تراكب موزون لجميع الحقائق المطابقة."""
        if not self._built:
            return []

        query_norm = np.linalg.norm(query_pv)
        if query_norm < 1e-10:
            return []
        q = query_pv / query_norm

        rel_types = [rel_type] if rel_type else list(self.banks.keys())
        all_results = []

        for rt in rel_types:
            if rt not in self.banks:
                continue
            bank = self.banks[rt]
            keys = bank['keys']
            values = bank['values']
            obj_words = bank['obj_words']

            if not keys:
                continue

            keys_matrix = np.array(keys)
            keys_norms = np.linalg.norm(keys_matrix, axis=1, keepdims=True)
            keys_normed = keys_matrix / np.maximum(keys_norms, 1e-10)

            resonances = np.dot(keys_normed, q)
            sharpened = np.exp(sharpening * resonances)
            total = sharpened.sum()
            if total < 1e-10:
                continue
            sharpened = sharpened / total

            # عتبة ديناميكية: 1/N (متوسط التوزيع المتساوي) مضروب في 2
            dynamic_cutoff = max(cutoff, 2.0 / max(len(keys), 1))
            above_cutoff = sharpened > dynamic_cutoff

            for i in np.where(above_cutoff)[0]:
                obj_word = obj_words[i] if i < len(obj_words) else ""
                if not obj_word or len(obj_word) < 2:
                    continue
                all_results.append((float(sharpened[i]), obj_word, rt, q))

        all_results.sort(key=lambda x: -x[0])
        return [(s, w, rt) for s, w, rt, _ in all_results[:top_k]]

    def query_bidirectional(self, query_word, rel_type=None, top_k=10):
        """استعلام ثنائي الاتجاه — يبحث في الفاعل والمفعول معاً."""
        if not self._built:
            return []

        rel_types = [rel_type] if rel_type else list(self.banks.keys())
        all_results = []

        for rt in rel_types:
            if rt not in self.banks:
                continue
            bank = self.banks[rt]
            for i, sw in enumerate(bank['subj_words']):
                obj_w = bank['obj_words'][i] if i < len(bank['obj_words']) else ""
                if sw == query_word:
                    all_results.append((1.0, obj_w, rt))
                elif obj_w == query_word:
                    all_results.append((1.0, sw, rt))

        all_results.sort(key=lambda x: -x[0])
        return all_results[:top_k]

    def query_by_word(self, query_word, rel_type=None, top_k=10):
        """استعلام بالكلمة المباشرة (اتجاه واحد: فاعل→مفعول).

        يُستخدم في المُسبِّب (reasoner) لتتبع سلاسل المعرفة.
        """
        if not self._built:
            return []

        rel_types = [rel_type] if rel_type else list(self.banks.keys())
        all_results = []

        for rt in rel_types:
            if rt not in self.banks:
                continue
            bank = self.banks[rt]
            for i, sw in enumerate(bank['subj_words']):
                if sw == query_word:
                    obj_word = bank['obj_words'][i] if i < len(bank['obj_words']) else ""
                    all_results.append((1.0, obj_word, rt))

        all_results.sort(key=lambda x: -x[0])
        return all_results[:top_k]

    def find_closest_words(self, query_pv, vocab, pv_matrix, rel_type=None, top_k=10, sharpening=3.0):
        """استعلام وإرجاع أقرب كلمات معجمية عبر مطابقة المتجه المُعاد بناؤه.

        يُنشئ reconstructed_pv عبر تراكب موزون للقيم، ثم يبحث عن أقرب كلمة في المعجم.
        """
        hologram_results = self.query(query_pv, rel_type=rel_type, top_k=top_k * 3, sharpening=sharpening)
        if not hologram_results:
            return []

        # تراكب موزون: بناء reconstructed_pv من نتائج الاستعلام
        word_scores = []
        for confidence, word, rt in hologram_results:
            if word and len(word) >= 2 and word in vocab.word2id:
                wid = vocab.word2id[word]
                if wid < len(pv_matrix):
                    # الرنين بين reconstructed والكلمة الفعلية
                    word_pv = pv_matrix[wid]
                    sim = np.dot(word_pv, query_pv) / (
                        max(np.linalg.norm(word_pv), 1e-10) *
                        max(np.linalg.norm(query_pv), 1e-10))
                    combined_score = confidence * max(0.0, float(sim))
                    word_scores.append((word, combined_score, rt))

        word_scores.sort(key=lambda x: -x[1])
        deduped = {}
        for w, s, rt in word_scores:
            if w not in deduped or s > deduped[w][0]:
                deduped[w] = (s, rt)
        return [(w, s, rt) for w, (s, rt) in deduped.items()][:top_k]

    def reconstruct_vector(self, query_pv, rel_type=None):
        """إعادة بناء متجه المفعول من استعلام الفاعل عبر التراكب الموزون.

        هذا هو "الجداء الهولوغرافي" — استرجاع الموجة المخزنة.

        Returns:
            np.ndarray: متجه 56D مُعاد بناؤه، أو None
        """
        if not self._built:
            return None

        q = query_pv / max(np.linalg.norm(query_pv), 1e-10)
        rel_types = [rel_type] if rel_type else list(self.banks.keys())
        reconstructed = np.zeros(TOTAL_DIM)
        total_weight = 0.0

        for rt in rel_types:
            if rt not in self.banks:
                continue
            bank = self.banks[rt]
            keys = np.array(bank['keys'])
            values = np.array(bank['values'])

            if len(keys) == 0:
                continue

            keys_norms = np.linalg.norm(keys, axis=1, keepdims=True)
            keys_normed = keys / np.maximum(keys_norms, 1e-10)
            resonances = np.dot(keys_normed, q)
            weights = np.exp(3.0 * resonances)
            weights = weights / (weights.sum() + 1e-10)

            for i, w in enumerate(weights):
                if w > 0.01:
                    reconstructed += w * values[i]
                    total_weight += w

        if total_weight > 1e-10:
            reconstructed /= total_weight
            nrm = np.linalg.norm(reconstructed)
            if nrm > 1e-10:
                reconstructed /= nrm
            return reconstructed
        return None

    def save(self, filename="holographic_kb.npz"):
        """حفظ البنك إلى ملف."""
        path = os.path.join(self.data_dir, filename)
        os.makedirs(self.data_dir, exist_ok=True)

        save_dict = {}
        for rt, bank in self.banks.items():
            if bank['keys']:
                save_dict[f"K_{rt}"] = np.array(bank['keys'])
                save_dict[f"V_{rt}"] = np.array(bank['values'])
                save_dict[f"SW_{rt}"] = np.array(bank['subj_words'])
                save_dict[f"OW_{rt}"] = np.array(bank['obj_words'])
        save_dict["fact_count"] = np.array([self.fact_count])

        np.savez_compressed(path, **save_dict)
        logger.info(f"  ✓ حفظ بنك المرشحات ({self.fact_count} حقيقة) إلى {path}")

    def load(self, filename="holographic_kb.npz"):
        """تحميل البنك من ملف."""
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            return False

        data = np.load(path, allow_pickle=True)
        self.banks = {}
        self.fact_count = int(data.get("fact_count", [0])[0])

        for rt in RELATION_TYPES:
            k_key = f"K_{rt}"
            if k_key in data:
                self.banks[rt] = {
                    'keys': [v for v in data[k_key]],
                    'values': [v for v in data[f"V_{rt}"]],
                    'subj_words': [str(w) for w in data.get(f"SW_{rt}", [])],
                    'obj_words': [str(w) for w in data.get(f"OW_{rt}", [])],
                }
        self._built = len(self.banks) > 0
        logger.info(f"  ✓ تحميل بنك المرشحات ({self.fact_count} حقيقة) من {path}")
        return self._built

    def build_from_texts(self, texts, vocab, pv_fn, max_facts=500):
        """استخراج الحقائق من النصوص الخام وتخزينها.

        يستخرج حقائق من أنماط جمل بسيطة:
        - 'X هو Y' / 'X is Y' → IS_A
        - 'X في Y' / 'X is in Y' → LOCATED_IN
        - 'X يستطيع Y' / 'X can Y' → CAPABLE_OF

        الفلترة: استبعاد الكلمات القصيرة والرموز والجسيمات النحوية.
        """
        stored = 0
        seen = set()
        _skipped_short = 0
        _skipped_novocab = 0

        # كلمات يجب تجاهلها
        STOP_WORDS = {
            'هو', 'هي', 'في', 'كان', 'كانت', 'على', 'عن', 'من', 'إلى', 'مع', 'أن', 'إن',
            'لم', 'لن', 'لا', 'ما', 'هذا', 'هذه', 'ذلك', 'تلك', 'كل', 'بعض',
            'the', 'is', 'a', 'an', 'in', 'of', 'to', 'for', 'with', 'and', 'or',
            'not', 'this', 'that', 'these', 'those', 'it', 'its',
        }

        for text in texts:
            if stored >= max_facts:
                break
            if not text or len(str(text)) < 20:
                continue
            sentences = re.split(r'[.!?\n]+', str(text))
            for sent in sentences:
                if stored >= max_facts:
                    break
                sent = sent.strip()
                if not sent or len(sent) < 6:
                    continue

                words = [w for w in sent.split() if len(w) >= 2]
                if len(words) < 3:
                    continue

                # تحديد نوع العلاقة من السياق
                rel_type = None
                if "هو" in words or "هي" in words:
                    rel_type = "IS_A"
                elif "في" in words:
                    rel_type = "LOCATED_IN"
                elif "يستطيع" in words or "يمكنه" in words or "يمكنها" in words:
                    rel_type = "CAPABLE_OF"
                elif "جزء" in words or ("من" in words and len(words) >= 4):
                    rel_type = "PART_OF"

                if rel_type is None:
                    continue

                # استخراج subject و object — فلترة الكلمات القصيرة والرموز
                non_particles = []
                for w in words:
                    if w in STOP_WORDS:
                        continue
                    # استبعاد الرموز والإيموجي
                    if not any(c.isalpha() or '\u0600' <= c <= '\u06ff' for c in w):
                        continue
                    if len(w) >= 3:
                        non_particles.append(w)

                if len(non_particles) < 2:
                    continue

                subj = non_particles[0]
                obj = non_particles[-1]

                if subj == obj:
                    continue
                if (subj, rel_type, obj) in seen:
                    continue
                if subj not in vocab.word2id or obj not in vocab.word2id:
                    _skipped_novocab += 1
                    continue

                subj_pv = pv_fn(subj)
                obj_pv = pv_fn(obj)
                if subj_pv is None or obj_pv is None:
                    continue
                # فلترة: تجاهل المتجهات الصفرية أو الضعيفة جداً
                if np.linalg.norm(subj_pv) < 0.01 or np.linalg.norm(obj_pv) < 0.01:
                    _skipped_short += 1
                    continue

                if self.store_fact(subj_pv, obj_pv, rel_type, subj_word=subj, obj_word=obj):
                    seen.add((subj, rel_type, obj))
                    stored += 1
                    if stored >= max_facts:
                        break

        self.save()
        logger.info(f"  ✓ بنك المرشحات: {stored} حقيقة (تجاوز: {_skipped_novocab} خارج المعجم, {_skipped_short} متجه ضعيف)")
        return stored

    def load_curated_facts(self, json_path, vocab, pv_fn):
        """تحميل حقائق منسقة من ملف JSON.

        التنسيق المتوقع:
        {
          "facts": [
            {"subject": "...", "relation": "...", "object": "..."},
            ...
          ]
        }

        تُخزَّن الحقائق في بنك CURATED (نوع 6).
        """
        if not os.path.exists(json_path):
            logger.warning(f"  ⚠ ملف الحقائق غير موجود: {json_path}")
            return 0

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        facts = data.get('facts', [])
        stored = 0

        for fact in facts:
            subj = fact.get('subject', '')
            obj = fact.get('object', '')

            if not subj or not obj:
                continue
            if vocab.get(subj) is None or vocab.get(obj) is None:
                continue

            subj_pv = pv_fn(subj)
            obj_pv = pv_fn(obj)
            if subj_pv is None or obj_pv is None:
                continue

            if self.store_fact(subj_pv, obj_pv, "CURATED", subj_word=subj, obj_word=obj):
                stored += 1

            # Also store with الـ prefix if the bare form exists
            prefixed = "ال" + subj
            if vocab.get(prefixed) is not None and prefixed != subj:
                prefixed_pv = pv_fn(prefixed)
                if prefixed_pv is not None:
                    if self.store_fact(prefixed_pv, obj_pv, "CURATED", subj_word=prefixed, obj_word=obj):
                        stored += 1

            # Also store English lowercase/capitalized variants
            if subj[0].islower():
                cap = subj[0].upper() + subj[1:]
                if cap in vocab.word2id and cap != subj:
                    cap_pv = pv_fn(cap)
                    if cap_pv is not None:
                        if self.store_fact(cap_pv, obj_pv, "CURATED", subj_word=cap, obj_word=obj):
                            stored += 1

        self.save()
        logger.info(f"  ✓ تحميل الحقائق المنسقة: {stored} حقيقة من {json_path}")
        return stored


def build_holographic_kb(vocab, pv_fn, corpus_texts=None, data_dir="data", max_facts=500, curated_path=None):
    """بناء أو تحميل بنك المرشحات الرنّان.

    Args:
        vocab: قاموس المفردات
        pv_fn: دالة word → pv
        corpus_texts: قائمة النصوص الخام (اختياري)
        data_dir: مجلد البيانات
        max_facts: أقصى عدد حقائق
        curated_path: مسار ملف الحقائق المنسقة (JSON)

    Returns:
        HolographicKB: قاعدة معرفية جاهزة
    """
    kb = HolographicKB(data_dir=data_dir)

    if kb.load():
        return kb

    if corpus_texts:
        kb.build_from_texts(corpus_texts, vocab, pv_fn, max_facts=max_facts)

    if curated_path:
        kb.load_curated_facts(curated_path, vocab, pv_fn)

    return kb

    return kb
