# -*- coding: utf-8 -*-
"""تطور المتجهات الطورية — تنظيم ذاتي عبر المزامنة الهيبيانية.

الفلسفة: البلورة تتشكّل حين تستقر الذرات في مواضع أقل طاقة.
كذلك الكلمات — كلما رنّت معاً في سياق، تتقارب متجهاتها الطورية.
هذا ليس backpropagation — إنه فيزياء:
  PV[w] ← PV[w] + η × resonance(w, context) × (PV[context_avg] − PV[w])

المتجهات لا تُعدَّل عبر مشتقات خطأ — بل عبر رنين:
- كلمات تظهر معاً كثيراً → متجهاتها تتقارب (مزامنة طورية)
- كلمات لا رنين بينها → متجهاتها تبقى كما هي
- التعلم محلي: كل كلمة تتعلم من سياقها المباشر فقط

أنماط التطور:
1. co_occurrence: تقارب الكلمات المتشاركة في السياق
2. contrastive: تباعد الكلمات المتنافسة (مثل:_hot vs _cold)
3. spectral_shift: إزاحة طورية بناءً على الملاحظة النصية
"""
import numpy as np
import logging
from src.physics.word_physics import compute_extended_phase_vector, phase_similarity
from src.physics.constants import TOTAL_DIM

logger = logging.getLogger(__name__)


class PhaseEvolution:
    """محرك التطور الطوري — تنظيم ذاتي للمتجهات."""

    def __init__(self, config=None):
        cfg = (config or {}).get('phase_evolution', {})
        self.lr_cooc = cfg.get('lr_cooc', 0.03)
        self.lr_contrast = cfg.get('lr_contrast', 0.015)
        self.lr_spectral = cfg.get('lr_spectral', 0.02)
        self.min_cooc = cfg.get('min_cooc', 2)
        self.max_shift = cfg.get('max_shift', 0.25)
        self.decay = cfg.get('shift_decay', 0.99)
        self.shifts = {}
        self.cooc_counts = {}
        self.epoch = 0

    def observe_cooccurrence(self, w1_pv, w2_pv, w1_id=None, w2_id=None, distance=1, strength=1.0):
        """مراقبة تشارك في السياق — تسجيل للتحديث اللاحق.

        Args:
            w1_pv: متجه الكلمة الأولى (TOTAL_DIM,)
            w2_pv: متجه الكلمة الثانية (TOTAL_DIM,)
            w1_id: معرّف الكلمة الأولى (اختياري)
            w2_id: معرّف الكلمة الثانية (اختياري)
            distance: المسافة بينهما في السياق
            strength: قوة الرنين بينهما
        """
        decay_factor = 1.0 / (1.0 + distance * 0.3)
        effective_strength = strength * decay_factor

        if w1_id is not None:
            if w1_id not in self.cooc_counts:
                self.cooc_counts[w1_id] = {}
            self.cooc_counts[w1_id][w2_id] = self.cooc_counts[w1_id].get(w2_id, 0) + 1

        sim = phase_similarity(w1_pv, w2_pv)
        shift_magnitude = self.lr_cooc * effective_strength * (1.0 - sim)

        shift = shift_magnitude * (w2_pv - w1_pv)
        shift_norm = np.linalg.norm(shift)
        if shift_norm > self.max_shift:
            shift = shift * (self.max_shift / shift_norm)

        if w1_id is not None:
            if w1_id not in self.shifts:
                self.shifts[w1_id] = np.zeros(TOTAL_DIM)
            self.shifts[w1_id] += shift

    def observe_contrast(self, w1_pv, w2_pv, w1_id=None, w2_id=None):
        """مراقبة تباين — كلمات متنافضة يجب أن تتباعد."""
        sim = phase_similarity(w1_pv, w2_pv)
        if sim > 0.5:
            repulsion = self.lr_contrast * (sim - 0.5)
            direction = w1_pv - w2_pv
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-10:
                shift = repulsion * direction / direction_norm
                shift_norm = np.linalg.norm(shift)
                if shift_norm > self.max_shift * 0.5:
                    shift = shift * (self.max_shift * 0.5 / shift_norm)
                if w1_id is not None:
                    if w1_id not in self.shifts:
                        self.shifts[w1_id] = np.zeros(TOTAL_DIM)
                    self.shifts[w1_id] += shift

    def observe_spectral_shift(self, word_pv, observed_context_pv, word_id=None):
        """إزاحة طورية بناءً على الملاحظة الطيفية للسياق.

        الكلمة التي تظهر في سياق مختلف عن متوسط سياقها المتوقع
        تُزاح نحو السياق الفعلي — هذا يعكس ما تتعلمه البلورات
        من المجال المحيط.
        """
        residue = observed_context_pv - word_pv
        residue_norm = np.linalg.norm(residue)
        if residue_norm > 1e-10:
            shift = self.lr_spectral * residue
            shift_norm = np.linalg.norm(shift)
            if shift_norm > self.max_shift:
                shift = shift * (self.max_shift / shift_norm)
            if word_id is not None:
                if word_id not in self.shifts:
                    self.shifts[word_id] = np.zeros(TOTAL_DIM)
                self.shifts[word_id] += shift

    def evolve(self, all_pv, vocab=None, window=5):
        """تطبيق التحديثات الطورية على مصفوفة المتجهات.

        Returns:
            np.ndarray: المصفوفة المحدّثة
            dict: إحصائيات التطور
        """
        stats = {
            'words_shifted': 0,
            'total_shift': 0.0,
            'max_shift': 0.0,
            'avg_shift': 0.0,
        }

        if not self.shifts:
            return all_pv, stats

        new_pv = all_pv.copy()

        for word_id, shift in self.shifts.items():
            if word_id is not None and word_id < len(new_pv):
                decayed_shift = shift * (self.decay ** self.epoch)
                shift_norm = np.linalg.norm(decayed_shift)
                if shift_norm > 1e-12:
                    if self.cooc_counts.get(word_id):
                        max_cooc = max(self.cooc_counts[word_id].values())
                        confidence = min(1.0, max_cooc / (self.min_cooc + 1))
                    else:
                        confidence = 0.3
                    effective_shift = decayed_shift * confidence
                    new_pv[word_id] += effective_shift
                    pv_norm = np.linalg.norm(new_pv[word_id])
                    if pv_norm > 1e-10:
                        new_pv[word_id] /= pv_norm
                    stats['words_shifted'] += 1
                    stats['total_shift'] += float(shift_norm)
                    stats['max_shift'] = max(stats['max_shift'], float(shift_norm))

        if stats['words_shifted'] > 0:
            stats['avg_shift'] = stats['total_shift'] / stats['words_shifted']

        self.shifts.clear()
        self.epoch += 1

        logger.info(f"  تطور طوري: {stats['words_shifted']} كلمة، "
                     f"متوسط إزاحة={stats['avg_shift']:.6f}")

        return new_pv, stats

    def evolve_from_corpus(self, all_pv, corpus_texts, vocab, window=5):
        """تطور ذاتي من كوربوس كامل — المزامنة الهيبيانية الجماعية.

        لكل تسلسل كلمات في الكوربس:
        1. نحسب متجهات السياق (متوسطPVs المحيطة)
        2. نقارن كل كلمة مع سياقها
        3. نُسجّل الأزاحة نحو السياق (تقارب رنيني)
        4. للكلمات المتنافسة، نُسجّل تباعد (تنافر طوري)

        هذا هوب فيزيائي — لا مشتقات، لا backprop.
        """
        if isinstance(corpus_texts, str):
            corpus_texts = [corpus_texts]

        total_pairs = 0
        for text in corpus_texts:
            words = text.split()
            for i, word in enumerate(words):
                wid = vocab.word2id.get(word)
                if wid is None or wid >= len(all_pv):
                    continue
                word_pv = all_pv[wid].copy()

                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                context_words = [words[j] for j in range(start, end) if j != i]
                context_ids = [vocab.word2id.get(cw) for cw in context_words
                               if vocab.word2id.get(cw) is not None
                               and vocab.word2id.get(cw) < len(all_pv)]

                if not context_ids:
                    continue

                context_pvs = all_pv[context_ids]
                context_avg = np.mean(context_pvs, axis=0)

                for j, cid in enumerate(context_ids):
                    dist = abs(i - (start + j))
                    self.observe_cooccurrence(
                        word_pv, all_pv[cid],
                        w1_id=wid, w2_id=cid,
                        distance=max(1, dist)
                    )
                    total_pairs += 1

                self.observe_spectral_shift(word_pv, context_avg, word_id=wid)

        logger.info(f"  تسجيل هيبياني: {total_pairs} زوج من الكوربوس")

        new_pv, stats = self.evolve(all_pv, vocab)
        return new_pv, stats