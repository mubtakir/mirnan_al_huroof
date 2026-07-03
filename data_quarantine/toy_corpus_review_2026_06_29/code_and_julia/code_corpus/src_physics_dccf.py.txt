"""DCCF — Dynamic Contextual Coupling Field.

حقل اقتران ديناميكي لحظي يُبنى من سياق التوليد الحالي فقط.
يوزن العلاقات الطويلة المدى طورياً باضمحلال المسافة.
"""

import numpy as np
from src.physics.word_physics import compute_extended_phase_vector
from src.physics.constants import TOTAL_DIM


class DCCF:
    """حقل اقتران ديناميكي — بديل فيزيائي للـ self-attention."""

    def __init__(self, decay_rate=0.5, mass_threshold=0.3):
        self.decay_rate = decay_rate
        self.mass_threshold = mass_threshold
        self._pv_cache = {}

    def _get_pv(self, word):
        if word not in self._pv_cache:
            self._pv_cache[word] = compute_extended_phase_vector(word)
        return self._pv_cache[word]

    def build_coupling(self, context_words, K_matrix=None, vocab=None):
        """بناء مصفوفة اقتران لحظية من سياق التوليد.

        Args:
            context_words: list[str] — كلمات السياق الحالي
            K_matrix: sparse matrix (اختياري) — K للاقترانات الإحصائية
            vocab: Vocabulary (اختياري) — للاستعلام من K

        Returns:
            coupling: np.ndarray (n, n) — مصفوفة الاقتران
            scores: dict — درجات لكل زوج
        """
        n = len(context_words)
        if n < 2:
            return np.eye(1), {}

        pvs = [self._get_pv(w) for w in context_words]
        coupling = np.zeros((n, n))
        scores = {}

        for i in range(n):
            for j in range(i + 1, n):
                if pvs[i] is None or pvs[j] is None:
                    continue
                norm_i = np.linalg.norm(pvs[i])
                norm_j = np.linalg.norm(pvs[j])
                if norm_i < 1e-10 or norm_j < 1e-10:
                    continue
                dot = float(np.dot(pvs[i], pvs[j]))
                phase_align = dot / (norm_i * norm_j)

                mass_sim = norm_i * norm_j
                dist_decay = np.exp(-self.decay_rate * abs(i - j))

                val = phase_align * mass_sim * dist_decay
                coupling[i, j] = val
                coupling[j, i] = val
                scores[(i, j)] = {
                    'phase_align': phase_align,
                    'mass_sim': mass_sim,
                    'dist_decay': dist_decay,
                    'total': val,
                }

        return coupling, scores

    def get_context_boost(self, word, context_words, context_ids=None, vocab=None):
        """حساب تعزيز DCCF لكلمة مرشحة ضد سياق التوليد.

        Returns:
            float: درجة الاقتران (0..1)
        """
        n = len(context_words)
        if n < 1:
            return 0.0

        pw = self._get_pv(word)
        if pw is None:
            return 0.0
        norm_w = np.linalg.norm(pw)
        if norm_w < 1e-10:
            return 0.0

        total = 0.0
        weight_sum = 0.0
        for i, cw in enumerate(context_words):
            cpv = self._get_pv(cw)
            if cpv is None:
                continue
            norm_c = np.linalg.norm(cpv)
            if norm_c < 1e-10:
                continue
            phase_align = float(np.dot(pw, cpv)) / (norm_w * norm_c)
            if phase_align < self.mass_threshold:
                continue
            dist_decay = np.exp(-self.decay_rate * (n - i))
            w = dist_decay * norm_c
            total += phase_align * w
            weight_sum += w

        return total / max(weight_sum, 1e-10)
