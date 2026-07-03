"""PPM — Prompt Phase Modulation.

حقل طوري خارجي مؤقت يزيح توازن التوليد حسب أمثلة الـ prompt، ويضمحل تلقائياً.
محاكاة فيزيائية لـ In-Context Learning / Few-Shot.
"""

import numpy as np
from src.physics.word_physics import compute_extended_phase_vector
from src.physics.constants import TOTAL_DIM


class PromptField:
    """حقل طوري خارجي يضمحل — يمتص أنماط الـ prompt ويعدل التوليد."""

    def __init__(self, decay_rate=0.1, strength=1.0, max_examples=10):
        self.decay_rate = decay_rate
        self.strength = strength
        self.max_examples = max_examples
        self.field = np.zeros(TOTAL_DIM)
        self.active = False
        self._pv_cache = {}
        self._example_count = 0

    def _get_pv(self, word):
        if word not in self._pv_cache:
            self._pv_cache[word] = compute_extended_phase_vector(word)
        return self._pv_cache[word]

    def absorb(self, prompt, examples=None):
        """ابتلاع prompt أو أمثلة كحقل خارجي.

        Args:
            prompt: str — النص الكامل (يمكن أن يحتوي أمثلة)
            examples: list[str] — أمثلة صريحة (اختياري)
        """
        if examples:
            sources = examples
        else:
            sources = [prompt]

        pvs = []
        for src in sources:
            for word in src.split():
                pv = self._get_pv(word)
                if pv is not None:
                    pvs.append(pv)

        if pvs:
            pv_array = np.array(pvs)
            self.field = self.strength * np.mean(pv_array, axis=0)
            self.active = True
            self._example_count = min(len(sources), self.max_examples)

    def modulate(self, candidate_pv):
        """تعديل طور المرشح حسب الحقل الخارجي.

        Returns:
            np.ndarray: الطور المعدّل
        """
        if not self.active:
            return candidate_pv
        return candidate_pv + self.field * 0.15

    def score(self, w_pv):
        """درجة التوافق بين المرشح والحقل.

        Returns:
            float: 0..1
        """
        if not self.active or np.linalg.norm(self.field) < 1e-10:
            return 0.0
        w_norm = np.linalg.norm(w_pv)
        f_norm = np.linalg.norm(self.field)
        if w_norm < 1e-10 or f_norm < 1e-10:
            return 0.0
        cos = float(np.dot(w_pv, self.field)) / (w_norm * f_norm)
        return max(0.0, cos)

    def step(self):
        """اضمحلال الحقل بعد كل خطوة توليد."""
        if self.active:
            self.field *= (1.0 - self.decay_rate)
            if np.linalg.norm(self.field) < 0.01:
                self.field = np.zeros(TOTAL_DIM)
                self.active = False

    def reset(self):
        """إعادة تعيين الحقل."""
        self.field = np.zeros(TOTAL_DIM)
        self.active = False
        self._example_count = 0
