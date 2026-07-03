"""PhaseReinforcement — تعزيز طوري ذاتي.

تعلّم بدون backprop:
- تعزيز المسارات الطورية التي أدّت إلى توليدات ناجحة
- إضعاف المسارات التي أدّت إلى توليدات ضعيفة
- يشبه Hebbian learning لكن في فضاء طوري
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class PhaseReinforcement:
    """تعزيز طوري — ذاكرة تعلّم فيزيائية."""

    def __init__(self, learning_rate=0.15, decay=0.005, max_traces=200):
        self.lr = learning_rate
        self.decay = decay
        self.max_traces = max_traces
        self._traces = {}  # word -> reinforced_pv
        self._trace_strengths = {}  # word -> strength
        self._step_count = 0

    def reinforce(self, word, pv, reward=1.0):
        """تعزيز المتجه الطوري لكلمة بعد توليد ناجح.

        Args:
            word: str — الكلمة
            pv: np.ndarray — متجهها الطوري
            reward: float — 0..1 (1 = نجاح كامل)
        """
        if pv is None or np.linalg.norm(pv) < 1e-10:
            return

        if word not in self._traces:
            self._traces[word] = pv.copy()
            self._trace_strengths[word] = 0.0

        old = self._traces[word]
        reinforced = old + self.lr * reward * (pv - old)
        reinforced = reinforced / max(np.linalg.norm(reinforced), 1e-10)
        self._traces[word] = reinforced
        self._trace_strengths[word] = min(1.0, self._trace_strengths[word] + reward * self.lr)

        self._step_count += 1
        self._prune()

    def weaken(self, word, pv, penalty=0.3):
        """إضعاف مسار طوري بعد توليد ضعيف."""
        if word not in self._traces:
            return
        self._trace_strengths[word] = max(0.0, self._trace_strengths[word] - penalty * self.lr)
        if self._trace_strengths[word] < 0.01:
            self._traces.pop(word, None)
            self._trace_strengths.pop(word, None)

    def apply(self, word, pv):
        """تطبيق التعزيز على متجه طوري — يعيد المتجه المعدّل.

        Args:
            word: str
            pv: np.ndarray — المتجه الأصلي

        Returns:
            np.ndarray — المتجه بعد التعزيز (أو الأصلي إن لم يوجد تعزيز)
        """
        if word not in self._traces:
            return pv
        strength = self._trace_strengths[word]
        if strength < 0.01:
            return pv
        alpha = strength * 0.5
        return (1.0 - alpha) * pv + alpha * self._traces[word]

    def get_strength(self, word):
        """قوة التعزيز لكلمة (0..1)."""
        return self._trace_strengths.get(word, 0.0)

    def reinforce_sentence(self, words, pvs, reward=0.5):
        """تعزيز جملة كاملة بعد توليدها."""
        for w, pv in zip(words, pvs):
            self.reinforce(w, pv, reward=reward)

    def _prune(self):
        """إزالة الآثار الضعيفة للحفاظ على الذاكرة."""
        if len(self._traces) <= self.max_traces:
            return
        sorted_words = sorted(self._trace_strengths.items(), key=lambda x: -x[1])
        keep = set(w for w, _ in sorted_words[:self.max_traces])
        for w in list(self._traces.keys()):
            if w not in keep:
                del self._traces[w]
                del self._trace_strengths[w]

    def decay_all(self):
        """اضمحلال جميع التعزيزات — لمنع التعلّم الزائد."""
        for w in list(self._trace_strengths.keys()):
            self._trace_strengths[w] *= (1.0 - self.decay)
            if self._trace_strengths[w] < 0.01:
                del self._traces[w]
                del self._trace_strengths[w]

    def state_dict(self):
        return {
            'traces': {w: self._trace_strengths.get(w, 0.0) for w in self._traces},
            'step': self._step_count,
            'size': len(self._traces),
        }

    def reset(self):
        self._traces.clear()
        self._trace_strengths.clear()
        self._step_count = 0
