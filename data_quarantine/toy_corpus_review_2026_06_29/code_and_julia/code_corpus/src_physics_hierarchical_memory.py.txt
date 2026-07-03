# -*- coding: utf-8 -*-
"""الذاكرة الهرمية الطيفية — ضغط السياق عبر أهرام طيفية.

الفلسفة: الموسيقى تُضغط هرمياً — من النغمة إلى الجملة إلى اللحن إلى السيمفونية.
كذلك السياق اللغوي:
- كلمة → متجه طوري (64D)
- عبارة → متوسط مرجح بالكتلة + توقيع FFT
- فقرة → متوسط الفقرات + طيف مُختزل
- محادثة → متجه واحد يلخص الموضوع العام

كل مستوى يضغط المستوى الأدنى — لا نحتاج окна انتباه بحجم 128K،
بل 4 مستويات من الضغط الطيفي، كل واحد يحمل الإشارات الجوهرية.
"""
import numpy as np
import logging
from src.physics.word_physics import compute_word_mass
from src.physics.constants import TOTAL_DIM, PHASE_DIM

logger = logging.getLogger(__name__)


class SpectralLevel:
    """مستوى هرمي طيفي واحد."""

    def __init__(self, name, max_entries, decay):
        self.name = name
        self.max_entries = max_entries
        self.decay = decay
        self.entries = []

    def add(self, pv_summary, spectrum, mass, text="", metadata=None):
        """إضافة مدخل مع اضمحلال."""
        self.entries.append({
            'pv': pv_summary,
            'spectrum': spectrum,
            'mass': mass,
            'text': text,
            'metadata': metadata or {},
            'age': 0,
        })
        for e in self.entries:
            e['age'] += 1
        self._apply_decay()
        while len(self.entries) > self.max_entries:
            self.entries.pop(0)

    def _apply_decay(self):
        """اضمحلال طبيعي — المداخل القديمة تفقد كتلتها."""
        for e in self.entries:
            e['mass'] *= self.decay

    def get_context_pv(self):
        """المتجه الطوري المرجح بالكتلة لهذا المستوى."""
        if not self.entries:
            return np.zeros(TOTAL_DIM)
        total_mass = sum(e['mass'] for e in self.entries)
        if total_mass < 1e-10:
            return np.zeros(TOTAL_DIM)
        weighted = np.zeros(TOTAL_DIM)
        for e in self.entries:
            weighted += e['pv'] * e['mass']
        return weighted / total_mass

    def get_context_spectrum(self):
        """المتوسط الطيفي المرجح."""
        if not self.entries:
            return np.zeros(32)
        total_mass = sum(e['mass'] for e in self.entries)
        if total_mass < 1e-10:
            return np.zeros(32)
        weighted = np.zeros(32)
        for e in self.entries:
            weighted += e['spectrum'] * e['mass']
        return weighted / total_mass


class HierarchicalMemory:
    """ذاكرة هرمية طيفية — 4 مستويات من الضغط."""

    def __init__(self, config=None):
        cfg = config or {}
        self.word_level = SpectralLevel('word', 100, 0.98)
        self.phrase_level = SpectralLevel('phrase', 40, 0.95)
        self.paragraph_level = SpectralLevel('paragraph', 15, 0.90)
        self.conversation_level = SpectralLevel('conversation', 5, 0.85)
        self._phrase_buffer = []
        self._phrase_pvs = []
        self._phrase_word_count = 0
        self._paragraph_buffer = []
        self._paragraph_pvs = []

    def add_word(self, word, pv, mass=None):
        """إضافة كلمة — تُجمع في عبارات تلقائياً."""
        if mass is None:
            mass = compute_word_mass(word)
        self._phrase_buffer.append(word)
        self._phrase_pvs.append(pv.copy())
        self._phrase_word_count += 1

        self.word_level.add(
            pv_summary=pv[:TOTAL_DIM].copy(),
            spectrum=self._compute_spectrum(pv[:PHASE_DIM]),
            mass=mass,
            text=word,
        )

        if self._phrase_word_count >= 5:
            self._flush_phrase()

    def _flush_phrase(self):
        """ضغط العبارة المُجمّعة في مستوى العبارة."""
        if not self._phrase_pvs:
            return

        pvs = np.array(self._phrase_pvs)
        masses = np.array([compute_word_mass(w) for w in self._phrase_buffer])
        total_mass = masses.sum()

        if total_mass > 1e-10:
            phrase_pv = np.average(pvs, axis=0, weights=masses)
        else:
            phrase_pv = np.mean(pvs, axis=0)

        phrase_pv_norm = np.linalg.norm(phrase_pv)
        if phrase_pv_norm > 1e-10:
            phrase_pv /= phrase_pv_norm

        combined_spectrum = np.mean(
            [self._compute_spectrum(p[:PHASE_DIM]) for p in pvs], axis=0
        )

        phrase_text = ' '.join(self._phrase_buffer)

        self.phrase_level.add(
            pv_summary=phrase_pv,
            spectrum=combined_spectrum,
            mass=total_mass,
            text=phrase_text,
        )

        self._paragraph_buffer.extend(self._phrase_buffer)
        self._paragraph_pvs.append(phrase_pv)

        self._phrase_buffer = []
        self._phrase_pvs = []
        self._phrase_word_count = 0

        if len(self._paragraph_pvs) >= 4:
            self._flush_paragraph()

    def _flush_paragraph(self):
        """ضغط الفقرة في مستوى الفقرة."""
        if not self._paragraph_pvs:
            return

        pvs = np.array(self._paragraph_pvs)
        para_pv = np.mean(pvs, axis=0)

        para_pv_norm = np.linalg.norm(para_pv)
        if para_pv_norm > 1e-10:
            para_pv /= para_pv_norm

        para_spectrum = np.mean(
            [self._compute_spectrum(p[:PHASE_DIM]) for p in pvs], axis=0
        )

        para_text = ' '.join(self._paragraph_buffer)

        self.paragraph_level.add(
            pv_summary=para_pv,
            spectrum=para_spectrum,
            mass=float(np.mean([np.linalg.norm(p) for p in pvs])),
            text=para_text,
        )

        if len(self.paragraph_level.entries) >= 3:
            self._flush_conversation()

        self._paragraph_buffer = []
        self._paragraph_pvs = []

    def _flush_conversation(self):
        """ضغط الفكرة في مستوى المحادثة."""
        if not self.paragraph_level.entries:
            return

        para_pvs = [e['pv'] for e in self.paragraph_level.entries[-3:]]
        conv_pv = np.mean(para_pvs, axis=0)

        conv_pv_norm = np.linalg.norm(conv_pv)
        if conv_pv_norm > 1e-10:
            conv_pv /= conv_pv_norm

        conv_spectrum = np.mean(
            [e['spectrum'] for e in self.paragraph_level.entries[-3:]], axis=0
        )

        self.conversation_level.add(
            pv_summary=conv_pv,
            spectrum=conv_spectrum,
            mass=float(np.mean([e['mass'] for e in self.paragraph_level.entries[-3:]])),
            text="[conversation]",
        )

    def _compute_spectrum(self, phase_vec):
        """حساب التوقيع الطيفي عبر FFT."""
        spectrum = np.fft.rfft(phase_vec)
        return np.abs(spectrum[:16])

    def get_hierarchical_context(self, level='all'):
        """الحصول على السياق الهرمي كمتجه مدمج.

        Args:
            level: 'word', 'phrase', 'paragraph', 'conversation', 'all'

        Returns:
            np.ndarray: المتجه الهرمي المدمج (TOTAL_DIM,)
        """
        if level == 'word':
            return self.word_level.get_context_pv()
        elif level == 'phrase':
            return self.phrase_level.get_context_pv()
        elif level == 'paragraph':
            return self.paragraph_level.get_context_pv()
        elif level == 'conversation':
            return self.conversation_level.get_context_pv()
        elif level == 'all':
            weights = {
                'word': 0.15,
                'phrase': 0.25,
                'paragraph': 0.30,
                'conversation': 0.30,
            }
            combined = np.zeros(TOTAL_DIM)
            levels = {
                'word': self.word_level,
                'phrase': self.phrase_level,
                'paragraph': self.paragraph_level,
                'conversation': self.conversation_level,
            }
            for name, level_obj in levels.items():
                pv = level_obj.get_context_pv()
                pv_norm = np.linalg.norm(pv)
                if pv_norm > 1e-10:
                    combined += weights[name] * pv
            combined_norm = np.linalg.norm(combined)
            if combined_norm > 1e-10:
                combined /= combined_norm
            return combined
        return np.zeros(TOTAL_DIM)

    def get_hierarchical_spectrum(self):
        """الحصول على الطيف الهرمي المدمج."""
        weights = [0.1, 0.2, 0.3, 0.4]
        levels = [
            self.word_level,
            self.phrase_level,
            self.paragraph_level,
            self.conversation_level,
        ]
        combined = np.zeros(16)
        for w, level in zip(weights, levels):
            spec = level.get_context_spectrum()
            if len(spec) >= 16:
                combined += w * spec[:16]
        return combined

    def compute_level_resonance(self, word_pv, level='all'):
        """حساب رنين كلمة مع السياق الهرمي.

        Args:
            word_pv: متجه الكلمة المرشحة
            level: مستوى السياق

        Returns:
            dict: رنين مع كل مستوى
        """
        result = {}
        levels = {
            'word': self.word_level,
            'phrase': self.phrase_level,
            'paragraph': self.paragraph_level,
            'conversation': self.conversation_level,
        }
        for name, level_obj in levels.items():
            context_pv = level_obj.get_context_pv()
            if np.linalg.norm(context_pv) > 1e-10 and np.linalg.norm(word_pv) > 1e-10:
                from src.physics.word_physics import phase_similarity
                sim = phase_similarity(word_pv[:22], context_pv[:22])
                result[name] = float(sim)
            else:
                result[name] = 0.0

        if level == 'all':
            weights = {'word': 0.15, 'phrase': 0.25, 'paragraph': 0.3, 'conversation': 0.3}
            result['weighted'] = sum(weights[k] * result[k] for k in weights)

        return result

    def reset(self):
        """إعادة تعيين كل المستويات."""
        self.word_level.entries.clear()
        self.phrase_level.entries.clear()
        self.paragraph_level.entries.clear()
        self.conversation_level.entries.clear()
        self._phrase_buffer = []
        self._phrase_pvs = []
        self._phrase_word_count = 0
        self._paragraph_buffer = []
        self._paragraph_pvs = []