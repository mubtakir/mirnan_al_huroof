# -*- coding: utf-8 -*-
"""Heterodyne Engine — محرك التغاير الترددي الطيفي الحقيقي.

المبدأ الفيزيائي:
  - كل كلمة لها طيف ترددي = متجهها 22D. كل بُعد قناة ترددية مستقلة
    (تركيز، حرارة، حركة، صلابة، شحنة...).
  - التغاير = لكل قناة ترددية: f_carrier[ch] ± f_mod[ch]
    النطاقات الجانبية تُنشأ لكل قناة على حدة.
  - المرشح الذي تقع مكوناته الطيفية ضمن نطاقات التغاير هو الرفيق الحقيقي.

هذا يحاكي مستقبل الراديو متعدد القنوات:
  - التردد الحامل = طيف الكلمة (22 قناة)
  - التضمين = طيف السياق (22 قناة)
  - النطاقات الجانبية = f_plus[ch], f_minus[ch] لكل قناة
  - الكشف = كم قناة من المرشح تطابقت مع النطاقات؟
"""

import numpy as np
from src.physics.word_physics import compute_word_phase_vector
from src.physics.letter_db import LetterDB

_letter_db = None


def _get_db():
    global _letter_db
    if _letter_db is None:
        _letter_db = LetterDB()
    return _letter_db


class HeterodyneEngine:
    """محرك التغاير الترددي الطيفي — 22 قناة ترددية مستقلة.

    f_carrier[ch] ± f_mod[ch]  ←  لكل قناة ترددية ch ∈ [0..21]
    """

    def __init__(self, bandwidth=0.15, context_window=8):
        self.bandwidth = bandwidth
        self.context_window = context_window
        self._spectrum_cache = {}

    def get_word_spectrum(self, word):
        """الطيف الترددي للكلمة = متجهها الطوري 22D (كل بُعد قناة ترددية)."""
        if word not in self._spectrum_cache:
            pv = compute_word_phase_vector(word)
            spectrum = np.abs(pv)
            self._spectrum_cache[word] = spectrum
        return self._spectrum_cache[word]

    def compute_sidebands(self, carrier_word, context_words):
        """النطاقات الجانبية الطيفية — 22 قناة × عدد كلمات السياق.

        لكل كلمة سياق ولكل قناة ترددية:
          f_plus[ch]  = f_carrier[ch] + f_context[ch]
          f_minus[ch] = abs(f_carrier[ch] - f_context[ch])

        Returns:
            list of (ctx_word, f_plus_array, f_minus_array, weight)
        """
        f_carrier = self.get_word_spectrum(carrier_word)
        sidebands = []
        ctx = context_words[-self.context_window:] if context_words else []

        for rank, ctx_word in enumerate(reversed(ctx), 1):
            if ctx_word == carrier_word:
                continue
            f_ctx = self.get_word_spectrum(ctx_word)
            f_plus = f_carrier + f_ctx
            f_minus = np.abs(f_carrier - f_ctx)
            weight = 1.0 / rank
            sidebands.append((ctx_word, f_plus, f_minus, weight))

        return sidebands

    def compute_candidate_resonance(self, candidate_word, candidate_spectrum, sidebands):
        """رنين طيفي: كم قناة ترددية من المرشح تقع ضمن نطاقات التغاير؟

        لكل قناة ترددية ch:
          - تحقق هل f_cand[ch] قريب من f_plus[ch] أو f_minus[ch]
          - القرب يُقاس نسبةً إلى عرض النطاق (bandwidth)
          - الرنين = متوسط التطابق عبر كل القنوات

        Returns:
            float: درجة الرنين [0, 1]
        """
        if not sidebands:
            return 0.0

        total_resonance = 0.0
        total_weight = 0.0

        for ctx_word, f_plus, f_minus, weight in sidebands:
            channel_resonance = np.zeros(22)
            for ch in range(22):
                dist_plus = abs(candidate_spectrum[ch] - f_plus[ch]) / max(f_plus[ch], 1e-10)
                dist_minus = abs(candidate_spectrum[ch] - f_minus[ch]) / max(f_minus[ch], 1e-10)
                closest = min(dist_plus, dist_minus)
                if closest < self.bandwidth:
                    channel_resonance[ch] = 1.0 - (closest / self.bandwidth)

            avg_channel_resonance = float(np.mean(channel_resonance))
            total_resonance += avg_channel_resonance * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0
        return total_resonance / total_weight

    def score_candidate(self, candidate_word, context_words):
        """تسجيل مرشح: طيف المرشح vs نطاقات تغاير السياق."""
        if not context_words or len(context_words) < 1:
            return 0.0

        candidate_spectrum = self.get_word_spectrum(candidate_word)
        sidebands = self.compute_sidebands(candidate_word, context_words)
        return self.compute_candidate_resonance(candidate_word, candidate_spectrum, sidebands)

    def compute_context_signature(self, context_words):
        """بصمة ترددية للسياق: مصفوفة (n_ctx, 44) — لكل كلمة: [f_plus_mean, f_minus_mean]."""
        if not context_words:
            return np.zeros((0, 44))

        n = min(len(context_words), self.context_window)
        ctx = context_words[-n:]
        signature = np.zeros((n, 44))

        for i, word in enumerate(ctx):
            sidebands = self.compute_sidebands(word, ctx[:i] + ctx[i + 1:])
            if sidebands:
                plus_vals = np.array([s[1] for s in sidebands])
                minus_vals = np.array([s[2] for s in sidebands])
                signature[i, :22] = np.mean(plus_vals, axis=0)
                signature[i, 22:] = np.mean(minus_vals, axis=0)
            else:
                spec = self.get_word_spectrum(word)
                signature[i, :22] = spec
                signature[i, 22:] = spec

        return signature

    def clear_cache(self):
        self._spectrum_cache.clear()
