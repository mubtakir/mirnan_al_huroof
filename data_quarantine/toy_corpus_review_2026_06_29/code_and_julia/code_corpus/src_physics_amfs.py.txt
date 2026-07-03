"""AMFS — Adaptive Mass & Frequency Shift.

تعديل الكتلة والتردد الذاتي للكلمة ديناميكياً حسب السياق.
محاكاة فيزيائية للـ contextual embeddings.
"""

import numpy as np
from src.physics.word_physics import compute_word_mass, compute_word_frequency, compute_extended_phase_vector
from src.physics.constants import TOTAL_DIM


def adapt_word(word, context_words=None, context_pvs=None):
    """تعديل كتلة وتردد الكلمة حسب السياق المحيط.

    Args:
        word: str — الكلمة
        context_words: list[str] — كلمات السياق
        context_pvs: list[np.ndarray] — متجهات الطور للسياق

    Returns:
        dict: {mass, freq, phase_shift, centrality}
    """
    base_mass = compute_word_mass(word)
    base_freq = compute_word_frequency(word)
    w_pv = compute_extended_phase_vector(word)

    if not context_pvs and context_words:
        context_pvs = [compute_extended_phase_vector(w) for w in context_words]
        context_pvs = [p for p in context_pvs if p is not None]

    if not context_pvs or len(context_pvs) < 1:
        return {'mass': base_mass, 'freq': base_freq, 'phase_shift': 0.0, 'centrality': 0.0}

    w_norm = np.linalg.norm(w_pv)
    if w_norm < 1e-10:
        return {'mass': base_mass, 'freq': base_freq, 'phase_shift': 0.0, 'centrality': 0.0}

    # المركزية النحوية: متوسط التوافق الطوري مع كل كلمات السياق
    aligns = []
    for cpv in context_pvs:
        c_norm = np.linalg.norm(cpv)
        if c_norm > 1e-10:
            aligns.append(float(np.dot(w_pv, cpv)) / (w_norm * c_norm))

    centrality = float(np.mean(aligns)) if aligns else 0.0

    # الكتلة المعدّلة: الكلمات ذات الدور المركزي تكتسب كتلة إضافية
    adapted_mass = base_mass * (1.0 + 0.3 * max(0.0, centrality))

    # الانزياح الطوري
    ctx_mean = np.mean(context_pvs, axis=0)
    ctx_norm = np.linalg.norm(ctx_mean)
    phase_shift = 0.0
    if ctx_norm > 1e-10:
        phase_shift = float(np.dot(w_pv, ctx_mean)) / (w_norm * ctx_norm)

    # التردد المعدّل: الانزياح الطوري يزيح التردد
    adapted_freq = base_freq * (1.0 + 0.15 * np.tanh(phase_shift))

    return {
        'mass': adapted_mass,
        'freq': adapted_freq,
        'phase_shift': phase_shift,
        'centrality': centrality,
    }
