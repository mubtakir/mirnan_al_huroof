"""Ψ_ω — Wave Spectrum Vector للتحليل الطيفي للكلمات.

يحول كل كلمة إلى حزمة موجية عبر FFT على متجهات حروفها.
الترددات العالية ← معنى مجرد/فلسفي، المنخفضة ← حسي/مادي.
التداخل البنّاء بين طيفين = رنين دلالي.
"""
import numpy as np
from src.physics.letter_db import LetterDB
from src.physics.word_physics import _normalize_letters

_letter_db = LetterDB()
_SPECTRAL_DIM = 6


def _get_letter_phase(letter):
    vec = _letter_db.get_vector(letter)
    return np.asarray(vec, dtype=np.float64) if vec is not None else np.zeros(22, dtype=np.float64)


def compute_word_spectrum(word):
    """حساب الطيف الترددي لكلمة.
    
    Returns dict مع: frequencies, amplitudes, phases, spectral_vector
    """
    normalized = _normalize_letters(word)
    if not normalized:
        return None
    letters = list(normalized)
    if len(letters) == 1:
        letters = letters * 2
    n = max(len(letters), 3)
    if len(letters) < n:
        letters = letters + [letters[-1]] * (n - len(letters))
    
    pvs = np.array([_get_letter_phase(l) for l in letters])
    fft_vals = np.fft.rfft(pvs, axis=0)
    magnitudes = np.abs(fft_vals)
    phases = np.angle(fft_vals)
    
    freqs = np.fft.rfftfreq(n)
    n_components = min(_SPECTRAL_DIM, len(freqs))
    
    top_indices = np.argsort(np.mean(magnitudes, axis=1))[-n_components:][::-1]
    
    spectral_vector = np.concatenate([
        freqs[top_indices],
        np.mean(magnitudes[top_indices], axis=1),
        np.mean(phases[top_indices], axis=1),
    ])
    
    return {
        'frequencies': freqs[top_indices].tolist(),
        'amplitudes': np.mean(magnitudes[top_indices], axis=1).tolist(),
        'phases': np.mean(phases[top_indices], axis=1).tolist(),
        'spectral_vector': spectral_vector,
    }


def spectral_resonance(word1, word2):
    """قياس الرنين الطيفي بين كلمتين (0.0 - 1.0).
    
    يعتمد على تشابه توزيع الترددات بعد تطبيع السعات.
    التداخل البنّاء = تشابه في الترددات المهيمنة.
    """
    s1 = compute_word_spectrum(word1)
    s2 = compute_word_spectrum(word2)
    if s1 is None or s2 is None:
        return 0.0
    v1 = s1['spectral_vector']
    v2 = s2['spectral_vector']
    dim = min(len(v1), len(v2))
    if dim == 0:
        return 0.0
    cos_sim = float(np.dot(v1[:dim], v2[:dim]) / (np.linalg.norm(v1[:dim]) * np.linalg.norm(v2[:dim]) + 1e-10))
    return max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))


def spectral_density(words):
    """كثافة الطيف — قياس التنوع الطيفي في مجموعة كلمات.
    
    كلما تنوعت الترددات ← أغنى دلالياً.
    """
    if not words or len(words) < 2:
        return 0.0
    spectra = []
    for w in words:
        s = compute_word_spectrum(w)
        if s is not None:
            spectra.append(s['spectral_vector'])
    if len(spectra) < 2:
        return 0.0
    sims = []
    for i in range(len(spectra)):
        for j in range(i + 1, len(spectra)):
            dim = min(len(spectra[i]), len(spectra[j]))
            if dim == 0:
                continue
            cs = float(np.dot(spectra[i][:dim], spectra[j][:dim]) / 
                       (np.linalg.norm(spectra[i][:dim]) * np.linalg.norm(spectra[j][:dim]) + 1e-10))
            sims.append(cs)
    if not sims:
        return 0.0
    return 1.0 - float(np.mean(sims))
