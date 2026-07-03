#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spectral_context.py — طبقة طيفية سياقية.

لكل كلمة، نحسب CS(w) = Σ_v K[w,v] × PV(v) / Σ_v K[w,v]
أي: متوسط المتجهات الطورية للكلمات التي تشاركها الظهور، مرجحاً بقوة الارتباط.

هذا يعطي "بصمة حقل دلالي" لكل كلمة — كلمتان في نفس الحقل (ملك، سلطة)
لها CS متقاربان حتى لو اختلفت حروفهما.
"""
import numpy as np
from scipy import sparse
from typing import Optional


def build_contextual_spectra(K, all_pv, normalize=True):
    """بناء الأطياف السياقية من مصفوفة K.

    Args:
        K: مصفوفة الارتباط (V×V, sparse CSR)
        all_pv: مصفوفة المتجهات الطورية (V×D, dense)
        normalize: تطبيع المتجهات الناتجة إلى unit length

    Returns:
        spectra: (V×D) — كل صف = طيف سياقي لكلمة
    """
    V = K.shape[0]
    D = all_pv.shape[1]
    
    # تطبيع صفوف K: كل صف sum=1
    row_sums = np.array(K.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1  # تجنب القسمة على صفر
    K_norm = K.multiply(1.0 / row_sums[:, np.newaxis])
    
    # CS = K_normalized @ PV
    spectra = K_norm @ all_pv  # (V, V) × (V, D) → (V, D)
    
    if normalize:
        norms = np.linalg.norm(spectra, axis=1, keepdims=True)
        norms[norms == 0] = 1
        spectra = spectra / norms
    
    return spectra


def context_spectral_score(word_id, context_ids, spectra):
    """درجة التوافق الطيفي السياقي.

    تحسب cos(CS(context) - CS(candidate)):
    - إذا كان للكلمة نفس الحقل الدلالي للسياق → درجة عالية
    - إذا اختلف الحقل → درجة منخفضة
    """
    if word_id is None or not context_ids:
        return 0.0
    
    # متوسط الطيف السياقي للسياق
    ctx_sum = np.zeros(spectra.shape[1])
    n = 0
    for cid in context_ids[-3:]:  # آخر 3 كلمات
        if cid is not None and cid < spectra.shape[0]:
            ctx_sum += spectra[cid]
            n += 1
    
    if n == 0:
        return 0.0
    
    ctx_spectrum = ctx_sum / n
    ctx_norm = np.linalg.norm(ctx_spectrum)
    if ctx_norm < 1e-10:
        return 0.0
    ctx_spectrum = ctx_spectrum / ctx_norm
    
    # تشابه مع مرشح
    if word_id >= spectra.shape[0]:
        return 0.0
    cand_spectrum = spectra[word_id]
    cand_norm = np.linalg.norm(cand_spectrum)
    if cand_norm < 1e-10:
        return 0.0
    
    cos_sim = float(np.dot(ctx_spectrum, cand_spectrum) / cand_norm)
    return cos_sim
